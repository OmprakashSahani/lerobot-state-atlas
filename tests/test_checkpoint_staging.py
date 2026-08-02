import hashlib
from pathlib import Path
import stat

import pytest
from safetensors.torch import save_file
import torch

from lerobot_state_atlas.checkpoint_comparison import (
    CheckpointStagingError,
    CheckpointVerificationError,
    load_staged_checkpoint_into_fresh_module,
    stage_runner_checkpoint,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2, 2)


def checkpoint(tmp_path: Path, kind="fine-tuned"):
    path = tmp_path / f"{kind}.safetensors"
    tensors = {
        "model.weight" if kind == "fine-tuned" else "weight": torch.arange(
            4, dtype=torch.float32
        ).reshape(2, 2),
        "model.bias" if kind == "fine-tuned" else "bias": torch.tensor([5.0, 6.0]),
    }
    save_file(tensors, path)
    data = path.read_bytes()
    return path, data


def test_secure_stage_and_streaming_fresh_module_load(tmp_path: Path) -> None:
    source, data = checkpoint(tmp_path)
    before = source.read_bytes()
    staged = stage_runner_checkpoint(
        source,
        expected_byte_count=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        staging_parent=tmp_path,
        checkpoint_kind="fine-tuned",
    )
    assert stat.S_IMODE(staged.staged_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(staged.staged_path.stat().st_mode) == 0o400
    module = Tiny()
    report = load_staged_checkpoint_into_fresh_module(module, staged, "fine-tuned")
    assert report.verified
    assert torch.equal(
        module.model.weight, torch.arange(4, dtype=torch.float32).reshape(2, 2)
    )
    assert source.read_bytes() == before
    directory = staged.staged_directory
    staged.cleanup()
    assert not directory.exists()


def test_base_reconciliation(tmp_path: Path) -> None:
    source, data = checkpoint(tmp_path, "base")
    with stage_runner_checkpoint(
        source,
        expected_byte_count=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        staging_parent=tmp_path,
        checkpoint_kind="base",
    ) as staged:
        report = load_staged_checkpoint_into_fresh_module(Tiny(), staged, "base")
        assert report.verified
        assert any("add-model-prefix" in value for value in report.transformations)


@pytest.mark.parametrize("failure", ["count", "hash"])
def test_staging_integrity_failure_leaves_no_stage(
    tmp_path: Path, failure: str
) -> None:
    source, data = checkpoint(tmp_path)
    with pytest.raises(CheckpointStagingError, match="mismatch"):
        stage_runner_checkpoint(
            source,
            expected_byte_count=len(data) + (1 if failure == "count" else 0),
            expected_sha256=(
                "0" * 64 if failure == "hash" else hashlib.sha256(data).hexdigest()
            ),
            staging_parent=tmp_path,
            checkpoint_kind="fine-tuned",
        )
    assert not list(tmp_path.glob(".checkpoint-stage-*"))


def test_symlink_directory_and_source_unchanged(tmp_path: Path) -> None:
    source, data = checkpoint(tmp_path)
    target = tmp_path / "target.safetensors"
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(CheckpointStagingError, match="non-symlink"):
        stage_runner_checkpoint(
            source,
            expected_byte_count=len(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            staging_parent=tmp_path,
            checkpoint_kind="fine-tuned",
        )


def test_staged_path_tampering_does_not_redirect_retained_descriptor(
    tmp_path: Path,
) -> None:
    source, data = checkpoint(tmp_path)
    staged = stage_runner_checkpoint(
        source,
        expected_byte_count=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        staging_parent=tmp_path,
        checkpoint_kind="fine-tuned",
    )
    staged.staged_path.chmod(0o600)
    staged.staged_path.unlink()
    staged.staged_path.write_bytes(b"replacement")
    report = load_staged_checkpoint_into_fresh_module(Tiny(), staged, "fine-tuned")
    assert report.verified
    staged.cleanup()


def test_malformed_and_nonfinite_fail_before_module_mutation(tmp_path: Path) -> None:
    source = tmp_path / "bad.safetensors"
    source.write_bytes(b"bad")
    data = source.read_bytes()
    with stage_runner_checkpoint(
        source,
        expected_byte_count=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        staging_parent=tmp_path,
        checkpoint_kind="fine-tuned",
    ) as staged:
        module = Tiny()
        before = {key: value.clone() for key, value in module.state_dict().items()}
        with pytest.raises(CheckpointVerificationError, match="malformed"):
            load_staged_checkpoint_into_fresh_module(module, staged, "fine-tuned")
        assert all(
            torch.equal(value, before[key])
            for key, value in module.state_dict().items()
        )


def test_cleanup_failure_preserves_recoverable_path(
    tmp_path: Path, monkeypatch
) -> None:
    source, data = checkpoint(tmp_path)
    staged = stage_runner_checkpoint(
        source,
        expected_byte_count=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        staging_parent=tmp_path,
        checkpoint_kind="fine-tuned",
    )
    import lerobot_state_atlas.checkpoint_comparison.checkpoint_staging as module

    monkeypatch.setattr(
        module.shutil, "rmtree", lambda path: (_ for _ in ()).throw(OSError("busy"))
    )
    with pytest.raises(CheckpointStagingError, match="recoverable path"):
        staged.cleanup()
    assert staged.staged_directory.exists()
