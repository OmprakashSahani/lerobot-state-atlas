from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from lerobot_state_atlas.checkpoint_comparison.checkpoints import (
    BASE_TIED_EMBEDDING,
    BASE_TIED_HEAD,
    BASE_UNUSED_EXPERT_HEAD,
    CheckpointVerificationError,
    load_verified_checkpoint,
    verify_checkpoint_for_module,
)


class TinyModule(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.model = torch.nn.Linear(2, 2, dtype=dtype)


class TimeModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.time_mlp_in = torch.nn.Linear(2, 2)
        self.model.time_mlp_out = torch.nn.Linear(2, 2)


class TiedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.paligemma_with_expert = torch.nn.Module()
        paligemma = torch.nn.Module()
        self.model.paligemma_with_expert.paligemma = paligemma
        paligemma.model = torch.nn.Module()
        paligemma.model.language_model = torch.nn.Module()
        paligemma.model.language_model.embed_tokens = torch.nn.Embedding(2, 3)


def save(path: Path, tensors: dict[str, torch.Tensor], metadata=None) -> bytes:
    save_file(tensors, path, metadata=metadata)
    return path.read_bytes()


def state_bytes(module: torch.nn.Module) -> dict[str, bytes]:
    return {
        key: value.detach().cpu().contiguous().numpy().tobytes()
        for key, value in module.state_dict().items()
    }


def test_valid_finetuned_strict_load_and_checkpoint_unchanged(tmp_path: Path) -> None:
    module = TinyModule()
    path = tmp_path / "fine.safetensors"
    checkpoint_bytes = save(
        path,
        {"model.weight": torch.full((2, 2), 3.0), "model.bias": torch.full((2,), 4.0)},
    )
    report = load_verified_checkpoint(module, path, "fine-tuned")
    assert report.verified is True
    assert report.loaded_tensor_count == 2
    assert report.checkpoint_sha256 == hashlib.sha256(checkpoint_bytes).hexdigest()
    assert torch.equal(module.model.weight, torch.full((2, 2), 3.0))
    assert torch.equal(module.model.bias, torch.full((2,), 4.0))
    assert path.read_bytes() == checkpoint_bytes


def test_load_uses_one_immutable_snapshot_even_if_path_is_replaced(
    tmp_path: Path, monkeypatch
) -> None:
    module = TinyModule()
    path = tmp_path / "fine.safetensors"
    original_bytes = save(
        path,
        {
            "model.weight": torch.full((2, 2), 3.0),
            "model.bias": torch.full((2,), 4.0),
        },
    )
    import lerobot_state_atlas.checkpoint_comparison.checkpoints as checkpoints

    real_snapshot = checkpoints._read_checkpoint_snapshot
    snapshot_calls = 0

    def snapshot_then_replace(snapshot_path: Path) -> bytes:
        nonlocal snapshot_calls
        snapshot_calls += 1
        snapshot = real_snapshot(snapshot_path)
        save_file(
            {
                "model.weight": torch.full((2, 2), 30.0),
                "model.bias": torch.full((2,), 40.0),
            },
            snapshot_path,
        )
        return snapshot

    monkeypatch.setattr(checkpoints, "_read_checkpoint_snapshot", snapshot_then_replace)
    report = load_verified_checkpoint(module, path, "fine-tuned")

    assert snapshot_calls == 1
    assert report.checkpoint_sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert torch.equal(module.model.weight, torch.full((2, 2), 3.0))
    assert torch.equal(module.model.bias, torch.full((2,), 4.0))
    assert path.read_bytes() != original_bytes


def test_metadata_and_tensor_loading_receive_identical_snapshot_object(
    tmp_path: Path, monkeypatch
) -> None:
    module = TinyModule()
    path = tmp_path / "fine.safetensors"
    checkpoint_bytes = save(
        path,
        {"model.weight": torch.ones(2, 2), "model.bias": torch.ones(2)},
    )
    import lerobot_state_atlas.checkpoint_comparison.checkpoints as checkpoints

    real_inspect = checkpoints._inspect_snapshot
    real_load = checkpoints.load_bytes
    observed_ids: list[int] = []

    def inspect_snapshot(snapshot: bytes):
        observed_ids.append(id(snapshot))
        return real_inspect(snapshot)

    def load_snapshot(snapshot: bytes):
        observed_ids.append(id(snapshot))
        assert (
            hashlib.sha256(snapshot).hexdigest()
            == hashlib.sha256(checkpoint_bytes).hexdigest()
        )
        return real_load(snapshot)

    monkeypatch.setattr(checkpoints, "_inspect_snapshot", inspect_snapshot)
    monkeypatch.setattr(checkpoints, "load_bytes", load_snapshot)
    load_verified_checkpoint(module, path, "fine-tuned")
    assert len(observed_ids) == 2
    assert observed_ids[0] == observed_ids[1]


def test_valid_base_prefix_reconciliation_and_dtype_cast(tmp_path: Path) -> None:
    module = TinyModule(dtype=torch.float32)
    path = tmp_path / "base.safetensors"
    save(
        path,
        {
            "weight": torch.ones(2, 2, dtype=torch.float16),
            "bias": torch.ones(2, dtype=torch.float16),
        },
    )
    report = load_verified_checkpoint(module, path, "base")
    assert report.source_dtype_counts == (("F16", 2),)
    assert module.model.weight.dtype == torch.float32
    assert all(item.transformation == "add-model-prefix" for item in report.tensors)


def test_action_time_mlp_renames(tmp_path: Path) -> None:
    module = TimeModule()
    tensors = {}
    for key, value in module.state_dict().items():
        source = (
            key.removeprefix("model.")
            .replace("time_mlp_in.", "action_time_mlp_in.")
            .replace("time_mlp_out.", "action_time_mlp_out.")
        )
        tensors[source] = torch.full_like(value, 2)
    path = tmp_path / "base.safetensors"
    save(path, tensors)
    report = load_verified_checkpoint(module, path, "base")
    assert report.verified
    assert all(
        "rename-action-time-mlp" in item.transformation for item in report.tensors
    )


def test_tied_paligemma_mapping_requires_verified_evidence(tmp_path: Path) -> None:
    module = TiedModule()
    path = tmp_path / "base.safetensors"
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save(path, {BASE_TIED_HEAD: tensor})
    report = verify_checkpoint_for_module(module, path, "base")
    assert not report.verified
    assert BASE_TIED_HEAD in report.unexpected_keys
    with pytest.raises(CheckpointVerificationError, match="unexpected"):
        load_verified_checkpoint(module, path, "base")

    save(path, {BASE_TIED_HEAD: tensor}, {BASE_TIED_EMBEDDING: BASE_TIED_HEAD})
    report = load_verified_checkpoint(module, path, "base")
    assert report.verified
    assert report.tensors[0].transformation == "verified-tied-head-to-embedding"
    assert torch.equal(
        module.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight,
        tensor,
    )


def test_explicit_tied_alias_contract_is_supported(tmp_path: Path) -> None:
    module = TiedModule()
    path = tmp_path / "base.safetensors"
    save(path, {BASE_TIED_HEAD: torch.ones(2, 3)})
    report = load_verified_checkpoint(
        module,
        path,
        "base",
        tied_alias_contract={BASE_TIED_EMBEDDING: BASE_TIED_HEAD},
    )
    assert report.verified


def test_known_unused_expert_head_drop_requires_policy(tmp_path: Path) -> None:
    module = TinyModule()
    path = tmp_path / "base.safetensors"
    tensors = {
        "weight": torch.ones(2, 2),
        "bias": torch.ones(2),
        BASE_UNUSED_EXPERT_HEAD: torch.ones(3, 3),
    }
    save(path, tensors)
    rejected = verify_checkpoint_for_module(module, path, "base")
    assert not rejected.verified
    assert BASE_UNUSED_EXPERT_HEAD in rejected.unexpected_keys
    accepted = load_verified_checkpoint(module, path, "base", drop_unused_lm_head=True)
    assert accepted.explicitly_dropped_keys == (BASE_UNUSED_EXPERT_HEAD,)


@pytest.mark.parametrize("case", ["missing", "unexpected", "shape"])
def test_verification_failures_leave_module_unchanged(
    tmp_path: Path, case: str
) -> None:
    module = TinyModule()
    before = state_bytes(module)
    tensors = {"model.weight": torch.ones(2, 2), "model.bias": torch.ones(2)}
    if case == "missing":
        tensors.pop("model.bias")
    elif case == "unexpected":
        tensors["model.extra"] = torch.ones(1)
    else:
        tensors["model.weight"] = torch.ones(3, 2)
    path = tmp_path / "fine.safetensors"
    save(path, tensors)
    report = verify_checkpoint_for_module(module, path, "fine-tuned")
    assert not report.verified
    with pytest.raises(CheckpointVerificationError, match="verification failed"):
        load_verified_checkpoint(module, path, "fine-tuned")
    assert state_bytes(module) == before


def test_duplicate_target_collision_is_reported(tmp_path: Path) -> None:
    module = torch.nn.Module()
    module.model = torch.nn.Module()
    module.model.time_mlp_in = torch.nn.Linear(2, 2, bias=False)
    path = tmp_path / "base.safetensors"
    save(
        path,
        {
            "time_mlp_in.weight": torch.ones(2, 2),
            "action_time_mlp_in.weight": torch.ones(2, 2),
        },
    )
    report = verify_checkpoint_for_module(module, path, "base")
    assert not report.verified
    assert report.duplicate_target_collisions


def test_state_proj_is_rejected_precisely(tmp_path: Path) -> None:
    module = TinyModule()
    path = tmp_path / "base.safetensors"
    save(
        path,
        {
            "weight": torch.ones(2, 2),
            "bias": torch.ones(2),
            "state_proj.weight": torch.ones(1),
        },
    )
    report = verify_checkpoint_for_module(module, path, "base")
    assert "state_proj.weight" in report.unexpected_keys


@pytest.mark.parametrize("kind", ["malformed", "symlink", "directory"])
def test_rejects_unsafe_or_malformed_checkpoint_path(tmp_path: Path, kind: str) -> None:
    module = TinyModule()
    path = tmp_path / "checkpoint.safetensors"
    if kind == "malformed":
        path.write_bytes(b"not safetensors")
    elif kind == "symlink":
        target = tmp_path / "target.safetensors"
        save(target, {"model.weight": torch.ones(2, 2), "model.bias": torch.ones(2)})
        path.symlink_to(target)
    else:
        path.mkdir()
    with pytest.raises(
        CheckpointVerificationError, match="checkpointPath|valid SafeTensors"
    ):
        verify_checkpoint_for_module(module, path, "fine-tuned")


def test_malformed_snapshot_fails_before_module_mutation(tmp_path: Path) -> None:
    module = TinyModule()
    before = state_bytes(module)
    path = tmp_path / "malformed.safetensors"
    path.write_bytes(b"not a safetensors snapshot")
    with pytest.raises(CheckpointVerificationError, match="snapshot.*not valid"):
        load_verified_checkpoint(module, path, "fine-tuned")
    assert state_bytes(module) == before


def test_nonfinite_tensor_is_rejected_before_mutation(tmp_path: Path) -> None:
    module = TinyModule()
    before = state_bytes(module)
    path = tmp_path / "fine.safetensors"
    save(
        path,
        {
            "model.weight": torch.tensor([[float("nan"), 0], [0, 0]]),
            "model.bias": torch.ones(2),
        },
    )
    with pytest.raises(CheckpointVerificationError, match="non-finite"):
        load_verified_checkpoint(module, path, "fine-tuned")
    assert state_bytes(module) == before


def test_unexpected_load_failure_restores_original_state(tmp_path: Path) -> None:
    class FailOnceModule(TinyModule):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def load_state_dict(self, state_dict, strict=True, assign=False):
            self.calls += 1
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
            if self.calls == 1:
                raise RuntimeError("simulated post-mutation failure")
            return result

    module = FailOnceModule()
    before = state_bytes(module)
    path = tmp_path / "fine.safetensors"
    save(
        path,
        {"model.weight": torch.full((2, 2), 9.0), "model.bias": torch.full((2,), 9.0)},
    )
    with pytest.raises(
        CheckpointVerificationError, match="original module state was restored"
    ):
        load_verified_checkpoint(module, path, "fine-tuned")
    assert state_bytes(module) == before


def test_verification_report_is_immutable(tmp_path: Path) -> None:
    module = TinyModule()
    path = tmp_path / "fine.safetensors"
    save(path, {"model.weight": torch.ones(2, 2), "model.bias": torch.ones(2)})
    report = verify_checkpoint_for_module(module, path, "fine-tuned")
    with pytest.raises(FrozenInstanceError):
        report.verified = False
