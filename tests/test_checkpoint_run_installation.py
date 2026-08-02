import json
from pathlib import Path

import pytest

from lerobot_state_atlas.checkpoint_comparison import (
    CheckpointComparisonRunInstallationError,
    install_checkpoint_comparison_run,
    validate_checkpoint_comparison,
)
from test_checkpoint_receipt import receipt


def documents() -> tuple[dict, dict]:
    root = Path(__file__).parent / "fixtures/checkpoint-comparison-v1.1-unavailable"
    return json.loads((root / "manifest.json").read_text()), json.loads(
        (root / "plans.json").read_text()
    )


def test_complete_run_installation_and_replacement(tmp_path: Path) -> None:
    manifest, plans = documents()
    destination = tmp_path / "run"
    result = install_checkpoint_comparison_run(
        destination, manifest, plans, receipt(), replace_existing=False
    )
    assert result.receipt_path.is_file()
    validate_checkpoint_comparison(result.comparison_directory)
    (destination / "old.txt").write_text("old")
    result = install_checkpoint_comparison_run(
        destination, manifest, plans, receipt(), replace_existing=True
    )
    assert not (destination / "old.txt").exists()
    assert not list(tmp_path.glob(".run.previous-*"))


def test_cleanup_failure_preserves_new_valid_run_and_partial_backup(
    monkeypatch, tmp_path: Path
) -> None:
    manifest, plans = documents()
    destination = tmp_path / "run"
    install_checkpoint_comparison_run(
        destination, manifest, plans, receipt(), replace_existing=False
    )
    (destination / "old.txt").write_text("old")
    from lerobot_state_atlas.checkpoint_comparison import run_installation

    original = run_installation.shutil.rmtree

    def partial(path, *args, **kwargs):
        path = Path(path)
        if ".previous-" in path.name:
            (path / "old.txt").unlink()
            raise OSError("injected partial cleanup")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(run_installation.shutil, "rmtree", partial)
    with pytest.raises(
        CheckpointComparisonRunInstallationError, match="may be partial"
    ):
        install_checkpoint_comparison_run(
            destination, manifest, plans, receipt(), replace_existing=True
        )
    validate_checkpoint_comparison(destination / "comparison")
    assert (destination / "run-receipt.json").is_file()
    assert len(list(tmp_path.glob(".run.previous-*"))) == 1


def test_rejects_file_symlink_and_immutable_demo_destinations(tmp_path: Path) -> None:
    manifest, plans = documents()
    destination = tmp_path / "file"
    destination.write_text("keep")
    with pytest.raises(ValueError, match="directory path"):
        install_checkpoint_comparison_run(
            destination, manifest, plans, receipt(), replace_existing=True
        )
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        install_checkpoint_comparison_run(
            link, manifest, plans, receipt(), replace_existing=True
        )
    with pytest.raises(ValueError, match="Immutable"):
        install_checkpoint_comparison_run(
            tmp_path / "demo-v1", manifest, plans, receipt(), replace_existing=False
        )
