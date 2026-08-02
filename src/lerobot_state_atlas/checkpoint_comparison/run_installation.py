"""Transactional installation of a complete checkpoint-comparison run directory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
from uuid import uuid4

from lerobot_state_atlas.checkpoint_comparison.artifact import (
    build_checkpoint_comparison_documents,
)
from lerobot_state_atlas.checkpoint_comparison.receipt import (
    build_checkpoint_comparison_run_receipt,
    load_checkpoint_comparison_run_receipt,
)
from lerobot_state_atlas.checkpoint_comparison.serialize import sha256_bytes
from lerobot_state_atlas.checkpoint_comparison.validate import (
    validate_checkpoint_comparison,
)


class CheckpointComparisonRunInstallationError(RuntimeError):
    """Raised when a complete run cannot be installed transactionally."""


@dataclass(frozen=True)
class CheckpointComparisonRunInstallation:
    run_directory: Path
    comparison_directory: Path
    receipt_path: Path
    manifest_sha256: str
    plans_sha256: str
    receipt_sha256: str


class _Phase(Enum):
    STAGING = auto()
    OLD_MOVED_INTACT = auto()
    NEW_INSTALLED = auto()
    BACKUP_CLEANUP_STARTED = auto()
    COMPLETE = auto()


def _reject_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError(
            "Checkpoint-comparison run destination must not be a symbolic link."
        )
    if destination.exists() and not destination.is_dir():
        raise ValueError(
            "Checkpoint-comparison run destination must be a directory path."
        )
    resolved = Path(os.path.abspath(destination))
    if resolved.name in {"demo-v1", "demo-v2"} or any(
        part in {"demo-v1", "demo-v2"} for part in resolved.parts
    ):
        raise ValueError(
            "Immutable demo-v1 and demo-v2 destinations cannot be replaced."
        )
    current = resolved.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(
                f"Checkpoint-comparison run destination parent must not be a symbolic link: {current}"
            )
        current = current.parent


def install_checkpoint_comparison_run(
    destination: str | Path,
    manifest: Mapping[str, Any],
    plans: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    replace_existing: bool,
) -> CheckpointComparisonRunInstallation:
    """Stage, validate, and atomically install the complete outer run directory."""
    destination = Path(destination)
    _reject_destination(destination)
    destination = Path(os.path.abspath(destination))
    if destination.exists() and not replace_existing:
        raise ValueError(
            "Checkpoint-comparison run destination exists and replaceExisting is false."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes, plans_bytes = build_checkpoint_comparison_documents(manifest, plans)
    receipt_bytes = build_checkpoint_comparison_run_receipt(receipt)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.run-", dir=destination.parent)
    )
    backup: Path | None = None
    phase = _Phase.STAGING
    try:
        comparison = staging / "comparison"
        comparison.mkdir()
        (comparison / "manifest.json").write_bytes(manifest_bytes)
        (comparison / "plans.json").write_bytes(plans_bytes)
        (staging / "run-receipt.json").write_bytes(receipt_bytes)
        validate_checkpoint_comparison(comparison)
        load_checkpoint_comparison_run_receipt(staging / "run-receipt.json")
        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.previous-{uuid4().hex}"
            )
            os.replace(destination, backup)
            phase = _Phase.OLD_MOVED_INTACT
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup is not None:
                os.replace(backup, destination)
                backup = None
            raise
        phase = _Phase.NEW_INSTALLED
        if backup is not None:
            phase = _Phase.BACKUP_CLEANUP_STARTED
            try:
                shutil.rmtree(backup)
            except BaseException as error:
                remaining = (
                    f" Remaining backup material is at {backup} and may be partial; "
                    "do not restore it blindly."
                    if backup.exists()
                    else " Backup cleanup raised after the backup disappeared."
                )
                raise CheckpointComparisonRunInstallationError(
                    f"Backup cleanup failed after the validated new run was installed. "
                    f"The new run remains installed at {destination}.{remaining} "
                    f"Cleanup error: {error!r}"
                ) from error
            backup = None
        phase = _Phase.COMPLETE
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if phase is _Phase.OLD_MOVED_INTACT and backup is not None:
            os.replace(backup, destination)
        raise
    return CheckpointComparisonRunInstallation(
        run_directory=destination,
        comparison_directory=destination / "comparison",
        receipt_path=destination / "run-receipt.json",
        manifest_sha256=sha256_bytes(manifest_bytes),
        plans_sha256=sha256_bytes(plans_bytes),
        receipt_sha256=sha256_bytes(receipt_bytes),
    )
