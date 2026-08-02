"""Secure runner-owned staging and bounded-memory SafeTensors loading."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Mapping
from uuid import uuid4

from safetensors import SafetensorError, safe_open
import torch

from lerobot_state_atlas.checkpoint_comparison.checkpoints import (
    BASE_UNUSED_EXPERT_HEAD,
    CheckpointKind,
    CheckpointVerificationError,
    _base_target_key,
)
from lerobot_state_atlas.checkpoint_comparison.models import (
    CheckpointTensorMetadata,
    CheckpointVerificationReport,
)


class CheckpointStagingError(RuntimeError):
    """Raised when a runner checkpoint cannot be staged or cleaned securely."""


# The runner creates two directories plus one staged file on this filesystem.
# Reserve sixteen 4 KiB allocation units for their inodes, directory entries,
# and filesystem bookkeeping; checkpoint content itself is accounted separately.
RUNNER_STAGING_METADATA_OVERHEAD_BYTES = 16 * 4096


def resolve_runner_staging_root(
    *,
    temp_directory: str | Path | None = None,
    usability_checker=None,
) -> Path:
    """Resolve the existing non-symlink filesystem root used for runner staging."""
    raw = Path(tempfile.gettempdir() if temp_directory is None else temp_directory)
    lexical = Path(os.path.abspath(raw))
    current = Path(lexical.anchor)
    try:
        for part in lexical.parts[1:]:
            current /= part
            if current.is_symlink():
                raise CheckpointStagingError(
                    f"system temporary directory contains symbolic-link path component {current}."
                )
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise CheckpointStagingError(
            f"system temporary directory does not exist: {lexical}."
        ) from error
    except OSError as error:
        raise CheckpointStagingError(
            f"system temporary directory could not be resolved safely: {lexical}: {error}"
        ) from error
    if not resolved.is_dir():
        raise CheckpointStagingError(
            f"system temporary directory must be an ordinary directory: {resolved}."
        )
    checker = usability_checker or (lambda path: os.access(path, os.W_OK | os.X_OK))
    if not checker(resolved):
        raise CheckpointStagingError(
            f"system temporary directory is not writable and searchable: {resolved}."
        )
    return resolved


@dataclass(frozen=True)
class StagedCheckpoint:
    checkpoint_kind: str
    source_path: Path
    staged_directory: Path
    staged_path: Path
    byte_count: int
    sha256: str
    stat_device: int
    stat_inode: int
    descriptor: int

    def __enter__(self) -> "StagedCheckpoint":
        return self

    def cleanup(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass
        try:
            shutil.rmtree(self.staged_directory)
        except OSError as error:
            raise CheckpointStagingError(
                f"Staged checkpoint cleanup failed; recoverable path remains at {self.staged_directory}: {error}"
            ) from error

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def stage_runner_checkpoint(
    source_path: str | Path,
    *,
    expected_byte_count: int,
    expected_sha256: str,
    staging_parent: str | Path,
    checkpoint_kind: str,
) -> StagedCheckpoint:
    """Copy once from a retained source descriptor into a private staged inode."""
    if checkpoint_kind not in {"base", "fine-tuned"}:
        raise CheckpointStagingError("checkpoint_kind must be 'base' or 'fine-tuned'.")
    if (
        isinstance(expected_byte_count, bool)
        or not isinstance(expected_byte_count, int)
        or expected_byte_count <= 0
    ):
        raise CheckpointStagingError("expected_byte_count must be a positive integer.")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise CheckpointStagingError(
            "expected_sha256 must be a lowercase SHA-256 digest."
        )
    source = Path(source_path)
    if source.is_symlink() or source.suffix != ".safetensors":
        raise CheckpointStagingError(
            "source checkpoint must be a non-symlink .safetensors file."
        )
    parent = Path(staging_parent)
    if parent.is_symlink() or not parent.is_dir():
        raise CheckpointStagingError(
            "staging_parent must be an existing non-symlink directory."
        )
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, source_flags)
    except OSError as error:
        raise CheckpointStagingError(
            f"Could not open source checkpoint safely: {error}"
        ) from error
    directory = Path(tempfile.mkdtemp(prefix=".checkpoint-stage-", dir=parent))
    os.chmod(directory, 0o700)
    staged_path = directory / f"{checkpoint_kind}-{uuid4().hex}.safetensors"
    staged_fd = -1
    retained_fd = -1
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointStagingError(
                "Source checkpoint descriptor is not a regular file."
            )
        staged_fd = os.open(staged_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        count = 0
        while True:
            chunk = os.read(source_fd, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            count += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(staged_fd, view)
                view = view[written:]
        after = os.fstat(source_fd)
        if _identity(before) != _identity(after):
            raise CheckpointStagingError("Source checkpoint changed during staging.")
        if count != expected_byte_count:
            raise CheckpointStagingError(
                f"Checkpoint byte count mismatch: expected {expected_byte_count}, copied {count}."
            )
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha256:
            raise CheckpointStagingError(
                f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, copied {actual_sha}."
            )
        os.fsync(staged_fd)
        os.close(staged_fd)
        staged_fd = -1
        os.chmod(staged_path, 0o400)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        retained_fd = os.open(staged_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(retained_fd)
        return StagedCheckpoint(
            checkpoint_kind,
            source.absolute(),
            directory,
            staged_path,
            count,
            actual_sha,
            info.st_dev,
            info.st_ino,
            retained_fd,
        )
    except BaseException:
        if retained_fd >= 0:
            os.close(retained_fd)
        if staged_fd >= 0:
            os.close(staged_fd)
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        os.close(source_fd)


def _digest_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 4 * 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def load_staged_checkpoint_into_fresh_module(
    module: torch.nn.Module,
    staged_checkpoint: StagedCheckpoint,
    checkpoint_kind: CheckpointKind,
    *,
    drop_unused_lm_head: bool = False,
    tied_alias_contract: Mapping[str, str] | None = None,
) -> CheckpointVerificationReport:
    """Validate fully, then copy one tensor at a time into a disposable fresh module."""
    if os.name != "posix" or not Path("/proc/self/fd").is_dir():
        raise CheckpointVerificationError(
            "Runner staged loading requires Linux /proc/self/fd support."
        )
    if checkpoint_kind != staged_checkpoint.checkpoint_kind:
        raise CheckpointVerificationError(
            "checkpoint kind does not match staged metadata."
        )
    info = os.fstat(staged_checkpoint.descriptor)
    if (info.st_dev, info.st_ino) != (
        staged_checkpoint.stat_device,
        staged_checkpoint.stat_inode,
    ):
        raise CheckpointVerificationError(
            "staged checkpoint descriptor identity changed."
        )
    if _digest_descriptor(staged_checkpoint.descriptor) != staged_checkpoint.sha256:
        raise CheckpointVerificationError(
            "staged checkpoint digest changed before loading."
        )
    fd_path = f"/proc/self/fd/{staged_checkpoint.descriptor}"
    expected = module.state_dict()
    expected_keys = set(expected)
    aliases = dict(tied_alias_contract or {})
    mapping = {}
    metadata_rows = []
    transformations = []
    dropped = []
    unexpected = []
    mismatches = []
    collisions = []
    sources = {}
    dtype_counts = Counter()
    try:
        with safe_open(fd_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = sorted(handle.keys())
            for key in keys:
                tensor = handle.get_tensor(key)
                dtype_counts[str(tensor.dtype).removeprefix("torch.")] += 1
                if (
                    torch.is_floating_point(tensor)
                    and not torch.isfinite(tensor).all().item()
                ):
                    raise CheckpointVerificationError(
                        f"checkpoint tensor {key!r} contains non-finite values."
                    )
                if checkpoint_kind == "fine-tuned":
                    target, transformation = key, "preserve-model-prefixed-key"
                    if not key.startswith("model."):
                        unexpected.append(key)
                elif (
                    key == BASE_UNUSED_EXPERT_HEAD
                    and f"model.{key}" not in expected_keys
                ):
                    if drop_unused_lm_head:
                        target, transformation = (
                            None,
                            "drop-known-unused-expert-lm-head",
                        )
                        dropped.append(key)
                    else:
                        target, transformation = (
                            f"model.{key}",
                            "unused-expert-lm-head-drop-not-allowed",
                        )
                else:
                    target, transformation = _base_target_key(
                        key, expected_keys, metadata, aliases
                    )
                if target is None:
                    if transformation != "drop-known-unused-expert-lm-head":
                        unexpected.append(key)
                elif target in sources:
                    collisions.append(f"{target}: {sources[target]}, {key}")
                else:
                    sources[target] = key
                    mapping[key] = target
                    if target not in expected_keys:
                        unexpected.append(key)
                    elif tuple(expected[target].shape) != tuple(tensor.shape):
                        mismatches.append(
                            f"{key} -> {target}: source {tuple(tensor.shape)}, target {tuple(expected[target].shape)}"
                        )
                metadata_rows.append(
                    CheckpointTensorMetadata(
                        key,
                        target,
                        tuple(tensor.shape),
                        str(tensor.dtype).removeprefix("torch."),
                        transformation,
                    )
                )
                transformations.append(f"{key}: {transformation}")
    except SafetensorError as error:
        raise CheckpointVerificationError(
            "staged checkpoint is malformed SafeTensors."
        ) from error
    mapped = {target for target in mapping.values() if target in expected_keys}
    missing = sorted(expected_keys - mapped)
    verified = not (missing or unexpected or mismatches or collisions)
    report = CheckpointVerificationReport(
        checkpoint_kind,
        staged_checkpoint.sha256,
        len(metadata_rows),
        len(expected),
        len(mapped) if verified else 0,
        tuple(sorted(dtype_counts.items())),
        tuple(metadata_rows),
        tuple(transformations),
        tuple(sorted(dropped)),
        tuple(missing),
        tuple(sorted(set(unexpected))),
        tuple(mismatches),
        tuple(collisions),
        verified,
    )
    if not verified:
        raise CheckpointVerificationError(
            f"Staged checkpoint verification failed: missing={report.missing_keys}, unexpected={report.unexpected_keys}, shapeMismatches={report.shape_mismatches}, collisions={report.duplicate_target_collisions}."
        )
    try:
        with (
            torch.no_grad(),
            safe_open(fd_path, framework="pt", device="cpu") as handle,
        ):
            targets = module.state_dict()
            for source_key, target_key in mapping.items():
                targets[target_key].copy_(handle.get_tensor(source_key))
    except BaseException as error:
        raise CheckpointVerificationError(
            "Fresh module loading failed; the module must be discarded and is not usable."
        ) from error
    if _digest_descriptor(staged_checkpoint.descriptor) != staged_checkpoint.sha256:
        raise CheckpointVerificationError(
            "staged checkpoint digest changed during loading; discard module."
        )
    return report
