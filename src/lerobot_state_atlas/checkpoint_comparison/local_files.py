"""Stable single-read snapshots for local comparison inputs."""

import os
from pathlib import Path
import stat


class StableFileSnapshotError(OSError):
    """Raised when a stable regular-file snapshot cannot be acquired."""


def _identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def read_stable_file_snapshot(path: Path) -> bytes:
    """Read one regular file once without following its final symlink.

    Descriptor metadata is compared before and after the read so in-place mutation
    during acquisition is rejected. The returned immutable bytes are the only
    content consumers should validate.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StableFileSnapshotError(f"could not open {path.name} safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StableFileSnapshotError(f"{path.name} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        if isinstance(error, StableFileSnapshotError):
            raise
        raise StableFileSnapshotError(f"could not read {path.name}") from error
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or len(content) != after.st_size:
        raise StableFileSnapshotError(f"{path.name} changed while it was being read")
    return content
