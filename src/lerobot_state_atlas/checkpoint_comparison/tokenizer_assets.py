"""Strict recursive verification and offline loading of a local PI05 tokenizer."""

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
    read_stable_file_snapshot,
)
from lerobot_state_atlas.checkpoint_comparison.runner_models import (
    CheckpointComparisonRunnerManifest,
    PI05ProcessorVerificationResult,
    PI05TokenizerVerificationResult,
    ResolvedRunnerInputs,
    TokenizerFileIdentity,
)


TOKENIZER_ID = "google/paligemma-3b-pt-224"


class PI05TokenizerVerificationError(ValueError):
    """Raised when a declared tokenizer directory is unsafe or inconsistent."""


class PI05TokenizerLoadingError(RuntimeError):
    """Raised when an offline verified tokenizer cannot be loaded safely."""


def _fail(field: str, message: str) -> None:
    raise PI05TokenizerVerificationError(f"{field} {message}")


def _canonical_identity(files: tuple[TokenizerFileIdentity, ...]) -> str:
    document = [
        {
            "path": item.relative_path,
            "byteCount": item.byte_count,
            "sha256": item.sha256,
        }
        for item in files
    ]
    encoded = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _declared_directory(
    manifest: CheckpointComparisonRunnerManifest, resolved: ResolvedRunnerInputs
) -> Path:
    matches = tuple(
        entry
        for entry in resolved.inventory
        if entry.logical_input_id == "processors.tokenizerDirectory"
    )
    if len(matches) != 1 or matches[0].kind != "directory":
        _fail(
            "processors.tokenizerDirectory",
            "must have one resolved directory inventory entry.",
        )
    base = manifest.manifest_path.parent.resolve()
    current = base
    for part in PurePosixPath(manifest.processors.tokenizer_directory.path).parts:
        current /= part
        if current.is_symlink():
            _fail(
                "processors.tokenizerDirectory.path",
                f"must not contain symbolic-link component {current}.",
            )
    resolved_path = current.resolve(strict=False)
    if resolved_path != matches[0].canonical_path:
        _fail(
            "processors.tokenizerDirectory.path",
            f"must resolve to {matches[0].canonical_path}.",
        )
    if not current.exists() or not current.is_dir():
        _fail("processors.tokenizerDirectory.path", "must be an ordinary directory.")
    return resolved_path


def verify_local_pi05_tokenizer_directory(
    manifest: CheckpointComparisonRunnerManifest,
    resolved_inputs: ResolvedRunnerInputs,
    processor_verification: PI05ProcessorVerificationResult,
    *,
    snapshot_reader: Callable[[Path], bytes] | None = None,
) -> PI05TokenizerVerificationResult:
    """Hash a complete symlink-free tokenizer tree without resolving the Hub."""
    if processor_verification.compatibility.tokenizer_repository_id != TOKENIZER_ID:
        _fail(
            "processor_verification.tokenizer_repository_id",
            f"must be {TOKENIZER_ID!r}.",
        )
    root = _declared_directory(manifest, resolved_inputs)
    reader = snapshot_reader or read_stable_file_snapshot
    candidates: list[tuple[str, Path]] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        for name in tuple(directory_names):
            child = directory_path / name
            if child.is_symlink():
                _fail(
                    "processors.tokenizerDirectory",
                    f"contains symbolic-link directory {child}.",
                )
            if not child.is_dir():
                _fail(
                    "processors.tokenizerDirectory",
                    f"contains special directory entry {child}.",
                )
        for name in file_names:
            child = directory_path / name
            relative = child.relative_to(root).as_posix()
            if not relative or any(
                part in {"", ".", ".."} for part in PurePosixPath(relative).parts
            ):
                _fail(
                    "processors.tokenizerDirectory",
                    f"contains unsafe relative path {relative!r}.",
                )
            if child.is_symlink():
                _fail(
                    "processors.tokenizerDirectory",
                    f"contains symbolic-link file {relative!r}.",
                )
            info = child.stat()
            if not stat.S_ISREG(info.st_mode):
                _fail(
                    "processors.tokenizerDirectory",
                    f"contains non-regular entry {relative!r}.",
                )
            candidates.append((relative, child))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        _fail("processors.tokenizerDirectory", "must contain tokenizer files.")
    files = []
    for relative, path in candidates:
        try:
            snapshot = reader(path)
        except StableFileSnapshotError as error:
            raise PI05TokenizerVerificationError(
                f"processors.tokenizerDirectory[{relative!r}] could not be acquired stably: {error}."
            ) from error
        if not isinstance(snapshot, bytes):
            _fail(
                f"processors.tokenizerDirectory[{relative!r}]",
                "snapshot reader must return bytes.",
            )
        files.append(
            TokenizerFileIdentity(
                relative, len(snapshot), hashlib.sha256(snapshot).hexdigest()
            )
        )
    identities = tuple(files)
    names = {item.relative_path for item in identities}
    if "tokenizer_config.json" not in names:
        _fail(
            "processors.tokenizerDirectory",
            "is incomplete: tokenizer_config.json is required.",
        )
    if not ({"tokenizer.json", "tokenizer.model"} & names):
        _fail(
            "processors.tokenizerDirectory",
            "is incomplete: tokenizer.json or tokenizer.model is required.",
        )
    identity = _canonical_identity(identities)
    declared = manifest.processors.tokenizer_directory.identity_sha256
    if identity != declared:
        _fail(
            "processors.tokenizerDirectory.identitySha256",
            f"expected {declared}; acquired {identity}.",
        )
    return PI05TokenizerVerificationResult(
        TOKENIZER_ID,
        root,
        identity,
        identities,
        True,
        (
            "Transformers supports equivalent fast-tokenizer and SentencePiece layouts; the installed loader performs the final capability check.",
        ),
        True,
    )


def _default_loader(path: Path, **kwargs: Any) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, **kwargs)


def _reverify_tree(verification: PI05TokenizerVerificationResult) -> None:
    expected = {item.relative_path: item for item in verification.files}
    observed_paths = []
    for path in verification.directory_path.rglob("*"):
        relative = path.relative_to(verification.directory_path).as_posix()
        if path.is_symlink():
            raise PI05TokenizerLoadingError(
                f"Verified tokenizer tree changed: {relative!r} is now a symbolic link."
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PI05TokenizerLoadingError(
                f"Verified tokenizer tree changed: {relative!r} is not a regular file."
            )
        observed_paths.append(relative)
        if relative not in expected:
            raise PI05TokenizerLoadingError(
                f"Verified tokenizer tree contains unexpected file {relative!r}."
            )
        try:
            snapshot = read_stable_file_snapshot(path)
        except StableFileSnapshotError as error:
            raise PI05TokenizerLoadingError(
                f"Verified tokenizer file {relative!r} could not be reacquired: {error}."
            ) from error
        identity = expected[relative]
        if (
            len(snapshot) != identity.byte_count
            or hashlib.sha256(snapshot).hexdigest() != identity.sha256
        ):
            raise PI05TokenizerLoadingError(
                f"Verified tokenizer file {relative!r} changed before loading."
            )
    if tuple(sorted(observed_paths)) != tuple(expected):
        raise PI05TokenizerLoadingError(
            "Verified tokenizer tree is missing one or more files."
        )


def load_verified_local_pi05_tokenizer(
    verification: PI05TokenizerVerificationResult,
    *,
    tokenizer_loader: Callable[..., object] | None = None,
) -> object:
    """Load only the verified local directory under temporary offline settings."""
    if not isinstance(verification, PI05TokenizerVerificationResult):
        raise PI05TokenizerLoadingError(
            "verification must be PI05TokenizerVerificationResult."
        )
    _reverify_tree(verification)
    loader = tokenizer_loader or _default_loader
    previous = {
        key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        tokenizer = loader(
            verification.directory_path,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise PI05TokenizerLoadingError(
            f"Offline tokenizer loading failed from {verification.directory_path}: {error}"
        ) from error
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if not callable(tokenizer):
        raise PI05TokenizerLoadingError("Loaded tokenizer must be callable.")
    for field in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, field, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PI05TokenizerLoadingError(
                f"Loaded tokenizer.{field} must be a nonnegative integer."
            )
    return tokenizer
