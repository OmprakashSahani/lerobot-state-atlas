"""Local-only PI05 SafeTensors inspection, reconciliation, and strict loading."""

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Literal

from safetensors import SafetensorError, deserialize
from safetensors.torch import load as load_bytes
import torch

from lerobot_state_atlas.checkpoint_comparison.models import (
    CheckpointTensorMetadata,
    CheckpointVerificationReport,
)


CheckpointKind = Literal["base", "fine-tuned"]
BASE_TIED_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
BASE_TIED_EMBEDDING = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)
TARGET_TIED_EMBEDDING = f"model.{BASE_TIED_EMBEDDING}"
BASE_UNUSED_EXPERT_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"


class CheckpointVerificationError(ValueError):
    """Raised when a local checkpoint cannot be verified or strictly loaded."""


def _checkpoint_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise CheckpointVerificationError("checkpointPath must not be a symbolic link.")
    path = Path(path.absolute())
    if path.suffix != ".safetensors":
        raise CheckpointVerificationError("checkpointPath must end with .safetensors.")
    if not path.exists():
        raise CheckpointVerificationError("checkpointPath does not exist.")
    if not path.is_file():
        raise CheckpointVerificationError("checkpointPath must be a regular file.")
    return path


def _read_checkpoint_snapshot(path: Path) -> bytes:
    """Read one stable regular-file snapshot without following a final symlink."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CheckpointVerificationError(
            "checkpoint snapshot could not be opened safely."
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointVerificationError(
                "checkpoint snapshot must come from a regular file."
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise CheckpointVerificationError(
            "checkpoint snapshot could not be read completely."
        ) from error
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(snapshot) != before.st_size:
        raise CheckpointVerificationError(
            "checkpoint snapshot changed while it was being read."
        )
    return snapshot


def _inspect_snapshot(
    snapshot: bytes,
) -> tuple[dict[str, tuple[tuple[int, ...], str]], dict[str, str]]:
    try:
        serialized_tensors = deserialize(snapshot)
        if len(snapshot) < 8:
            raise ValueError("missing SafeTensors header length")
        header_length = int.from_bytes(snapshot[:8], byteorder="little", signed=False)
        header_end = 8 + header_length
        if header_end > len(snapshot):
            raise ValueError("SafeTensors header exceeds snapshot length")
        header = json.loads(snapshot[8:header_end].decode("utf-8"))
        raw_metadata = header.get("__metadata__", {})
        if not isinstance(raw_metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_metadata.items()
        ):
            raise ValueError("SafeTensors metadata must contain string pairs")
        metadata = dict(raw_metadata)
        tensors = {
            key: (
                tuple(int(value) for value in information["shape"]),
                str(information["dtype"]),
            )
            for key, information in serialized_tensors
        }
    except (
        SafetensorError,
        ValueError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise CheckpointVerificationError(
            "checkpoint snapshot is not valid SafeTensors content."
        ) from error
    return tensors, metadata


def _base_target_key(
    key: str,
    expected_keys: set[str],
    metadata: Mapping[str, str],
    tied_alias_contract: Mapping[str, str],
) -> tuple[str | None, str]:
    if key.startswith("state_proj."):
        return None, "rejected-state-projection"
    if key == BASE_TIED_HEAD:
        metadata_proves_alias = metadata.get(BASE_TIED_EMBEDDING) == BASE_TIED_HEAD
        contract_proves_alias = (
            tied_alias_contract.get(BASE_TIED_EMBEDDING) == BASE_TIED_HEAD
        )
        if not metadata_proves_alias and not contract_proves_alias:
            return None, "missing-tied-alias-evidence"
        return TARGET_TIED_EMBEDDING, "verified-tied-head-to-embedding"
    renamed = key
    if "action_time_mlp_in." in renamed:
        renamed = renamed.replace("action_time_mlp_in.", "time_mlp_in.", 1)
    if "action_time_mlp_out." in renamed:
        renamed = renamed.replace("action_time_mlp_out.", "time_mlp_out.", 1)
    target = f"model.{renamed}"
    transformation = "add-model-prefix"
    if renamed != key:
        transformation = "rename-action-time-mlp-and-add-model-prefix"
    if target not in expected_keys:
        return target, transformation
    return target, transformation


def _prepare_verification(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    checkpoint_kind: CheckpointKind,
    *,
    drop_unused_lm_head: bool,
    tied_alias_contract: Mapping[str, str] | None,
) -> tuple[CheckpointVerificationReport, bytes, dict[str, str]]:
    if checkpoint_kind not in {"base", "fine-tuned"}:
        raise CheckpointVerificationError(
            "checkpointKind must be 'base' or 'fine-tuned'."
        )
    if not isinstance(drop_unused_lm_head, bool):
        raise CheckpointVerificationError("dropUnusedLmHead must be boolean.")
    path = _checkpoint_path(checkpoint_path)
    snapshot = _read_checkpoint_snapshot(path)
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    source, metadata = _inspect_snapshot(snapshot)
    expected = module.state_dict()
    expected_keys = set(expected)
    aliases = dict(tied_alias_contract or {})
    mapping: dict[str, str] = {}
    tensor_metadata: list[CheckpointTensorMetadata] = []
    transformations: list[str] = []
    dropped: list[str] = []
    unexpected: list[str] = []
    shape_mismatches: list[str] = []
    collisions: list[str] = []
    target_sources: dict[str, str] = {}
    for key in sorted(source):
        shape, dtype = source[key]
        target: str | None
        transformation: str
        if checkpoint_kind == "fine-tuned":
            target = key
            transformation = "preserve-model-prefixed-key"
            if not key.startswith("model."):
                unexpected.append(key)
                transformation = "rejected-missing-model-prefix"
        else:
            if key == BASE_UNUSED_EXPERT_HEAD and f"model.{key}" not in expected_keys:
                if drop_unused_lm_head:
                    target = None
                    transformation = "drop-known-unused-expert-lm-head"
                    dropped.append(key)
                else:
                    target = f"model.{key}"
                    transformation = "unused-expert-lm-head-drop-not-allowed"
            else:
                target, transformation = _base_target_key(
                    key, expected_keys, metadata, aliases
                )
        if target is None:
            if transformation not in {"drop-known-unused-expert-lm-head"}:
                unexpected.append(key)
        elif target in target_sources:
            collisions.append(f"{target}: {target_sources[target]}, {key}")
        else:
            target_sources[target] = key
            mapping[key] = target
            if target not in expected_keys:
                unexpected.append(key)
            elif tuple(expected[target].shape) != shape:
                shape_mismatches.append(
                    f"{key} -> {target}: source {shape}, target {tuple(expected[target].shape)}"
                )
        tensor_metadata.append(
            CheckpointTensorMetadata(key, target, shape, dtype, transformation)
        )
        transformations.append(f"{key}: {transformation}")
    mapped_expected = {target for target in mapping.values() if target in expected_keys}
    missing = sorted(expected_keys - mapped_expected)
    verified = not (missing or unexpected or shape_mismatches or collisions)
    dtype_counts = Counter(dtype for _, dtype in source.values())
    report = CheckpointVerificationReport(
        checkpoint_kind=checkpoint_kind,
        checkpoint_sha256=snapshot_sha256,
        source_tensor_count=len(source),
        expected_tensor_count=len(expected),
        loaded_tensor_count=len(mapped_expected) if verified else 0,
        source_dtype_counts=tuple(sorted(dtype_counts.items())),
        tensors=tuple(tensor_metadata),
        transformations=tuple(transformations),
        explicitly_dropped_keys=tuple(sorted(dropped)),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(sorted(set(unexpected))),
        shape_mismatches=tuple(shape_mismatches),
        duplicate_target_collisions=tuple(collisions),
        verified=verified,
    )
    return report, snapshot, mapping


def verify_checkpoint_for_module(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    checkpoint_kind: CheckpointKind,
    *,
    drop_unused_lm_head: bool = False,
    tied_alias_contract: Mapping[str, str] | None = None,
) -> CheckpointVerificationReport:
    """Inspect and reconcile a local checkpoint without mutating the module."""
    return _prepare_verification(
        module,
        checkpoint_path,
        checkpoint_kind,
        drop_unused_lm_head=drop_unused_lm_head,
        tied_alias_contract=tied_alias_contract,
    )[0]


def _report_failure(report: CheckpointVerificationReport) -> str:
    return (
        "Checkpoint verification failed: "
        f"missing={report.missing_keys}, unexpected={report.unexpected_keys}, "
        f"shapeMismatches={report.shape_mismatches}, "
        f"duplicateTargets={report.duplicate_target_collisions}."
    )


def load_verified_checkpoint(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    checkpoint_kind: CheckpointKind,
    *,
    drop_unused_lm_head: bool = False,
    tied_alias_contract: Mapping[str, str] | None = None,
) -> CheckpointVerificationReport:
    """Verify completely, then strictly load, restoring state on any load failure."""
    report, snapshot, mapping = _prepare_verification(
        module,
        checkpoint_path,
        checkpoint_kind,
        drop_unused_lm_head=drop_unused_lm_head,
        tied_alias_contract=tied_alias_contract,
    )
    if not report.verified:
        raise CheckpointVerificationError(_report_failure(report))
    try:
        source = load_bytes(snapshot)
    except SafetensorError as error:
        raise CheckpointVerificationError(
            "checkpoint tensors could not be loaded from the verified snapshot."
        ) from error
    for key, tensor in source.items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all().item():
            raise CheckpointVerificationError(
                f"checkpoint tensor {key!r} contains non-finite floating values."
            )
    reconciled = {
        mapping[key]: tensor for key, tensor in source.items() if key in mapping
    }
    original = {
        key: value.detach().clone() for key, value in module.state_dict().items()
    }
    try:
        incompatible = module.load_state_dict(reconciled, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise CheckpointVerificationError(
                "strict load returned missing or unexpected keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
            )
    except BaseException as load_error:
        try:
            module.load_state_dict(original, strict=True)
        except BaseException as restore_error:
            raise CheckpointVerificationError(
                f"checkpoint load failed ({load_error!r}) and original module state "
                f"could not be restored ({restore_error!r})."
            ) from load_error
        raise CheckpointVerificationError(
            "checkpoint load failed; the original module state was restored."
        ) from load_error
    return report
