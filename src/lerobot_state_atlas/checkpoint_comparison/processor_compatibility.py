"""Local-only PI05 processor configuration and learned-state verification."""

from collections.abc import Callable, Mapping
import copy
from dataclasses import replace
import hashlib
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
import torch

from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
    read_stable_file_snapshot,
)
from lerobot_state_atlas.checkpoint_comparison.observation import COMPONENT_NAMES
from lerobot_state_atlas.checkpoint_comparison.runner_models import (
    CheckpointComparisonRunnerManifest,
    InputInventoryEntry,
    PI05ProcessorCompatibilityResult,
    PI05ProcessorVerificationResult,
    ProcessorConfigTransformation,
    ProcessorNormalizationContract,
    ProcessorStateSummary,
    ProcessorStepSummary,
    ResolvedRunnerInputs,
    RunnerFileInput,
)


SUPPORTED_LEROBOT_VERSION = "0.6.0"
TOKENIZER_REPOSITORY_ID = "google/paligemma-3b-pt-224"
KNOWN_PREPROCESSOR_JSON_SHA256 = (
    "42919e0c739665186ee2121f52adcbfff9eefb5fdc9601edd8a14cfaaa21d902"
)
KNOWN_POSTPROCESSOR_JSON_SHA256 = (
    "03af86cc58ef3893701df4d2fe8cd9dbd333fddfb9c0b69c7642ea1a9d984b1d"
)
PREPROCESSOR_STATE_FILE = "policy_preprocessor_step_3_normalizer_processor.safetensors"
POSTPROCESSOR_STATE_FILE = (
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
)
PREPROCESSOR_ORDER = (
    "rename_observations_processor",
    "to_batch_processor",
    "relative_actions_processor",
    "normalizer_processor",
    "pi05_prepare_state_tokenizer_processor_step",
    "tokenizer_processor",
    "device_processor",
)
POSTPROCESSOR_ORDER = (
    "unnormalizer_processor",
    "absolute_actions_processor",
    "device_processor",
)
CAMERA_FEATURES = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top",
)
STATE_KEYS = ("observation.state.q01", "observation.state.q99")
ACTION_KEYS = ("action.q01", "action.q99")
SUPPORTED_STAT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


class PI05ProcessorCompatibilityError(ValueError):
    """Raised when serialized PI05 processor configurations are incompatible."""


class PI05ProcessorVerificationError(ValueError):
    """Raised when local processor assets fail snapshot or state verification."""


def _compat_fail(path: str, message: str) -> None:
    raise PI05ProcessorCompatibilityError(f"{path} {message}")


def _verify_fail(path: str, message: str) -> None:
    raise PI05ProcessorVerificationError(f"{path} {message}")


def _object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _compat_fail(path, "must be an object.")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing:
        _compat_fail(path, f"is missing fields: {', '.join(sorted(missing))}.")
    if extra:
        _compat_fail(path, f"has unsupported fields: {', '.join(sorted(extra))}.")
    return value


def _canonical_bytes(value: Mapping[str, Any], path: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PI05ProcessorCompatibilityError(
            f"{path} cannot be serialized canonically: {error}."
        ) from error


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _compat_fail(path, "must be a boolean.")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _compat_fail(path, "must be a number.")
    result = float(value)
    if not isfinite(result):
        _compat_fail(path, "must be finite.")
    return result


def _step(
    entry: Any, side: str, index: int
) -> tuple[str, Mapping[str, Any], str | None]:
    path = f"{side}.steps[{index}]"
    if not isinstance(entry, dict):
        _compat_fail(path, "must be an object.")
    allowed = {"registry_name", "config", "state_file"}
    missing = {"registry_name", "config"} - set(entry)
    extra = set(entry) - allowed
    if missing:
        _compat_fail(path, f"is missing fields: {', '.join(sorted(missing))}.")
    if extra:
        _compat_fail(path, f"has unsupported fields: {', '.join(sorted(extra))}.")
    name = entry["registry_name"]
    if not isinstance(name, str) or not name:
        _compat_fail(f"{path}.registry_name", "must be a non-empty string.")
    config = entry["config"]
    if not isinstance(config, dict):
        _compat_fail(f"{path}.config", "must be an object.")
    state_file = entry.get("state_file")
    if state_file is not None and (not isinstance(state_file, str) or not state_file):
        _compat_fail(f"{path}.state_file", "must be a non-empty string.")
    return name, config, state_file


def _validate_features(config: Mapping[str, Any], side: str, index: int) -> None:
    path = f"{side}.steps[{index}].config"
    expected_fields = {"eps", "features", "norm_map"}
    _object(config, expected_fields, path)
    _finite_number(config["eps"], f"{path}.eps")
    features = config["features"]
    if not isinstance(features, dict):
        _compat_fail(f"{path}.features", "must be an object.")
    expected_names = (
        ("observation.state", *CAMERA_FEATURES, "action")
        if side == "preprocessor"
        else ("action",)
    )
    if tuple(features) != expected_names:
        _compat_fail(
            f"{path}.features",
            f"must contain the ordered features {list(expected_names)!r}; received {list(features)!r}.",
        )
    for name in expected_names:
        feature = _object(features[name], {"type", "shape"}, f"{path}.features.{name}")
        expected_type = (
            "VISUAL"
            if name in CAMERA_FEATURES
            else "STATE"
            if name == "observation.state"
            else "ACTION"
        )
        if feature["type"] != expected_type:
            _compat_fail(f"{path}.features.{name}.type", f"must be {expected_type!r}.")
        expected_shape = [3, 224, 224] if name in CAMERA_FEATURES else [14]
        shape = feature["shape"]
        if not isinstance(shape, list) or len(shape) != len(expected_shape):
            _compat_fail(f"{path}.features.{name}.shape", f"must be {expected_shape}.")
        for shape_index, dimension in enumerate(shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                _compat_fail(
                    f"{path}.features.{name}.shape[{shape_index}]",
                    "must be an integer.",
                )
        if shape != expected_shape:
            _compat_fail(f"{path}.features.{name}.shape", f"must be {expected_shape}.")
    norm_map = _object(
        config["norm_map"], {"VISUAL", "STATE", "ACTION"}, f"{path}.norm_map"
    )
    expected_norms = {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}
    for feature_type, expected_mode in expected_norms.items():
        if norm_map[feature_type] != expected_mode:
            _compat_fail(
                f"{path}.norm_map.{feature_type}",
                f"must be {expected_mode!r}; received {norm_map[feature_type]!r}.",
            )


def _validate_preprocessor(
    source: Mapping[str, Any], effective: dict[str, Any]
) -> tuple[
    tuple[ProcessorStepSummary, ...], tuple[ProcessorConfigTransformation, ...], str
]:
    root = _object(source, {"name", "steps"}, "preprocessor")
    if root["name"] != "policy_preprocessor":
        _compat_fail("preprocessor.name", "must be 'policy_preprocessor'.")
    steps = root["steps"]
    if not isinstance(steps, list) or len(steps) != len(PREPROCESSOR_ORDER):
        _compat_fail(
            "preprocessor.steps",
            f"must contain exactly {len(PREPROCESSOR_ORDER)} steps.",
        )
    parsed = [_step(entry, "preprocessor", index) for index, entry in enumerate(steps)]
    names = [item[0] for item in parsed]
    alias_count = names.count("delta_actions_processor")
    target_count = names.count("relative_actions_processor")
    if alias_count and target_count:
        _compat_fail(
            "preprocessor.steps",
            "must not contain both legacy and effective relative-action step names.",
        )
    if alias_count > 1 or target_count > 1:
        _compat_fail("preprocessor.steps", "contains duplicate relative-action steps.")
    transformations: list[ProcessorConfigTransformation] = []
    summaries: list[ProcessorStepSummary] = []
    effective_steps = effective["steps"]
    tokenizer_id = ""
    for index, ((name, config, state_file), expected) in enumerate(
        zip(parsed, PREPROCESSOR_ORDER, strict=True)
    ):
        path = f"preprocessor.steps[{index}]"
        effective_name = name
        if index == 2 and name == "delta_actions_processor":
            relative_config = _object(
                config, {"enabled", "exclude_joints", "action_names"}, f"{path}.config"
            )
            enabled = _bool(relative_config["enabled"], f"{path}.config.enabled")
            if enabled:
                _compat_fail(
                    f"{path}.registry_name",
                    "legacy rename is allowed only when enabled is false.",
                )
            effective_name = "relative_actions_processor"
            effective_steps[index]["registry_name"] = effective_name
            transformations.append(
                ProcessorConfigTransformation(
                    "preprocessor",
                    f"steps[{index}].registry_name",
                    "legacy-disabled-delta-actions-step-v1",
                    name,
                    effective_name,
                    "LeRobot 0.6.0 renamed the disabled relative-action processor registry entry.",
                )
            )
        if effective_name != expected:
            _compat_fail(
                f"{path}.registry_name",
                f"must resolve to {expected!r}; received {name!r}.",
            )
        enabled: bool | None = None
        learned = False
        logical_id = None
        if expected in {
            "rename_observations_processor",
            "to_batch_processor",
            "pi05_prepare_state_tokenizer_processor_step",
        }:
            expected_config = (
                {"rename_map"} if expected == "rename_observations_processor" else set()
            )
            checked = _object(config, expected_config, f"{path}.config")
            if (
                expected == "rename_observations_processor"
                and checked["rename_map"] != {}
            ):
                _compat_fail(f"{path}.config.rename_map", "must be empty.")
        elif expected == "relative_actions_processor":
            checked = _object(
                config, {"enabled", "exclude_joints", "action_names"}, f"{path}.config"
            )
            enabled = _bool(checked["enabled"], f"{path}.config.enabled")
            if enabled:
                _compat_fail(
                    f"{path}.config.enabled",
                    "must be false for absolute action targets.",
                )
            if (
                checked["exclude_joints"] != ["gripper"]
                or checked["action_names"] is not None
            ):
                _compat_fail(
                    f"{path}.config",
                    "must preserve the verified disabled PI05 relative-action configuration.",
                )
        elif expected == "normalizer_processor":
            _validate_features(config, "preprocessor", index)
            if state_file != PREPROCESSOR_STATE_FILE:
                _compat_fail(
                    f"{path}.state_file", f"must be {PREPROCESSOR_STATE_FILE!r}."
                )
            learned = True
            logical_id = "processors.preprocessorState"
        elif expected == "tokenizer_processor":
            checked = _object(
                config,
                {
                    "max_length",
                    "task_key",
                    "padding_side",
                    "padding",
                    "truncation",
                    "tokenizer_name",
                },
                f"{path}.config",
            )
            if checked != {
                "max_length": 200,
                "task_key": "task",
                "padding_side": "right",
                "padding": "max_length",
                "truncation": True,
                "tokenizer_name": TOKENIZER_REPOSITORY_ID,
            }:
                _compat_fail(
                    f"{path}.config",
                    "must match the verified PI05 task-tokenizer declaration.",
                )
            tokenizer_id = TOKENIZER_REPOSITORY_ID
        elif expected == "device_processor":
            checked = _object(config, {"device", "float_dtype"}, f"{path}.config")
            if checked["device"] != "cuda" or checked["float_dtype"] is not None:
                _compat_fail(
                    f"{path}.config", "must declare device='cuda' and float_dtype=null."
                )
        if expected != "normalizer_processor" and state_file is not None:
            _compat_fail(
                f"{path}.state_file", "is allowed only for the normalizer step."
            )
        summaries.append(
            ProcessorStepSummary(
                "preprocessor",
                index,
                name,
                effective_name,
                expected,
                enabled,
                learned,
                logical_id,
            )
        )
    return tuple(summaries), tuple(transformations), tokenizer_id


def _validate_postprocessor(
    source: Mapping[str, Any],
) -> tuple[ProcessorStepSummary, ...]:
    root = _object(source, {"name", "steps"}, "postprocessor")
    if root["name"] != "policy_postprocessor":
        _compat_fail("postprocessor.name", "must be 'policy_postprocessor'.")
    steps = root["steps"]
    if not isinstance(steps, list) or len(steps) != len(POSTPROCESSOR_ORDER):
        _compat_fail(
            "postprocessor.steps",
            f"must contain exactly {len(POSTPROCESSOR_ORDER)} steps.",
        )
    summaries = []
    for index, (entry, expected) in enumerate(
        zip(steps, POSTPROCESSOR_ORDER, strict=True)
    ):
        name, config, state_file = _step(entry, "postprocessor", index)
        path = f"postprocessor.steps[{index}]"
        if name != expected:
            _compat_fail(
                f"{path}.registry_name", f"must be {expected!r}; received {name!r}."
            )
        enabled = None
        learned = False
        logical_id = None
        if expected == "unnormalizer_processor":
            _validate_features(config, "postprocessor", index)
            if state_file != POSTPROCESSOR_STATE_FILE:
                _compat_fail(
                    f"{path}.state_file", f"must be {POSTPROCESSOR_STATE_FILE!r}."
                )
            learned = True
            logical_id = "processors.postprocessorState"
        elif expected == "absolute_actions_processor":
            checked = _object(config, {"enabled"}, f"{path}.config")
            enabled = _bool(checked["enabled"], f"{path}.config.enabled")
            if enabled:
                _compat_fail(
                    f"{path}.config.enabled",
                    "must be false for absolute action targets.",
                )
        else:
            checked = _object(config, {"device", "float_dtype"}, f"{path}.config")
            if checked["device"] != "cpu" or checked["float_dtype"] is not None:
                _compat_fail(
                    f"{path}.config", "must declare device='cpu' and float_dtype=null."
                )
        if expected != "unnormalizer_processor" and state_file is not None:
            _compat_fail(
                f"{path}.state_file", "is allowed only for the unnormalizer step."
            )
        summaries.append(
            ProcessorStepSummary(
                "postprocessor",
                index,
                name,
                name,
                expected,
                enabled,
                learned,
                logical_id,
            )
        )
    return tuple(summaries)


def adapt_pi05_processor_configs(
    preprocessor_config: Mapping[str, Any],
    postprocessor_config: Mapping[str, Any],
    *,
    installed_lerobot_version: str,
) -> PI05ProcessorCompatibilityResult:
    """Validate and canonically adapt the audited PI05 processor JSON objects."""
    if installed_lerobot_version != SUPPORTED_LEROBOT_VERSION:
        _compat_fail("installed_lerobot_version", "must be exactly '0.6.0'.")
    if not isinstance(preprocessor_config, dict):
        _compat_fail("preprocessor", "must be an object.")
    if not isinstance(postprocessor_config, dict):
        _compat_fail("postprocessor", "must be an object.")
    source_pre = copy.deepcopy(preprocessor_config)
    source_post = copy.deepcopy(postprocessor_config)
    effective_pre = copy.deepcopy(preprocessor_config)
    effective_post = copy.deepcopy(postprocessor_config)
    pre_steps, transformations, tokenizer_id = _validate_preprocessor(
        source_pre, effective_pre
    )
    post_steps = _validate_postprocessor(source_post)
    source_pre_bytes = _canonical_bytes(source_pre, "preprocessor")
    source_post_bytes = _canonical_bytes(source_post, "postprocessor")
    effective_pre_bytes = _canonical_bytes(effective_pre, "effective_preprocessor")
    effective_post_bytes = _canonical_bytes(effective_post, "effective_postprocessor")
    normalization = ProcessorNormalizationContract(
        "IDENTITY",
        "QUANTILES",
        "QUANTILES",
        COMPONENT_NAMES,
        COMPONENT_NAMES,
        STATE_KEYS,
        ACTION_KEYS,
        14,
        14,
    )
    return PI05ProcessorCompatibilityResult(
        installed_lerobot_version,
        hashlib.sha256(source_pre_bytes).hexdigest(),
        hashlib.sha256(source_post_bytes).hexdigest(),
        hashlib.sha256(effective_pre_bytes).hexdigest(),
        hashlib.sha256(effective_post_bytes).hexdigest(),
        effective_pre_bytes.decode("utf-8"),
        effective_post_bytes.decode("utf-8"),
        transformations,
        pre_steps,
        post_steps,
        tokenizer_id,
        True,
        False,
        False,
        False,
        normalization,
    )


def _inventory(resolved: ResolvedRunnerInputs, logical_id: str) -> InputInventoryEntry:
    matches = tuple(
        entry for entry in resolved.inventory if entry.logical_input_id == logical_id
    )
    if len(matches) != 1:
        _verify_fail(
            "resolved_inputs.inventory",
            f"must contain exactly one {logical_id!r} entry.",
        )
    return matches[0]


def _acquire(
    manifest: CheckpointComparisonRunnerManifest,
    resolved: ResolvedRunnerInputs,
    logical_id: str,
    declared: RunnerFileInput,
    reader: Callable[[Path], bytes],
) -> tuple[bytes, Path]:
    entry = _inventory(resolved, logical_id)
    if (
        entry.expected_byte_count != declared.byte_count
        or entry.expected_sha256 != declared.sha256
        or entry.kind != "file"
    ):
        _verify_fail(
            logical_id,
            "manifest declaration must match the resolved file inventory identity.",
        )
    base = manifest.manifest_path.parent.resolve()
    current = base
    for part in PurePosixPath(declared.path).parts:
        current = current / part
        if current.is_symlink():
            _verify_fail(
                f"{logical_id}.path",
                f"must not contain symbolic-link component {current}.",
            )
    resolved_path = current.resolve(strict=False)
    if resolved_path != entry.canonical_path:
        _verify_fail(
            f"{logical_id}.path",
            f"must resolve to inventoried path {entry.canonical_path}; received {resolved_path}.",
        )
    if not current.exists():
        _verify_fail(f"{logical_id}.path", f"does not exist: {current}.")
    info = current.stat()
    if not stat.S_ISREG(info.st_mode):
        _verify_fail(
            f"{logical_id}.path", f"must be an ordinary regular file: {current}."
        )
    try:
        snapshot = reader(current)
    except StableFileSnapshotError as error:
        raise PI05ProcessorVerificationError(
            f"{logical_id}.path {current} could not be acquired as a stable snapshot: {error}."
        ) from error
    except OSError as error:
        raise PI05ProcessorVerificationError(
            f"{logical_id}.path {current} could not be read: {error}."
        ) from error
    if not isinstance(snapshot, bytes):
        _verify_fail(f"{logical_id}.snapshot", "reader must return immutable bytes.")
    if len(snapshot) != declared.byte_count:
        _verify_fail(
            f"{logical_id}.byteCount",
            f"expected {declared.byte_count}; acquired {len(snapshot)}.",
        )
    digest = hashlib.sha256(snapshot).hexdigest()
    if digest != declared.sha256:
        _verify_fail(
            f"{logical_id}.sha256", f"expected {declared.sha256}; acquired {digest}."
        )
    return snapshot, resolved_path


def _parse_json(snapshot: bytes, logical_id: str, path: Path) -> Mapping[str, Any]:
    try:
        text = snapshot.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PI05ProcessorVerificationError(
            f"{logical_id} at {path} contains malformed UTF-8."
        ) from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise PI05ProcessorVerificationError(
            f"{logical_id} at {path} contains malformed JSON."
        ) from error
    if not isinstance(value, dict):
        _verify_fail(logical_id, f"at {path} must contain a top-level object.")
    return value


def _state(
    snapshot: bytes,
    logical_id: str,
    path: Path,
    allowed_keys: set[str],
    required_keys: set[str],
) -> tuple[ProcessorStateSummary, dict[str, torch.Tensor]]:
    try:
        tensors = load_safetensors(snapshot)
    except (SafetensorError, ValueError, TypeError) as error:
        raise PI05ProcessorVerificationError(
            f"{logical_id} at {path} is malformed SafeTensors data: {error}."
        ) from error
    keys = tuple(sorted(tensors))
    missing = required_keys - set(keys)
    unexpected = set(keys) - allowed_keys
    if missing:
        _verify_fail(
            logical_id,
            f"at {path} is missing tensor keys: {', '.join(sorted(missing))}.",
        )
    if unexpected:
        _verify_fail(
            logical_id,
            f"at {path} has unexpected tensor keys: {', '.join(sorted(unexpected))}.",
        )
    if ("action.q01" in tensors) != ("action.q99" in tensors):
        _verify_fail(
            logical_id,
            f"at {path} must contain both action.q01 and action.q99 when either is present.",
        )
    for key in keys:
        tensor = tensors[key]
        if not isinstance(tensor, torch.Tensor):
            _verify_fail(f"{logical_id}.{key}", "must be a tensor.")
        if tensor.dtype not in SUPPORTED_STAT_DTYPES:
            _verify_fail(
                f"{logical_id}.{key}.dtype",
                f"must be floating; received {tensor.dtype}.",
            )
        if tuple(tensor.shape) != (14,):
            _verify_fail(
                f"{logical_id}.{key}.shape",
                f"must be [14]; received {list(tensor.shape)}.",
            )
        if not torch.isfinite(tensor).all().item():
            _verify_fail(f"{logical_id}.{key}", "must contain only finite values.")
    for prefix in ("observation.state", "action"):
        low = tensors.get(f"{prefix}.q01")
        high = tensors.get(f"{prefix}.q99")
        if low is not None and high is not None:
            if not torch.all(low.to(torch.float64) <= high.to(torch.float64)).item():
                _verify_fail(
                    f"{logical_id}.{prefix}", "requires q01 <= q99 elementwise."
                )
    summary = ProcessorStateSummary(
        logical_id,
        path,
        len(snapshot),
        hashlib.sha256(snapshot).hexdigest(),
        keys,
        tuple((key, tuple(tensors[key].shape)) for key in keys),
        tuple((key, str(tensors[key].dtype).removeprefix("torch.")) for key in keys),
        True,
    )
    return summary, tensors


def verify_pi05_processor_assets(
    manifest: CheckpointComparisonRunnerManifest,
    resolved_inputs: ResolvedRunnerInputs,
    *,
    snapshot_reader: Callable[[Path], bytes] | None = None,
    installed_lerobot_version: str = SUPPORTED_LEROBOT_VERSION,
) -> PI05ProcessorVerificationResult:
    """Verify four fine-tuned processor assets without constructing live processors."""
    if not isinstance(manifest, CheckpointComparisonRunnerManifest):
        _verify_fail("manifest", "must be a CheckpointComparisonRunnerManifest.")
    if not isinstance(resolved_inputs, ResolvedRunnerInputs):
        _verify_fail("resolved_inputs", "must be ResolvedRunnerInputs.")
    if manifest.manifest_sha256 != resolved_inputs.manifest_sha256:
        _verify_fail(
            "resolved_inputs.manifestSha256", "must match the runner manifest."
        )
    reader = snapshot_reader or read_stable_file_snapshot
    assets = (
        ("processors.preprocessorConfig", manifest.processors.preprocessor_config),
        ("processors.postprocessorConfig", manifest.processors.postprocessor_config),
        ("processors.preprocessorState", manifest.processors.preprocessor_state),
        ("processors.postprocessorState", manifest.processors.postprocessor_state),
    )
    acquired = {
        logical_id: _acquire(manifest, resolved_inputs, logical_id, declared, reader)
        for logical_id, declared in assets
    }
    pre_bytes, pre_path = acquired["processors.preprocessorConfig"]
    post_bytes, post_path = acquired["processors.postprocessorConfig"]
    if hashlib.sha256(pre_bytes).hexdigest() != KNOWN_PREPROCESSOR_JSON_SHA256:
        _verify_fail(
            "processors.preprocessorConfig.sha256",
            "must match the audited fine-tuned PI05 preprocessor JSON identity.",
        )
    if hashlib.sha256(post_bytes).hexdigest() != KNOWN_POSTPROCESSOR_JSON_SHA256:
        _verify_fail(
            "processors.postprocessorConfig.sha256",
            "must match the audited fine-tuned PI05 postprocessor JSON identity.",
        )
    pre_config = _parse_json(pre_bytes, "processors.preprocessorConfig", pre_path)
    post_config = _parse_json(post_bytes, "processors.postprocessorConfig", post_path)
    try:
        compatibility = adapt_pi05_processor_configs(
            pre_config, post_config, installed_lerobot_version=installed_lerobot_version
        )
    except PI05ProcessorCompatibilityError as error:
        raise PI05ProcessorVerificationError(
            "processor configuration compatibility failed for "
            f"processors.preprocessorConfig at {pre_path} and "
            f"processors.postprocessorConfig at {post_path}: {error}"
        ) from error
    compatibility = replace(
        compatibility,
        source_preprocessor_sha256=hashlib.sha256(pre_bytes).hexdigest(),
        source_postprocessor_sha256=hashlib.sha256(post_bytes).hexdigest(),
    )
    pre_state_bytes, pre_state_path = acquired["processors.preprocessorState"]
    post_state_bytes, post_state_path = acquired["processors.postprocessorState"]
    pre_summary, pre_tensors = _state(
        pre_state_bytes,
        "processors.preprocessorState",
        pre_state_path,
        set(STATE_KEYS + ACTION_KEYS),
        set(STATE_KEYS),
    )
    post_summary, post_tensors = _state(
        post_state_bytes,
        "processors.postprocessorState",
        post_state_path,
        set(ACTION_KEYS),
        set(ACTION_KEYS),
    )
    if all(key in pre_tensors for key in ACTION_KEYS):
        for key in ACTION_KEYS:
            if not torch.equal(
                pre_tensors[key].to(device="cpu", dtype=torch.float64),
                post_tensors[key].to(device="cpu", dtype=torch.float64),
            ):
                _verify_fail(
                    f"crossState.{key}",
                    "must match exactly between preprocessor and postprocessor state.",
                )
    return PI05ProcessorVerificationResult(
        "dream-machines-actuator-unboxing-pi05-processors-v1",
        compatibility,
        pre_summary,
        post_summary,
        True,
        ("base-pi05", "fine-tuned-pi05"),
        "Base π0.5 and Fine-tuned π0.5 must use this identical verified fine-tuned robot-specific processor contract.",
    )
