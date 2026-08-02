"""Narrow PI05 fine-tuned configuration compatibility for LeRobot 0.6.0."""

import copy
import hashlib
import json
from math import isfinite
from typing import Any, Mapping

from lerobot_state_atlas.checkpoint_comparison.models import (
    CompatibilityTransformation,
    PI05CompatibilityResult,
)


SUPPORTED_LEROBOT_VERSIONS = frozenset({"0.6.0"})
INCOMPATIBLE_FIELDS = frozenset(
    {
        "type",
        "image_resize_to",
        "drop_unused_lm_head",
        "scheduler_type",
        "scheduler_decay_ratio",
        "scheduler_max_decay_steps",
    }
)
RUNTIME_OVERRIDE_FIELDS = frozenset(
    {"device", "dtype", "compile_model", "gradient_checkpointing"}
)
BOOLEAN_CONSTRUCTOR_FIELDS = frozenset(
    {
        "use_amp",
        "use_peft",
        "push_to_hub",
        "use_relative_actions",
        "gradient_checkpointing",
        "compile_model",
        "freeze_vision_encoder",
        "train_expert_only",
    }
)
OPTIONAL_BOOLEAN_CONSTRUCTOR_FIELDS = frozenset({"private"})
PI05_CONSTRUCTOR_FIELDS = frozenset(
    {
        "n_obs_steps",
        "input_features",
        "output_features",
        "device",
        "use_amp",
        "use_peft",
        "push_to_hub",
        "repo_id",
        "private",
        "tags",
        "license",
        "pretrained_path",
        "pretrained_revision",
        "paligemma_variant",
        "action_expert_variant",
        "dtype",
        "chunk_size",
        "n_action_steps",
        "max_state_dim",
        "max_action_dim",
        "num_inference_steps",
        "time_sampling_beta_alpha",
        "time_sampling_beta_beta",
        "time_sampling_scale",
        "time_sampling_offset",
        "min_period",
        "max_period",
        "use_relative_actions",
        "relative_exclude_joints",
        "action_feature_names",
        "rtc_config",
        "image_resolution",
        "empty_cameras",
        "tokenizer_max_length",
        "normalization_mapping",
        "gradient_checkpointing",
        "compile_model",
        "compile_mode",
        "freeze_vision_encoder",
        "train_expert_only",
        "optimizer_lr",
        "optimizer_betas",
        "optimizer_eps",
        "optimizer_weight_decay",
        "optimizer_grad_clip_norm",
        "scheduler_warmup_steps",
        "scheduler_decay_steps",
        "scheduler_decay_lr",
    }
)
INTEGER_FIELDS = frozenset(
    {
        "n_obs_steps",
        "chunk_size",
        "n_action_steps",
        "max_state_dim",
        "max_action_dim",
        "num_inference_steps",
        "empty_cameras",
        "tokenizer_max_length",
        "scheduler_warmup_steps",
        "scheduler_decay_steps",
        "scheduler_max_decay_steps",
    }
)
NUMBER_FIELDS = frozenset(
    {
        "time_sampling_beta_alpha",
        "time_sampling_beta_beta",
        "time_sampling_scale",
        "time_sampling_offset",
        "min_period",
        "max_period",
        "optimizer_lr",
        "optimizer_eps",
        "optimizer_weight_decay",
        "optimizer_grad_clip_norm",
        "scheduler_decay_lr",
        "scheduler_decay_ratio",
    }
)


class PI05CompatibilityError(ValueError):
    """Raised when PI05 configuration equivalence cannot be proven."""


def _fail(field: str, message: str) -> None:
    raise PI05CompatibilityError(f"{field} {message}")


def _validate_json_value(value: Any, field: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        _fail(field, "must be finite.")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(field, "must contain only string keys.")
            _validate_json_value(item, f"{field}.{key}")
        return
    _fail(field, "must contain only JSON-compatible values.")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _freeze(value: Any) -> object:
    if isinstance(value, dict):
        return tuple((key, _freeze(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _frozen_items(value: Mapping[str, Any]) -> tuple[tuple[str, object], ...]:
    return tuple((key, _freeze(value[key])) for key in sorted(value))


def adapt_pi05_finetuned_config(
    source_config: Mapping[str, Any],
    installed_lerobot_version: str,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> PI05CompatibilityResult:
    """Adapt one decoded PI05 config only for the supported LeRobot runtime."""
    if installed_lerobot_version not in SUPPORTED_LEROBOT_VERSIONS:
        _fail(
            "installedLeRobotVersion",
            f"must be one of {sorted(SUPPORTED_LEROBOT_VERSIONS)!r}.",
        )
    if not isinstance(source_config, Mapping):
        _fail("config", "must be an object.")
    source = copy.deepcopy(dict(source_config))
    for key, value in source.items():
        if not isinstance(key, str):
            _fail("config", "must contain only string keys.")
        _validate_json_value(value, f"config.{key}")
    unknown = set(source) - PI05_CONSTRUCTOR_FIELDS - INCOMPATIBLE_FIELDS
    if unknown:
        _fail("config", f"has unsupported fields: {', '.join(sorted(unknown))}.")
    for field in sorted(BOOLEAN_CONSTRUCTOR_FIELDS & set(source)):
        if not isinstance(source[field], bool):
            _fail(f"source_config.{field}", "must be a boolean.")
    for field in sorted(OPTIONAL_BOOLEAN_CONSTRUCTOR_FIELDS & set(source)):
        value = source[field]
        if value is not None and not isinstance(value, bool):
            _fail(f"source_config.{field}", "must be a boolean or null.")
    if source.get("type") != "pi05":
        _fail("config.type", "must be 'pi05'.")
    for field in ("image_resolution", "image_resize_to"):
        value = source.get(field)
        if not isinstance(value, list) or len(value) != 2:
            _fail(f"config.{field}", "must contain exactly two integer dimensions.")
        for index, dimension in enumerate(value):
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                _fail(f"config.{field}[{index}]", "must be an integer.")
            if dimension <= 0:
                _fail(f"config.{field}[{index}]", "must be greater than zero.")
    for field in sorted(INTEGER_FIELDS & set(source)):
        value = source[field]
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(f"config.{field}", "must be an integer.")
    for field in sorted(NUMBER_FIELDS & set(source)):
        value = source[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"config.{field}", "must be a number.")
        if not isfinite(float(value)):
            _fail(f"config.{field}", "must be finite.")
    if "scheduler_type" in source and (
        not isinstance(source["scheduler_type"], str) or not source["scheduler_type"]
    ):
        _fail("config.scheduler_type", "must be a non-empty string.")
    if "optimizer_betas" in source:
        betas = source["optimizer_betas"]
        if not isinstance(betas, list) or len(betas) != 2:
            _fail("config.optimizer_betas", "must contain exactly two numbers.")
        for index, beta in enumerate(betas):
            if isinstance(beta, bool) or not isinstance(beta, (int, float)):
                _fail(f"config.optimizer_betas[{index}]", "must be a number.")
            if not isfinite(float(beta)):
                _fail(f"config.optimizer_betas[{index}]", "must be finite.")
    if "image_resize_to" not in source:
        _fail("config.image_resize_to", "is required for compatibility verification.")
    if source.get("image_resize_to") != source.get("image_resolution"):
        _fail("config.image_resize_to", "must exactly equal config.image_resolution.")
    drop_unused = source.get("drop_unused_lm_head")
    if not isinstance(drop_unused, bool):
        _fail("config.drop_unused_lm_head", "must be boolean.")
    effective = copy.deepcopy(source)
    transformations: list[CompatibilityTransformation] = []
    for field, operation, detail in (
        ("type", "remove", "Serialization discriminator; verified as pi05."),
        ("image_resize_to", "remove", "Equivalent to image_resolution."),
        (
            "drop_unused_lm_head",
            "preserve-as-load-policy",
            "Removed from constructor fields.",
        ),
        ("scheduler_type", "remove", "Training-only serialization field."),
        ("scheduler_decay_ratio", "remove", "Training-only serialization field."),
        ("scheduler_max_decay_steps", "remove", "Training-only serialization field."),
    ):
        if field in effective:
            effective.pop(field)
            transformations.append(
                CompatibilityTransformation(field, operation, detail)
            )
    overrides = dict(runtime_overrides or {})
    unknown_overrides = set(overrides) - RUNTIME_OVERRIDE_FIELDS
    if unknown_overrides:
        _fail(
            "runtimeOverrides",
            f"has unsupported fields: {', '.join(sorted(unknown_overrides))}.",
        )
    for key, value in overrides.items():
        field = f"runtimeOverrides.{key}"
        if key in {"compile_model", "gradient_checkpointing"}:
            if not isinstance(value, bool):
                _fail(field, "must be boolean.")
        elif key == "device":
            if not isinstance(value, str) or not value:
                _fail(field, "must be a non-empty string.")
        elif key == "dtype":
            if not isinstance(value, str) or value not in {
                "float32",
                "float16",
                "bfloat16",
            }:
                _fail(field, "must be 'float32', 'float16', or 'bfloat16'.")
        effective[key] = copy.deepcopy(value)
        transformations.append(
            CompatibilityTransformation(
                key, "runtime-override", "Explicit caller override."
            )
        )
    source_bytes = _canonical_bytes(source)
    effective_bytes = _canonical_bytes(effective)
    return PI05CompatibilityResult(
        source_config=_frozen_items(source),
        effective_config=_frozen_items(effective),
        transformations=tuple(transformations),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        effective_sha256=hashlib.sha256(effective_bytes).hexdigest(),
        installed_lerobot_version=installed_lerobot_version,
        drop_unused_lm_head=drop_unused,
        runtime_overrides=_frozen_items(overrides),
        unknown_fields_resolved=True,
    )
