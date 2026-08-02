import copy

import pytest

from lerobot_state_atlas.checkpoint_comparison.compatibility import (
    BOOLEAN_CONSTRUCTOR_FIELDS,
    OPTIONAL_BOOLEAN_CONSTRUCTOR_FIELDS,
    PI05CompatibilityError,
    adapt_pi05_finetuned_config,
)


def config() -> dict:
    return {
        "type": "pi05",
        "image_resolution": [224, 224],
        "image_resize_to": [224, 224],
        "drop_unused_lm_head": True,
        "scheduler_type": "cosine",
        "scheduler_decay_ratio": 0.1,
        "scheduler_max_decay_steps": 100,
        "chunk_size": 50,
        "max_action_dim": 32,
        "gradient_checkpointing": False,
    }


def test_adapts_known_fields_without_mutating_source_and_hashes_deterministically() -> (
    None
):
    source = config()
    original = copy.deepcopy(source)
    first = adapt_pi05_finetuned_config(
        source,
        "0.6.0",
        {"device": "cuda", "dtype": "bfloat16", "compile_model": False},
    )
    second = adapt_pi05_finetuned_config(
        source,
        "0.6.0",
        {"device": "cuda", "dtype": "bfloat16", "compile_model": False},
    )
    assert source == original
    assert first == second
    assert first.source_sha256 == second.source_sha256
    assert first.effective_sha256 == second.effective_sha256
    assert first.drop_unused_lm_head is True
    assert first.unknown_fields_resolved is True
    effective = dict(first.effective_config)
    assert not {
        "type",
        "image_resize_to",
        "drop_unused_lm_head",
        "scheduler_type",
        "scheduler_decay_ratio",
        "scheduler_max_decay_steps",
    } & set(effective)
    assert effective["device"] == "cuda"
    assert tuple(item.field for item in first.transformations) == (
        "type",
        "image_resize_to",
        "drop_unused_lm_head",
        "scheduler_type",
        "scheduler_decay_ratio",
        "scheduler_max_decay_steps",
        "device",
        "dtype",
        "compile_model",
    )


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (lambda value: value.update(type="pi0"), "config.type"),
        (
            lambda value: value.update(image_resize_to=[128, 128]),
            "config.image_resize_to",
        ),
        (
            lambda value: value.update(drop_unused_lm_head=1),
            "config.drop_unused_lm_head",
        ),
        (
            lambda value: value.update(drop_unused_lm_head="true"),
            "config.drop_unused_lm_head",
        ),
        (
            lambda value: value.update(scheduler_decay_ratio=True),
            "config.scheduler_decay_ratio",
        ),
        (
            lambda value: value.update(scheduler_max_decay_steps=False),
            "config.scheduler_max_decay_steps",
        ),
        (
            lambda value: value.update(
                image_resolution=[True, 224], image_resize_to=[True, 224]
            ),
            r"config.image_resolution\[0\]",
        ),
        (
            lambda value: value.update(optimizer_betas=[0.9, False]),
            r"config.optimizer_betas\[1\]",
        ),
        (lambda value: value.update(chunk_size=True), "config.chunk_size"),
        (lambda value: value.update(unknown_field=1), "config has unsupported fields"),
        (lambda value: value.update(optimizer_lr=float("nan")), "config.optimizer_lr"),
    ],
)
def test_rejects_unproven_or_malformed_config(mutation, path: str) -> None:
    source = config()
    mutation(source)
    with pytest.raises(PI05CompatibilityError, match=path):
        adapt_pi05_finetuned_config(source, "0.6.0")


def test_training_only_fields_are_removed_and_recorded() -> None:
    result = adapt_pi05_finetuned_config(config(), "0.6.0")
    operations = {(item.field, item.operation) for item in result.transformations}
    assert ("scheduler_type", "remove") in operations
    assert ("scheduler_decay_ratio", "remove") in operations
    assert ("scheduler_max_decay_steps", "remove") in operations


def test_rejects_unsupported_installed_lerobot_version() -> None:
    with pytest.raises(PI05CompatibilityError, match="installedLeRobotVersion"):
        adapt_pi05_finetuned_config(config(), "0.7.0")


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"device": ""}, "runtimeOverrides.device"),
        ({"dtype": "float64"}, "runtimeOverrides.dtype"),
        ({"compile_model": 1}, "runtimeOverrides.compile_model"),
        ({"gradient_checkpointing": "yes"}, "runtimeOverrides.gradient_checkpointing"),
        ({"unknown": True}, "runtimeOverrides has unsupported fields"),
    ],
)
def test_rejects_invalid_runtime_overrides(overrides: dict, path: str) -> None:
    with pytest.raises(PI05CompatibilityError, match=path):
        adapt_pi05_finetuned_config(config(), "0.6.0", overrides)


def test_frozen_result_does_not_share_nested_source_values() -> None:
    source = config()
    result = adapt_pi05_finetuned_config(source, "0.6.0")
    source["image_resolution"][0] = 1
    assert dict(result.source_config)["image_resolution"] == (224, 224)


@pytest.mark.parametrize("field", sorted(BOOLEAN_CONSTRUCTOR_FIELDS))
@pytest.mark.parametrize("value", [False, True])
def test_all_boolean_constructor_fields_accept_literal_booleans(
    field: str, value: bool
) -> None:
    source = config()
    source[field] = value
    result = adapt_pi05_finetuned_config(source, "0.6.0")
    assert dict(result.effective_config)[field] is value


@pytest.mark.parametrize("field", sorted(BOOLEAN_CONSTRUCTOR_FIELDS))
@pytest.mark.parametrize("value", [0, 1, "false", "yes", 0.0, [], {}])
def test_all_boolean_constructor_fields_reject_substitutes_without_mutation(
    field: str, value: object
) -> None:
    source = config()
    source[field] = value
    original = copy.deepcopy(source)
    with pytest.raises(
        PI05CompatibilityError,
        match=rf"^source_config\.{field} must be a boolean\.$",
    ):
        adapt_pi05_finetuned_config(source, "0.6.0")
    assert source == original


def test_installed_pi05_boolean_field_audit_is_complete() -> None:
    assert BOOLEAN_CONSTRUCTOR_FIELDS == {
        "use_amp",
        "use_peft",
        "push_to_hub",
        "use_relative_actions",
        "gradient_checkpointing",
        "compile_model",
        "freeze_vision_encoder",
        "train_expert_only",
    }
    assert OPTIONAL_BOOLEAN_CONSTRUCTOR_FIELDS == {"private"}


@pytest.mark.parametrize("value", [None, False, True])
def test_optional_boolean_constructor_field_accepts_only_boolean_or_null(
    value: bool | None,
) -> None:
    source = config()
    source["private"] = value
    result = adapt_pi05_finetuned_config(source, "0.6.0")
    assert dict(result.effective_config)["private"] is value


@pytest.mark.parametrize("value", [0, 1, "false", 0.0, [], {}])
def test_optional_boolean_constructor_field_rejects_other_values_without_mutation(
    value: object,
) -> None:
    source = config()
    source["private"] = value
    original = copy.deepcopy(source)
    with pytest.raises(
        PI05CompatibilityError,
        match=r"^source_config\.private must be a boolean or null\.$",
    ):
        adapt_pi05_finetuned_config(source, "0.6.0")
    assert source == original
