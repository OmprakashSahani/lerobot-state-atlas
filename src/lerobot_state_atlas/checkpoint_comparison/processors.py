"""Construction of locally verified PI05 processor pairs without Hub resolution."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable

from safetensors.torch import load as load_safetensors

from lerobot_state_atlas.checkpoint_comparison.local_files import (
    read_stable_file_snapshot,
)
from lerobot_state_atlas.checkpoint_comparison.runner_models import (
    PI05ProcessorVerificationResult,
    PI05TokenizerVerificationResult,
)


class PI05ProcessorConstructionError(RuntimeError):
    """Raised when verified processor material cannot be consumed exactly."""


@dataclass(frozen=True)
class VerifiedPI05ProcessorPair:
    preprocessor: object
    postprocessor: object
    processor_identity: str
    tokenizer_directory_identity_sha256: str
    shared_for_policy_ids: tuple[str, str]
    preprocessor_step_names: tuple[str, ...]
    postprocessor_step_names: tuple[str, ...]


def _default_pipeline_factory(config, state_dict, overrides):
    # Importing the policy processor module registers the PI05-specific step.
    import lerobot.policies.pi05.processor_pi05  # noqa: F401
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    return PolicyProcessorPipeline.from_config(
        config, state_dict=state_dict, overrides=overrides
    )


def _load_verified_state(
    summary,
    loader: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    snapshot = read_stable_file_snapshot(summary.source_path)
    if (
        len(snapshot) != summary.byte_count
        or hashlib.sha256(snapshot).hexdigest() != summary.sha256
    ):
        raise PI05ProcessorConstructionError(
            f"Verified processor state changed before construction: {summary.logical_input_id}."
        )
    state = loader(snapshot)
    if tuple(sorted(state)) != summary.tensor_keys:
        raise PI05ProcessorConstructionError(
            f"State loader did not consume the exact verified keys for {summary.logical_input_id}."
        )
    return state


def build_verified_pi05_processor_pair(
    processor_verification: PI05ProcessorVerificationResult,
    tokenizer_verification: PI05TokenizerVerificationResult,
    tokenizer: object,
    *,
    pipeline_factory: Callable[..., object] | None = None,
    state_loader: Callable[[bytes], dict[str, Any]] | None = None,
    device: str = "cpu",
) -> VerifiedPI05ProcessorPair:
    """Build both pipelines from canonical configs and exact verified state bytes."""
    if (
        processor_verification.compatibility.tokenizer_repository_id
        != tokenizer_verification.repository_id
    ):
        raise PI05ProcessorConstructionError(
            "Processor and tokenizer identities do not match."
        )
    if not callable(tokenizer):
        raise PI05ProcessorConstructionError(
            "tokenizer must be the callable verified local tokenizer."
        )
    if device not in {"cpu"} and not device.startswith("cuda:"):
        raise PI05ProcessorConstructionError(
            "device must be 'cpu' or an indexed CUDA device."
        )
    compatibility = processor_verification.compatibility
    try:
        pre_config = json.loads(compatibility.effective_preprocessor_json)
        post_config = json.loads(compatibility.effective_postprocessor_json)
    except json.JSONDecodeError as error:
        raise PI05ProcessorConstructionError(
            "Verified effective processor JSON is malformed."
        ) from error
    loader = state_loader or load_safetensors
    pre_state = _load_verified_state(processor_verification.preprocessor_state, loader)
    post_state = _load_verified_state(
        processor_verification.postprocessor_state, loader
    )
    factory = pipeline_factory or _default_pipeline_factory
    tokenizer_override = {"tokenizer": tokenizer, "tokenizer_name": None}
    try:
        preprocessor = factory(
            pre_config,
            {"policy_preprocessor_step_3_normalizer_processor": pre_state},
            {
                "tokenizer_processor": tokenizer_override,
                "device_processor": {"device": device},
            },
        )
        postprocessor = factory(
            post_config,
            {"policy_postprocessor_step_0_unnormalizer_processor": post_state},
            {"device_processor": {"device": "cpu"}},
        )
    except Exception as error:
        raise PI05ProcessorConstructionError(
            f"Local processor construction failed: {error}"
        ) from error
    expected_pre = tuple(
        step.effective_step_name for step in compatibility.preprocessor_steps
    )
    expected_post = tuple(
        step.effective_step_name for step in compatibility.postprocessor_steps
    )
    for side, pipeline, expected in (
        ("preprocessor", preprocessor, expected_pre),
        ("postprocessor", postprocessor, expected_post),
    ):
        steps = getattr(pipeline, "steps", None)
        if steps is None:
            raise PI05ProcessorConstructionError(
                f"Constructed {side} does not expose ordered steps."
            )
        actual = tuple(
            getattr(step, "_registry_name", type(step).__name__) for step in steps
        )
        if actual != expected:
            raise PI05ProcessorConstructionError(
                f"Constructed {side} step order mismatch: expected {expected}, received {actual}."
            )
    return VerifiedPI05ProcessorPair(
        preprocessor,
        postprocessor,
        processor_verification.processor_identity,
        tokenizer_verification.directory_identity_sha256,
        processor_verification.shared_for_policy_ids,
        expected_pre,
        expected_post,
    )
