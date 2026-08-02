"""Deterministic, injectable PI05 comparison orchestration for cloud runners."""

from collections.abc import Callable, Mapping
import hashlib
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from lerobot_state_atlas.checkpoint_comparison.models import (
    BoundCameraInput,
    BoundPolicyObservationInput,
    DeterministicNoiseProvenance,
    InferredPolicyPlan,
    PolicyComparisonInferenceResult,
    PolicyComparisonObservation,
)
from lerobot_state_atlas.checkpoint_comparison.schema import (
    ACTION_DIMENSION,
    BASE_POLICY_ID,
    BASE_POLICY_LABEL,
    CHUNK_LENGTH,
    FINE_TUNED_POLICY_ID,
    FINE_TUNED_POLICY_LABEL,
    NOISE_SHAPE,
)


MAX_TORCH_GENERATOR_SEED = 2**63 - 1
SUPPORTED_NOISE_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)
CAMERA_FEATURE_NAMES = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top",
)
CAMERA_INPUT_SHAPE = (1, 3, 224, 224)


class PolicyComparisonInferenceError(ValueError):
    """Raised when deterministic comparison orchestration cannot complete."""


def _fail(field: str, message: str) -> None:
    raise PolicyComparisonInferenceError(f"{field} {message}")


def _clone_supported(value: Any, path: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            _fail(path, "must be finite.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "must contain only string mapping keys.")
            result[key] = _clone_supported(item, f"{path}.{key}")
        return result
    if isinstance(value, list):
        return [
            _clone_supported(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _clone_supported(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    _fail(path, f"contains unsupported mutable value type {type(value).__name__!r}.")


def _equal_supported(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equal_supported(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _equal_supported(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _normalized_device(value: str | torch.device, field: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as error:
        raise PolicyComparisonInferenceError(
            f"{field} must identify a valid Torch device: {error}"
        ) from error
    if device.type == "cuda" and device.index is None:
        if not torch.cuda.is_available():
            _fail(field, "uses unindexed CUDA but no current CUDA device is available.")
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def validate_processed_observation_device(
    value: Any,
    expected_device: str | torch.device,
    *,
    path: str = "processed_observation",
) -> None:
    """Require every recursively contained tensor to use one exact device."""
    expected = _normalized_device(expected_device, "expected_device")

    def visit(item: Any, item_path: str) -> None:
        if isinstance(item, torch.Tensor):
            actual = _normalized_device(item.device, item_path)
            if actual != expected:
                _fail(
                    item_path,
                    f"tensor device must be {expected}, received {actual}.",
                )
            return
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not isfinite(item):
                _fail(item_path, "must be finite.")
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    _fail(item_path, "must contain only string mapping keys.")
                visit(nested, f"{item_path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{item_path}[{index}]")
            return
        _fail(
            item_path,
            f"contains unsupported mutable value type {type(item).__name__!r}.",
        )

    visit(value, path)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    byte_view = canonical.view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _canonical_raw_observation(
    observation: PolicyComparisonObservation,
    bound_input: BoundPolicyObservationInput,
) -> dict[str, Any]:
    """Build policy input from validated values plus provenance-bound cameras.

    Camera provenance is declared by the external preparation boundary. This
    validates that declaration against the observation; it does not prove that the
    tensor values were cryptographically derived from the source image bytes.
    """
    if not isinstance(bound_input, BoundPolicyObservationInput):
        _fail("bound_input", "must be a BoundPolicyObservationInput.")
    if bound_input.observation_id != observation.observation_id:
        _fail(
            "bound_input.observation_id",
            "must match observation.observation_id.",
        )
    if len(observation.cameras) != len(CAMERA_FEATURE_NAMES):
        _fail(
            "observation.cameras",
            f"must contain exactly {len(CAMERA_FEATURE_NAMES)} validated cameras.",
        )
    if not isinstance(bound_input.cameras, tuple) or len(bound_input.cameras) != len(
        CAMERA_FEATURE_NAMES
    ):
        _fail(
            "bound_input.cameras",
            f"must contain exactly {len(CAMERA_FEATURE_NAMES)} ordered cameras.",
        )
    canonical: dict[str, Any] = {
        observation.state.feature_name: torch.tensor(
            observation.state.values, dtype=torch.float32
        ).reshape(1, ACTION_DIMENSION)
    }
    for index, expected_feature in enumerate(CAMERA_FEATURE_NAMES):
        field = f"bound_input.cameras[{index}]"
        validated_camera = observation.cameras[index]
        if validated_camera.feature_name != expected_feature:
            _fail(
                f"observation.cameras[{index}].feature_name",
                f"must be {expected_feature!r}.",
            )
        candidate = bound_input.cameras[index]
        if not isinstance(candidate, BoundCameraInput):
            _fail(field, "must be a BoundCameraInput.")
        if candidate.feature_name != expected_feature:
            _fail(f"{field}.feature_name", f"must be {expected_feature!r}.")
        if candidate.source_sha256 != validated_camera.sha256:
            _fail(
                f"{field}.source_sha256",
                "must match the validated camera SHA-256.",
            )
        if (
            isinstance(candidate.source_byte_count, bool)
            or not isinstance(candidate.source_byte_count, int)
            or candidate.source_byte_count != validated_camera.byte_count
        ):
            _fail(
                f"{field}.source_byte_count",
                "must match the validated camera byte count.",
            )
        if not isinstance(candidate.source_path, Path):
            _fail(f"{field}.source_path", "must be a pathlib.Path.")
        if candidate.source_path != validated_camera.path:
            _fail(
                f"{field}.source_path",
                "must match the validated resolved camera path.",
            )
        tensor = candidate.tensor
        if not isinstance(tensor, torch.Tensor):
            _fail(f"{field}.tensor", "must be a torch.Tensor.")
        if tuple(tensor.shape) != CAMERA_INPUT_SHAPE:
            _fail(
                f"{field}.tensor",
                f"shape must be {list(CAMERA_INPUT_SHAPE)}, received {list(tensor.shape)}.",
            )
        if tensor.device.type != "cpu":
            _fail(f"{field}.tensor.device", "must be cpu.")
        if tensor.requires_grad:
            _fail(f"{field}.tensor.requires_grad", "must be false.")
        if tensor.dtype != torch.uint8 and not torch.is_floating_point(tensor):
            _fail(f"{field}.tensor.dtype", "must be uint8 or floating point.")
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all().item():
            _fail(f"{field}.tensor", "must contain only finite values.")
        canonical[expected_feature] = tensor.detach().clone()
    canonical["task"] = observation.prompt
    if tuple(canonical) != (
        "observation.state",
        *CAMERA_FEATURE_NAMES,
        "task",
    ):
        _fail("observation", "does not use the canonical policy-input fields.")
    return canonical


def build_deterministic_noise(
    seed: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, DeterministicNoiseProvenance]:
    """Build the explicit shared PI05 noise tensor with a dedicated generator."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        _fail("noise_seed", "must be an integer.")
    if seed < 0 or seed > MAX_TORCH_GENERATOR_SEED:
        _fail(
            "noise_seed",
            f"must be between 0 and {MAX_TORCH_GENERATOR_SEED} inclusive.",
        )
    if not isinstance(dtype, torch.dtype) or dtype not in SUPPORTED_NOISE_DTYPES:
        _fail("noise_dtype", "must be a supported floating torch dtype.")
    try:
        normalized_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise PolicyComparisonInferenceError(
            f"noise_device {device!r} is not a valid torch device."
        ) from error
    if normalized_device.type not in {"cpu", "cuda"}:
        _fail("noise_device", "must use the cpu or cuda generator backend.")
    if normalized_device.type == "cuda" and not torch.cuda.is_available():
        _fail("noise_device", f"{str(normalized_device)!r} is unavailable.")
    try:
        generator = torch.Generator(device=normalized_device)
        generator.manual_seed(seed)
        noise = torch.randn(
            NOISE_SHAPE,
            generator=generator,
            device=normalized_device,
            dtype=dtype,
        )
    except (RuntimeError, TypeError) as error:
        raise PolicyComparisonInferenceError(
            f"noise_device {str(normalized_device)!r} cannot construct deterministic noise."
        ) from error
    if not torch.isfinite(noise).all().item():
        _fail("noise", "contains non-finite values.")
    provenance = DeterministicNoiseProvenance(
        seed=seed,
        generator=f"torch.Generator({normalized_device.type}) with torch.randn",
        construction_device=str(normalized_device),
        dtype=str(dtype).removeprefix("torch."),
        shape=NOISE_SHAPE,
        sha256=_tensor_sha256(noise),
    )
    return noise, provenance


def _invoke_policy(
    policy: Any,
    policy_id: str,
    processed_observation: Any,
    canonical_noise: torch.Tensor,
    num_inference_steps: int | None,
) -> torch.Tensor:
    method = getattr(policy, "predict_action_chunk", None)
    if not callable(method):
        _fail(policy_id, "must expose callable predict_action_chunk.")
    policy_input = _clone_supported(processed_observation, f"{policy_id}.observation")
    input_snapshot = _clone_supported(policy_input, f"{policy_id}.observation")
    noise = canonical_noise.detach().clone()
    noise_snapshot = noise.detach().clone()
    kwargs: dict[str, Any] = {"noise": noise}
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    try:
        output = method(policy_input, **kwargs)
    except TypeError as error:
        if not _equal_supported(policy_input, input_snapshot):
            _fail(policy_id, "mutated its processed observation input before failing.")
        if not torch.equal(noise, noise_snapshot):
            _fail(policy_id, "mutated its explicit noise tensor before failing.")
        raise PolicyComparisonInferenceError(
            f"{policy_id}.predict_action_chunk has an incompatible explicit-noise signature: {error}"
        ) from error
    except Exception as error:
        if not _equal_supported(policy_input, input_snapshot):
            _fail(policy_id, "mutated its processed observation input before failing.")
        if not torch.equal(noise, noise_snapshot):
            _fail(policy_id, "mutated its explicit noise tensor before failing.")
        raise PolicyComparisonInferenceError(
            f"{policy_id}.predict_action_chunk failed: {error}"
        ) from error
    if not _equal_supported(policy_input, input_snapshot):
        _fail(policy_id, "mutated its processed observation input.")
    if not torch.equal(noise, noise_snapshot):
        _fail(policy_id, "mutated its explicit noise tensor.")
    if not isinstance(output, torch.Tensor):
        _fail(policy_id, "must return a torch.Tensor.")
    if tuple(output.shape) != (1, CHUNK_LENGTH, ACTION_DIMENSION):
        _fail(
            policy_id,
            f"output shape must be [1, {CHUNK_LENGTH}, {ACTION_DIMENSION}], received {list(output.shape)}.",
        )
    if not torch.is_floating_point(output):
        _fail(policy_id, "output tensor must have a floating dtype.")
    if not torch.isfinite(output).all().item():
        _fail(policy_id, "output tensor must contain only finite values.")
    if output.untyped_storage().data_ptr() == noise.untyped_storage().data_ptr():
        _fail(policy_id, "output tensor must not alias the explicit noise tensor.")
    return output.detach().to(device="cpu").clone()


def _postprocess_chunk(
    raw_chunk: torch.Tensor,
    postprocessor: Callable[[torch.Tensor], Any],
    policy_id: str,
) -> tuple[tuple[float, ...], ...]:
    processed_steps: list[torch.Tensor] = []
    for index in range(CHUNK_LENGTH):
        step = raw_chunk[:, index, :].detach().clone()
        try:
            result = postprocessor(step)
        except Exception as error:
            raise PolicyComparisonInferenceError(
                f"{policy_id}.postprocessing[{index}] failed: {error}"
            ) from error
        field = f"{policy_id}.postprocessing[{index}]"
        if not isinstance(result, torch.Tensor):
            _fail(field, "must return a torch.Tensor.")
        if tuple(result.shape) != (1, ACTION_DIMENSION):
            _fail(
                field,
                f"output shape must be [1, {ACTION_DIMENSION}], received {list(result.shape)}.",
            )
        if not torch.is_floating_point(result):
            _fail(field, "output tensor must have a floating dtype.")
        if not torch.isfinite(result).all().item():
            _fail(field, "output tensor must contain only finite values.")
        processed_steps.append(result.detach().to(device="cpu").clone().squeeze(0))
    stacked = torch.stack(processed_steps, dim=0)
    return tuple(tuple(float(value) for value in row.tolist()) for row in stacked)


def _prepare_comparison(
    observation: PolicyComparisonObservation,
    bound_input: BoundPolicyObservationInput,
    preprocessor: Callable[[Any], Any],
    postprocessor: Callable[[torch.Tensor], Any],
    noise_seed: int,
    noise_device: str | torch.device,
    noise_dtype: torch.dtype,
    num_inference_steps: int | None,
    processed_observation_validator: Callable[[Any], None] | None,
) -> tuple[Any, Any, torch.Tensor, DeterministicNoiseProvenance]:
    if not isinstance(observation, PolicyComparisonObservation):
        _fail("observation", "must be a validated PolicyComparisonObservation.")
    if num_inference_steps is not None and (
        isinstance(num_inference_steps, bool)
        or not isinstance(num_inference_steps, int)
        or num_inference_steps <= 0
    ):
        _fail("num_inference_steps", "must be a positive integer when supplied.")
    if not callable(preprocessor):
        _fail("preprocessor", "must be callable.")
    if not callable(postprocessor):
        _fail("postprocessor", "must be callable.")
    raw_input = _canonical_raw_observation(observation, bound_input)
    try:
        processed = preprocessor(raw_input)
    except Exception as error:
        raise PolicyComparisonInferenceError(f"preprocessor failed: {error}") from error
    canonical_processed = _clone_supported(processed, "processed_observation")
    if processed_observation_validator is not None:
        try:
            processed_observation_validator(
                _clone_supported(canonical_processed, "processed_observation")
            )
        except Exception as error:
            raise PolicyComparisonInferenceError(
                f"processed_observation validation failed: {error}"
            ) from error
    canonical_snapshot = _clone_supported(canonical_processed, "processed_observation")
    canonical_noise, noise_provenance = build_deterministic_noise(
        noise_seed, device=noise_device, dtype=noise_dtype
    )
    expected_noise_device = _normalized_device(noise_device, "noise_device")
    actual_noise_device = _normalized_device(canonical_noise.device, "noise.device")
    if actual_noise_device != expected_noise_device:
        _fail(
            "noise.device",
            f"must be {expected_noise_device}, received {actual_noise_device}.",
        )
    return canonical_processed, canonical_snapshot, canonical_noise, noise_provenance


def _comparison_result(
    observation: PolicyComparisonObservation,
    raw_outputs: list[tuple[str, str, torch.Tensor]],
    postprocessor: Callable[[torch.Tensor], Any],
    noise_provenance: DeterministicNoiseProvenance,
    num_inference_steps: int | None,
) -> PolicyComparisonInferenceResult:
    fps = observation.dataset.fps
    if not isfinite(fps) or fps <= 0:
        _fail("observation.dataset.fps", "must be finite and greater than zero.")
    relative_times = tuple(index / fps for index in range(CHUNK_LENGTH))
    plans = tuple(
        InferredPolicyPlan(
            policy_id=policy_id,
            label=label,
            relative_times_seconds=relative_times,
            actions=_postprocess_chunk(raw, postprocessor, policy_id),
        )
        for policy_id, label, raw in raw_outputs
    )
    return PolicyComparisonInferenceResult(
        observation_id=observation.observation_id,
        policies=(plans[0], plans[1]),
        noise=noise_provenance,
        action_dimension=ACTION_DIMENSION,
        chunk_length=CHUNK_LENGTH,
        num_inference_steps=num_inference_steps,
        shared_preprocessing=True,
        shared_postprocessing=True,
    )


def run_policy_comparison(
    observation: PolicyComparisonObservation,
    *,
    bound_input: BoundPolicyObservationInput,
    preprocessor: Callable[[Any], Any],
    postprocessor: Callable[[torch.Tensor], Any],
    base_policy: Any,
    fine_tuned_policy: Any,
    noise_seed: int,
    noise_device: str | torch.device = "cpu",
    noise_dtype: torch.dtype = torch.float32,
    num_inference_steps: int | None = None,
    processed_observation_validator: Callable[[Any], None] | None = None,
) -> PolicyComparisonInferenceResult:
    """Run a fair sequential comparison using caller-supplied policies/processors."""
    canonical_processed, canonical_snapshot, canonical_noise, noise_provenance = (
        _prepare_comparison(
            observation,
            bound_input,
            preprocessor,
            postprocessor,
            noise_seed,
            noise_device,
            noise_dtype,
            num_inference_steps,
            processed_observation_validator,
        )
    )
    raw_outputs: list[tuple[str, str, torch.Tensor]] = []
    for policy_id, label, policy in (
        (BASE_POLICY_ID, BASE_POLICY_LABEL, base_policy),
        (FINE_TUNED_POLICY_ID, FINE_TUNED_POLICY_LABEL, fine_tuned_policy),
    ):
        raw_outputs.append(
            (
                policy_id,
                label,
                _invoke_policy(
                    policy,
                    policy_id,
                    canonical_processed,
                    canonical_noise,
                    num_inference_steps,
                ),
            )
        )
        if not _equal_supported(canonical_processed, canonical_snapshot):
            _fail(
                "processed_observation", "canonical value was mutated during inference."
            )
    return _comparison_result(
        observation, raw_outputs, postprocessor, noise_provenance, num_inference_steps
    )


def run_sequential_policy_comparison(
    observation: PolicyComparisonObservation,
    *,
    bound_input: BoundPolicyObservationInput,
    preprocessor: Callable[[Any], Any],
    postprocessor: Callable[[torch.Tensor], Any],
    base_policy_factory: Callable[[], Any],
    fine_tuned_policy_factory: Callable[[], Any],
    noise_seed: int,
    noise_device: str | torch.device = "cpu",
    noise_dtype: torch.dtype = torch.float32,
    num_inference_steps: int | None = None,
    processed_observation_validator: Callable[[Any], None] | None = None,
) -> PolicyComparisonInferenceResult:
    """Run policies through non-overlapping context-managed lifecycles."""
    if not callable(base_policy_factory):
        _fail("base_policy_factory", "must be callable.")
    if not callable(fine_tuned_policy_factory):
        _fail("fine_tuned_policy_factory", "must be callable.")
    canonical, snapshot, noise, provenance = _prepare_comparison(
        observation,
        bound_input,
        preprocessor,
        postprocessor,
        noise_seed,
        noise_device,
        noise_dtype,
        num_inference_steps,
        processed_observation_validator,
    )
    outputs: list[tuple[str, str, torch.Tensor]] = []
    factories = (
        (BASE_POLICY_ID, BASE_POLICY_LABEL, base_policy_factory),
        (FINE_TUNED_POLICY_ID, FINE_TUNED_POLICY_LABEL, fine_tuned_policy_factory),
    )
    for policy_id, label, factory in factories:
        try:
            manager = factory()
        except Exception as error:
            raise PolicyComparisonInferenceError(
                f"{policy_id}.factory failed: {error}"
            ) from error
        if not hasattr(manager, "__enter__") or not hasattr(manager, "__exit__"):
            _fail(f"{policy_id}.factory", "must return a context manager.")
        policy = None
        try:
            with manager as policy:
                raw = _invoke_policy(
                    policy, policy_id, canonical, noise, num_inference_steps
                )
        except PolicyComparisonInferenceError:
            raise
        except Exception as error:
            raise PolicyComparisonInferenceError(
                f"{policy_id}.lifecycle failed: {error}"
            ) from error
        finally:
            policy = None
            manager = None
        outputs.append((policy_id, label, raw))
        if not _equal_supported(canonical, snapshot):
            _fail(
                "processed_observation", "canonical value was mutated during inference."
            )
    return _comparison_result(
        observation, outputs, postprocessor, provenance, num_inference_steps
    )
