from dataclasses import FrozenInstanceError, replace
import inspect
from pathlib import Path

import pytest
import torch

from lerobot_state_atlas.checkpoint_comparison.inference import (
    MAX_TORCH_GENERATOR_SEED,
    PolicyComparisonInferenceError,
    build_deterministic_noise,
    run_policy_comparison,
)
from lerobot_state_atlas.checkpoint_comparison.models import (
    BoundCameraInput,
    BoundPolicyObservationInput,
    ObservationCamera,
    ObservationDatasetIdentity,
    ObservationRecordedGroundTruth,
    ObservationState,
    PolicyComparisonObservation,
)
from lerobot_state_atlas.checkpoint_comparison.observation import COMPONENT_NAMES


def observation() -> PolicyComparisonObservation:
    cameras = tuple(
        ObservationCamera(
            feature_name=feature_name,
            filename=f"camera-{index}.png",
            path=Path(f"/synthetic/camera-{index}.png"),
            width=224,
            height=224,
            channels=3,
            byte_count=100 + index,
            sha256=str(index) * 64,
        )
        for index, feature_name in enumerate(
            (
                "observation.images.left_wrist",
                "observation.images.right_wrist",
                "observation.images.top",
            )
        )
    )
    return PolicyComparisonObservation(
        manifest_path=Path("/synthetic/observation.json"),
        manifest_sha256="f" * 64,
        manifest_byte_count=1,
        observation_id="observation-123",
        dataset=ObservationDatasetIdentity(
            repository_id="example/dataset",
            revision="a" * 40,
            episode_index=1,
            frame_index=10,
            timestamp_seconds=0.2,
            fps=50.0,
            task="Actuator Unboxing",
        ),
        prompt="Unbox the actuator.",
        state=ObservationState(
            feature_name="observation.state",
            dtype="float32",
            component_names=COMPONENT_NAMES,
            values=tuple(float(index) for index in range(14)),
        ),
        cameras=cameras,
        recorded_ground_truth=ObservationRecordedGroundTruth(
            available=False, reason="Synthetic orchestration test."
        ),
    )


def bound_input(
    source: PolicyComparisonObservation | None = None,
) -> BoundPolicyObservationInput:
    source = source or observation()
    return BoundPolicyObservationInput(
        observation_id=source.observation_id,
        cameras=tuple(
            BoundCameraInput(
                feature_name=camera.feature_name,
                tensor=torch.full((1, 3, 224, 224), index, dtype=torch.uint8),
                source_sha256=camera.sha256,
                source_byte_count=camera.byte_count,
                source_path=camera.path,
            )
            for index, camera in enumerate(source.cameras)
        ),
    )


class CapturingPolicy:
    def __init__(self, value: float, order: list[str], policy_id: str) -> None:
        self.value = value
        self.order = order
        self.policy_id = policy_id
        self.observations = []
        self.noises = []
        self.steps = []

    def predict_action_chunk(
        self, processed_observation, *, noise, num_inference_steps=None
    ):
        self.order.append(self.policy_id)
        self.observations.append(processed_observation)
        self.noises.append(noise)
        self.steps.append(num_inference_steps)
        return torch.full((1, 50, 14), self.value)


class CountingPostprocessor:
    def __init__(self) -> None:
        self.inputs = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value)
        return value + 10


def run(**overrides):
    order: list[str] = []
    base = CapturingPolicy(1.0, order, "base-pi05")
    fine = CapturingPolicy(2.0, order, "fine-tuned-pi05")
    postprocessor = CountingPostprocessor()
    calls = []

    def preprocessor(value):
        calls.append(value)
        return {**value, "nested": [torch.tensor([3.0]), ("prompt",)]}

    source = observation()
    kwargs = {
        "observation": source,
        "bound_input": bound_input(source),
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "base_policy": base,
        "fine_tuned_policy": fine,
        "noise_seed": 42,
    }
    kwargs.update(overrides)
    result = run_policy_comparison(**kwargs)
    return result, base, fine, postprocessor, calls, order


def test_noise_is_deterministic_explicit_and_does_not_change_global_rng() -> None:
    global_state = torch.random.get_rng_state().clone()
    first, first_provenance = build_deterministic_noise(123)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    second, second_provenance = build_deterministic_noise(123)
    other, other_provenance = build_deterministic_noise(124)
    assert tuple(first.shape) == (1, 50, 32)
    assert torch.equal(first, second)
    assert first_provenance == second_provenance
    assert first_provenance.sha256 != other_provenance.sha256
    assert not torch.equal(first, other)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("seed", [True, -1, MAX_TORCH_GENERATOR_SEED + 1, 1.5])
def test_noise_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(PolicyComparisonInferenceError, match="noise_seed"):
        build_deterministic_noise(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, "float32"])
def test_noise_rejects_nonfloating_or_non_torch_dtype(dtype: object) -> None:
    with pytest.raises(PolicyComparisonInferenceError, match="noise_dtype"):
        build_deterministic_noise(1, dtype=dtype)  # type: ignore[arg-type]


@pytest.mark.parametrize("device", ["meta", "not-a-device", "cuda"])
def test_noise_rejects_unsupported_or_unavailable_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment.")
    with pytest.raises(PolicyComparisonInferenceError, match="noise_device"):
        build_deterministic_noise(1, device=device)


def test_shared_preprocessing_policy_order_noise_and_postprocessing() -> None:
    result, base, fine, postprocessor, calls, order = run(num_inference_steps=7)
    assert len(calls) == 1
    assert order == ["base-pi05", "fine-tuned-pi05"]
    assert torch.equal(base.noises[0], fine.noises[0])
    assert base.noises[0].data_ptr() != fine.noises[0].data_ptr()
    assert (
        base.observations[0]["observation.state"].data_ptr()
        != fine.observations[0]["observation.state"].data_ptr()
    )
    assert torch.equal(
        base.observations[0]["nested"][0], fine.observations[0]["nested"][0]
    )
    assert (
        base.observations[0]["nested"][0].data_ptr()
        != fine.observations[0]["nested"][0].data_ptr()
    )
    assert base.steps == fine.steps == [7]
    assert len(postprocessor.inputs) == 100
    assert all(tuple(value.shape) == (1, 14) for value in postprocessor.inputs)
    assert tuple(plan.policy_id for plan in result.policies) == (
        "base-pi05",
        "fine-tuned-pi05",
    )
    assert tuple(plan.label for plan in result.policies) == (
        "Base π0.5",
        "Fine-tuned π0.5",
    )
    assert result.policies[0].actions[0] == tuple(11.0 for _ in range(14))
    assert result.policies[1].actions[0] == tuple(12.0 for _ in range(14))
    assert result.policies[0].relative_times_seconds == tuple(i / 50 for i in range(50))
    assert result.noise.shape == (1, 50, 32)
    assert result.shared_preprocessing and result.shared_postprocessing
    assert calls[0]["task"] == observation().prompt
    assert torch.equal(
        calls[0]["observation.state"],
        torch.tensor(observation().state.values).reshape(1, 14),
    )
    for feature_name in (
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.top",
    ):
        assert torch.equal(
            base.observations[0][feature_name], fine.observations[0][feature_name]
        )
        assert (
            base.observations[0][feature_name].data_ptr()
            != fine.observations[0][feature_name].data_ptr()
        )


def test_processed_observation_validator_receives_clone() -> None:
    seen = []

    def validator(value):
        seen.append(value)
        value["observation.state"].add_(99)

    _, base, _, _, _, _ = run(processed_observation_validator=validator)
    assert len(seen) == 1
    assert torch.equal(
        base.observations[0]["observation.state"],
        torch.tensor(observation().state.values).reshape(1, 14),
    )


def test_rejects_unsupported_mutable_processed_value() -> None:
    with pytest.raises(PolicyComparisonInferenceError, match="unsupported mutable"):
        run(preprocessor=lambda value: {"bad": object()})


def test_public_api_has_no_unbound_raw_observation_parameter() -> None:
    assert "raw_observation" not in inspect.signature(run_policy_comparison).parameters
    with pytest.raises(TypeError, match="raw_observation"):
        run(raw_observation={"observation.state": torch.full((1, 14), 99.0)})


def test_state_and_prompt_are_constructed_only_from_validated_observation() -> None:
    source = observation()
    seen = []

    def preprocessor(value):
        seen.append(value)
        return value

    run(observation=source, bound_input=bound_input(source), preprocessor=preprocessor)
    assert seen[0]["task"] == source.prompt
    assert torch.equal(
        seen[0]["observation.state"],
        torch.tensor(source.state.values, dtype=torch.float32).reshape(1, 14),
    )
    assert set(seen[0]) == {
        "observation.state",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.top",
        "task",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_sha256", "f" * 64, r"cameras\[0\]\.source_sha256"),
        ("source_byte_count", 999, r"cameras\[0\]\.source_byte_count"),
        ("source_byte_count", True, r"cameras\[0\]\.source_byte_count"),
        ("feature_name", "observation.images.top", r"cameras\[0\]\.feature_name"),
        ("source_path", Path("/other/image.png"), r"cameras\[0\]\.source_path"),
    ],
)
def test_rejects_mismatched_camera_provenance_before_preprocessing(
    field: str, value: object, message: str
) -> None:
    source = observation()
    binding = bound_input(source)
    cameras = list(binding.cameras)
    cameras[0] = replace(cameras[0], **{field: value})
    invalid = replace(binding, cameras=tuple(cameras))
    preprocessor_calls = []
    policy_order = []

    with pytest.raises(PolicyComparisonInferenceError, match=message):
        run(
            observation=source,
            bound_input=invalid,
            preprocessor=lambda value: preprocessor_calls.append(value) or value,
            base_policy=CapturingPolicy(1, policy_order, "base"),
        )
    assert preprocessor_calls == []
    assert policy_order == []


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "reordered"])
def test_rejects_invalid_camera_collection_before_preprocessing(kind: str) -> None:
    source = observation()
    binding = bound_input(source)
    cameras = list(binding.cameras)
    if kind == "missing":
        cameras.pop()
    elif kind == "extra":
        cameras.append(cameras[-1])
    elif kind == "duplicate":
        cameras[1] = cameras[0]
    else:
        cameras[0], cameras[1] = cameras[1], cameras[0]
    preprocessor_calls = []

    with pytest.raises(PolicyComparisonInferenceError, match="bound_input.cameras"):
        run(
            observation=source,
            bound_input=replace(binding, cameras=tuple(cameras)),
            preprocessor=lambda value: preprocessor_calls.append(value) or value,
        )
    assert preprocessor_calls == []


def test_rejects_binding_for_another_observation_before_preprocessing() -> None:
    source = observation()
    binding = replace(bound_input(source), observation_id="different-observation")
    calls = []
    with pytest.raises(
        PolicyComparisonInferenceError, match="bound_input.observation_id"
    ):
        run(
            observation=source,
            bound_input=binding,
            preprocessor=lambda value: calls.append(value) or value,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (object(), r"cameras\[0\]\.tensor must be a torch.Tensor"),
        (torch.zeros(3, 224, 224), r"cameras\[0\]\.tensor shape"),
        (
            torch.zeros(1, 3, 224, 224, dtype=torch.int32),
            r"cameras\[0\]\.tensor.dtype",
        ),
        (
            torch.zeros(1, 3, 224, 224, device="meta"),
            r"cameras\[0\]\.tensor.device",
        ),
        (
            torch.full((1, 3, 224, 224), float("nan")),
            r"cameras\[0\]\.tensor must contain only finite",
        ),
        (
            torch.zeros(1, 3, 224, 224, requires_grad=True),
            r"cameras\[0\]\.tensor.requires_grad",
        ),
    ],
)
def test_rejects_invalid_camera_tensor_contract(tensor: object, message: str) -> None:
    binding = bound_input()
    cameras = list(binding.cameras)
    cameras[0] = replace(cameras[0], tensor=tensor)
    with pytest.raises(PolicyComparisonInferenceError, match=message):
        run(bound_input=replace(binding, cameras=tuple(cameras)))


def test_valid_float_camera_tensor_is_accepted() -> None:
    binding = bound_input()
    cameras = list(binding.cameras)
    cameras[0] = replace(
        cameras[0], tensor=torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    )
    result, *_ = run(bound_input=replace(binding, cameras=tuple(cameras)))
    assert result.observation_id == observation().observation_id


def test_result_identity_timing_and_sources_share_one_validated_observation(
    tmp_path: Path,
) -> None:
    source = observation()
    manifest_path = tmp_path / "observation.json"
    manifest_path.write_bytes(b"validated manifest bytes")
    cameras = []
    camera_bytes = {}
    for index, camera in enumerate(source.cameras):
        path = tmp_path / f"camera-{index}.png"
        content = f"camera-{index}-bytes".encode()
        path.write_bytes(content)
        camera_bytes[path] = content
        cameras.append(replace(camera, path=path.resolve()))
    source = replace(
        source,
        manifest_path=manifest_path.resolve(),
        cameras=tuple(cameras),
        dataset=replace(source.dataset, fps=25.0),
    )
    result, *_ = run(observation=source, bound_input=bound_input(source))
    assert result.observation_id == source.observation_id
    assert result.policies[0].relative_times_seconds == tuple(
        index / 25.0 for index in range(50)
    )
    assert manifest_path.read_bytes() == b"validated manifest bytes"
    assert {path: path.read_bytes() for path in camera_bytes} == camera_bytes


def test_mutating_policy_input_is_detected_and_stops_second_policy() -> None:
    order = []

    class MutatingPolicy:
        def predict_action_chunk(self, value, *, noise):
            order.append("base")
            value["observation.state"].add_(1)
            return torch.zeros(1, 50, 14)

    fine = CapturingPolicy(2, order, "fine")
    with pytest.raises(PolicyComparisonInferenceError, match="base-pi05 mutated"):
        run_policy_comparison(
            observation(),
            bound_input=bound_input(),
            preprocessor=lambda value: value,
            postprocessor=lambda value: value,
            base_policy=MutatingPolicy(),
            fine_tuned_policy=fine,
            noise_seed=1,
        )
    assert order == ["base"]


def test_mutating_noise_is_detected() -> None:
    class MutatingNoisePolicy:
        def predict_action_chunk(self, value, *, noise):
            noise.zero_()
            return torch.zeros(1, 50, 14)

    with pytest.raises(PolicyComparisonInferenceError, match="explicit noise"):
        run(base_policy=MutatingNoisePolicy())


def test_mutation_is_reported_even_when_policy_then_raises() -> None:
    class MutatingFailingPolicy:
        def predict_action_chunk(self, value, *, noise):
            value["observation.state"].add_(1)
            raise RuntimeError("after mutation")

    with pytest.raises(PolicyComparisonInferenceError, match="mutated.*before failing"):
        run(base_policy=MutatingFailingPolicy())


def test_incompatible_policy_signature_is_clear() -> None:
    class NoNoisePolicy:
        def predict_action_chunk(self, value):
            return torch.zeros(1, 50, 14)

    with pytest.raises(
        PolicyComparisonInferenceError, match="incompatible explicit-noise"
    ):
        run(base_policy=NoNoisePolicy())


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ([], "torch.Tensor"),
        (torch.zeros(50, 14), r"received \[50, 14\]"),
        (torch.zeros(2, 50, 14), r"received \[2, 50, 14\]"),
        (torch.zeros(1, 49, 14), r"received \[1, 49, 14\]"),
        (torch.zeros(1, 50, 13), r"received \[1, 50, 13\]"),
        (torch.zeros(1, 50, 14, dtype=torch.int64), "floating dtype"),
        (torch.full((1, 50, 14), float("nan")), "finite values"),
        (torch.full((1, 50, 14), float("inf")), "finite values"),
        (torch.full((1, 50, 14), -float("inf")), "finite values"),
    ],
)
def test_rejects_invalid_policy_output(output: object, message: str) -> None:
    class Policy:
        def predict_action_chunk(self, value, *, noise):
            return output

    with pytest.raises(PolicyComparisonInferenceError, match=message):
        run(base_policy=Policy())


def test_policy_output_must_not_alias_noise() -> None:
    class AliasingPolicy:
        def predict_action_chunk(self, value, *, noise):
            return noise[:, :, 1:15]

    with pytest.raises(PolicyComparisonInferenceError, match="must not alias"):
        run(base_policy=AliasingPolicy())


def test_policy_failure_returns_no_partial_result() -> None:
    order = []
    base = CapturingPolicy(1.0, order, "base")

    class FailingPolicy:
        def predict_action_chunk(self, value, *, noise):
            order.append("fine")
            raise RuntimeError("failed")

    with pytest.raises(PolicyComparisonInferenceError, match="fine-tuned-pi05.*failed"):
        run(base_policy=base, fine_tuned_policy=FailingPolicy())
    assert order == ["base", "fine"]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ([], "torch.Tensor"),
        (torch.zeros(14), r"received \[14\]"),
        (torch.zeros(1, 13), r"received \[1, 13\]"),
        (torch.zeros(1, 14, dtype=torch.int64), "floating dtype"),
        (torch.full((1, 14), float("nan")), "finite values"),
    ],
)
def test_rejects_invalid_postprocessor_output_with_step_path(
    result, message: str
) -> None:
    with pytest.raises(
        PolicyComparisonInferenceError,
        match=rf"base-pi05\.postprocessing\[0\].*{message}",
    ):
        run(postprocessor=lambda value: result)


def test_postprocessor_mutation_cannot_change_other_steps_or_policy_output() -> None:
    seen = []

    def mutating_postprocessor(value):
        seen.append(value.clone())
        value.add_(100)
        return value

    result, *_ = run(postprocessor=mutating_postprocessor)
    assert all(torch.equal(value, torch.ones(1, 14)) for value in seen[:50])
    assert all(torch.equal(value, torch.full((1, 14), 2.0)) for value in seen[50:])
    assert result.policies[0].actions[0] == tuple(101.0 for _ in range(14))


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_rejects_invalid_num_inference_steps(value: object) -> None:
    with pytest.raises(PolicyComparisonInferenceError, match="num_inference_steps"):
        run(num_inference_steps=value)


def test_result_models_are_immutable_and_contain_no_first_checkpoint_wording() -> None:
    result, *_ = run()
    with pytest.raises(FrozenInstanceError):
        result.chunk_length = 1
    assert "first checkpoint" not in repr(result).lower()
