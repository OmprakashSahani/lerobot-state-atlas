from contextlib import contextmanager
import gc
import weakref

import pytest
import torch

from lerobot_state_atlas.checkpoint_comparison import (
    PI05PolicyAdapterError,
    PI05PredictActionChunkAdapter,
    PolicyComparisonInferenceError,
    run_sequential_policy_comparison,
    validate_processed_observation_device,
)
from lerobot_state_atlas.checkpoint_comparison.models import (
    DeterministicNoiseProvenance,
)
import lerobot_state_atlas.checkpoint_comparison.inference as inference_module
from test_checkpoint_inference import bound_input, observation


class Policy:
    def __init__(self, value, events, name, *, fail=False):
        self.value = value
        self.events = events
        self.name = name
        self.fail = fail
        self.inputs = []
        self.noises = []

    def predict_action_chunk(self, processed, *, noise, num_inference_steps=None):
        self.events.append(f"{self.name}:run")
        self.inputs.append(processed)
        self.noises.append(noise)
        if self.fail:
            raise RuntimeError("inference exploded")
        return torch.full((1, 50, 14), self.value)


def factory(policy, events, *, cleanup_failure=False):
    @contextmanager
    def manager():
        events.append(f"{policy.name}:enter")
        try:
            yield policy
        finally:
            events.append(f"{policy.name}:exit")
            if cleanup_failure:
                raise RuntimeError("cleanup exploded")

    def build():
        events.append(f"{policy.name}:factory")
        return manager()

    return build


def test_sequential_lifecycle_preprocesses_once_and_releases_base_first() -> None:
    events = []
    base = Policy(1.0, events, "base")
    fine = Policy(2.0, events, "fine")
    preprocessing = []
    postprocessing = []
    rng = torch.random.get_rng_state().clone()
    result = run_sequential_policy_comparison(
        observation(),
        bound_input=bound_input(),
        preprocessor=lambda value: preprocessing.append(value) or value,
        postprocessor=lambda value: postprocessing.append(value) or value,
        base_policy_factory=factory(base, events),
        fine_tuned_policy_factory=factory(fine, events),
        noise_seed=12,
        num_inference_steps=7,
    )
    assert events == [
        "base:factory",
        "base:enter",
        "base:run",
        "base:exit",
        "fine:factory",
        "fine:enter",
        "fine:run",
        "fine:exit",
    ]
    assert len(preprocessing) == 1
    assert len(postprocessing) == 100
    assert torch.equal(base.noises[0], fine.noises[0])
    assert base.noises[0].data_ptr() != fine.noises[0].data_ptr()
    assert (
        base.inputs[0]["observation.state"].data_ptr()
        != fine.inputs[0]["observation.state"].data_ptr()
    )
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert tuple(plan.policy_id for plan in result.policies) == (
        "base-pi05",
        "fine-tuned-pi05",
    )


def test_recursive_processed_observation_device_validation_has_precise_path() -> None:
    valid = {
        "state": torch.zeros(1),
        "nested": ["metadata", (torch.ones(1),)],
    }
    validate_processed_observation_device(valid, "cpu")
    invalid = {
        "state": torch.zeros(1),
        "nested": ["metadata", (torch.empty(1, device="meta"),)],
    }
    with pytest.raises(
        PolicyComparisonInferenceError,
        match=r"processed_observation\.nested\[1\]\[0\].*cpu.*meta",
    ):
        validate_processed_observation_device(invalid, "cpu")

    with pytest.raises(
        PolicyComparisonInferenceError,
        match=r"processed_observation\.nested\.cpu_value.*cuda:1.*cpu",
    ):
        validate_processed_observation_device(
            {"nested": {"cpu_value": torch.zeros(1)}}, "cuda:1"
        )


def test_processed_observation_rejects_wrong_cuda_index_without_cuda_allocation() -> (
    None
):
    class DeviceReportingTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:2")

    reported_cuda_tensor = torch.empty(1).as_subclass(DeviceReportingTensor)
    with pytest.raises(
        PolicyComparisonInferenceError,
        match=r"processed_observation\.tokens.*cuda:1.*cuda:2",
    ):
        validate_processed_observation_device(
            {"tokens": reported_cuda_tensor}, "cuda:1"
        )


def test_processed_device_validation_precedes_policy_factory() -> None:
    called = False

    def policy_factory():
        nonlocal called
        called = True
        raise AssertionError("must not be called")

    with pytest.raises(PolicyComparisonInferenceError, match="tensor device"):
        run_sequential_policy_comparison(
            observation(),
            bound_input=bound_input(),
            preprocessor=lambda value: {
                "nested": {"wrong": torch.empty(1, device="meta")}
            },
            postprocessor=lambda value: value,
            base_policy_factory=policy_factory,
            fine_tuned_policy_factory=policy_factory,
            noise_seed=1,
            processed_observation_validator=lambda value: (
                validate_processed_observation_device(value, "cpu")
            ),
        )
    assert not called


def test_mismatched_constructed_noise_device_precedes_policy_factory(
    monkeypatch,
) -> None:
    called = False

    def policy_factory():
        nonlocal called
        called = True
        raise AssertionError("must not be called")

    monkeypatch.setattr(
        inference_module,
        "build_deterministic_noise",
        lambda *args, **kwargs: (
            torch.zeros((1, 50, 32)),
            DeterministicNoiseProvenance(
                seed=1,
                generator="fake",
                construction_device="cpu",
                dtype="float32",
                shape=(1, 50, 32),
                sha256="0" * 64,
            ),
        ),
    )
    with pytest.raises(
        PolicyComparisonInferenceError, match="noise.device.*cuda:1.*cpu"
    ):
        run_sequential_policy_comparison(
            observation(),
            bound_input=bound_input(),
            preprocessor=lambda value: value,
            postprocessor=lambda value: value,
            base_policy_factory=policy_factory,
            fine_tuned_policy_factory=policy_factory,
            noise_seed=1,
            noise_device="cuda:1",
        )
    assert not called


def test_sequential_adapters_release_models_before_next_factory() -> None:
    events = []
    resident = set()
    base_reference = None
    captured_base_output = None

    class Wrapped:
        def __init__(self, name, value):
            self.name = name
            self.value = value
            resident.add(name)
            events.append(f"{name}:enter")

        def predict_action_chunk(self, _processed, *, noise, num_steps=None):
            nonlocal captured_base_output
            events.append(f"{self.name}:inference")
            output = torch.full((1, 50, 14), self.value)
            if self.name == "base":
                captured_base_output = output
            return output

        def __del__(self):
            resident.discard(self.name)

    def make_factory(name, value):
        def build():
            nonlocal base_reference
            events.append(f"{name}:construct")
            if name == "fine":
                gc.collect()
                assert base_reference() is None
                assert resident == set()

            @contextmanager
            def manager():
                nonlocal base_reference
                model = Wrapped(name, value)
                if name == "base":
                    base_reference = weakref.ref(model)
                adapter = PI05PredictActionChunkAdapter(model)
                try:
                    yield adapter
                finally:
                    events.append(f"{name}:adapter-release")
                    adapter.release()
                    model = None
                    gc.collect()
                    events.append(f"{name}:cleanup")

            return manager()

        return build

    result = run_sequential_policy_comparison(
        observation(),
        bound_input=bound_input(),
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
        base_policy_factory=make_factory("base", 1.0),
        fine_tuned_policy_factory=make_factory("fine", 2.0),
        noise_seed=4,
    )
    assert events == [
        "base:construct",
        "base:enter",
        "base:inference",
        "base:adapter-release",
        "base:cleanup",
        "fine:construct",
        "fine:enter",
        "fine:inference",
        "fine:adapter-release",
        "fine:cleanup",
    ]
    assert resident == set()
    assert captured_base_output is not None
    assert torch.equal(captured_base_output, torch.ones((1, 50, 14)))
    assert result.policies[0].actions[0][0] == 1.0
    assert not any(
        isinstance(value, PI05PredictActionChunkAdapter)
        for value in result.__dict__.values()
    )


@pytest.mark.parametrize("failure", ["inference", "output-validation"])
def test_sequential_adapter_releases_after_policy_failure(failure) -> None:
    events = []
    fine_called = False

    class Wrapped:
        def predict_action_chunk(self, _processed, *, noise, num_steps=None):
            if failure == "inference":
                raise RuntimeError("inference exploded")
            return torch.zeros((1, 49, 14))

    @contextmanager
    def manager():
        adapter = PI05PredictActionChunkAdapter(Wrapped())
        try:
            yield adapter
        finally:
            adapter.release()
            events.append("released")

    def fine_factory():
        nonlocal fine_called
        fine_called = True
        return manager()

    with pytest.raises(PolicyComparisonInferenceError):
        run_sequential_policy_comparison(
            observation(),
            bound_input=bound_input(),
            preprocessor=lambda value: value,
            postprocessor=lambda value: value,
            base_policy_factory=lambda: manager(),
            fine_tuned_policy_factory=fine_factory,
            noise_seed=1,
        )
    assert events == ["released"]
    assert not fine_called


def test_base_release_failure_prevents_fine_factory() -> None:
    fine_called = False

    @contextmanager
    def manager():
        adapter = PI05PredictActionChunkAdapter(Policy(1.0, [], "base"))
        try:
            yield adapter
        finally:
            adapter.release()
            raise RuntimeError("release exploded")

    def fine_factory():
        nonlocal fine_called
        fine_called = True
        return manager()

    with pytest.raises(PolicyComparisonInferenceError, match="release exploded"):
        run_sequential_policy_comparison(
            observation(),
            bound_input=bound_input(),
            preprocessor=lambda value: value,
            postprocessor=lambda value: value,
            base_policy_factory=lambda: manager(),
            fine_tuned_policy_factory=fine_factory,
            noise_seed=1,
        )
    assert not fine_called


def test_fine_release_failure_prevents_success() -> None:
    @contextmanager
    def manager(name):
        adapter = PI05PredictActionChunkAdapter(Policy(1.0, [], name))
        try:
            yield adapter
        finally:
            adapter.release()
            if name == "fine":
                raise RuntimeError("fine release exploded")

    with pytest.raises(PolicyComparisonInferenceError, match="fine release exploded"):
        run_sequential_policy_comparison(
            observation(),
            bound_input=bound_input(),
            preprocessor=lambda value: value,
            postprocessor=lambda value: value,
            base_policy_factory=lambda: manager("base"),
            fine_tuned_policy_factory=lambda: manager("fine"),
            noise_seed=1,
        )


@pytest.mark.parametrize("failure", ["factory", "enter", "inference", "cleanup"])
def test_base_failure_prevents_fine_factory(failure: str) -> None:
    events = []
    base = Policy(1.0, events, "base", fail=failure == "inference")
    fine_called = False

    @contextmanager
    def base_manager():
        events.append("base:enter")
        if failure == "enter":
            raise RuntimeError("enter exploded")
        try:
            yield base
        finally:
            events.append("base:exit")
            if failure == "cleanup":
                raise RuntimeError("cleanup exploded")

    def base_factory():
        if failure == "factory":
            raise RuntimeError("factory exploded")
        return base_manager()

    def fine_factory():
        nonlocal fine_called
        fine_called = True
        return factory(Policy(2, events, "fine"), events)()

    with pytest.raises(PolicyComparisonInferenceError):
        run_sequential_policy_comparison(
            observation(),
            bound_input=bound_input(),
            preprocessor=lambda value: value,
            postprocessor=lambda value: value,
            base_policy_factory=base_factory,
            fine_tuned_policy_factory=fine_factory,
            noise_seed=1,
        )
    assert not fine_called
    if failure == "inference":
        assert "base:exit" in events


def test_fine_failure_returns_no_partial_result_and_cleans_context() -> None:
    events = []
    with pytest.raises(PolicyComparisonInferenceError, match="fine-tuned-pi05"):
        run_sequential_policy_comparison(
            observation(),
            bound_input=bound_input(),
            preprocessor=lambda value: value,
            postprocessor=lambda value: value,
            base_policy_factory=factory(Policy(1.0, events, "base"), events),
            fine_tuned_policy_factory=factory(
                Policy(2.0, events, "fine", fail=True), events
            ),
            noise_seed=1,
        )
    assert events[-1] == "fine:exit"


def test_pi05_adapter_translates_override_without_forwarding_original_name() -> None:
    class Wrapped:
        def __init__(self):
            self.kwargs = None

        def predict_action_chunk(self, observation, *, noise, num_steps=None):
            self.kwargs = (observation, noise, num_steps)
            return "result"

    wrapped = Wrapped()
    adapter = PI05PredictActionChunkAdapter(wrapped)
    noise = object()
    assert (
        adapter.predict_action_chunk("obs", noise=noise, num_inference_steps=9)
        == "result"
    )
    assert wrapped.kwargs == ("obs", noise, 9)
    adapter.predict_action_chunk("obs", noise=noise)
    assert wrapped.kwargs == ("obs", noise, None)


@pytest.mark.parametrize("value", [True, 0, -1])
def test_pi05_adapter_rejects_invalid_override(value) -> None:
    class Wrapped:
        def predict_action_chunk(self, observation, *, noise, num_steps=None):
            return None

    with pytest.raises(PI05PolicyAdapterError, match="positive integer"):
        PI05PredictActionChunkAdapter(Wrapped()).predict_action_chunk(
            {}, noise=torch.zeros(1), num_inference_steps=value
        )


def test_pi05_adapter_diagnoses_signature_but_preserves_internal_type_error() -> None:
    class WrongSignature:
        def predict_action_chunk(self, observation):
            return None

    with pytest.raises(PI05PolicyAdapterError, match="incompatible"):
        PI05PredictActionChunkAdapter(WrongSignature()).predict_action_chunk(
            {}, noise=object()
        )

    class InternalFailure:
        def predict_action_chunk(self, observation, *, noise):
            raise TypeError("internal bug")

    with pytest.raises(TypeError, match="internal bug"):
        PI05PredictActionChunkAdapter(InternalFailure()).predict_action_chunk(
            {}, noise=object()
        )
