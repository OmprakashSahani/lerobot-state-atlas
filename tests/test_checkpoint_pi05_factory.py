from contextlib import contextmanager
import gc
import weakref
import pytest
import torch

from lerobot_state_atlas.checkpoint_comparison import (
    PI05PolicyAdapterError,
    PI05PolicyFactoryError,
    PI05PredictActionChunkAdapter,
    create_pi05_policy_factory,
)
from test_checkpoint_compatibility import config
from lerobot_state_atlas.checkpoint_comparison.compatibility import (
    adapt_pi05_finetuned_config,
)
from test_checkpoint_runner_manifest import _manifest
from lerobot_state_atlas.checkpoint_comparison import (
    load_checkpoint_comparison_runner_manifest,
)


class FakePolicy:
    def __init__(self, events):
        self.events = events

    def to(self, **kwargs):
        self.events.append(("to", kwargs))
        return self

    def eval(self):
        self.events.append("eval")

    def predict_action_chunk(self, observation, *, noise, num_steps=None):
        return torch.zeros((1, 50, 14))


def test_factory_stages_loads_releases_moves_yields_and_cleans(
    tmp_path, monkeypatch
) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    compatibility = adapt_pi05_finetuned_config(
        config(),
        "0.6.0",
        {"compile_model": False, "gradient_checkpointing": False},
    )
    events = []
    original_release = PI05PredictActionChunkAdapter.release

    def recorded_release(adapter):
        events.append("release")
        original_release(adapter)

    monkeypatch.setattr(PI05PredictActionChunkAdapter, "release", recorded_release)

    @contextmanager
    def stager(*args, **kwargs):
        events.append(("stage", kwargs["checkpoint_kind"]))
        yield "staged"
        events.append("stage-exit")

    def loader(policy, staged, kind, **kwargs):
        events.append(("load", kind, staged, kwargs["drop_unused_lm_head"]))

    factory = create_pi05_policy_factory(
        policy_id="base-pi05",
        effective_config=compatibility,
        checkpoint_input=manifest.base_checkpoint,
        checkpoint_path=tmp_path / "base.safetensors",
        runtime=manifest.runtime,
        staging_parent=tmp_path,
        policy_constructor=lambda config: (
            events.append(("construct", config["device"])) or FakePolicy(events)
        ),
        checkpoint_stager=stager,
        checkpoint_loader=loader,
        cleanup_hooks=(lambda: events.append("cleanup"),),
    )
    with factory() as adapter:
        assert adapter.predict_action_chunk(
            {}, noise=torch.zeros(1), num_inference_steps=3
        ).shape == (1, 50, 14)
    assert events[:5] == [
        ("stage", "base"),
        ("construct", "cpu"),
        ("load", "base", "staged", compatibility.drop_unused_lm_head),
        ("to", {"device": "cuda:0", "dtype": torch.bfloat16}),
        "eval",
    ]
    assert events[-3:] == ["release", "stage-exit", "cleanup"]
    with pytest.raises(PI05PolicyAdapterError, match="has been released"):
        adapter.predict_action_chunk({}, noise=torch.zeros(1))


def test_factory_releases_model_before_cleanup_hook(tmp_path) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    compatibility = adapt_pi05_finetuned_config(config(), "0.6.0")
    model_reference = None
    events = []

    def constructor(_config):
        nonlocal model_reference
        model = FakePolicy(events)
        model_reference = weakref.ref(model)
        return model

    @contextmanager
    def stager(*args, **kwargs):
        yield "staged"

    def cleanup():
        gc.collect()
        events.append(("cleanup-model-alive", model_reference() is not None))

    factory = create_pi05_policy_factory(
        policy_id="base-pi05",
        effective_config=compatibility,
        checkpoint_input=manifest.base_checkpoint,
        checkpoint_path=tmp_path / "base.safetensors",
        runtime=manifest.runtime,
        staging_parent=tmp_path,
        policy_constructor=constructor,
        checkpoint_stager=stager,
        checkpoint_loader=lambda *args, **kwargs: None,
        cleanup_hooks=(cleanup,),
    )
    with factory():
        pass
    assert model_reference() is None
    assert events[-1] == ("cleanup-model-alive", False)


def test_adapter_release_is_idempotent_and_irrevocable() -> None:
    policy = FakePolicy([])
    adapter = PI05PredictActionChunkAdapter(policy)
    adapter.release()
    adapter.release()
    with pytest.raises(
        PI05PolicyAdapterError, match="PI05 policy adapter has been released"
    ):
        adapter.predict_action_chunk({}, noise=torch.zeros(1))


def test_release_failure_runs_cleanup_and_preserves_original_failure(
    tmp_path, monkeypatch
) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    compatibility = adapt_pi05_finetuned_config(config(), "0.6.0")
    events = []
    original_release = PI05PredictActionChunkAdapter.release

    def failing_release(adapter):
        original_release(adapter)
        events.append("release")
        raise RuntimeError("release exploded")

    monkeypatch.setattr(PI05PredictActionChunkAdapter, "release", failing_release)

    @contextmanager
    def stager(*args, **kwargs):
        yield "staged"

    factory = create_pi05_policy_factory(
        policy_id="base-pi05",
        effective_config=compatibility,
        checkpoint_input=manifest.base_checkpoint,
        checkpoint_path=tmp_path / "base.safetensors",
        runtime=manifest.runtime,
        staging_parent=tmp_path,
        policy_constructor=lambda _config: FakePolicy(events),
        checkpoint_stager=stager,
        checkpoint_loader=lambda *args, **kwargs: None,
        cleanup_hooks=(lambda: events.append("cleanup"),),
    )
    with pytest.raises(
        PI05PolicyFactoryError,
        match=r"release exploded.*original failure.*inference exploded",
    ):
        with factory():
            raise RuntimeError("inference exploded")
    assert events[-2:] == ["release", "cleanup"]


def test_cleanup_failure_preserves_primary_diagnostic(tmp_path) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    compatibility = adapt_pi05_finetuned_config(config(), "0.6.0")

    @contextmanager
    def stager(*args, **kwargs):
        yield "staged"

    def fail_load(*args, **kwargs):
        raise RuntimeError("load failed")

    factory = create_pi05_policy_factory(
        policy_id="fine-tuned-pi05",
        effective_config=compatibility,
        checkpoint_input=manifest.fine_tuned_checkpoint,
        checkpoint_path=tmp_path / "fine.safetensors",
        runtime=manifest.runtime,
        staging_parent=tmp_path,
        policy_constructor=lambda config: FakePolicy([]),
        checkpoint_stager=stager,
        checkpoint_loader=fail_load,
        cleanup_hooks=(lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")),),
    )
    with pytest.raises(PI05PolicyFactoryError, match=r"cleanup failed.*load failed"):
        with factory():
            pass
