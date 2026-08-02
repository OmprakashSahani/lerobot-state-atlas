from types import SimpleNamespace

import pytest

from lerobot_state_atlas.checkpoint_comparison import (
    PI05ProcessorConstructionError,
    build_verified_pi05_processor_pair,
    verify_local_pi05_tokenizer_directory,
    verify_pi05_processor_assets,
)
from test_checkpoint_tokenizer_assets import setup_tokenizer


def verified(tmp_path, monkeypatch):
    manifest, resolved, _, _, _ = setup_tokenizer(tmp_path, monkeypatch)
    processor = verify_pi05_processor_assets(manifest, resolved)
    tokenizer = verify_local_pi05_tokenizer_directory(manifest, resolved, processor)
    return processor, tokenizer


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, *args, **kwargs):
        return {}


def test_builds_exact_configs_states_tokenizer_and_step_order(
    tmp_path, monkeypatch
) -> None:
    processor, tokenizer = verified(tmp_path, monkeypatch)
    calls = []

    def factory(config, state, overrides):
        calls.append((config, state, overrides))
        return SimpleNamespace(
            steps=[
                SimpleNamespace(_registry_name=step["registry_name"])
                for step in config["steps"]
            ]
        )

    pair = build_verified_pi05_processor_pair(
        processor,
        tokenizer,
        Tokenizer(),
        pipeline_factory=factory,
        device="cuda:3",
    )
    assert len(calls) == 2
    assert calls[0][0]["steps"][2]["registry_name"] == "relative_actions_processor"
    assert calls[0][2]["tokenizer_processor"]["tokenizer"].pad_token_id == 0
    assert set(calls[0][1]) == {"policy_preprocessor_step_3_normalizer_processor"}
    assert set(calls[1][1]) == {"policy_postprocessor_step_0_unnormalizer_processor"}
    assert calls[0][2]["device_processor"]["device"] == "cuda:3"
    assert calls[1][2]["device_processor"]["device"] == "cpu"
    assert pair.shared_for_policy_ids == ("base-pi05", "fine-tuned-pi05")


def test_rejects_identity_tokenizer_device_and_step_mismatch(
    tmp_path, monkeypatch
) -> None:
    processor, tokenizer = verified(tmp_path, monkeypatch)
    with pytest.raises(PI05ProcessorConstructionError, match="identities"):
        build_verified_pi05_processor_pair(
            processor,
            __import__("dataclasses").replace(tokenizer, repository_id="other"),
            Tokenizer(),
            pipeline_factory=lambda *a: None,
        )
    with pytest.raises(PI05ProcessorConstructionError, match="callable"):
        build_verified_pi05_processor_pair(
            processor, tokenizer, object(), pipeline_factory=lambda *a: None
        )
    with pytest.raises(PI05ProcessorConstructionError, match="device"):
        build_verified_pi05_processor_pair(
            processor,
            tokenizer,
            Tokenizer(),
            device="mps",
            pipeline_factory=lambda *a: None,
        )

    def wrong(config, state, overrides):
        return SimpleNamespace(steps=[])

    with pytest.raises(PI05ProcessorConstructionError, match="step order mismatch"):
        build_verified_pi05_processor_pair(
            processor, tokenizer, Tokenizer(), pipeline_factory=wrong
        )


def test_changed_state_and_loader_key_mismatch_fail_before_usable_pair(
    tmp_path, monkeypatch
) -> None:
    processor, tokenizer = verified(tmp_path, monkeypatch)
    processor.preprocessor_state.source_path.write_bytes(b"changed")
    with pytest.raises(
        PI05ProcessorConstructionError, match="changed before construction"
    ):
        build_verified_pi05_processor_pair(
            processor, tokenizer, Tokenizer(), pipeline_factory=lambda *a: None
        )

    processor, tokenizer = verified(tmp_path / "other", monkeypatch)
    with pytest.raises(PI05ProcessorConstructionError, match="exact verified keys"):
        build_verified_pi05_processor_pair(
            processor,
            tokenizer,
            Tokenizer(),
            state_loader=lambda data: {"wrong": object()},
            pipeline_factory=lambda *a: None,
        )


def test_factory_failure_is_not_hidden(tmp_path, monkeypatch) -> None:
    processor, tokenizer = verified(tmp_path, monkeypatch)

    def fail(*args):
        raise RuntimeError("construction exploded")

    with pytest.raises(PI05ProcessorConstructionError, match="construction exploded"):
        build_verified_pi05_processor_pair(
            processor, tokenizer, Tokenizer(), pipeline_factory=fail
        )
