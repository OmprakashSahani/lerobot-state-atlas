from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import socket

import pytest
from safetensors.torch import save as save_safetensors
import torch

from lerobot_state_atlas.checkpoint_comparison import (
    PI05ProcessorCompatibilityError,
    PI05ProcessorVerificationError,
    adapt_pi05_processor_configs,
    load_checkpoint_comparison_runner_manifest,
    resolve_checkpoint_comparison_runner_inputs,
    verify_pi05_processor_assets,
)
from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
)
from lerobot_state_atlas.checkpoint_comparison.observation import COMPONENT_NAMES
import lerobot_state_atlas.checkpoint_comparison.processor_compatibility as processor_module
from test_checkpoint_runner_manifest import _document, _write_inputs


def preprocessor_config() -> dict:
    features = {
        "observation.state": {"type": "STATE", "shape": [14]},
        "observation.images.left_wrist": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.right_wrist": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 224, 224]},
        "action": {"type": "ACTION", "shape": [14]},
    }
    norms = {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}
    return {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {}},
            },
            {"registry_name": "to_batch_processor", "config": {}},
            {
                "registry_name": "delta_actions_processor",
                "config": {
                    "enabled": False,
                    "exclude_joints": ["gripper"],
                    "action_names": None,
                },
            },
            {
                "registry_name": "normalizer_processor",
                "config": {"eps": 1e-8, "features": features, "norm_map": norms},
                "state_file": "policy_preprocessor_step_3_normalizer_processor.safetensors",
            },
            {
                "registry_name": "pi05_prepare_state_tokenizer_processor_step",
                "config": {},
            },
            {
                "registry_name": "tokenizer_processor",
                "config": {
                    "max_length": 200,
                    "task_key": "task",
                    "padding_side": "right",
                    "padding": "max_length",
                    "truncation": True,
                    "tokenizer_name": "google/paligemma-3b-pt-224",
                },
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cuda", "float_dtype": None},
            },
        ],
    }


def postprocessor_config() -> dict:
    return {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "eps": 1e-8,
                    "features": {"action": {"type": "ACTION", "shape": [14]}},
                    "norm_map": {
                        "VISUAL": "IDENTITY",
                        "STATE": "QUANTILES",
                        "ACTION": "QUANTILES",
                    },
                },
                "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            },
            {
                "registry_name": "absolute_actions_processor",
                "config": {"enabled": False},
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cpu", "float_dtype": None},
            },
        ],
    }


def state_bytes(
    *, include_pre_action: bool = True, bad: tuple[str, torch.Tensor] | None = None
):
    q01 = torch.arange(14, dtype=torch.float32)
    q99 = q01 + 10
    pre = {"observation.state.q01": q01, "observation.state.q99": q99}
    if include_pre_action:
        pre.update({"action.q01": q01 + 20, "action.q99": q99 + 20})
    post = {"action.q01": q01 + 20, "action.q99": q99 + 20}
    if bad:
        side, tensor = bad
        target = pre if side.startswith("pre:") else post
        key = side.split(":", 1)[1]
        target[key] = tensor
    return save_safetensors(pre), save_safetensors(post)


def runner_assets(tmp_path: Path, monkeypatch, *, pre=None, post=None, states=None):
    contents = _write_inputs(tmp_path)
    document = _document(contents)
    pre = pre or preprocessor_config()
    post = post or postprocessor_config()
    pre_bytes = json.dumps(pre, indent=2).encode() + b"\n"
    post_bytes = json.dumps(post, indent=2).encode() + b"\n"
    pre_state, post_state = states or state_bytes()
    replacements = {
        "processors.preprocessorConfig": (
            document["processors"]["preprocessorConfig"],
            pre_bytes,
        ),
        "processors.postprocessorConfig": (
            document["processors"]["postprocessorConfig"],
            post_bytes,
        ),
        "processors.preprocessorState": (
            document["processors"]["preprocessorState"],
            pre_state,
        ),
        "processors.postprocessorState": (
            document["processors"]["postprocessorState"],
            post_state,
        ),
    }
    for _, (entry, content) in replacements.items():
        path = tmp_path / entry["path"]
        path.write_bytes(content)
        entry["byteCount"] = len(content)
        entry["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path = tmp_path / "runner.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        processor_module,
        "KNOWN_PREPROCESSOR_JSON_SHA256",
        hashlib.sha256(pre_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        processor_module,
        "KNOWN_POSTPROCESSOR_JSON_SHA256",
        hashlib.sha256(post_bytes).hexdigest(),
    )
    manifest = load_checkpoint_comparison_runner_manifest(manifest_path)
    return manifest, resolve_checkpoint_comparison_runner_inputs(manifest), replacements


def test_valid_compatibility_is_deterministic_immutable_and_does_not_mutate() -> None:
    pre = preprocessor_config()
    post = postprocessor_config()
    before = copy.deepcopy((pre, post))
    first = adapt_pi05_processor_configs(pre, post, installed_lerobot_version="0.6.0")
    second = adapt_pi05_processor_configs(pre, post, installed_lerobot_version="0.6.0")
    assert first == second
    assert (pre, post) == before
    assert first.use_relative_actions is False
    assert first.delta_actions_preprocessor_enabled is False
    assert first.absolute_actions_postprocessor_enabled is False
    assert first.tokenizer_repository_id == "google/paligemma-3b-pt-224"
    assert first.normalization.state_component_names == COMPONENT_NAMES
    assert first.normalization.action_component_names == COMPONENT_NAMES
    assert first.transformations[0].source_value == "delta_actions_processor"
    assert first.transformations[0].effective_value == "relative_actions_processor"
    with pytest.raises(FrozenInstanceError):
        first.installed_lerobot_version = "x"  # type: ignore[misc]


def test_unsupported_version_and_non_objects_rejected() -> None:
    with pytest.raises(
        PI05ProcessorCompatibilityError, match="installed_lerobot_version"
    ):
        adapt_pi05_processor_configs(
            preprocessor_config(),
            postprocessor_config(),
            installed_lerobot_version="0.7.0",
        )
    with pytest.raises(
        PI05ProcessorCompatibilityError, match="preprocessor must be an object"
    ):
        adapt_pi05_processor_configs(
            [], postprocessor_config(), installed_lerobot_version="0.6.0"
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda p: p["steps"].pop(), "exactly 7"),
        (
            lambda p: p["steps"].__setitem__(
                0, {"registry_name": "unknown", "config": {}}
            ),
            r"steps\[0\].registry_name",
        ),
        (lambda p: p["steps"].reverse(), r"steps\[0\].registry_name"),
        (
            lambda p: p["steps"][2]["config"].__setitem__("enabled", True),
            "enabled is false",
        ),
        (
            lambda p: p["steps"][2]["config"].__setitem__("enabled", 0),
            "must be a boolean",
        ),
        (lambda p: p["steps"][2]["config"].pop("enabled"), "missing fields: enabled"),
        (
            lambda p: p["steps"][2].__setitem__(
                "registry_name", "delta_action_processor"
            ),
            "must resolve",
        ),
        (
            lambda p: p["steps"][3]["config"]["norm_map"].__setitem__(
                "VISUAL", "QUANTILES"
            ),
            "norm_map.VISUAL",
        ),
        (
            lambda p: p["steps"][3]["config"]["norm_map"].__setitem__(
                "STATE", "MEAN_STD"
            ),
            "norm_map.STATE",
        ),
        (
            lambda p: p["steps"][3]["config"]["features"][
                "observation.state"
            ].__setitem__("shape", [True]),
            r"shape\[0\]",
        ),
        (
            lambda p: p["steps"][3]["config"]["features"].__setitem__(
                "observation.state", p["steps"][3]["config"]["features"].pop("action")
            ),
            "ordered features",
        ),
        (
            lambda p: p["steps"][5]["config"].__setitem__(
                "tokenizer_name", "other/tokenizer"
            ),
            "task-tokenizer",
        ),
    ],
)
def test_invalid_preprocessor_contract(mutation, match: str) -> None:
    pre = preprocessor_config()
    mutation(pre)
    with pytest.raises(PI05ProcessorCompatibilityError, match=match):
        adapt_pi05_processor_configs(
            pre, postprocessor_config(), installed_lerobot_version="0.6.0"
        )


def test_both_alias_names_and_duplicates_are_rejected() -> None:
    pre = preprocessor_config()
    pre["steps"].insert(3, copy.deepcopy(pre["steps"][2]))
    pre["steps"][3]["registry_name"] = "relative_actions_processor"
    with pytest.raises(PI05ProcessorCompatibilityError, match="exactly 7"):
        adapt_pi05_processor_configs(
            pre, postprocessor_config(), installed_lerobot_version="0.6.0"
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda p: p["steps"][1]["config"].__setitem__("enabled", True),
            "must be false",
        ),
        (lambda p: p["steps"][1]["config"].pop("enabled"), "missing fields"),
        (
            lambda p: p["steps"][1]["config"].__setitem__("enabled", 0),
            "must be a boolean",
        ),
        (lambda p: p["steps"].reverse(), r"steps\[0\].registry_name"),
        (
            lambda p: p["steps"][0]["config"]["features"]["action"].__setitem__(
                "shape", [13]
            ),
            r"must be \[14\]",
        ),
    ],
)
def test_invalid_postprocessor_contract(mutation, match: str) -> None:
    post = postprocessor_config()
    mutation(post)
    with pytest.raises(PI05ProcessorCompatibilityError, match=match):
        adapt_pi05_processor_configs(
            preprocessor_config(), post, installed_lerobot_version="0.6.0"
        )


def test_valid_asset_verification_and_snapshot_binding(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, resolved, replacements = runner_assets(tmp_path, monkeypatch)
    source_bytes = {
        tmp_path / entry["path"]: content
        for entry, content in (item for item in replacements.values())
    }
    calls = []

    def reader(path: Path) -> bytes:
        calls.append(path)
        return path.read_bytes()

    result = verify_pi05_processor_assets(manifest, resolved, snapshot_reader=reader)
    repeated = verify_pi05_processor_assets(
        manifest, resolved, snapshot_reader=lambda path: path.read_bytes()
    )
    assert result == repeated
    assert len(calls) == 4 and len(set(calls)) == 4
    assert result.cross_state_consistent
    assert result.shared_for_policy_ids == ("base-pi05", "fine-tuned-pi05")
    assert "identical verified" in result.fairness_statement
    assert result.preprocessor_state.tensor_keys == (
        "action.q01",
        "action.q99",
        "observation.state.q01",
        "observation.state.q99",
    )
    assert result.postprocessor_state.tensor_keys == ("action.q01", "action.q99")
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes

    def assert_public_has_no_tensor_or_mutable(value):
        assert not isinstance(value, (torch.Tensor, dict, list))
        if is_dataclass(value):
            for field in fields(value):
                assert_public_has_no_tensor_or_mutable(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                assert_public_has_no_tensor_or_mutable(item)

    assert_public_has_no_tensor_or_mutable(result)


def test_path_replacement_after_acquisition_does_not_change_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, resolved, _ = runner_assets(tmp_path, monkeypatch)
    first_path = next(
        entry.canonical_path
        for entry in resolved.inventory
        if entry.logical_input_id == "processors.preprocessorConfig"
    )
    original = first_path.read_bytes()

    def reader(path: Path) -> bytes:
        content = path.read_bytes()
        if path == first_path:
            path.write_bytes(b"x" * len(content))
        return content

    result = verify_pi05_processor_assets(manifest, resolved, snapshot_reader=reader)
    assert (
        result.compatibility.source_preprocessor_sha256
        == hashlib.sha256(original).hexdigest()
    )


def test_acquisition_mutation_is_rejected(tmp_path: Path, monkeypatch) -> None:
    manifest, resolved, _ = runner_assets(tmp_path, monkeypatch)
    calls = 0

    def reader(path: Path) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise StableFileSnapshotError("changed while it was being read")
        return path.read_bytes()

    with pytest.raises(PI05ProcessorVerificationError, match="changed while"):
        verify_pi05_processor_assets(manifest, resolved, snapshot_reader=reader)
    assert calls == 3


@pytest.mark.parametrize("kind", ["direct", "dangling", "intermediate"])
def test_processor_asset_symlinks_rejected(
    tmp_path: Path, monkeypatch, kind: str
) -> None:
    manifest, resolved, _ = runner_assets(tmp_path, monkeypatch)
    path = next(
        entry.canonical_path
        for entry in resolved.inventory
        if entry.logical_input_id == "processors.preprocessorState"
    )
    if kind == "intermediate":
        parent = path.parent
        stored = tmp_path / "stored-fine-tuned"
        parent.rename(stored)
        parent.symlink_to(stored, target_is_directory=True)
    else:
        content = path.read_bytes()
        target = tmp_path / ("state-target" if kind == "direct" else "missing-target")
        if kind == "direct":
            target.write_bytes(content)
        path.unlink()
        path.symlink_to(target)
    with pytest.raises(PI05ProcessorVerificationError, match="symbolic-link"):
        verify_pi05_processor_assets(manifest, resolved)


def test_manifest_byte_count_and_digest_mismatch(tmp_path: Path, monkeypatch) -> None:
    manifest, resolved, _ = runner_assets(tmp_path, monkeypatch)
    path = next(
        entry.canonical_path
        for entry in resolved.inventory
        if entry.logical_input_id == "processors.postprocessorState"
    )
    original = path.read_bytes()
    path.write_bytes(original + b"x")
    with pytest.raises(PI05ProcessorVerificationError, match="byteCount"):
        verify_pi05_processor_assets(manifest, resolved)
    path.write_bytes(b"x" * len(original))
    with pytest.raises(PI05ProcessorVerificationError, match="sha256"):
        verify_pi05_processor_assets(manifest, resolved)


@pytest.mark.parametrize(
    ("states", "match"),
    [
        ((b"bad", state_bytes()[1]), "malformed SafeTensors"),
        (
            (
                save_safetensors({"observation.state.q99": torch.ones(14)}),
                state_bytes()[1],
            ),
            "missing tensor keys",
        ),
        (
            (
                save_safetensors({"observation.state.q01": torch.zeros(14)}),
                state_bytes()[1],
            ),
            "missing tensor keys",
        ),
        (
            (
                save_safetensors(
                    {
                        "observation.state.q01": torch.zeros(14),
                        "observation.state.q99": torch.ones(14),
                        "visual.q01": torch.zeros(3),
                    }
                ),
                state_bytes()[1],
            ),
            "unexpected tensor keys",
        ),
        (state_bytes(bad=("pre:observation.state.q01", torch.zeros(13))), "shape"),
        (
            state_bytes(
                bad=("pre:observation.state.q01", torch.zeros(14, dtype=torch.int64))
            ),
            "dtype",
        ),
        (
            state_bytes(
                bad=("pre:observation.state.q01", torch.zeros(14, dtype=torch.bool))
            ),
            "dtype",
        ),
        (
            state_bytes(
                bad=("pre:observation.state.q01", torch.full((14,), float("nan")))
            ),
            "finite",
        ),
        (
            state_bytes(
                bad=("pre:observation.state.q01", torch.full((14,), float("inf")))
            ),
            "finite",
        ),
        (
            state_bytes(bad=("post:action.q99", torch.full((14,), float("-inf")))),
            "finite",
        ),
        (
            state_bytes(bad=("pre:observation.state.q01", torch.full((14,), 100.0))),
            "q01 <= q99",
        ),
        (
            state_bytes(bad=("pre:action.q01", torch.full((14,), 21.0))),
            "crossState.action.q01",
        ),
    ],
)
def test_invalid_learned_state(tmp_path: Path, monkeypatch, states, match: str) -> None:
    manifest, resolved, _ = runner_assets(tmp_path, monkeypatch, states=states)
    with pytest.raises(PI05ProcessorVerificationError, match=match):
        verify_pi05_processor_assets(manifest, resolved)


def test_malformed_config_bytes_and_no_external_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, resolved, _ = runner_assets(tmp_path, monkeypatch)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: pytest.fail("network called"),
    )
    result = verify_pi05_processor_assets(manifest, resolved)
    assert result.compatibility.tokenizer_verification_pending is True
    assert "first checkpoint" not in repr(result).lower()
