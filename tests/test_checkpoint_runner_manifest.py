from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lerobot_state_atlas.checkpoint_comparison import (
    CheckpointComparisonRunnerManifestError,
    load_checkpoint_comparison_runner_manifest,
    resolve_checkpoint_comparison_runner_inputs,
)
import lerobot_state_atlas.checkpoint_comparison.runner_manifest as runner_manifest


SHA = "a" * 64


def _write_inputs(root: Path) -> dict[str, bytes]:
    contents = {
        "inputs/observation/manifest.json": b"observation",
        "inputs/base/model.safetensors": b"base",
        "inputs/fine-tuned/model.safetensors": b"fine",
        "inputs/fine-tuned/config.json": b"config",
        "inputs/fine-tuned/pre.json": b"pre",
        "inputs/fine-tuned/pre.safetensors": b"pre-state",
        "inputs/fine-tuned/post.json": b"post",
        "inputs/fine-tuned/post.safetensors": b"post-state",
        "inputs/robot/robot.urdf": b"urdf",
    }
    for relative, value in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    (root / "inputs/tokenizer").mkdir(parents=True, exist_ok=True)
    return contents


def _file(path: str, contents: dict[str, bytes]) -> dict[str, object]:
    data = contents[path]
    return {
        "path": path,
        "byteCount": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _document(contents: dict[str, bytes]) -> dict[str, object]:
    base = _file("inputs/base/model.safetensors", contents)
    base.update(
        repositoryId="lerobot/pi05_base",
        revision="b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
    )
    fine = _file("inputs/fine-tuned/model.safetensors", contents)
    fine.update(
        repositoryId="DreamMachines/actuator_unboxing_4h_diverse_fullft_bs256",
        revision="6c50dbbccd576e4e384ed51a8244272aab5f3c62",
    )
    urdf = _file("inputs/robot/robot.urdf", contents)
    return {
        "schema": {
            "name": "lerobot-state-atlas.checkpoint-comparison-runner",
            "major": 1,
            "minor": 0,
        },
        "dataset": {
            "repositoryId": "DreamMachines/actuator_unboxing_4h_diverse",
            "revision": "e973df866c80f52884cc68355579043cab828e78",
        },
        "observationManifest": _file("inputs/observation/manifest.json", contents),
        "checkpoints": {"base": base, "fineTuned": fine},
        "configuration": _file("inputs/fine-tuned/config.json", contents),
        "processors": {
            "preprocessorConfig": _file("inputs/fine-tuned/pre.json", contents),
            "preprocessorState": _file("inputs/fine-tuned/pre.safetensors", contents),
            "postprocessorConfig": _file("inputs/fine-tuned/post.json", contents),
            "postprocessorState": _file("inputs/fine-tuned/post.safetensors", contents),
            "tokenizerDirectory": {"path": "inputs/tokenizer", "identitySha256": SHA},
        },
        "robot": {
            "urdfPath": urdf["path"],
            "urdfByteCount": urdf["byteCount"],
            "urdfSha256": urdf["sha256"],
            "upstreamRevisionIdentity": "12f5368aefd0381461f2c7ffbb5611b4e8c90de9",
            "leftArmTransform": {
                "translationXyz": [0, 0.4, 0],
                "rotationRpy": [0, 0, 0],
            },
            "rightArmTransform": {
                "translationXyz": [0, -0.4, 0],
                "rotationRpy": [0, 0, 0],
            },
            "calibratedArmTransforms": False,
        },
        "runtime": {
            "device": "cuda:0",
            "modelDtype": "bfloat16",
            "noiseDtype": "float32",
            "noiseSeed": 123,
            "numInferenceSteps": 10,
            "minimumFreeVramBytes": 1,
            "minimumAvailableRamBytes": 1,
            "minimumFreeDiskBytes": 1,
        },
        "projection": {
            "mode": "available",
            "jointLimitPolicy": "reject",
            "unavailableReason": None,
            "acknowledgeUncalibratedArmTransforms": True,
            "acknowledgeRecordedLimitViolations": False,
        },
        "output": {
            "runDirectory": "output/run",
            "bundleId": "comparison-example",
            "replaceExisting": False,
        },
    }


def _manifest(tmp_path: Path, mutate=None):
    contents = _write_inputs(tmp_path)
    document = _document(contents)
    if mutate:
        mutate(document)
    path = tmp_path / "runner.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_valid_manifest_and_inventory_are_immutable_and_repeatable(
    tmp_path: Path,
) -> None:
    path, _ = _manifest(tmp_path)
    before = path.read_bytes()
    manifest = load_checkpoint_comparison_runner_manifest(path)
    first = resolve_checkpoint_comparison_runner_inputs(manifest)
    second = resolve_checkpoint_comparison_runner_inputs(manifest)
    assert first == second
    assert tuple(item.logical_input_id for item in first.inventory) == (
        "observationManifest",
        "checkpoints.base",
        "checkpoints.fineTuned",
        "configuration",
        "processors.preprocessorConfig",
        "processors.preprocessorState",
        "processors.postprocessorConfig",
        "processors.postprocessorState",
        "robot.urdf",
        "processors.tokenizerDirectory",
    )
    assert all(item.content_hash_pending for item in first.inventory)
    assert first.inventory[-1].kind == "directory"
    assert path.read_bytes() == before
    assert not first.output_run_directory.exists()
    with pytest.raises(FrozenInstanceError):
        manifest.runtime.device = "cuda:1"  # type: ignore[misc]


def test_standalone_schema_accepts_valid_document_and_rejects_boolean_versions(
    tmp_path: Path,
) -> None:
    _, document = _manifest(tmp_path)
    schema_path = (
        Path(__file__).parents[1]
        / "schemas/checkpoint-comparison-runner-v1.schema.json"
    )
    validator = Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    assert list(validator.iter_errors(document)) == []
    document["schema"]["major"] = True
    assert list(validator.iter_errors(document))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (("schema", "major"), True, "schema.major"),
        (("schema", "minor"), False, "schema.minor"),
        (("schema", "minor"), 1, "unsupported"),
        (("schema", "major"), 2, "unsupported"),
        (("dataset", "repositoryId"), "other/data", "dataset.repositoryId"),
        (("dataset", "revision"), "A" * 40, "dataset.revision"),
        (("checkpoints", "base", "revision"), "a" * 39, "checkpoints.base.revision"),
        (("runtime", "device"), "cpu", "runtime.device"),
        (("runtime", "noiseSeed"), True, "runtime.noiseSeed"),
        (("runtime", "noiseSeed"), 2**63, "runtime.noiseSeed"),
        (("runtime", "modelDtype"), "float16", "runtime.modelDtype"),
        (("runtime", "noiseDtype"), "bfloat16", "runtime.noiseDtype"),
        (("runtime", "numInferenceSteps"), 0, "runtime.numInferenceSteps"),
        (("runtime", "minimumFreeVramBytes"), False, "runtime.minimumFreeVramBytes"),
    ],
)
def test_strict_fields(
    tmp_path: Path, field: tuple[str, ...], value: object, match: str
) -> None:
    def mutate(document):
        target = document
        for part in field[:-1]:
            target = target[part]
        target[field[-1]] = value

    path, _ = _manifest(tmp_path, mutate)
    with pytest.raises(CheckpointComparisonRunnerManifestError, match=match):
        load_checkpoint_comparison_runner_manifest(path)


@pytest.mark.parametrize(
    "bad", ["/tmp/x", "../x", "inputs/../x", "inputs//x", "inputs\\x"]
)
def test_unsafe_input_paths(tmp_path: Path, bad: str) -> None:
    path, _ = _manifest(tmp_path, lambda d: d["configuration"].update(path=bad))
    with pytest.raises(
        CheckpointComparisonRunnerManifestError, match="configuration.path"
    ):
        load_checkpoint_comparison_runner_manifest(path)


def test_manifest_malformed_utf8_and_json(tmp_path: Path) -> None:
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff")
    with pytest.raises(CheckpointComparisonRunnerManifestError, match="UTF-8"):
        load_checkpoint_comparison_runner_manifest(binary)
    binary.write_bytes(b"{")
    with pytest.raises(CheckpointComparisonRunnerManifestError, match="JSON"):
        load_checkpoint_comparison_runner_manifest(binary)


@pytest.mark.parametrize("dangling", [False, True])
def test_manifest_symlink_rejected(tmp_path: Path, dangling: bool) -> None:
    target = tmp_path / "missing.json"
    if not dangling:
        target.write_text("{}", encoding="utf-8")
    link = tmp_path / "runner.json"
    link.symlink_to(target)
    with pytest.raises(CheckpointComparisonRunnerManifestError, match="symbolic link"):
        load_checkpoint_comparison_runner_manifest(link)


def test_manifest_snapshot_is_acquired_once_and_bound(
    monkeypatch, tmp_path: Path
) -> None:
    path, document = _manifest(tmp_path)
    snapshot = path.read_bytes()
    calls = 0

    def acquire(requested):
        nonlocal calls
        calls += 1
        path.write_text(json.dumps({"schema": {}}), encoding="utf-8")
        return snapshot

    monkeypatch.setattr(runner_manifest, "read_stable_file_snapshot", acquire)
    loaded = load_checkpoint_comparison_runner_manifest(path)
    assert calls == 1
    assert loaded.dataset.repository_id == document["dataset"]["repositoryId"]


def test_snapshot_mutation_error_is_translated(monkeypatch, tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)

    def fail(_):
        raise runner_manifest.StableFileSnapshotError("changed while it was being read")

    monkeypatch.setattr(runner_manifest, "read_stable_file_snapshot", fail)
    with pytest.raises(CheckpointComparisonRunnerManifestError, match="changed while"):
        load_checkpoint_comparison_runner_manifest(path)


def test_resolution_reads_no_input_contents(monkeypatch, tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail(f"read {self}"))
    resolve_checkpoint_comparison_runner_inputs(manifest)


def test_size_mismatch_and_input_types(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    manifest.configuration.path  # retain frozen source model
    (tmp_path / manifest.configuration.path).write_bytes(b"changed-size")
    with pytest.raises(
        CheckpointComparisonRunnerManifestError, match="configuration.byteCount"
    ):
        resolve_checkpoint_comparison_runner_inputs(manifest)


def test_direct_and_intermediate_input_symlinks_rejected(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    config = tmp_path / "inputs/fine-tuned/config.json"
    target = tmp_path / "target"
    target.write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(target)
    with pytest.raises(CheckpointComparisonRunnerManifestError, match="symbolic-link"):
        resolve_checkpoint_comparison_runner_inputs(manifest)


def test_projection_acknowledgements_and_unavailable_reason(tmp_path: Path) -> None:
    path, _ = _manifest(
        tmp_path,
        lambda d: d["projection"].update(acknowledgeUncalibratedArmTransforms=False),
    )
    with pytest.raises(
        CheckpointComparisonRunnerManifestError, match="acknowledgeUncalibrated"
    ):
        load_checkpoint_comparison_runner_manifest(path)
    path, _ = _manifest(
        tmp_path,
        lambda d: d["projection"].update(
            mode="unavailable",
            unavailableReason=" blocked ",
            acknowledgeUncalibratedArmTransforms=False,
        ),
    )
    with pytest.raises(
        CheckpointComparisonRunnerManifestError, match="surrounding whitespace"
    ):
        load_checkpoint_comparison_runner_manifest(path)


def test_output_symlink_traversal_and_demo_rejected(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    (tmp_path / "output").symlink_to(tmp_path / "inputs", target_is_directory=True)
    with pytest.raises(CheckpointComparisonRunnerManifestError, match="symbolic-link"):
        resolve_checkpoint_comparison_runner_inputs(manifest)
    path, _ = _manifest(
        tmp_path,
        lambda d: d["output"].update(runDirectory="apps/web/public/atlas-data/demo-v1"),
    )
    with pytest.raises(CheckpointComparisonRunnerManifestError, match="immutable"):
        resolve_checkpoint_comparison_runner_inputs(
            load_checkpoint_comparison_runner_manifest(path)
        )


@pytest.mark.parametrize(
    ("output_path", "logical_id", "relationship"),
    [
        ("inputs/tokenizer", "processors.tokenizerDirectory", "equal"),
        ("inputs", "observationManifest", "output-contains-input"),
        (
            "inputs/tokenizer/generated-run",
            "processors.tokenizerDirectory",
            "output-inside-input",
        ),
        ("inputs/base", "checkpoints.base", "output-contains-input"),
        (
            "inputs/fine-tuned",
            "checkpoints.fineTuned",
            "output-contains-input",
        ),
        (
            "inputs/fine-tuned/pre.json",
            "processors.preprocessorConfig",
            "equal",
        ),
        ("inputs/observation", "observationManifest", "output-contains-input"),
        ("inputs/robot", "robot.urdf", "output-contains-input"),
        ("runner.json", "runnerManifest", "equal"),
    ],
)
@pytest.mark.parametrize("replace_existing", [False, True])
def test_output_must_be_disjoint_from_every_static_input(
    tmp_path: Path,
    output_path: str,
    logical_id: str,
    relationship: str,
    replace_existing: bool,
) -> None:
    path, _ = _manifest(
        tmp_path,
        lambda document: document["output"].update(
            runDirectory=output_path, replaceExisting=replace_existing
        ),
    )
    before = {
        item.relative_to(tmp_path): item.read_bytes()
        for item in tmp_path.rglob("*")
        if item.is_file()
    }
    manifest = load_checkpoint_comparison_runner_manifest(path)
    with pytest.raises(CheckpointComparisonRunnerManifestError) as caught:
        resolve_checkpoint_comparison_runner_inputs(manifest)
    message = str(caught.value)
    assert "output.runDirectory" in message
    assert logical_id in message
    assert relationship in message
    assert {
        item.relative_to(tmp_path): item.read_bytes()
        for item in tmp_path.rglob("*")
        if item.is_file()
    } == before
    assert not any(tmp_path.rglob(".checkpoint-stage-*"))
    assert not any(tmp_path.rglob("*.previous-*"))


def test_similar_nonancestor_input_name_and_sibling_output_are_accepted(
    tmp_path: Path,
) -> None:
    path, _ = _manifest(
        tmp_path,
        lambda document: document["output"].update(runDirectory="inputs-old/run"),
    )
    resolved = resolve_checkpoint_comparison_runner_inputs(
        load_checkpoint_comparison_runner_manifest(path)
    )
    assert resolved.output_run_directory == tmp_path / "inputs-old/run"

    sibling_path, _ = _manifest(
        tmp_path,
        lambda document: document["output"].update(runDirectory="output/run"),
    )
    sibling = resolve_checkpoint_comparison_runner_inputs(
        load_checkpoint_comparison_runner_manifest(sibling_path)
    )
    assert sibling.output_run_directory == tmp_path / "output/run"
