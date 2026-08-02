from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from lerobot_state_atlas.checkpoint_comparison.artifact import (
    build_checkpoint_comparison_documents,
    build_checkpoint_comparison_v1_1,
    install_checkpoint_comparison_bundle,
)
from lerobot_state_atlas.checkpoint_comparison.models import PolicyIdentity
from lerobot_state_atlas.checkpoint_comparison.projection import (
    unavailable_policy_comparison_trajectory_result,
)
from lerobot_state_atlas.checkpoint_comparison.serialize import (
    deterministic_json_bytes,
    sha256_bytes,
)
from lerobot_state_atlas.checkpoint_comparison.validate import (
    CheckpointComparisonValidationError,
    validate_checkpoint_comparison,
)
from test_checkpoint_projection import (
    inference,
    observation,
    project,
    provenance,
    robot,
)


SCHEMA_PATH = (
    Path(__file__).parents[1] / "schemas/checkpoint-comparison-v1.1.schema.json"
)
V1_FIXTURE = Path(__file__).parent / "fixtures/checkpoint-comparison-v1"
V1_1_AVAILABLE_FIXTURE = (
    Path(__file__).parent / "fixtures/checkpoint-comparison-v1.1-available"
)
V1_1_UNAVAILABLE_FIXTURE = (
    Path(__file__).parent / "fixtures/checkpoint-comparison-v1.1-unavailable"
)


def policies() -> tuple[PolicyIdentity, PolicyIdentity]:
    return (
        PolicyIdentity("base-pi05", "Base π0.5", "lerobot/pi05_base", "b" * 40),
        PolicyIdentity(
            "fine-tuned-pi05",
            "Fine-tuned π0.5",
            "example/fine-tuned",
            "c" * 40,
        ),
    )


def available_documents() -> tuple[dict, dict]:
    model = robot(upper=0.12)
    trajectory = project(
        robot_model=model,
        robot_provenance=provenance(model),
        joint_limit_policy="allow-with-recorded-violations",
    )
    return build_checkpoint_comparison_v1_1(
        observation(),
        inference(),
        trajectory,
        bundle_id="synthetic-checkpoint-comparison-v1.1-available",
        policies=policies(),
        joint_limit_policy="allow-with-recorded-violations",
    )


def unavailable_documents() -> tuple[dict, dict]:
    obs = observation()
    result = inference()
    trajectory = unavailable_policy_comparison_trajectory_result(
        obs, result, reason="Synthetic fixture has no configured robot projection."
    )
    return build_checkpoint_comparison_v1_1(
        obs,
        result,
        trajectory,
        bundle_id="synthetic-checkpoint-comparison-v1.1-unavailable",
        policies=policies(),
        joint_limit_policy="reject",
    )


def write_bundle(path: Path, documents: tuple[dict, dict]) -> None:
    path.mkdir()
    manifest_bytes, plans_bytes = build_checkpoint_comparison_documents(*documents)
    (path / "manifest.json").write_bytes(manifest_bytes)
    (path / "plans.json").write_bytes(plans_bytes)


@pytest.mark.parametrize(
    ("fixture", "available"),
    [(V1_1_AVAILABLE_FIXTURE, True), (V1_1_UNAVAILABLE_FIXTURE, False)],
)
def test_tracked_v1_1_fixture_validates(fixture: Path, available: bool) -> None:
    before = {path.name: path.read_bytes() for path in fixture.iterdir()}
    manifest = validate_checkpoint_comparison(fixture)
    plans = json.loads((fixture / "plans.json").read_bytes())
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    assert list(validator.iter_errors(manifest)) == []
    assert list(validator.iter_errors(plans)) == []
    assert manifest["schema"]["minor"] == 1
    assert plans["trajectoryProjection"]["available"] is available
    assert {path.name: path.read_bytes() for path in fixture.iterdir()} == before


@pytest.mark.parametrize("factory", [available_documents, unavailable_documents])
def test_v1_1_documents_validate_with_python_and_json_schema(tmp_path, factory) -> None:
    manifest, plans = factory()
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    assert list(validator.iter_errors(manifest)) == []
    assert list(validator.iter_errors(plans)) == []
    bundle = tmp_path / "bundle"
    write_bundle(bundle, (manifest, plans))
    assert validate_checkpoint_comparison(bundle)["schema"] == {
        "name": "lerobot-state-atlas.checkpoint-comparison",
        "major": 1,
        "minor": 1,
    }


def test_v1_1_serialization_is_deterministic_and_retains_authoritative_actions() -> (
    None
):
    first = available_documents()
    second = available_documents()
    assert tuple(map(deterministic_json_bytes, first)) == tuple(
        map(deterministic_json_bytes, second)
    )
    assert first[1]["plans"][0]["actions"] == [
        list(row) for row in inference().policies[0].actions
    ]
    assert "targetStates" not in json.dumps(first[1])
    assert deterministic_json_bytes(first[1]).endswith(b"\n")


def test_available_projection_has_shared_provenance_and_ordered_violation() -> None:
    _, plans = available_documents()
    projection = plans["trajectoryProjection"]
    assert projection["sharedConfiguration"] is True
    assert [item["policyId"] for item in projection["plans"]] == [
        "base-pi05",
        "fine-tuned-pi05",
    ]
    assert projection["actionInterpretation"]["useRelativeActions"] is False
    assert (
        projection["initialState"]["initialStateParticipatesInTransformation"] is False
    )
    assert projection["robot"]["rotationComponentOrder"] == ["X", "Y", "Z", "W"]
    assert projection["robot"]["calibratedGripperGeometry"] is False
    assert "not physical jaw widths" in projection["robot"]["gripperSemanticDisclaimer"]
    violations = projection["plans"][1]["jointLimitViolations"]
    assert violations
    assert violations == sorted(
        violations,
        key=lambda item: (
            item["stepIndex"],
            item["componentName"],
            item["violationKind"],
        ),
    )


def test_unavailable_projection_has_no_fabricated_fields() -> None:
    _, plans = unavailable_documents()
    assert plans["trajectoryProjection"] == {
        "available": False,
        "reason": "Synthetic fixture has no configured robot projection.",
    }


@pytest.mark.parametrize("version", [(1, 2), (2, 0), (-1, 1)])
def test_unsupported_versions_do_not_fall_back(tmp_path: Path, version) -> None:
    manifest, plans = available_documents()
    manifest["schema"]["major"], manifest["schema"]["minor"] = version
    plans["schema"]["major"], plans["schema"]["minor"] = version
    bundle = tmp_path / "bundle"
    write_bundle(bundle, (manifest, plans))
    with pytest.raises(CheckpointComparisonValidationError, match="unsupported"):
        validate_checkpoint_comparison(bundle)


def test_v1_0_bytes_and_validation_remain_unchanged(tmp_path: Path) -> None:
    original_manifest = (V1_FIXTURE / "manifest.json").read_bytes()
    original_plans = (V1_FIXTURE / "plans.json").read_bytes()
    manifest = json.loads(original_manifest)
    plans = json.loads(original_plans)
    assert build_checkpoint_comparison_documents(manifest, plans) == (
        original_manifest,
        original_plans,
    )
    validate_checkpoint_comparison(V1_FIXTURE)
    plans["trajectoryProjection"] = {"available": False, "reason": "not v1.0"}
    bundle = tmp_path / "bundle"
    write_bundle(bundle, (manifest, plans))
    with pytest.raises(CheckpointComparisonValidationError, match="unsupported fields"):
        validate_checkpoint_comparison(bundle)


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (lambda p: p["plans"].reverse(), r"trajectoryProjection\.plans\[0\]\.policyId"),
        (
            lambda p: p["plans"][0]["relativeTimesSeconds"].__setitem__(1, 0.021),
            "relativeTimesSeconds",
        ),
        (
            lambda p: p["initialState"].__setitem__("sha256", "0" * 64),
            "initialState.sha256",
        ),
        (lambda p: p["robot"].__setitem__("targetLink", "other"), "targetLink"),
        (lambda p: p["robot"]["leftJointMapping"].reverse(), "leftJointMapping"),
        (lambda p: p["robot"].__setitem__("urdfSha256", "BAD"), "urdfSha256"),
        (
            lambda p: p["robot"].__setitem__("calibratedGripperGeometry", True),
            "calibratedGripperGeometry",
        ),
        (
            lambda p: p["plans"][0]["left"]["orientationsXyzw"][0].__setitem__(3, 2.0),
            "unit quaternion",
        ),
        (
            lambda p: p["plans"][1]["jointLimitViolations"][0].__setitem__(
                "componentName", "right_gripper.pos"
            ),
            "componentName",
        ),
        (
            lambda p: p["plans"][0]["left"]["positionsXyz"].pop(),
            "positionsXyz",
        ),
        (
            lambda p: p["robot"].__setitem__(
                "generatedGripperSemantics", "physical-jaw-width"
            ),
            "generatedGripperSemantics",
        ),
    ],
)
def test_v1_1_cross_model_corruption_is_rejected(tmp_path, mutation, path) -> None:
    manifest, plans = available_documents()
    mutation(plans["trajectoryProjection"])
    bundle = tmp_path / "bundle"
    write_bundle(bundle, (manifest, plans))
    with pytest.raises(CheckpointComparisonValidationError, match=path):
        validate_checkpoint_comparison(bundle)


def test_reject_limit_policy_cannot_serialize_violations(tmp_path: Path) -> None:
    manifest, plans = available_documents()
    plans["trajectoryProjection"]["jointLimitPolicy"] = "reject"
    bundle = tmp_path / "bundle"
    write_bundle(bundle, (manifest, plans))
    with pytest.raises(CheckpointComparisonValidationError, match="must be empty"):
        validate_checkpoint_comparison(bundle)


def test_unavailable_projection_rejects_whitespace_reason_and_extra_data(
    tmp_path: Path,
) -> None:
    for mutation, message in (
        (lambda value: value.update(reason="  "), "reason"),
        (lambda value: value.update(robot={}), "unsupported fields"),
    ):
        manifest, plans = unavailable_documents()
        mutation(plans["trajectoryProjection"])
        bundle = tmp_path / message.replace(" ", "-")
        write_bundle(bundle, (manifest, plans))
        with pytest.raises(CheckpointComparisonValidationError, match=message):
            validate_checkpoint_comparison(bundle)


def test_nonfinite_projection_value_is_rejected(tmp_path: Path) -> None:
    manifest, plans = available_documents()
    plans["trajectoryProjection"]["plans"][0]["right"]["generatedRawGripperTargets"][
        0
    ] = float("nan")
    plans_bytes = (
        json.dumps(
            plans, sort_keys=True, separators=(",", ":"), allow_nan=True
        ).encode()
        + b"\n"
    )
    manifest["payloads"][0]["byteSize"] = len(plans_bytes)
    manifest["payloads"][0]["sha256"] = sha256_bytes(plans_bytes)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_bytes(deterministic_json_bytes(manifest))
    (bundle / "plans.json").write_bytes(plans_bytes)
    with pytest.raises(
        CheckpointComparisonValidationError,
        match="generatedRawGripperTargets.*finite",
    ):
        validate_checkpoint_comparison(bundle)


@pytest.mark.parametrize(("field", "value"), [("major", True), ("minor", False)])
def test_v1_1_boolean_versions_are_rejected(
    tmp_path: Path, field: str, value: bool
) -> None:
    manifest, plans = available_documents()
    manifest["schema"][field] = value
    bundle = tmp_path / "bundle"
    write_bundle(bundle, (manifest, plans))
    with pytest.raises(
        CheckpointComparisonValidationError,
        match=rf"manifest\.schema\.{field} must be an integer",
    ):
        validate_checkpoint_comparison(bundle)


def test_builder_rejects_identity_mismatch_without_mutating_sources() -> None:
    obs = observation()
    result = inference()
    trajectory = project()
    snapshots = (repr(obs), repr(result), repr(trajectory))
    with pytest.raises(ValueError, match="trajectory_result.observation_id"):
        build_checkpoint_comparison_v1_1(
            obs,
            result,
            replace(trajectory, observation_id="other"),
            bundle_id="bad",
            policies=policies(),
            joint_limit_policy="reject",
        )
    assert (repr(obs), repr(result), repr(trajectory)) == snapshots


def test_atomic_v1_1_installation_validates_bundle(tmp_path: Path) -> None:
    manifest, plans = available_documents()
    destination = tmp_path / "comparison"
    export = install_checkpoint_comparison_bundle(destination, manifest, plans)
    assert export.output_path == destination
    assert validate_checkpoint_comparison(destination)["schema"]["minor"] == 1


def test_v1_1_backup_cleanup_failure_preserves_installed_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = available_documents()
    destination = tmp_path / "comparison"
    destination.mkdir()
    (destination / "old.bin").write_bytes(b"old-v1.1")
    import lerobot_state_atlas.checkpoint_comparison.artifact as artifact

    real_rmtree = artifact.shutil.rmtree

    def fail_cleanup(path, *args, **kwargs):
        if ".previous-" in Path(path).name:
            raise OSError("simulated v1.1 backup cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(artifact.shutil, "rmtree", fail_cleanup)
    with pytest.raises(
        artifact.CheckpointComparisonInstallError,
        match="new bundle remains installed",
    ):
        install_checkpoint_comparison_bundle(destination, manifest, plans)

    assert validate_checkpoint_comparison(destination)["schema"]["minor"] == 1
    backups = list(tmp_path.glob(".comparison.previous-*"))
    assert len(backups) == 1
    assert (backups[0] / "old.bin").read_bytes() == b"old-v1.1"


def test_fixture_language_contains_no_false_claims() -> None:
    text = json.dumps(available_documents(), ensure_ascii=False).lower()
    assert "first checkpoint" not in text
    assert "physical jaw width" in text
    assert 'calibratedgrippergeometry": true' not in text
