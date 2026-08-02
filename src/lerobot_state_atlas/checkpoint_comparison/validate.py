"""Strict structural and cross-file validation for comparison schemas v1.0/v1.1."""

from collections.abc import Mapping
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import re
from typing import Any

from lerobot_state_atlas.checkpoint_comparison.schema import (
    ACTION_DIMENSION,
    BASE_POLICY_ID,
    BASE_POLICY_LABEL,
    CHUNK_LENGTH,
    FINE_TUNED_POLICY_ID,
    FINE_TUNED_POLICY_LABEL,
    MANIFEST_FILENAME,
    NOISE_SHAPE,
    PLANS_FILENAME,
    SCHEMA_NAME,
    SUPPORTED_SCHEMA_VERSIONS,
)
from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
    read_stable_file_snapshot,
)
from lerobot_state_atlas.checkpoint_comparison.serialize import sha256_bytes


class CheckpointComparisonValidationError(ValueError):
    """Raised when a checkpoint-comparison bundle violates its contract."""


def _fail(label: str, message: str) -> None:
    raise CheckpointComparisonValidationError(f"{label} {message}")


def _object(
    value: Any,
    fields: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(label, "must be an object.")
    missing = fields - set(value)
    unsupported = set(value) - fields - (optional or set())
    if missing:
        _fail(label, f"is missing fields: {', '.join(sorted(missing))}.")
    if unsupported:
        _fail(label, f"has unsupported fields: {', '.join(sorted(unsupported))}.")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(label, "must be a non-empty string.")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(label, "must be an integer.")
    if minimum is not None and value < minimum:
        _fail(label, f"must be greater than or equal to {minimum}.")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(label, "must be a number.")
    normalized = float(value)
    if not isfinite(normalized):
        _fail(label, "must be finite.")
    return normalized


def _sha(value: Any, label: str, *, length: int) -> str:
    digest = _string(value, label)
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", digest):
        _fail(label, f"must be a full lowercase {length}-character hexadecimal SHA.")
    return digest


def _schema(value: Any, label: str) -> tuple[int, int]:
    schema = _object(value, {"name", "major", "minor"}, label)
    if schema["name"] != SCHEMA_NAME:
        _fail(f"{label}.name", f"must be {SCHEMA_NAME!r}.")
    major = _integer(schema["major"], f"{label}.major")
    minor = _integer(schema["minor"], f"{label}.minor")
    if (major, minor) not in SUPPORTED_SCHEMA_VERSIONS:
        _fail(label, f"version {major}.{minor} is unsupported.")
    return major, minor


def _safe_filename(value: Any, label: str) -> str:
    filename = _string(value, label)
    path = PurePosixPath(filename)
    if (
        "\\" in filename
        or "://" in filename
        or filename.startswith("//")
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != filename
    ):
        _fail(label, "must be a safe bundle-relative POSIX path.")
    return filename


def _policy_list(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        _fail(label, "must contain exactly two policies.")
    expected = (
        (BASE_POLICY_ID, BASE_POLICY_LABEL),
        (FINE_TUNED_POLICY_ID, FINE_TUNED_POLICY_LABEL),
    )
    result: list[Mapping[str, Any]] = []
    for index, (policy_id, policy_label) in enumerate(expected):
        item_label = f"{label}[{index}]"
        item = _object(
            value[index],
            {"policyId", "label", "repositoryId", "revision"},
            item_label,
        )
        if item["policyId"] != policy_id:
            _fail(f"{item_label}.policyId", f"must be {policy_id!r}.")
        if item["label"] != policy_label:
            _fail(f"{item_label}.label", f"must be {policy_label!r}.")
        _string(item["repositoryId"], f"{item_label}.repositoryId")
        _sha(item["revision"], f"{item_label}.revision", length=40)
        result.append(item)
    return result


def _actions(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != CHUNK_LENGTH:
        _fail(label, f"must contain exactly {CHUNK_LENGTH} action steps.")
    for step_index, step in enumerate(value):
        step_label = f"{label}[{step_index}]"
        if not isinstance(step, list) or len(step) != ACTION_DIMENSION:
            _fail(step_label, f"must contain exactly {ACTION_DIMENSION} values.")
        for component_index, component in enumerate(step):
            _number(component, f"{step_label}[{component_index}]")


def _times(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != CHUNK_LENGTH:
        _fail(label, f"must contain exactly {CHUNK_LENGTH} relative times.")
    times = [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if times[0] != 0.0:
        _fail(f"{label}[0]", "must be 0.0.")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        _fail(label, "must be strictly increasing.")


def _recorded(value: Any, label: str) -> Mapping[str, Any]:
    recorded = _object(
        value,
        {"available", "reason"},
        label,
        optional={"relativeTimesSeconds", "actions"},
    )
    if not isinstance(recorded["available"], bool):
        _fail(f"{label}.available", "must be boolean.")
    if recorded["available"]:
        if recorded["reason"] is not None:
            _fail(
                f"{label}.reason", "must be null when recorded actions are available."
            )
        if "relativeTimesSeconds" not in recorded or "actions" not in recorded:
            _fail(
                label, "must include relativeTimesSeconds and actions when available."
            )
        _times(recorded["relativeTimesSeconds"], f"{label}.relativeTimesSeconds")
        _actions(recorded["actions"], f"{label}.actions")
    else:
        _string(recorded["reason"], f"{label}.reason")
        if "relativeTimesSeconds" in recorded or "actions" in recorded:
            _fail(label, "must not include action data when unavailable.")
    return recorded


COMPONENT_NAMES = tuple(
    [*(f"left_joint_{index}.pos" for index in range(1, 7)), "left_gripper.pos"]
    + [*(f"right_joint_{index}.pos" for index in range(1, 7)), "right_gripper.pos"]
)
ARM_COMPONENT_NAMES = tuple(name for name in COMPONENT_NAMES if "gripper" not in name)
QUATERNION_NORM_TOLERANCE = 1e-9
INITIAL_STATE_HASH_REPRESENTATION = "deterministic-json-array-v1"


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(label, "must be boolean.")
    return value


def _string_list(value: Any, expected: tuple[str, ...], label: str) -> None:
    if not isinstance(value, list) or len(value) != len(expected):
        _fail(label, f"must contain exactly {len(expected)} entries.")
    for index, expected_value in enumerate(expected):
        if value[index] != expected_value:
            _fail(f"{label}[{index}]", f"must be {expected_value!r}.")


def _number_vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        _fail(label, f"must contain exactly {size} values.")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _number_rows(value: Any, rows: int, width: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != rows:
        _fail(label, f"must contain exactly {rows} rows.")
    for index, row in enumerate(value):
        _number_vector(row, width, f"{label}[{index}]")


def _validate_interpretation(value: Any, label: str) -> Mapping[str, Any]:
    result = _object(
        value,
        {
            "interpretationId",
            "interpretationVersion",
            "useRelativeActions",
            "deltaActionsPreprocessorEnabled",
            "absoluteActionsPostprocessorEnabled",
            "componentNames",
            "initialStateParticipatesInTransformation",
            "transformationsPerformed",
        },
        label,
    )
    if result["interpretationId"] != "pi05-postprocessed-absolute-position-targets":
        _fail(f"{label}.interpretationId", "must identify PI05 absolute targets.")
    if result["interpretationVersion"] != "1.0":
        _fail(f"{label}.interpretationVersion", "must be '1.0'.")
    for field in (
        "useRelativeActions",
        "deltaActionsPreprocessorEnabled",
        "absoluteActionsPostprocessorEnabled",
        "initialStateParticipatesInTransformation",
    ):
        if _boolean(result[field], f"{label}.{field}") is not False:
            _fail(f"{label}.{field}", "must be false.")
    _string_list(result["componentNames"], COMPONENT_NAMES, f"{label}.componentNames")
    transformations = result["transformationsPerformed"]
    if not isinstance(transformations, list) or not transformations:
        _fail(f"{label}.transformationsPerformed", "must be a non-empty array.")
    for index, item in enumerate(transformations):
        _string(item, f"{label}.transformationsPerformed[{index}]")
    return result


def _validate_initial_state(value: Any, label: str) -> Mapping[str, Any]:
    state = _object(
        value,
        {
            "featureName",
            "componentNames",
            "values",
            "hashRepresentation",
            "sha256",
            "initialStateParticipatesInTransformation",
        },
        label,
    )
    if state["featureName"] != "observation.state":
        _fail(f"{label}.featureName", "must be 'observation.state'.")
    _string_list(state["componentNames"], COMPONENT_NAMES, f"{label}.componentNames")
    _number_vector(state["values"], ACTION_DIMENSION, f"{label}.values")
    if state["hashRepresentation"] != INITIAL_STATE_HASH_REPRESENTATION:
        _fail(
            f"{label}.hashRepresentation",
            f"must be {INITIAL_STATE_HASH_REPRESENTATION!r}.",
        )
    digest = _sha(state["sha256"], f"{label}.sha256", length=64)
    expected = sha256_bytes(
        json.dumps(
            state["values"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    if digest != expected:
        _fail(f"{label}.sha256", "does not match the canonical initial-state values.")
    _boolean(
        state["initialStateParticipatesInTransformation"],
        f"{label}.initialStateParticipatesInTransformation",
    )
    return state


def _validate_mapping(value: Any, arm: str, label: str) -> None:
    if not isinstance(value, list) or len(value) != 6:
        _fail(label, "must contain exactly six joint mappings.")
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        mapping = _object(item, {"urdfJointName", "componentName"}, item_label)
        _string(mapping["urdfJointName"], f"{item_label}.urdfJointName")
        expected = f"{arm}_joint_{index + 1}.pos"
        if mapping["componentName"] != expected:
            _fail(f"{item_label}.componentName", f"must be {expected!r}.")


def _validate_transform(value: Any, label: str) -> None:
    transform = _object(value, {"translationXyz", "rotationRpy"}, label)
    _number_vector(transform["translationXyz"], 3, f"{label}.translationXyz")
    _number_vector(transform["rotationRpy"], 3, f"{label}.rotationRpy")


def _validate_robot(value: Any, label: str) -> Mapping[str, Any]:
    robot = _object(
        value,
        {
            "robotModelName",
            "rootLink",
            "targetLink",
            "urdfSha256",
            "fkImplementationId",
            "fkImplementationVersion",
            "leftJointMapping",
            "rightJointMapping",
            "leftArmTransform",
            "rightArmTransform",
            "calibratedArmTransforms",
            "lengthUnit",
            "angularUnit",
            "handedness",
            "outputCoordinateFrame",
            "rotationRepresentation",
            "rotationComponentOrder",
            "generatedGripperSemantics",
            "gripperSemanticDisclaimer",
            "calibratedGripperGeometry",
        },
        label,
        optional={"upstreamRevisionIdentity"},
    )
    for field in (
        "robotModelName",
        "rootLink",
        "targetLink",
        "fkImplementationId",
        "fkImplementationVersion",
        "lengthUnit",
        "angularUnit",
        "handedness",
        "outputCoordinateFrame",
        "generatedGripperSemantics",
        "gripperSemanticDisclaimer",
    ):
        _string(robot[field], f"{label}.{field}")
    if "upstreamRevisionIdentity" in robot:
        _string(robot["upstreamRevisionIdentity"], f"{label}.upstreamRevisionIdentity")
    _sha(robot["urdfSha256"], f"{label}.urdfSha256", length=64)
    _validate_mapping(robot["leftJointMapping"], "left", f"{label}.leftJointMapping")
    _validate_mapping(robot["rightJointMapping"], "right", f"{label}.rightJointMapping")
    _validate_transform(robot["leftArmTransform"], f"{label}.leftArmTransform")
    _validate_transform(robot["rightArmTransform"], f"{label}.rightArmTransform")
    _boolean(robot["calibratedArmTransforms"], f"{label}.calibratedArmTransforms")
    if robot["rotationRepresentation"] != "quaternion":
        _fail(f"{label}.rotationRepresentation", "must be 'quaternion'.")
    _string_list(
        robot["rotationComponentOrder"],
        ("X", "Y", "Z", "W"),
        f"{label}.rotationComponentOrder",
    )
    if (
        _boolean(
            robot["calibratedGripperGeometry"], f"{label}.calibratedGripperGeometry"
        )
        is not False
    ):
        _fail(f"{label}.calibratedGripperGeometry", "must be false.")
    if robot["generatedGripperSemantics"] != "raw-device-specific-target":
        _fail(
            f"{label}.generatedGripperSemantics",
            "must be 'raw-device-specific-target'.",
        )
    disclaimer = robot["gripperSemanticDisclaimer"].lower()
    if "raw" not in disclaimer or "not physical jaw widths" not in disclaimer:
        _fail(
            f"{label}.gripperSemanticDisclaimer",
            "must state that targets are raw values and not physical jaw widths.",
        )
    return robot


def _validate_arm(value: Any, arm: str, target_link: str, label: str) -> None:
    trajectory = _object(
        value,
        {
            "armId",
            "targetLink",
            "positionsXyz",
            "orientationsXyzw",
            "generatedRawGripperTargets",
        },
        label,
    )
    if trajectory["armId"] != arm:
        _fail(f"{label}.armId", f"must be {arm!r}.")
    if trajectory["targetLink"] != target_link:
        _fail(
            f"{label}.targetLink",
            "does not match trajectoryProjection.robot.targetLink.",
        )
    _number_rows(trajectory["positionsXyz"], CHUNK_LENGTH, 3, f"{label}.positionsXyz")
    orientations = trajectory["orientationsXyzw"]
    _number_rows(orientations, CHUNK_LENGTH, 4, f"{label}.orientationsXyzw")
    for index, quaternion in enumerate(orientations):
        norm = sum(float(item) ** 2 for item in quaternion) ** 0.5
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            _fail(f"{label}.orientationsXyzw[{index}]", "must be a unit quaternion.")
    _number_vector(
        trajectory["generatedRawGripperTargets"],
        CHUNK_LENGTH,
        f"{label}.generatedRawGripperTargets",
    )


def _validate_violation(value: Any, policy_id: str, label: str) -> tuple[int, int, str]:
    violation = _object(
        value,
        {
            "policyId",
            "stepIndex",
            "componentName",
            "urdfJointName",
            "value",
            "bound",
            "violationKind",
        },
        label,
    )
    if violation["policyId"] != policy_id:
        _fail(f"{label}.policyId", "must match the containing projected plan.")
    step = _integer(violation["stepIndex"], f"{label}.stepIndex", minimum=0)
    if step >= CHUNK_LENGTH:
        _fail(f"{label}.stepIndex", f"must be less than {CHUNK_LENGTH}.")
    component = violation["componentName"]
    if component not in ARM_COMPONENT_NAMES:
        _fail(
            f"{label}.componentName", "must identify a canonical arm joint component."
        )
    _string(violation["urdfJointName"], f"{label}.urdfJointName")
    _number(violation["value"], f"{label}.value")
    _number(violation["bound"], f"{label}.bound")
    kind = violation["violationKind"]
    if kind not in {"lower", "upper"}:
        _fail(f"{label}.violationKind", "must be 'lower' or 'upper'.")
    return step, COMPONENT_NAMES.index(component), kind


def _validate_trajectory_projection(value: Any, plans: Mapping[str, Any]) -> None:
    label = "plans.trajectoryProjection"
    if not isinstance(value, dict):
        _fail(label, "must be an object.")
    available = value.get("available")
    if not isinstance(available, bool):
        _fail(f"{label}.available", "must be boolean.")
    if not available:
        unavailable = _object(value, {"available", "reason"}, label)
        reason = _string(unavailable["reason"], f"{label}.reason")
        if reason != reason.strip():
            _fail(
                f"{label}.reason", "must be normalized without surrounding whitespace."
            )
        return
    projection = _object(
        value,
        {
            "available",
            "sharedConfiguration",
            "actionInterpretation",
            "initialState",
            "robot",
            "jointLimitPolicy",
            "plans",
        },
        label,
    )
    if projection["sharedConfiguration"] is not True:
        _fail(f"{label}.sharedConfiguration", "must be true.")
    interpretation = _validate_interpretation(
        projection["actionInterpretation"], f"{label}.actionInterpretation"
    )
    state = _validate_initial_state(projection["initialState"], f"{label}.initialState")
    if (
        state["initialStateParticipatesInTransformation"]
        != interpretation["initialStateParticipatesInTransformation"]
    ):
        _fail(
            f"{label}.initialState.initialStateParticipatesInTransformation",
            "must agree with actionInterpretation.",
        )
    robot = _validate_robot(projection["robot"], f"{label}.robot")
    limit_policy = projection["jointLimitPolicy"]
    if limit_policy not in {"reject", "allow-with-recorded-violations"}:
        _fail(
            f"{label}.jointLimitPolicy",
            "must be 'reject' or 'allow-with-recorded-violations'.",
        )
    projected_plans = projection["plans"]
    if not isinstance(projected_plans, list) or len(projected_plans) != 2:
        _fail(f"{label}.plans", "must contain exactly two projected plans.")
    expected_ids = (BASE_POLICY_ID, FINE_TUNED_POLICY_ID)
    for index, policy_id in enumerate(expected_ids):
        plan_label = f"{label}.plans[{index}]"
        projected = _object(
            projected_plans[index],
            {
                "policyId",
                "relativeTimesSeconds",
                "left",
                "right",
                "jointLimitViolations",
            },
            plan_label,
        )
        if projected["policyId"] != policy_id:
            _fail(f"{plan_label}.policyId", f"must be {policy_id!r}.")
        _times(projected["relativeTimesSeconds"], f"{plan_label}.relativeTimesSeconds")
        if (
            projected["relativeTimesSeconds"]
            != plans["plans"][index]["relativeTimesSeconds"]
        ):
            _fail(
                f"{plan_label}.relativeTimesSeconds",
                "must match the authoritative plan.",
            )
        _validate_arm(
            projected["left"], "left", robot["targetLink"], f"{plan_label}.left"
        )
        _validate_arm(
            projected["right"], "right", robot["targetLink"], f"{plan_label}.right"
        )
        violations = projected["jointLimitViolations"]
        if not isinstance(violations, list):
            _fail(f"{plan_label}.jointLimitViolations", "must be an array.")
        keys = [
            _validate_violation(
                item, policy_id, f"{plan_label}.jointLimitViolations[{violation_index}]"
            )
            for violation_index, item in enumerate(violations)
        ]
        if keys != sorted(keys):
            _fail(
                f"{plan_label}.jointLimitViolations",
                "must be deterministically ordered.",
            )
        if limit_policy == "reject" and violations:
            _fail(
                f"{plan_label}.jointLimitViolations", "must be empty for reject policy."
            )


def _load_manifest_snapshot(content: bytes, filename: str) -> Any:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointComparisonValidationError(
            f"Invalid manifest UTF-8: {filename}."
        ) from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise CheckpointComparisonValidationError(
            f"Invalid manifest JSON: {filename}."
        ) from error


def _load_json_snapshot(content: bytes, filename: str) -> Any:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointComparisonValidationError(
            f"Invalid plans payload UTF-8: {filename}."
        ) from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise CheckpointComparisonValidationError(
            f"Invalid plans payload JSON: {filename}."
        ) from error


def _validate_manifest(value: Any) -> tuple[Mapping[str, Any], tuple[int, int]]:
    manifest = _object(
        value,
        {
            "schema",
            "bundleId",
            "dataset",
            "observation",
            "comparison",
            "noise",
            "policies",
            "recordedGroundTruth",
            "payloads",
        },
        "manifest",
    )
    version = _schema(manifest["schema"], "manifest.schema")
    _string(manifest["bundleId"], "manifest.bundleId")
    dataset = _object(
        manifest["dataset"], {"repositoryId", "revision"}, "manifest.dataset"
    )
    _string(dataset["repositoryId"], "manifest.dataset.repositoryId")
    _sha(dataset["revision"], "manifest.dataset.revision", length=40)
    observation = _object(
        manifest["observation"], {"observationId"}, "manifest.observation"
    )
    _string(observation["observationId"], "manifest.observation.observationId")
    comparison = _object(
        manifest["comparison"],
        {"actionDimension", "chunkLength"},
        "manifest.comparison",
    )
    action_dimension = _integer(
        comparison["actionDimension"], "manifest.comparison.actionDimension"
    )
    if action_dimension != ACTION_DIMENSION:
        _fail("manifest.comparison.actionDimension", f"must be {ACTION_DIMENSION}.")
    chunk_length = _integer(
        comparison["chunkLength"], "manifest.comparison.chunkLength"
    )
    if chunk_length != CHUNK_LENGTH:
        _fail("manifest.comparison.chunkLength", f"must be {CHUNK_LENGTH}.")
    noise = _object(
        manifest["noise"], {"shape", "dtype", "seed", "sha256"}, "manifest.noise"
    )
    shape = noise["shape"]
    if not isinstance(shape, list) or len(shape) != len(NOISE_SHAPE):
        _fail("manifest.noise.shape", f"must be {list(NOISE_SHAPE)}.")
    normalized_shape = [
        _integer(value, f"manifest.noise.shape[{index}]")
        for index, value in enumerate(shape)
    ]
    if normalized_shape != list(NOISE_SHAPE):
        _fail("manifest.noise.shape", f"must be {list(NOISE_SHAPE)}.")
    _string(noise["dtype"], "manifest.noise.dtype")
    _integer(noise["seed"], "manifest.noise.seed", minimum=0)
    _sha(noise["sha256"], "manifest.noise.sha256", length=64)
    _policy_list(manifest["policies"], "manifest.policies")
    _recorded(manifest["recordedGroundTruth"], "manifest.recordedGroundTruth")
    payloads = manifest["payloads"]
    if not isinstance(payloads, list) or len(payloads) != 1:
        _fail("manifest.payloads", "must contain exactly one plans payload.")
    payload = _object(
        payloads[0],
        {"kind", "filename", "encoding", "byteSize", "sha256"},
        "manifest.payloads[0]",
    )
    if payload["kind"] != "plans":
        _fail("manifest.payloads[0].kind", "must be 'plans'.")
    _safe_filename(payload["filename"], "manifest.payloads[0].filename")
    if payload["filename"] != PLANS_FILENAME:
        _fail("manifest.payloads[0].filename", f"must be {PLANS_FILENAME!r}.")
    if payload["encoding"] != "json":
        _fail("manifest.payloads[0].encoding", "must be 'json'.")
    _integer(payload["byteSize"], "manifest.payloads[0].byteSize", minimum=1)
    _sha(payload["sha256"], "manifest.payloads[0].sha256", length=64)
    return manifest, version


def _validate_plans(value: Any, version: tuple[int, int]) -> Mapping[str, Any]:
    plans = _object(
        value,
        {
            "schema",
            "observationId",
            "actionDimension",
            "chunkLength",
            "plans",
            "recordedGroundTruth",
        },
        "plans",
        optional={"trajectoryProjection"},
    )
    plans_version = _schema(plans["schema"], "plans.schema")
    if plans_version != version:
        field = "major" if plans_version[0] != version[0] else "minor"
        _fail(
            f"manifest.schema.{field}",
            f"does not match plans.schema.{field}.",
        )
    if version == (1, 1) and "trajectoryProjection" not in plans:
        _fail("plans", "is missing fields: trajectoryProjection.")
    if version == (1, 0) and "trajectoryProjection" in plans:
        _fail("plans", "has unsupported fields: trajectoryProjection.")
    _string(plans["observationId"], "plans.observationId")
    action_dimension = _integer(plans["actionDimension"], "plans.actionDimension")
    if action_dimension != ACTION_DIMENSION:
        _fail("plans.actionDimension", f"must be {ACTION_DIMENSION}.")
    chunk_length = _integer(plans["chunkLength"], "plans.chunkLength")
    if chunk_length != CHUNK_LENGTH:
        _fail("plans.chunkLength", f"must be {CHUNK_LENGTH}.")
    plan_values = plans["plans"]
    if not isinstance(plan_values, list) or len(plan_values) != 2:
        _fail("plans.plans", "must contain exactly two generated policy plans.")
    for index, policy_id in enumerate((BASE_POLICY_ID, FINE_TUNED_POLICY_ID)):
        label = f"plans.plans[{index}]"
        plan = _object(
            plan_values[index],
            {"policyId", "relativeTimesSeconds", "actions"},
            label,
        )
        if plan["policyId"] != policy_id:
            _fail(f"{label}.policyId", f"must be {policy_id!r}.")
        _times(plan["relativeTimesSeconds"], f"{label}.relativeTimesSeconds")
        _actions(plan["actions"], f"{label}.actions")
    _recorded(plans["recordedGroundTruth"], "plans.recordedGroundTruth")
    if version == (1, 1):
        _validate_trajectory_projection(plans["trajectoryProjection"], plans)
    return plans


def validate_checkpoint_comparison(path: str | Path) -> Mapping[str, Any]:
    """Validate a complete comparison bundle and return its manifest."""
    bundle = Path(path).resolve()
    manifest_path = bundle / MANIFEST_FILENAME
    if manifest_path.is_symlink():
        _fail("manifest", f"{MANIFEST_FILENAME!r} must not be a symbolic link.")
    if not manifest_path.exists():
        _fail("manifest", f"is missing required file {MANIFEST_FILENAME!r}.")
    if not manifest_path.is_file():
        _fail("manifest", f"{MANIFEST_FILENAME!r} must be a regular file.")
    try:
        manifest_snapshot = read_stable_file_snapshot(manifest_path)
    except StableFileSnapshotError as error:
        raise CheckpointComparisonValidationError(
            f"manifest {MANIFEST_FILENAME!r} could not be acquired as a stable "
            f"file snapshot: {error}."
        ) from error
    manifest, version = _validate_manifest(
        _load_manifest_snapshot(manifest_snapshot, MANIFEST_FILENAME)
    )
    reference = manifest["payloads"][0]
    declared_plans_path = bundle / reference["filename"]
    plans_path = declared_plans_path.resolve()
    try:
        plans_path.relative_to(bundle)
    except ValueError:
        _fail(
            "manifest.payloads[0].filename",
            f"({reference['filename']!r}) resolves outside the comparison bundle.",
        )
    if declared_plans_path.is_symlink():
        _fail(
            "manifest.payloads[0].filename",
            f"({reference['filename']!r}) must be a regular file, not a symlink.",
        )
    if not plans_path.exists():
        _fail(
            "manifest.payloads[0].filename",
            f"references missing payload {reference['filename']!r}.",
        )
    if not plans_path.is_file():
        _fail(
            "manifest.payloads[0].filename",
            f"({reference['filename']!r}) must reference a regular file.",
        )
    try:
        plans_snapshot = read_stable_file_snapshot(plans_path)
    except StableFileSnapshotError as error:
        raise CheckpointComparisonValidationError(
            f"manifest.payloads[0].filename ({reference['filename']!r}) "
            f"could not be acquired as a stable file snapshot: {error}."
        ) from error
    byte_count = len(plans_snapshot)
    if byte_count != reference["byteSize"]:
        _fail(
            "manifest.payloads[0].byteSize",
            f"declares {reference['byteSize']} bytes but {reference['filename']} has {byte_count}.",
        )
    digest = sha256_bytes(plans_snapshot)
    if digest != reference["sha256"]:
        _fail(
            "manifest.payloads[0].sha256",
            f"does not match {reference['filename']}.",
        )
    plans = _validate_plans(
        _load_json_snapshot(plans_snapshot, reference["filename"]), version
    )
    if plans["observationId"] != manifest["observation"]["observationId"]:
        _fail(
            "plans.observationId", "does not match manifest.observation.observationId."
        )
    if plans["actionDimension"] != manifest["comparison"]["actionDimension"]:
        _fail(
            "plans.actionDimension",
            "does not match manifest.comparison.actionDimension.",
        )
    if plans["chunkLength"] != manifest["comparison"]["chunkLength"]:
        _fail("plans.chunkLength", "does not match manifest.comparison.chunkLength.")
    manifest_policy_ids = [item["policyId"] for item in manifest["policies"]]
    plans_policy_ids = [item["policyId"] for item in plans["plans"]]
    if plans_policy_ids != manifest_policy_ids:
        _fail("plans.plans", "policy IDs do not match manifest.policies.")
    if plans["recordedGroundTruth"] != manifest["recordedGroundTruth"]:
        _fail(
            "plans.recordedGroundTruth", "does not match manifest.recordedGroundTruth."
        )
    return manifest
