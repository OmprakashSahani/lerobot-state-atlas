"""Deterministic construction and atomic installation of comparison bundles."""

from collections.abc import Mapping
import copy
from enum import Enum, auto
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from lerobot_state_atlas.checkpoint_comparison.models import (
    AvailablePolicyComparisonProjection,
    CheckpointComparisonExport,
    PolicyComparisonInferenceResult,
    PolicyComparisonObservation,
    PolicyComparisonTrajectoryResult,
    PolicyIdentity,
    ProjectedArmTrajectory,
    UnavailablePolicyComparisonProjection,
)
from lerobot_state_atlas.checkpoint_comparison.schema import (
    ACTION_DIMENSION,
    BASE_POLICY_ID,
    BASE_POLICY_LABEL,
    CHUNK_LENGTH,
    FINE_TUNED_POLICY_ID,
    FINE_TUNED_POLICY_LABEL,
    MANIFEST_FILENAME,
    PLANS_FILENAME,
)
from lerobot_state_atlas.checkpoint_comparison.serialize import (
    deterministic_json_bytes,
    sha256_bytes,
)
from lerobot_state_atlas.checkpoint_comparison.validate import (
    validate_checkpoint_comparison,
)


class CheckpointComparisonInstallError(RuntimeError):
    """Raised when atomic comparison installation or rollback fails."""


def _recorded_ground_truth(observation: PolicyComparisonObservation) -> dict[str, Any]:
    recorded = observation.recorded_ground_truth
    if not recorded.available:
        return {"available": False, "reason": recorded.reason}
    return {
        "available": True,
        "reason": None,
        "relativeTimesSeconds": list(recorded.relative_times_seconds or ()),
        "actions": [list(row) for row in recorded.actions or ()],
    }


def _arm_document(arm: ProjectedArmTrajectory) -> dict[str, Any]:
    return {
        "armId": arm.arm_id,
        "targetLink": arm.target_link,
        "positionsXyz": [list(row) for row in arm.positions_xyz],
        "orientationsXyzw": [list(row) for row in arm.orientations_xyzw],
        "generatedRawGripperTargets": list(arm.generated_raw_gripper_targets),
    }


def _initial_state_sha256(values: tuple[float, ...]) -> str:
    return sha256_bytes(deterministic_json_bytes(list(values)))


def build_checkpoint_comparison_v1_1(
    observation: PolicyComparisonObservation,
    inference_result: PolicyComparisonInferenceResult,
    trajectory_result: PolicyComparisonTrajectoryResult,
    *,
    bundle_id: str,
    policies: tuple[PolicyIdentity, PolicyIdentity],
    joint_limit_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble validated in-memory v1.1 documents from immutable domain results."""
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError("bundle_id must be a non-empty string.")
    if observation.observation_id != inference_result.observation_id:
        raise ValueError(
            "inference_result.observation_id must match observation.observation_id."
        )
    if trajectory_result.observation_id != observation.observation_id:
        raise ValueError(
            "trajectory_result.observation_id must match observation.observation_id."
        )
    if (
        inference_result.action_dimension != ACTION_DIMENSION
        or trajectory_result.action_dimension != ACTION_DIMENSION
    ):
        raise ValueError(
            f"comparison action dimensions must both be {ACTION_DIMENSION}."
        )
    if (
        inference_result.chunk_length != CHUNK_LENGTH
        or trajectory_result.chunk_length != CHUNK_LENGTH
    ):
        raise ValueError(f"comparison chunk lengths must both be {CHUNK_LENGTH}.")
    expected = (
        (BASE_POLICY_ID, BASE_POLICY_LABEL),
        (FINE_TUNED_POLICY_ID, FINE_TUNED_POLICY_LABEL),
    )
    if tuple((item.policy_id, item.label) for item in policies) != expected:
        raise ValueError("policies must be ordered Base π0.5 then Fine-tuned π0.5.")
    if (
        tuple((item.policy_id, item.label) for item in inference_result.policies)
        != expected
    ):
        raise ValueError(
            "inference_result.policies must be ordered Base π0.5 then Fine-tuned π0.5."
        )

    schema = {
        "name": "lerobot-state-atlas.checkpoint-comparison",
        "major": 1,
        "minor": 1,
    }
    recorded = _recorded_ground_truth(observation)
    plans: dict[str, Any] = {
        "schema": schema,
        "observationId": observation.observation_id,
        "actionDimension": ACTION_DIMENSION,
        "chunkLength": CHUNK_LENGTH,
        "plans": [
            {
                "policyId": plan.policy_id,
                "relativeTimesSeconds": list(plan.relative_times_seconds),
                "actions": [list(row) for row in plan.actions],
            }
            for plan in inference_result.policies
        ],
        "recordedGroundTruth": recorded,
    }
    projection = trajectory_result.projection
    if isinstance(projection, UnavailablePolicyComparisonProjection):
        if projection.available is not False:
            raise ValueError(
                "unavailable trajectory projection available flag must be false."
            )
        plans["trajectoryProjection"] = {
            "available": False,
            "reason": projection.reason,
        }
    elif isinstance(projection, AvailablePolicyComparisonProjection):
        if projection.available is not True:
            raise ValueError(
                "available trajectory projection available flag must be true."
            )
        if trajectory_result.shared_projection_configuration is not True:
            raise ValueError(
                "trajectory_result.shared_projection_configuration must be true."
            )
        if (
            tuple((item.policy_id, item.label) for item in projection.policies)
            != expected
        ):
            raise ValueError(
                "trajectory projection policies must be ordered Base π0.5 then Fine-tuned π0.5."
            )
        interpretations = tuple(
            item.action_interpretation for item in projection.policies
        )
        if interpretations[0] != interpretations[1]:
            raise ValueError(
                "trajectory projection action interpretation must be shared."
            )
        interpretation = interpretations[0]
        robot = projection.robot
        plans["trajectoryProjection"] = {
            "available": True,
            "sharedConfiguration": True,
            "actionInterpretation": {
                "interpretationId": interpretation.interpretation_id,
                "interpretationVersion": interpretation.interpretation_version,
                "useRelativeActions": interpretation.use_relative_actions,
                "deltaActionsPreprocessorEnabled": interpretation.delta_actions_preprocessor_enabled,
                "absoluteActionsPostprocessorEnabled": interpretation.absolute_actions_postprocessor_enabled,
                "componentNames": list(interpretation.component_names),
                "initialStateParticipatesInTransformation": interpretation.initial_state_participates,
                "transformationsPerformed": list(interpretation.transformations),
            },
            "initialState": {
                "featureName": observation.state.feature_name,
                "componentNames": list(observation.state.component_names),
                "values": list(observation.state.values),
                "hashRepresentation": "deterministic-json-array-v1",
                "sha256": _initial_state_sha256(observation.state.values),
                "initialStateParticipatesInTransformation": interpretation.initial_state_participates,
            },
            "robot": {
                "robotModelName": robot.robot_model_name,
                "rootLink": robot.root_link,
                "targetLink": robot.target_link,
                "urdfSha256": robot.urdf_sha256,
                **(
                    {"upstreamRevisionIdentity": robot.upstream_revision}
                    if robot.upstream_revision is not None
                    else {}
                ),
                "fkImplementationId": robot.fk_implementation_id,
                "fkImplementationVersion": robot.fk_implementation_version,
                "leftJointMapping": [
                    {"urdfJointName": joint, "componentName": component}
                    for joint, component in robot.left_joint_mapping
                ],
                "rightJointMapping": [
                    {"urdfJointName": joint, "componentName": component}
                    for joint, component in robot.right_joint_mapping
                ],
                "leftArmTransform": {
                    "translationXyz": list(robot.left_transform_translation_xyz),
                    "rotationRpy": list(robot.left_transform_rotation_rpy),
                },
                "rightArmTransform": {
                    "translationXyz": list(robot.right_transform_translation_xyz),
                    "rotationRpy": list(robot.right_transform_rotation_rpy),
                },
                "calibratedArmTransforms": robot.calibrated_arm_transforms,
                "lengthUnit": robot.length_unit,
                "angularUnit": robot.angle_unit,
                "handedness": robot.handedness,
                "outputCoordinateFrame": robot.output_coordinate_frame,
                "rotationRepresentation": "quaternion",
                "rotationComponentOrder": ["X", "Y", "Z", "W"],
                "generatedGripperSemantics": "raw-device-specific-target",
                "gripperSemanticDisclaimer": (
                    f"{robot.gripper_semantic_disclaimer.rstrip()} Generated gripper "
                    "targets are raw device-specific unproven values and not physical "
                    "jaw widths."
                ),
                "calibratedGripperGeometry": robot.calibrated_gripper_geometry,
            },
            "jointLimitPolicy": joint_limit_policy,
            "plans": [
                {
                    "policyId": item.policy_id,
                    "relativeTimesSeconds": list(item.relative_times_seconds),
                    "left": _arm_document(item.left),
                    "right": _arm_document(item.right),
                    "jointLimitViolations": [
                        {
                            "policyId": violation.policy_id,
                            "stepIndex": violation.step_index,
                            "componentName": violation.component_name,
                            "urdfJointName": violation.urdf_joint_name,
                            "value": violation.value,
                            "bound": violation.bound,
                            "violationKind": violation.violation_kind,
                        }
                        for violation in item.joint_limit_violations
                    ],
                }
                for item in projection.policies
            ],
        }
    else:
        raise ValueError(
            "trajectory_result.projection has an unsupported projection form."
        )

    manifest: dict[str, Any] = {
        "schema": schema,
        "bundleId": bundle_id,
        "dataset": {
            "repositoryId": observation.dataset.repository_id,
            "revision": observation.dataset.revision,
        },
        "observation": {"observationId": observation.observation_id},
        "comparison": {
            "actionDimension": ACTION_DIMENSION,
            "chunkLength": CHUNK_LENGTH,
        },
        "noise": {
            "shape": list(inference_result.noise.shape),
            "dtype": inference_result.noise.dtype,
            "seed": inference_result.noise.seed,
            "sha256": inference_result.noise.sha256,
        },
        "policies": [
            {
                "policyId": item.policy_id,
                "label": item.label,
                "repositoryId": item.repository_id,
                "revision": item.revision,
            }
            for item in policies
        ],
        "recordedGroundTruth": recorded,
        "payloads": [],
    }
    plans_bytes = deterministic_json_bytes(plans)
    manifest["payloads"] = [
        {
            "kind": "plans",
            "filename": PLANS_FILENAME,
            "encoding": "json",
            "byteSize": len(plans_bytes),
            "sha256": sha256_bytes(plans_bytes),
        }
    ]
    from lerobot_state_atlas.checkpoint_comparison.validate import (
        _validate_manifest,
        _validate_plans,
    )

    validated_manifest, version = _validate_manifest(manifest)
    _validate_plans(plans, version)
    return copy.deepcopy(dict(validated_manifest)), copy.deepcopy(plans)


def install_checkpoint_comparison_v1_1_bundle(
    destination: str | Path,
    observation: PolicyComparisonObservation,
    inference_result: PolicyComparisonInferenceResult,
    trajectory_result: PolicyComparisonTrajectoryResult,
    *,
    bundle_id: str,
    policies: tuple[PolicyIdentity, PolicyIdentity],
    joint_limit_policy: str,
) -> CheckpointComparisonExport:
    """Assemble, stage, validate, and atomically install a v1.1 bundle."""
    manifest, plans = build_checkpoint_comparison_v1_1(
        observation,
        inference_result,
        trajectory_result,
        bundle_id=bundle_id,
        policies=policies,
        joint_limit_policy=joint_limit_policy,
    )
    return install_checkpoint_comparison_bundle(destination, manifest, plans)


def build_checkpoint_comparison_documents(
    manifest: Mapping[str, Any], plans: Mapping[str, Any]
) -> tuple[bytes, bytes]:
    """Return deterministic manifest and plans bytes with a computed reference."""
    normalized_manifest = copy.deepcopy(dict(manifest))
    plans_bytes = deterministic_json_bytes(plans)
    normalized_manifest["payloads"] = [
        {
            "kind": "plans",
            "filename": PLANS_FILENAME,
            "encoding": "json",
            "byteSize": len(plans_bytes),
            "sha256": sha256_bytes(plans_bytes),
        }
    ]
    return deterministic_json_bytes(normalized_manifest), plans_bytes


def _reject_immutable_destination(destination: Path) -> None:
    parts = destination.resolve().parts
    immutable_suffixes = (
        ("apps", "web", "public", "atlas-data", "demo-v1"),
        ("apps", "web", "public", "atlas-data", "demo-v2"),
    )
    if any(tuple(parts[-len(suffix) :]) == suffix for suffix in immutable_suffixes):
        raise ValueError(
            "Checkpoint-comparison bundles must not replace immutable demo-v1 or demo-v2."
        )


class _InstallationPhase(Enum):
    BEFORE_DESTINATION_REPLACEMENT = auto()
    OLD_DESTINATION_MOVED_TO_BACKUP = auto()
    NEW_BUNDLE_INSTALLED = auto()
    BACKUP_CLEANUP_NOT_STARTED = auto()
    BACKUP_CLEANUP_STARTED = auto()
    INSTALLATION_FULLY_COMPLETE = auto()


def _raise_after_backup_cleanup_failure(
    destination: Path,
    backup_path: Path,
    cleanup_error: BaseException,
) -> None:
    """Preserve the installed bundle once destructive backup cleanup has started."""
    if not destination.is_dir():
        raise CheckpointComparisonInstallError(
            "Backup cleanup failed after cleanup began, and the validated new bundle "
            f"is unexpectedly unavailable at {destination}. The backup at "
            f"{backup_path} may be partial and must not be blindly restored. "
            f"Cleanup error: {cleanup_error!r}. Manual filesystem recovery is required."
        ) from cleanup_error
    if backup_path.exists() or backup_path.is_symlink():
        raise CheckpointComparisonInstallError(
            "Backup cleanup failed after cleanup began. The complete validated new "
            f"bundle remains installed at {destination}. Remaining backup material "
            f"is preserved at {backup_path}, may be partial, and must not be blindly "
            f"restored. Cleanup error: {cleanup_error!r}. Inspect or remove the "
            "remaining backup manually after confirming it is no longer needed."
        ) from cleanup_error
    raise CheckpointComparisonInstallError(
        "Backup cleanup raised after the backup disappeared. The complete validated "
        f"new bundle remains installed at {destination}; no backup recovery path "
        f"remains. Cleanup error: {cleanup_error!r}. Confirm the destination and "
        "investigate the cleanup error manually."
    ) from cleanup_error


def install_checkpoint_comparison_bundle(
    destination: str | Path,
    manifest: Mapping[str, Any],
    plans: Mapping[str, Any],
) -> CheckpointComparisonExport:
    """Validate a staged bundle and atomically install it with rollback."""
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError(
            "Checkpoint-comparison destination must not be a symbolic link."
        )
    destination = Path(os.path.abspath(destination))
    _reject_immutable_destination(destination)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Checkpoint-comparison destination must be a directory path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes, plans_bytes = build_checkpoint_comparison_documents(manifest, plans)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    backup_path: Path | None = None
    phase = _InstallationPhase.BEFORE_DESTINATION_REPLACEMENT
    try:
        (temporary_path / MANIFEST_FILENAME).write_bytes(manifest_bytes)
        (temporary_path / PLANS_FILENAME).write_bytes(plans_bytes)
        validated_manifest = validate_checkpoint_comparison(temporary_path)
        if destination.exists():
            backup_path = destination.with_name(
                f".{destination.name}.previous-{uuid4().hex}"
            )
            os.replace(destination, backup_path)
            phase = _InstallationPhase.OLD_DESTINATION_MOVED_TO_BACKUP
        try:
            os.replace(temporary_path, destination)
        except BaseException:
            if backup_path is not None:
                os.replace(backup_path, destination)
                backup_path = None
            raise
        phase = _InstallationPhase.NEW_BUNDLE_INSTALLED
        phase = _InstallationPhase.BACKUP_CLEANUP_NOT_STARTED
        if backup_path is not None:
            phase = _InstallationPhase.BACKUP_CLEANUP_STARTED
            try:
                shutil.rmtree(backup_path)
            except BaseException as cleanup_error:
                _raise_after_backup_cleanup_failure(
                    destination,
                    backup_path,
                    cleanup_error,
                )
            backup_path = None
        phase = _InstallationPhase.INSTALLATION_FULLY_COMPLETE
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        if (
            phase is _InstallationPhase.OLD_DESTINATION_MOVED_TO_BACKUP
            and backup_path is not None
        ):
            os.replace(backup_path, destination)
        raise
    return CheckpointComparisonExport(
        output_path=destination,
        bundle_id=str(validated_manifest["bundleId"]),
        manifest_byte_count=len(manifest_bytes),
        plans_byte_count=len(plans_bytes),
        plans_sha256=sha256_bytes(plans_bytes),
    )
