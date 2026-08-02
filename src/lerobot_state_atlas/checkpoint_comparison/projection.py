"""Comparison-specific action rollout and deterministic dual-arm FK projection."""

from dataclasses import dataclass
from math import isfinite
import re
from typing import Protocol

import torch

from lerobot_state_atlas.checkpoint_comparison.models import (
    ActionInterpretationProvenance,
    ActionRolloutResult,
    AvailablePolicyComparisonProjection,
    InferredPolicyPlan,
    JointLimitViolation,
    PolicyComparisonInferenceResult,
    PolicyComparisonObservation,
    PolicyComparisonTrajectoryResult,
    ProjectedArmTrajectory,
    ProjectedPolicyTrajectory,
    RobotProjectionProvenance,
    UnavailablePolicyComparisonProjection,
)
from lerobot_state_atlas.checkpoint_comparison.observation import COMPONENT_NAMES
from lerobot_state_atlas.checkpoint_comparison.schema import (
    ACTION_DIMENSION,
    BASE_POLICY_ID,
    BASE_POLICY_LABEL,
    CHUNK_LENGTH,
    FINE_TUNED_POLICY_ID,
    FINE_TUNED_POLICY_LABEL,
)
from lerobot_state_atlas.orientation import rotation_matrices_to_quaternions_xyzw
from lerobot_state_atlas.trajectory import (
    build_trlc_dk1_gripper_component_name,
    build_trlc_dk1_joint_component_map,
    compute_tool_trajectory,
)
from lerobot_state_atlas.transforms import RigidTransform, transform_tool_trajectory
from lerobot_state_atlas.urdf import RobotModel


ABSOLUTE_INTERPRETATION_ID = "pi05-postprocessed-absolute-position-targets"
ABSOLUTE_INTERPRETATION_VERSION = "1.0"
ABSOLUTE_TRANSFORMATIONS = (
    "validate-postprocessed-action-targets",
    "preserve-actions-as-absolute-target-states",
)
JOINT_LIMIT_POLICIES = frozenset({"reject", "allow-with-recorded-violations"})
QUATERNION_NORM_TOLERANCE = 1e-9


class PolicyComparisonProjectionError(ValueError):
    """Raised when comparison actions cannot be projected truthfully."""


def _fail(path: str, message: str) -> None:
    raise PolicyComparisonProjectionError(f"{path} {message}")


class ActionRolloutAdapter(Protocol):
    """Explicit action-interpretation boundary used before FK."""

    def rollout(
        self,
        *,
        initial_state: tuple[float, ...],
        actions: tuple[tuple[float, ...], ...],
        component_names: tuple[str, ...],
        fps: float,
    ) -> ActionRolloutResult: ...


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number.")
    normalized = float(value)
    if not isfinite(normalized):
        _fail(path, "must be finite.")
    return normalized


def _state_row(value: object, path: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != ACTION_DIMENSION:
        received = (
            len(value) if isinstance(value, (tuple, list)) else type(value).__name__
        )
        _fail(
            path,
            f"must contain exactly {ACTION_DIMENSION} values; received {received}.",
        )
    return tuple(
        _finite_number(component, f"{path}[{index}]")
        for index, component in enumerate(value)
    )


def _action_chunk(value: object, path: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != CHUNK_LENGTH:
        received = (
            len(value) if isinstance(value, (tuple, list)) else type(value).__name__
        )
        _fail(path, f"must contain exactly {CHUNK_LENGTH} rows; received {received}.")
    return tuple(_state_row(row, f"{path}[{index}]") for index, row in enumerate(value))


def _validate_component_names(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        _fail(path, "must be the canonical ordered component sequence.")
    names = tuple(value)
    if names != COMPONENT_NAMES:
        _fail(path, "must match the canonical ordered 14-component contract.")
    return names


def _positive_fps(value: object, path: str) -> float:
    fps = _finite_number(value, path)
    if fps <= 0:
        _fail(path, "must be greater than zero.")
    return fps


def _validate_interpretation(
    value: ActionInterpretationProvenance, path: str
) -> ActionInterpretationProvenance:
    if not isinstance(value, ActionInterpretationProvenance):
        _fail(path, "must be ActionInterpretationProvenance.")
    if value.interpretation_id != ABSOLUTE_INTERPRETATION_ID:
        _fail(
            f"{path}.interpretation_id",
            f"must be {ABSOLUTE_INTERPRETATION_ID!r}.",
        )
    if value.interpretation_version != ABSOLUTE_INTERPRETATION_VERSION:
        _fail(
            f"{path}.interpretation_version",
            f"must be {ABSOLUTE_INTERPRETATION_VERSION!r}.",
        )
    for field in (
        "use_relative_actions",
        "delta_actions_preprocessor_enabled",
        "absolute_actions_postprocessor_enabled",
        "initial_state_participates",
    ):
        field_value = getattr(value, field)
        if not isinstance(field_value, bool):
            _fail(f"{path}.{field}", "must be boolean.")
        if field_value:
            _fail(f"{path}.{field}", "must be false for this interpretation.")
    _validate_component_names(value.component_names, f"{path}.component_names")
    if value.transformations != ABSOLUTE_TRANSFORMATIONS:
        _fail(
            f"{path}.transformations",
            f"must be {ABSOLUTE_TRANSFORMATIONS!r}.",
        )
    return value


@dataclass(frozen=True)
class PI05AbsolutePositionTargetsAdapter:
    """Verified passthrough for disabled-relative PI05 postprocessed targets."""

    interpretation_provenance: ActionInterpretationProvenance

    def __post_init__(self) -> None:
        _validate_interpretation(
            self.interpretation_provenance, "interpretation_provenance"
        )

    def rollout(
        self,
        *,
        initial_state: tuple[float, ...],
        actions: tuple[tuple[float, ...], ...],
        component_names: tuple[str, ...],
        fps: float,
    ) -> ActionRolloutResult:
        normalized_names = _validate_component_names(component_names, "component_names")
        normalized_initial = _state_row(initial_state, "initial_state")
        normalized_actions = _action_chunk(actions, "actions")
        normalized_fps = _positive_fps(fps, "fps")
        return ActionRolloutResult(
            interpretation=self.interpretation_provenance,
            target_states=tuple(tuple(row) for row in normalized_actions),
            component_names=normalized_names,
            fps=normalized_fps,
            initial_state=tuple(normalized_initial),
        )


def _validate_binding(
    observation: PolicyComparisonObservation,
    inference_result: PolicyComparisonInferenceResult,
) -> tuple[float, tuple[InferredPolicyPlan, InferredPolicyPlan]]:
    if not isinstance(observation, PolicyComparisonObservation):
        _fail("observation", "must be a validated PolicyComparisonObservation.")
    if not isinstance(inference_result, PolicyComparisonInferenceResult):
        _fail("inference_result", "must be PolicyComparisonInferenceResult.")
    if inference_result.observation_id != observation.observation_id:
        _fail(
            "inference_result.observation_id",
            "must match observation.observation_id.",
        )
    if (
        isinstance(inference_result.action_dimension, bool)
        or inference_result.action_dimension != ACTION_DIMENSION
    ):
        _fail("inference_result.action_dimension", f"must be {ACTION_DIMENSION}.")
    if (
        isinstance(inference_result.chunk_length, bool)
        or inference_result.chunk_length != CHUNK_LENGTH
    ):
        _fail("inference_result.chunk_length", f"must be {CHUNK_LENGTH}.")
    _validate_component_names(
        observation.state.component_names, "observation.state.component_names"
    )
    if observation.state.feature_name != "observation.state":
        _fail("observation.state.feature_name", "must be 'observation.state'.")
    _state_row(observation.state.values, "observation.state.values")
    fps = _positive_fps(observation.dataset.fps, "observation.dataset.fps")
    policies = inference_result.policies
    if not isinstance(policies, tuple) or len(policies) != 2:
        _fail("inference_result.policies", "must contain exactly two ordered policies.")
    expected = (
        (BASE_POLICY_ID, BASE_POLICY_LABEL),
        (FINE_TUNED_POLICY_ID, FINE_TUNED_POLICY_LABEL),
    )
    normalized: list[InferredPolicyPlan] = []
    expected_times = tuple(index / fps for index in range(CHUNK_LENGTH))
    for index, (policy_id, label) in enumerate(expected):
        path = f"inference_result.policies[{index}]"
        plan = policies[index]
        if not isinstance(plan, InferredPolicyPlan):
            _fail(path, "must be InferredPolicyPlan.")
        if plan.policy_id != policy_id:
            _fail(f"{path}.policy_id", f"must be {policy_id!r}.")
        if plan.label != label:
            _fail(f"{path}.label", f"must be {label!r}.")
        if plan.relative_times_seconds != expected_times:
            _fail(
                f"{path}.relative_times_seconds",
                "must equal 0/fps through 49/fps exactly.",
            )
        _action_chunk(plan.actions, f"{path}.actions")
        normalized.append(plan)
    return fps, (normalized[0], normalized[1])


def _validate_mapping(
    model: RobotModel,
    mapping: tuple[tuple[str, str], ...],
    expected: tuple[tuple[str, str], ...],
    path: str,
) -> None:
    if mapping != expected:
        _fail(path, "must match the canonical ordered six-joint mapping.")
    for index, (joint_name, _) in enumerate(mapping):
        try:
            joint = model.joint(joint_name)
        except KeyError as error:
            raise PolicyComparisonProjectionError(
                f"{path}[{index}].joint_name references unknown URDF joint {joint_name!r}."
            ) from error
        if joint.joint_type not in {"revolute", "continuous"}:
            _fail(
                f"{path}[{index}].joint_name",
                f"must reference a revolute or continuous joint; {joint_name!r} is {joint.joint_type!r}.",
            )


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string.")
    return value


def _validate_robot_provenance(
    model: RobotModel,
    provenance: RobotProjectionProvenance,
    target_link: str,
    left_transform: RigidTransform,
    right_transform: RigidTransform,
) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(model, RobotModel):
        _fail("robot_model", "must be RobotModel.")
    if not isinstance(provenance, RobotProjectionProvenance):
        _fail("robot_provenance", "must be RobotProjectionProvenance.")
    if provenance.robot_model_name != model.name:
        _fail("robot_provenance.robot_model_name", "must match robot_model.name.")
    if provenance.root_link != model.root_link:
        _fail("robot_provenance.root_link", "must match robot_model.root_link.")
    if not isinstance(target_link, str) or not target_link:
        _fail("target_link", "must be a non-empty string.")
    if target_link not in model.links:
        _fail("target_link", f"references unknown robot link {target_link!r}.")
    if provenance.target_link != target_link:
        _fail("robot_provenance.target_link", "must match target_link.")
    if not re.fullmatch(r"[0-9a-f]{64}", provenance.urdf_sha256):
        _fail(
            "robot_provenance.urdf_sha256",
            "must be a lowercase 64-character SHA-256.",
        )
    if provenance.upstream_revision is not None:
        _nonempty(provenance.upstream_revision, "robot_provenance.upstream_revision")
    for field in (
        "fk_implementation_id",
        "fk_implementation_version",
        "length_unit",
        "angle_unit",
        "handedness",
        "output_coordinate_frame",
        "gripper_semantic_disclaimer",
    ):
        _nonempty(getattr(provenance, field), f"robot_provenance.{field}")
    if provenance.rotation_representation != "unit-quaternion":
        _fail(
            "robot_provenance.rotation_representation",
            "must be 'unit-quaternion'.",
        )
    if provenance.rotation_component_order != ("x", "y", "z", "w"):
        _fail(
            "robot_provenance.rotation_component_order",
            "must be ('x', 'y', 'z', 'w').",
        )
    if provenance.calibrated_gripper_geometry is not False:
        _fail("robot_provenance.calibrated_gripper_geometry", "must be false.")
    if not isinstance(provenance.calibrated_arm_transforms, bool):
        _fail("robot_provenance.calibrated_arm_transforms", "must be boolean.")
    if not isinstance(left_transform, RigidTransform):
        _fail("left_arm_transform", "must be RigidTransform.")
    if not isinstance(right_transform, RigidTransform):
        _fail("right_arm_transform", "must be RigidTransform.")
    expected_left_transform = (
        tuple(left_transform.translation_xyz),
        tuple(left_transform.rotation_rpy),
    )
    expected_right_transform = (
        tuple(right_transform.translation_xyz),
        tuple(right_transform.rotation_rpy),
    )
    if (
        provenance.left_transform_translation_xyz,
        provenance.left_transform_rotation_rpy,
    ) != expected_left_transform:
        _fail("robot_provenance.left_transform", "must match left_arm_transform.")
    if (
        provenance.right_transform_translation_xyz,
        provenance.right_transform_rotation_rpy,
    ) != expected_right_transform:
        _fail("robot_provenance.right_transform", "must match right_arm_transform.")
    left_mapping = build_trlc_dk1_joint_component_map("left")
    right_mapping = build_trlc_dk1_joint_component_map("right")
    expected_left = tuple(left_mapping.items())
    expected_right = tuple(right_mapping.items())
    _validate_mapping(
        model,
        provenance.left_joint_mapping,
        expected_left,
        "robot_provenance.left_joint_mapping",
    )
    _validate_mapping(
        model,
        provenance.right_joint_mapping,
        expected_right,
        "robot_provenance.right_joint_mapping",
    )
    return left_mapping, right_mapping


def _validate_rollout(
    rollout: object,
    *,
    policy_id: str,
    initial_state: tuple[float, ...],
    fps: float,
) -> ActionRolloutResult:
    path = f"{policy_id}.rollout"
    if not isinstance(rollout, ActionRolloutResult):
        _fail(path, "must return ActionRolloutResult.")
    _validate_interpretation(rollout.interpretation, f"{path}.interpretation")
    _validate_component_names(rollout.component_names, f"{path}.component_names")
    target_states = _action_chunk(rollout.target_states, f"{path}.target_states")
    if _positive_fps(rollout.fps, f"{path}.fps") != fps:
        _fail(f"{path}.fps", "must match observation.dataset.fps.")
    normalized_initial = _state_row(rollout.initial_state, f"{path}.initial_state")
    if normalized_initial != initial_state:
        _fail(f"{path}.initial_state", "must match observation.state.values.")
    return ActionRolloutResult(
        interpretation=rollout.interpretation,
        target_states=target_states,
        component_names=COMPONENT_NAMES,
        fps=fps,
        initial_state=initial_state,
    )


def _limit_violations(
    model: RobotModel,
    mapping: dict[str, str],
    target_states: tuple[tuple[float, ...], ...],
    policy_id: str,
) -> tuple[JointLimitViolation, ...]:
    indices = {name: index for index, name in enumerate(COMPONENT_NAMES)}
    violations: list[JointLimitViolation] = []
    for step_index, row in enumerate(target_states):
        for joint_name, component_name in mapping.items():
            joint = model.joint(joint_name)
            value = row[indices[component_name]]
            if joint.lower_limit is not None and value < joint.lower_limit:
                violations.append(
                    JointLimitViolation(
                        policy_id=policy_id,
                        step_index=step_index,
                        component_name=component_name,
                        urdf_joint_name=joint_name,
                        value=value,
                        bound=joint.lower_limit,
                        violation_kind="lower",
                    )
                )
            if joint.upper_limit is not None and value > joint.upper_limit:
                violations.append(
                    JointLimitViolation(
                        policy_id=policy_id,
                        step_index=step_index,
                        component_name=component_name,
                        urdf_joint_name=joint_name,
                        value=value,
                        bound=joint.upper_limit,
                        violation_kind="upper",
                    )
                )
    return tuple(violations)


def _reject_first_violation(violation: JointLimitViolation, model: RobotModel) -> None:
    joint = model.joint(violation.urdf_joint_name)
    _fail(
        f"{violation.policy_id}.target_states[{violation.step_index}].{violation.component_name}",
        f"value {violation.value} violates URDF joint {violation.urdf_joint_name!r} "
        f"allowed range [{joint.lower_limit}, {joint.upper_limit}].",
    )


def _tensor_rows(
    tensor: torch.Tensor, width: int, path: str
) -> tuple[tuple[float, ...], ...]:
    if tuple(tensor.shape) != (CHUNK_LENGTH, width):
        _fail(
            path,
            f"must have shape [{CHUNK_LENGTH}, {width}], received {list(tensor.shape)}.",
        )
    values = tensor.detach().to(device="cpu", dtype=torch.float64).clone()
    if not torch.isfinite(values).all().item():
        _fail(path, "must contain only finite values.")
    return tuple(tuple(float(item) for item in row) for row in values.tolist())


def _project_arm(
    states: tuple[tuple[float, ...], ...],
    *,
    policy_id: str,
    model: RobotModel,
    mapping: dict[str, str],
    arm: str,
    target_link: str,
    transform: RigidTransform,
) -> ProjectedArmTrajectory:
    state_tensor = torch.tensor(states, dtype=torch.float64)
    try:
        local = compute_tool_trajectory(
            state_tensor,
            COMPONENT_NAMES,
            model,
            mapping,
            arm=arm,
            link_name=target_link,
            gripper_component_name=build_trlc_dk1_gripper_component_name(arm),
        )
        world = transform_tool_trajectory(local, transform)
        orientations = rotation_matrices_to_quaternions_xyzw(world.rotation_matrices)
    except Exception as error:
        raise PolicyComparisonProjectionError(
            f"{policy_id}.{arm}.fk failed: {error}"
        ) from error
    positions_xyz = _tensor_rows(world.positions, 3, f"{policy_id}.{arm}.positions_xyz")
    orientations_xyzw = _tensor_rows(
        orientations, 4, f"{policy_id}.{arm}.orientations_xyzw"
    )
    for index, quaternion in enumerate(orientations_xyzw):
        norm = sum(component * component for component in quaternion) ** 0.5
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            _fail(
                f"{policy_id}.{arm}.orientations_xyzw[{index}]",
                "must be a unit quaternion.",
            )
    if world.recorded_gripper_values is None:
        _fail(
            f"{policy_id}.{arm}.generated_raw_gripper_targets",
            "were not preserved by FK.",
        )
    gripper = world.recorded_gripper_values.detach().to(
        device="cpu", dtype=torch.float64
    )
    if tuple(gripper.shape) != (CHUNK_LENGTH,):
        _fail(
            f"{policy_id}.{arm}.generated_raw_gripper_targets",
            f"must have shape [{CHUNK_LENGTH}], received {list(gripper.shape)}.",
        )
    if not torch.isfinite(gripper).all().item():
        _fail(f"{policy_id}.{arm}.generated_raw_gripper_targets", "must be finite.")
    return ProjectedArmTrajectory(
        arm_id=arm,
        target_link=target_link,
        positions_xyz=positions_xyz,  # type: ignore[arg-type]
        orientations_xyzw=orientations_xyzw,  # type: ignore[arg-type]
        generated_raw_gripper_targets=tuple(float(value) for value in gripper.tolist()),
    )


def project_policy_comparison_trajectories(
    observation: PolicyComparisonObservation,
    inference_result: PolicyComparisonInferenceResult,
    *,
    rollout_adapter: ActionRolloutAdapter,
    robot_model: RobotModel,
    robot_provenance: RobotProjectionProvenance,
    left_arm_transform: RigidTransform,
    right_arm_transform: RigidTransform,
    joint_limit_policy: str,
    target_link: str = "tool0",
) -> PolicyComparisonTrajectoryResult:
    """Project exactly two generated policy plans using shared verified geometry."""
    fps, plans = _validate_binding(observation, inference_result)
    if not callable(getattr(rollout_adapter, "rollout", None)):
        _fail("rollout_adapter", "must expose callable rollout.")
    if (
        not isinstance(joint_limit_policy, str)
        or joint_limit_policy not in JOINT_LIMIT_POLICIES
    ):
        _fail(
            "joint_limit_policy",
            f"must be one of {sorted(JOINT_LIMIT_POLICIES)!r}; received {joint_limit_policy!r}.",
        )
    left_mapping, right_mapping = _validate_robot_provenance(
        robot_model,
        robot_provenance,
        target_link,
        left_arm_transform,
        right_arm_transform,
    )
    initial_state = _state_row(observation.state.values, "observation.state.values")
    prepared: list[
        tuple[InferredPolicyPlan, ActionRolloutResult, tuple[JointLimitViolation, ...]]
    ] = []
    for plan in plans:
        independent_actions = _action_chunk(plan.actions, f"{plan.policy_id}.actions")
        try:
            rollout_value = rollout_adapter.rollout(
                initial_state=tuple(initial_state),
                actions=tuple(tuple(row) for row in independent_actions),
                component_names=tuple(COMPONENT_NAMES),
                fps=fps,
            )
        except PolicyComparisonProjectionError as error:
            raise PolicyComparisonProjectionError(
                f"{plan.policy_id}.rollout failed: {error}"
            ) from error
        except Exception as error:
            raise PolicyComparisonProjectionError(
                f"{plan.policy_id}.rollout failed: {error}"
            ) from error
        rollout = _validate_rollout(
            rollout_value,
            policy_id=plan.policy_id,
            initial_state=initial_state,
            fps=fps,
        )
        violations = (
            *_limit_violations(
                robot_model,
                left_mapping,
                rollout.target_states,
                plan.policy_id,
            ),
            *_limit_violations(
                robot_model,
                right_mapping,
                rollout.target_states,
                plan.policy_id,
            ),
        )
        violations = tuple(
            sorted(
                violations,
                key=lambda item: (
                    item.step_index,
                    COMPONENT_NAMES.index(item.component_name),
                ),
            )
        )
        if joint_limit_policy == "reject" and violations:
            _reject_first_violation(violations[0], robot_model)
        prepared.append((plan, rollout, violations))

    projected: list[ProjectedPolicyTrajectory] = []
    for plan, rollout, violations in prepared:
        left = _project_arm(
            rollout.target_states,
            policy_id=plan.policy_id,
            model=robot_model,
            mapping=left_mapping,
            arm="left",
            target_link=target_link,
            transform=left_arm_transform,
        )
        right = _project_arm(
            rollout.target_states,
            policy_id=plan.policy_id,
            model=robot_model,
            mapping=right_mapping,
            arm="right",
            target_link=target_link,
            transform=right_arm_transform,
        )
        projected.append(
            ProjectedPolicyTrajectory(
                policy_id=plan.policy_id,
                label=plan.label,
                relative_times_seconds=tuple(plan.relative_times_seconds),
                left=left,
                right=right,
                joint_limit_violations=violations,
                action_interpretation=rollout.interpretation,
            )
        )
    return PolicyComparisonTrajectoryResult(
        observation_id=observation.observation_id,
        action_dimension=ACTION_DIMENSION,
        chunk_length=CHUNK_LENGTH,
        projection=AvailablePolicyComparisonProjection(
            available=True,
            policies=(projected[0], projected[1]),
            robot=robot_provenance,
        ),
        shared_projection_configuration=True,
    )


def unavailable_policy_comparison_trajectory_result(
    observation: PolicyComparisonObservation,
    inference_result: PolicyComparisonInferenceResult,
    *,
    reason: str,
) -> PolicyComparisonTrajectoryResult:
    """Return an explicitly unavailable projection without fabricated geometry."""
    _validate_binding(observation, inference_result)
    normalized_reason = _nonempty(reason, "reason").strip()
    return PolicyComparisonTrajectoryResult(
        observation_id=observation.observation_id,
        action_dimension=ACTION_DIMENSION,
        chunk_length=CHUNK_LENGTH,
        projection=UnavailablePolicyComparisonProjection(
            available=False,
            reason=normalized_reason,
        ),
        shared_projection_configuration=False,
    )
