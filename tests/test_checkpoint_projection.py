from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path

import pytest
import torch

import lerobot_state_atlas.checkpoint_comparison.projection as projection_module
from lerobot_state_atlas.checkpoint_comparison.models import (
    ActionInterpretationProvenance,
    DeterministicNoiseProvenance,
    InferredPolicyPlan,
    ObservationDatasetIdentity,
    ObservationRecordedGroundTruth,
    ObservationState,
    PolicyComparisonInferenceResult,
    PolicyComparisonObservation,
    RobotProjectionProvenance,
    UnavailablePolicyComparisonProjection,
)
from lerobot_state_atlas.checkpoint_comparison.observation import COMPONENT_NAMES
from lerobot_state_atlas.checkpoint_comparison.projection import (
    ABSOLUTE_INTERPRETATION_ID,
    ABSOLUTE_INTERPRETATION_VERSION,
    ABSOLUTE_TRANSFORMATIONS,
    PI05AbsolutePositionTargetsAdapter,
    PolicyComparisonProjectionError,
    project_policy_comparison_trajectories,
    unavailable_policy_comparison_trajectory_result,
)
from lerobot_state_atlas.trajectory import ToolTrajectory
from lerobot_state_atlas.transforms import RigidTransform
from lerobot_state_atlas.urdf import JointDefinition, RobotModel


def interpretation() -> ActionInterpretationProvenance:
    return ActionInterpretationProvenance(
        interpretation_id=ABSOLUTE_INTERPRETATION_ID,
        interpretation_version=ABSOLUTE_INTERPRETATION_VERSION,
        use_relative_actions=False,
        delta_actions_preprocessor_enabled=False,
        absolute_actions_postprocessor_enabled=False,
        component_names=COMPONENT_NAMES,
        initial_state_participates=False,
        transformations=ABSOLUTE_TRANSFORMATIONS,
    )


def adapter() -> PI05AbsolutePositionTargetsAdapter:
    return PI05AbsolutePositionTargetsAdapter(interpretation())


def observation(*, fps: float = 50.0) -> PolicyComparisonObservation:
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
            fps=fps,
            task="Actuator Unboxing",
        ),
        prompt="Unbox the actuator.",
        state=ObservationState(
            feature_name="observation.state",
            dtype="float32",
            component_names=COMPONENT_NAMES,
            values=tuple(float(index + 1) / 100 for index in range(14)),
        ),
        cameras=(),
        recorded_ground_truth=ObservationRecordedGroundTruth(
            available=False, reason="Synthetic projection test."
        ),
    )


def action_rows(base: float = 0.0) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(base + step / 1000 + component / 100 for component in range(14))
        for step in range(50)
    )


def inference(*, fps: float = 50.0) -> PolicyComparisonInferenceResult:
    times = tuple(index / fps for index in range(50))
    return PolicyComparisonInferenceResult(
        observation_id="observation-123",
        policies=(
            InferredPolicyPlan(
                policy_id="base-pi05",
                label="Base π0.5",
                relative_times_seconds=times,
                actions=action_rows(0.0),
            ),
            InferredPolicyPlan(
                policy_id="fine-tuned-pi05",
                label="Fine-tuned π0.5",
                relative_times_seconds=times,
                actions=action_rows(0.1),
            ),
        ),
        noise=DeterministicNoiseProvenance(
            seed=1,
            generator="synthetic",
            construction_device="cpu",
            dtype="float32",
            shape=(1, 50, 32),
            sha256="b" * 64,
        ),
        action_dimension=14,
        chunk_length=50,
        num_inference_steps=10,
        shared_preprocessing=True,
        shared_postprocessing=True,
    )


def robot(*, lower: float = -10.0, upper: float = 10.0) -> RobotModel:
    links = ["base_link"]
    joints = []
    parent = "base_link"
    for index in range(1, 7):
        child = f"link{index}"
        links.append(child)
        joints.append(
            JointDefinition(
                name=f"joint{index}",
                joint_type="revolute",
                parent_link=parent,
                child_link=child,
                origin_xyz=(0.1, 0.0, 0.01 * index),
                origin_rpy=(0.0, 0.0, 0.0),
                axis=(0.0, 0.0, 1.0),
                lower_limit=lower,
                upper_limit=upper,
            )
        )
        parent = child
    links.append("tool0")
    joints.append(
        JointDefinition(
            name="tool_joint",
            joint_type="fixed",
            parent_link=parent,
            child_link="tool0",
            origin_xyz=(0.25, 0.0, 0.0),
            origin_rpy=(math.pi / 2, 0.0, 0.0),
            axis=None,
            lower_limit=None,
            upper_limit=None,
        )
    )
    return RobotModel(
        name="synthetic-trlc",
        root_link="base_link",
        links=tuple(links),
        joints=tuple(joints),
        mesh_paths=(),
    )


LEFT_TRANSFORM = RigidTransform(
    translation_xyz=(0.0, 0.4, 0.0), rotation_rpy=(0.0, 0.0, 0.0)
)
RIGHT_TRANSFORM = RigidTransform(
    translation_xyz=(0.0, -0.3, 0.1), rotation_rpy=(0.0, 0.0, 0.2)
)


def provenance(model: RobotModel | None = None) -> RobotProjectionProvenance:
    model = model or robot()
    return RobotProjectionProvenance(
        robot_model_name=model.name,
        root_link=model.root_link,
        target_link="tool0",
        urdf_sha256="c" * 64,
        upstream_revision="synthetic-v1",
        fk_implementation_id="lerobot-state-atlas.compute-tool-trajectory",
        fk_implementation_version="1",
        left_joint_mapping=tuple(
            (f"joint{index}", f"left_joint_{index}.pos") for index in range(1, 7)
        ),
        right_joint_mapping=tuple(
            (f"joint{index}", f"right_joint_{index}.pos") for index in range(1, 7)
        ),
        left_transform_translation_xyz=tuple(LEFT_TRANSFORM.translation_xyz),
        left_transform_rotation_rpy=tuple(LEFT_TRANSFORM.rotation_rpy),
        right_transform_translation_xyz=tuple(RIGHT_TRANSFORM.translation_xyz),
        right_transform_rotation_rpy=tuple(RIGHT_TRANSFORM.rotation_rpy),
        length_unit="metre",
        angle_unit="radian",
        handedness="right-handed",
        output_coordinate_frame="explicit-synthetic-shared-world",
        rotation_representation="unit-quaternion",
        rotation_component_order=("x", "y", "z", "w"),
        gripper_semantic_disclaimer=(
            "Generated raw device-specific targets; no calibrated jaw geometry."
        ),
        calibrated_gripper_geometry=False,
        calibrated_arm_transforms=False,
    )


def project(**overrides):
    model = overrides.pop("robot_model", robot())
    kwargs = {
        "observation": observation(),
        "inference_result": inference(),
        "rollout_adapter": adapter(),
        "robot_model": model,
        "robot_provenance": provenance(model),
        "left_arm_transform": LEFT_TRANSFORM,
        "right_arm_transform": RIGHT_TRANSFORM,
        "joint_limit_policy": "reject",
    }
    kwargs.update(overrides)
    return project_policy_comparison_trajectories(**kwargs)


def test_absolute_adapter_preserves_actions_without_using_initial_state() -> None:
    actions = action_rows()
    initial = tuple(100.0 + index for index in range(14))
    result = adapter().rollout(
        initial_state=initial,
        actions=actions,
        component_names=COMPONENT_NAMES,
        fps=50.0,
    )
    assert result.target_states == actions
    assert result.target_states[1][0] == actions[1][0]
    assert result.target_states[1][0] != initial[0] + actions[1][0]
    assert result.interpretation.initial_state_participates is False
    assert result.initial_state == initial


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_relative_actions", True),
        ("delta_actions_preprocessor_enabled", True),
        ("absolute_actions_postprocessor_enabled", True),
        ("initial_state_participates", True),
        ("interpretation_id", "unknown"),
        ("interpretation_version", "2.0"),
        ("component_names", tuple(reversed(COMPONENT_NAMES))),
        ("transformations", ("guess",)),
    ],
)
def test_absolute_adapter_rejects_unverified_provenance(
    field: str, value: object
) -> None:
    with pytest.raises(PolicyComparisonProjectionError, match=field):
        PI05AbsolutePositionTargetsAdapter(replace(interpretation(), **{field: value}))


@pytest.mark.parametrize(
    ("initial,actions,path"),
    [
        ((0.0,) * 13, action_rows(), "initial_state"),
        ((0.0,) * 14, action_rows()[:-1], "actions"),
        (
            (0.0,) * 14,
            ((True,) + action_rows()[0][1:],) + action_rows()[1:],
            r"actions\[0\]\[0\]",
        ),
        (
            (0.0,) * 14,
            ((float("nan"),) + action_rows()[0][1:],) + action_rows()[1:],
            r"actions\[0\]\[0\]",
        ),
        (
            (0.0,) * 14,
            ((float("inf"),) + action_rows()[0][1:],) + action_rows()[1:],
            r"actions\[0\]\[0\]",
        ),
        (
            (0.0,) * 14,
            ((-float("inf"),) + action_rows()[0][1:],) + action_rows()[1:],
            r"actions\[0\]\[0\]",
        ),
    ],
)
def test_absolute_adapter_rejects_malformed_values(initial, actions, path: str) -> None:
    with pytest.raises(PolicyComparisonProjectionError, match=path):
        adapter().rollout(
            initial_state=initial,
            actions=actions,
            component_names=COMPONENT_NAMES,
            fps=50.0,
        )


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (
            lambda obs, result: replace(result, observation_id="other"),
            "inference_result.observation_id",
        ),
        (
            lambda obs, result: replace(
                result, policies=tuple(reversed(result.policies))
            ),
            r"policies\[0\]\.policy_id",
        ),
        (
            lambda obs, result: replace(
                result,
                policies=(
                    replace(result.policies[0], label="First checkpoint"),
                    result.policies[1],
                ),
            ),
            r"policies\[0\]\.label",
        ),
        (
            lambda obs, result: replace(result, action_dimension=13),
            "action_dimension",
        ),
        (
            lambda obs, result: replace(result, chunk_length=49),
            "chunk_length",
        ),
        (
            lambda obs, result: replace(
                result,
                policies=(
                    replace(result.policies[0], relative_times_seconds=(0.0,) * 50),
                    result.policies[1],
                ),
            ),
            "relative_times_seconds",
        ),
        (
            lambda obs, result: replace(
                obs,
                state=replace(
                    obs.state, component_names=tuple(reversed(COMPONENT_NAMES))
                ),
            ),
            "component_names",
        ),
    ],
)
def test_projection_rejects_binding_errors_before_fk(
    mutation, path, monkeypatch
) -> None:
    obs = observation()
    result = inference()
    changed = mutation(obs, result)
    if isinstance(changed, PolicyComparisonObservation):
        obs = changed
    else:
        result = changed
    fk_calls = []
    monkeypatch.setattr(
        projection_module,
        "compute_tool_trajectory",
        lambda *args, **kwargs: fk_calls.append((args, kwargs)),
    )
    with pytest.raises(PolicyComparisonProjectionError, match=path):
        project(observation=obs, inference_result=result)
    assert fk_calls == []


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_projection_rejects_invalid_fps(fps: float) -> None:
    with pytest.raises(PolicyComparisonProjectionError, match="dataset.fps"):
        project(observation=observation(fps=fps), inference_result=inference())


def test_projection_outputs_two_deterministic_dual_arm_trajectories() -> None:
    first = project()
    second = project()
    assert first == second
    assert first.projection.available is True
    policies = first.projection.policies  # type: ignore[union-attr]
    assert tuple(item.policy_id for item in policies) == (
        "base-pi05",
        "fine-tuned-pi05",
    )
    assert tuple(item.label for item in policies) == (
        "Base π0.5",
        "Fine-tuned π0.5",
    )
    for policy in policies:
        assert len(policy.relative_times_seconds) == 50
        for arm in (policy.left, policy.right):
            assert len(arm.positions_xyz) == 50
            assert len(arm.orientations_xyzw) == 50
            assert len(arm.generated_raw_gripper_targets) == 50
            assert all(
                math.isclose(
                    math.sqrt(sum(component * component for component in quaternion)),
                    1.0,
                    abs_tol=1e-9,
                )
                for quaternion in arm.orientations_xyzw
            )
    assert policies[0].left.positions_xyz != policies[0].right.positions_xyz
    assert policies[0].left.generated_raw_gripper_targets == tuple(
        row[6] for row in action_rows()
    )
    assert policies[0].right.generated_raw_gripper_targets == tuple(
        row[13] for row in action_rows()
    )
    assert first.projection.robot.calibrated_gripper_geometry is False  # type: ignore[union-attr]
    assert "first checkpoint" not in repr(first).lower()


def test_fixed_tool_transform_is_included() -> None:
    model = robot()
    result = project(robot_model=model, robot_provenance=provenance(model))
    position = result.projection.policies[0].left.positions_xyz[0]  # type: ignore[union-attr]
    assert position[1] != pytest.approx(0.4)


def test_reject_limit_accepts_boundaries_and_rejects_first_violation() -> None:
    model = robot(lower=-1.0, upper=1.0)
    rows = [list(row) for row in action_rows()]
    rows[0][0] = -1.0
    rows[0][1] = 1.0
    good = inference()
    good = replace(
        good,
        policies=(
            replace(good.policies[0], actions=tuple(tuple(row) for row in rows)),
            good.policies[1],
        ),
    )
    project(
        robot_model=model, robot_provenance=provenance(model), inference_result=good
    )

    rows[3][2] = 1.01
    bad = replace(
        good,
        policies=(
            replace(good.policies[0], actions=tuple(tuple(row) for row in rows)),
            good.policies[1],
        ),
    )
    with pytest.raises(
        PolicyComparisonProjectionError,
        match=r"base-pi05.*target_states\[3\].*left_joint_3\.pos.*joint3.*\[-1\.0, 1\.0\]",
    ):
        project(
            robot_model=model, robot_provenance=provenance(model), inference_result=bad
        )


def test_allow_limit_records_all_violations_without_clipping_grippers() -> None:
    model = robot(lower=-1.0, upper=1.0)
    rows = [list(row) for row in action_rows()]
    rows[1][0] = -1.5
    rows[1][7] = 1.5
    rows[2][1] = 2.0
    rows[2][6] = 500.0
    rows[2][13] = -500.0
    result = inference()
    result = replace(
        result,
        policies=(
            replace(result.policies[0], actions=tuple(tuple(row) for row in rows)),
            result.policies[1],
        ),
    )
    projected = project(
        robot_model=model,
        robot_provenance=provenance(model),
        inference_result=result,
        joint_limit_policy="allow-with-recorded-violations",
    )
    policy = projected.projection.policies[0]  # type: ignore[union-attr]
    assert tuple(
        (item.step_index, item.component_name, item.value, item.violation_kind)
        for item in policy.joint_limit_violations
    ) == (
        (1, "left_joint_1.pos", -1.5, "lower"),
        (1, "right_joint_1.pos", 1.5, "upper"),
        (2, "left_joint_2.pos", 2.0, "upper"),
    )
    assert policy.left.generated_raw_gripper_targets[2] == 500.0
    assert policy.right.generated_raw_gripper_targets[2] == -500.0


def test_unknown_limit_policy_is_rejected() -> None:
    with pytest.raises(PolicyComparisonProjectionError, match="joint_limit_policy"):
        project(joint_limit_policy="clip")


@pytest.mark.parametrize(
    ("change", "path"),
    [
        ({"root_link": "other"}, "root_link"),
        ({"target_link": "missing"}, "target_link"),
        ({"urdf_sha256": "ABC"}, "urdf_sha256"),
        ({"left_joint_mapping": ()}, "left_joint_mapping"),
        ({"rotation_representation": "wxyz"}, "rotation_representation"),
        ({"calibrated_gripper_geometry": True}, "calibrated_gripper_geometry"),
        ({"calibrated_arm_transforms": "false"}, "calibrated_arm_transforms"),
    ],
)
def test_robot_provenance_is_cross_checked(change: dict, path: str) -> None:
    with pytest.raises(PolicyComparisonProjectionError, match=path):
        project(robot_provenance=replace(provenance(), **change))


def test_invalid_target_link_and_transform_mismatch_are_rejected() -> None:
    with pytest.raises(PolicyComparisonProjectionError, match="target_link"):
        project(target_link="missing")
    with pytest.raises(PolicyComparisonProjectionError, match="left_transform"):
        project(left_arm_transform=RigidTransform((1.0, 0.0, 0.0)))


def test_nonfinite_fk_output_is_rejected(monkeypatch) -> None:
    real_fk = projection_module.compute_tool_trajectory

    def invalid_fk(*args, **kwargs):
        trajectory = real_fk(*args, **kwargs)
        positions = trajectory.positions.clone()
        positions[0, 0] = float("nan")
        return ToolTrajectory(
            arm=trajectory.arm,
            link_name=trajectory.link_name,
            positions=positions,
            rotation_matrices=trajectory.rotation_matrices,
            recorded_gripper_values=trajectory.recorded_gripper_values,
        )

    monkeypatch.setattr(projection_module, "compute_tool_trajectory", invalid_fk)
    with pytest.raises(PolicyComparisonProjectionError, match="left.*fk.*finite"):
        project()


def test_adapter_cannot_mutate_sources_or_other_policy_inputs() -> None:
    source_observation = observation()
    source_inference = inference()
    original_state = source_observation.state.values
    original_actions = source_inference.policies[0].actions
    seen = []

    class AttemptingMutationAdapter:
        def rollout(self, *, initial_state, actions, component_names, fps):
            seen.append((initial_state, actions))
            with pytest.raises(TypeError):
                actions[0][0] = 999  # type: ignore[index]
            return adapter().rollout(
                initial_state=initial_state,
                actions=actions,
                component_names=component_names,
                fps=fps,
            )

    result = project(
        observation=source_observation,
        inference_result=source_inference,
        rollout_adapter=AttemptingMutationAdapter(),
    )
    assert len(seen) == 2
    assert seen[0][1] is not seen[1][1]
    assert source_observation.state.values == original_state
    assert source_inference.policies[0].actions == original_actions
    with pytest.raises(FrozenInstanceError):
        result.chunk_length = 1
    assert not any(
        isinstance(value, torch.Tensor)
        for policy in result.projection.policies  # type: ignore[union-attr]
        for arm in (policy.left, policy.right)
        for value in (arm.positions_xyz, arm.orientations_xyzw)
    )


def test_unavailable_projection_is_explicit_and_contains_no_geometry() -> None:
    result = unavailable_policy_comparison_trajectory_result(
        observation(), inference(), reason="  Camera calibration is unavailable.  "
    )
    assert result.projection == UnavailablePolicyComparisonProjection(
        available=False, reason="Camera calibration is unavailable."
    )
    assert not hasattr(result.projection, "policies")
    assert not hasattr(result.projection, "robot")
    assert result.shared_projection_configuration is False


@pytest.mark.parametrize("reason", ["", "   "])
def test_unavailable_projection_requires_reason(reason: str) -> None:
    with pytest.raises(PolicyComparisonProjectionError, match="reason"):
        unavailable_policy_comparison_trajectory_result(
            observation(), inference(), reason=reason
        )


def test_unavailable_projection_still_requires_identity_binding() -> None:
    with pytest.raises(PolicyComparisonProjectionError, match="observation_id"):
        unavailable_policy_comparison_trajectory_result(
            observation(), replace(inference(), observation_id="other"), reason="No FK."
        )
