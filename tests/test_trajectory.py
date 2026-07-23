import math

import pytest
import torch

from lerobot_state_atlas.trajectory import (
    build_trlc_dk1_joint_component_map,
    compute_tool_trajectory,
)
from lerobot_state_atlas.urdf import JointDefinition, RobotModel


def make_model() -> RobotModel:
    return RobotModel(
        name="trajectory_robot",
        root_link="base_link",
        links=("base_link", "arm_link", "tool0"),
        joints=(
            JointDefinition(
                name="joint1",
                joint_type="revolute",
                parent_link="base_link",
                child_link="arm_link",
                origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
                axis=(0.0, 0.0, 1.0),
                lower_limit=None,
                upper_limit=None,
            ),
            JointDefinition(
                name="tool_joint",
                joint_type="fixed",
                parent_link="arm_link",
                child_link="tool0",
                origin_xyz=(1.0, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
                axis=None,
                lower_limit=None,
                upper_limit=None,
            ),
        ),
        mesh_paths=(),
    )


def test_build_trlc_dk1_joint_component_map() -> None:
    mapping = build_trlc_dk1_joint_component_map("left")

    assert mapping == {
        "joint1": "left_joint_1.pos",
        "joint2": "left_joint_2.pos",
        "joint3": "left_joint_3.pos",
        "joint4": "left_joint_4.pos",
        "joint5": "left_joint_5.pos",
        "joint6": "left_joint_6.pos",
    }


def test_build_trlc_dk1_joint_component_map_rejects_arm() -> None:
    with pytest.raises(
        ValueError,
        match="either 'left' or 'right'",
    ):
        build_trlc_dk1_joint_component_map("center")


def test_compute_tool_trajectory() -> None:
    states = torch.tensor(
        [
            [0.0, 10.0],
            [math.pi / 2.0, 20.0],
        ],
        dtype=torch.float32,
    )

    trajectory = compute_tool_trajectory(
        states,
        component_names=(
            "left_joint_1.pos",
            "unused_component",
        ),
        model=make_model(),
        joint_component_map={
            "joint1": "left_joint_1.pos",
        },
        arm="left",
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert trajectory.arm == "left"
    assert trajectory.link_name == "tool0"
    assert trajectory.num_frames == 2

    torch.testing.assert_close(
        trajectory.positions,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_compute_tool_trajectory_rejects_non_matrix() -> None:
    with pytest.raises(
        ValueError,
        match="States must have shape",
    ):
        compute_tool_trajectory(
            torch.tensor([0.0]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
        )


def test_compute_tool_trajectory_rejects_empty_states() -> None:
    with pytest.raises(
        ValueError,
        match="at least one frame",
    ):
        compute_tool_trajectory(
            torch.empty((0, 1)),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
        )


def test_compute_tool_trajectory_rejects_wrong_name_count() -> None:
    with pytest.raises(
        ValueError,
        match="component names",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0, 1.0]]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
        )


def test_compute_tool_trajectory_rejects_duplicate_names() -> None:
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0, 1.0]]),
            component_names=(
                "left_joint_1.pos",
                "left_joint_1.pos",
            ),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
        )


def test_compute_tool_trajectory_rejects_unknown_mapped_joint() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown mapped joints",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0]]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "missing_joint": "left_joint_1.pos",
            },
            arm="left",
        )


def test_compute_tool_trajectory_requires_joint_mapping() -> None:
    with pytest.raises(
        ValueError,
        match="Missing state-component mappings",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0]]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={},
            arm="left",
        )


def test_compute_tool_trajectory_requires_state_component() -> None:
    with pytest.raises(
        ValueError,
        match="Missing state components",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0]]),
            component_names=("other_component",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
        )


def test_compute_tool_trajectory_rejects_unknown_target_link() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown target link",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0]]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
            link_name="missing_link",
        )


def test_compute_tool_trajectory_matches_across_chunks() -> None:
    states = torch.tensor(
        [
            [0.0],
            [math.pi / 4.0],
            [math.pi / 2.0],
            [math.pi],
            [-math.pi / 2.0],
        ],
        dtype=torch.float32,
    )

    unchunked = compute_tool_trajectory(
        states,
        component_names=("left_joint_1.pos",),
        model=make_model(),
        joint_component_map={
            "joint1": "left_joint_1.pos",
        },
        arm="left",
        chunk_size=100,
    )

    chunked = compute_tool_trajectory(
        states,
        component_names=("left_joint_1.pos",),
        model=make_model(),
        joint_component_map={
            "joint1": "left_joint_1.pos",
        },
        arm="left",
        chunk_size=2,
    )

    torch.testing.assert_close(
        chunked.positions,
        unchunked.positions,
    )


def test_compute_tool_trajectory_rejects_invalid_chunk_size() -> None:
    with pytest.raises(
        ValueError,
        match="Chunk size must be greater than zero",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0]]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
            chunk_size=0,
        )


def test_compute_tool_trajectory_preserves_episode_indices() -> None:
    states = torch.tensor(
        [
            [0.0],
            [math.pi / 2.0],
            [math.pi],
        ],
        dtype=torch.float32,
    )
    episode_indices = torch.tensor(
        [7, 7, 3],
        dtype=torch.int64,
    )

    trajectory = compute_tool_trajectory(
        states,
        component_names=("left_joint_1.pos",),
        model=make_model(),
        joint_component_map={
            "joint1": "left_joint_1.pos",
        },
        arm="left",
        episode_indices=episode_indices,
    )

    assert trajectory.num_frames == 3
    assert trajectory.num_episodes == 2
    assert torch.equal(
        trajectory.episode_indices,
        episode_indices,
    )


def test_compute_tool_trajectory_rejects_non_vector_episode_indices() -> None:
    with pytest.raises(
        ValueError,
        match="Episode indices must be one-dimensional",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0], [1.0]]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
            episode_indices=torch.tensor([[1], [1]]),
        )


def test_compute_tool_trajectory_rejects_episode_index_count() -> None:
    with pytest.raises(
        ValueError,
        match="must match the number of state frames",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0], [1.0]]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
            episode_indices=torch.tensor([1]),
        )


@pytest.mark.parametrize(
    "episode_indices",
    (
        torch.tensor([1.0, 1.0]),
        torch.tensor([True, False]),
        torch.tensor([1.0 + 0.0j, 2.0 + 0.0j]),
    ),
)
def test_compute_tool_trajectory_rejects_noninteger_episode_indices(
    episode_indices: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match="must use an integer dtype",
    ):
        compute_tool_trajectory(
            torch.tensor([[0.0], [1.0]]),
            component_names=("left_joint_1.pos",),
            model=make_model(),
            joint_component_map={
                "joint1": "left_joint_1.pos",
            },
            arm="left",
            episode_indices=episode_indices,
        )
