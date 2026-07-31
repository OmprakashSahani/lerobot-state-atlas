import math

import pytest
import torch

from lerobot_state_atlas.trajectory import ToolTrajectory
from lerobot_state_atlas.transforms import (
    RigidTransform,
    transform_tool_trajectory,
)


def make_trajectory(
    positions: torch.Tensor,
    rotation_matrices: torch.Tensor | None = None,
) -> ToolTrajectory:
    return ToolTrajectory(
        arm="left",
        link_name="tool0",
        positions=positions,
        rotation_matrices=(
            torch.eye(3, dtype=torch.float64).expand(positions.shape[0], -1, -1).clone()
            if rotation_matrices is None
            else rotation_matrices
        ),
        episode_indices=torch.tensor(
            [4, 7],
            dtype=torch.int64,
        ),
    )


def test_transform_tool_trajectory_applies_translation_and_rotation() -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
    )
    transform = RigidTransform(
        translation_xyz=(0.5, -0.25, 1.0),
        rotation_rpy=(0.0, 0.0, math.pi / 2.0),
    )

    transformed = transform_tool_trajectory(
        trajectory,
        transform,
    )

    torch.testing.assert_close(
        transformed.positions,
        torch.tensor(
            [
                [0.5, 0.75, 1.0],
                [-0.5, -0.25, 1.0],
            ],
            dtype=torch.float64,
        ),
        atol=1e-12,
        rtol=1e-12,
    )
    assert transformed.arm == "left"
    assert transformed.link_name == "tool0"
    assert transformed.episode_indices is not trajectory.episode_indices
    torch.testing.assert_close(
        transformed.episode_indices,
        trajectory.episode_indices,
    )
    torch.testing.assert_close(
        transformed.rotation_matrices,
        transform.rotation_matrix().expand(2, -1, -1),
        atol=1e-12,
        rtol=1e-12,
    )


def test_identity_rotation_preserves_trajectory_rotations() -> None:
    rotations = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    trajectory = make_trajectory(torch.zeros((2, 3)), rotations)

    transformed = transform_tool_trajectory(trajectory, RigidTransform())

    torch.testing.assert_close(transformed.rotation_matrices, rotations)


def test_translation_only_preserves_trajectory_rotations() -> None:
    rotations = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    trajectory = make_trajectory(torch.zeros((2, 3)), rotations)

    transformed = transform_tool_trajectory(
        trajectory,
        RigidTransform(translation_xyz=(1.0, -2.0, 3.0)),
    )

    torch.testing.assert_close(transformed.rotation_matrices, rotations)


def test_base_rotation_left_composes_noncommuting_tool_rotation() -> None:
    tool_rotation_x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    trajectory = make_trajectory(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        tool_rotation_x.expand(2, -1, -1).clone().to(dtype=torch.float32),
    )
    original_positions = trajectory.positions.clone()
    original_rotations = trajectory.rotation_matrices.clone()
    base_transform = RigidTransform(rotation_rpy=(0.0, 0.0, math.pi / 2.0))
    base_rotation_z = base_transform.rotation_matrix()

    transformed = transform_tool_trajectory(trajectory, base_transform)

    expected = (base_rotation_z @ tool_rotation_x).expand(2, -1, -1)
    wrong_order = (tool_rotation_x @ base_rotation_z).expand(2, -1, -1)
    torch.testing.assert_close(transformed.rotation_matrices, expected)
    assert not torch.allclose(transformed.rotation_matrices, wrong_order)
    assert transformed.positions.shape[0] == transformed.rotation_matrices.shape[0]
    assert transformed.positions.device.type == "cpu"
    assert transformed.rotation_matrices.device.type == "cpu"
    assert transformed.positions.dtype == torch.float64
    assert transformed.rotation_matrices.dtype == torch.float64
    torch.testing.assert_close(trajectory.positions, original_positions)
    torch.testing.assert_close(trajectory.rotation_matrices, original_rotations)


@pytest.mark.parametrize(
    ("translation_xyz", "rotation_rpy"),
    [
        ((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, float("inf"), 0.0)),
        ((0.0, 0.0), (0.0, 0.0, 0.0)),
    ],
)
def test_rigid_transform_rejects_invalid_parameters(
    translation_xyz: tuple[float, ...],
    rotation_rpy: tuple[float, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="three finite values",
    ):
        RigidTransform(
            translation_xyz=translation_xyz,
            rotation_rpy=rotation_rpy,
        )
