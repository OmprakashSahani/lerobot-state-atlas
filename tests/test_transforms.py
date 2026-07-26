import math

import pytest
import torch

from lerobot_state_atlas.trajectory import ToolTrajectory
from lerobot_state_atlas.transforms import (
    RigidTransform,
    transform_tool_trajectory,
)


def make_trajectory(positions: torch.Tensor) -> ToolTrajectory:
    return ToolTrajectory(
        arm="left",
        link_name="tool0",
        positions=positions,
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
