from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, sin

import torch
from torch import Tensor

from lerobot_state_atlas.trajectory import ToolTrajectory


Vector3 = tuple[float, float, float]


def _validated_vector3(
    values: Sequence[float],
    *,
    name: str,
) -> Vector3:
    normalized = tuple(float(value) for value in values)

    if len(normalized) != 3 or not all(isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain three finite values.")

    return (
        normalized[0],
        normalized[1],
        normalized[2],
    )


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform from a local coordinate frame into a world frame."""

    translation_xyz: Sequence[float] = (0.0, 0.0, 0.0)
    rotation_rpy: Sequence[float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_xyz",
            _validated_vector3(
                self.translation_xyz,
                name="Translation",
            ),
        )
        object.__setattr__(
            self,
            "rotation_rpy",
            _validated_vector3(
                self.rotation_rpy,
                name="Rotation",
            ),
        )

    def rotation_matrix(self) -> Tensor:
        """Return Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
        roll, pitch, yaw = self.rotation_rpy

        rotation_x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos(roll), -sin(roll)],
                [0.0, sin(roll), cos(roll)],
            ],
            dtype=torch.float64,
        )
        rotation_y = torch.tensor(
            [
                [cos(pitch), 0.0, sin(pitch)],
                [0.0, 1.0, 0.0],
                [-sin(pitch), 0.0, cos(pitch)],
            ],
            dtype=torch.float64,
        )
        rotation_z = torch.tensor(
            [
                [cos(yaw), -sin(yaw), 0.0],
                [sin(yaw), cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        return rotation_z @ rotation_y @ rotation_x


def transform_tool_trajectory(
    trajectory: ToolTrajectory,
    transform: RigidTransform,
) -> ToolTrajectory:
    """Transform a tool trajectory into another coordinate frame."""
    positions = trajectory.positions

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Trajectory positions must have shape (num_points, 3).")

    values = positions.detach().to(
        device="cpu",
        dtype=torch.float64,
    )

    if not torch.isfinite(values).all().item():
        raise ValueError("Trajectory positions must contain only finite values.")

    rotation = transform.rotation_matrix()
    translation = torch.tensor(
        transform.translation_xyz,
        dtype=torch.float64,
    )
    transformed_positions = values @ rotation.T + translation
    local_rotations = trajectory.rotation_matrices.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    transformed_rotations = rotation @ local_rotations

    episode_indices = (
        None
        if trajectory.episode_indices is None
        else trajectory.episode_indices.detach().to(device="cpu").clone()
    )

    return ToolTrajectory(
        arm=trajectory.arm,
        link_name=trajectory.link_name,
        positions=transformed_positions,
        rotation_matrices=transformed_rotations,
        episode_indices=episode_indices,
    )
