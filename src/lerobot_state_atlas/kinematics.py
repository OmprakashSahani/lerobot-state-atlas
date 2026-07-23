from collections import defaultdict, deque
from collections.abc import Mapping
import math

import torch
from torch import Tensor

from lerobot_state_atlas.urdf import JointDefinition, RobotModel, Vector3


def _identity_transform(*, dtype: torch.dtype) -> Tensor:
    return torch.eye(4, dtype=dtype)


def _rpy_rotation(
    rpy: Vector3,
    *,
    dtype: torch.dtype,
) -> Tensor:
    roll, pitch, yaw = rpy

    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    rotation_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ],
        dtype=dtype,
    )
    rotation_y = torch.tensor(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=dtype,
    )
    rotation_z = torch.tensor(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )

    return rotation_z @ rotation_y @ rotation_x


def _origin_transform(
    joint: JointDefinition,
    *,
    dtype: torch.dtype,
) -> Tensor:
    transform = _identity_transform(dtype=dtype)
    transform[:3, :3] = _rpy_rotation(
        joint.origin_rpy,
        dtype=dtype,
    )
    transform[:3, 3] = torch.tensor(
        joint.origin_xyz,
        dtype=dtype,
    )
    return transform


def _normalized_axis(
    joint: JointDefinition,
    *,
    dtype: torch.dtype,
) -> Tensor:
    if joint.axis is None:
        raise ValueError(f"Joint {joint.name} requires an axis.")

    axis = torch.tensor(joint.axis, dtype=dtype)
    magnitude = torch.linalg.vector_norm(axis)

    if magnitude.item() == 0.0:
        raise ValueError(f"Joint {joint.name} has a zero-length axis.")

    return axis / magnitude


def _revolute_transform(
    joint: JointDefinition,
    position: float,
    *,
    dtype: torch.dtype,
) -> Tensor:
    axis = _normalized_axis(joint, dtype=dtype)
    x, y, z = axis.unbind()

    cosine = math.cos(position)
    sine = math.sin(position)
    one_minus_cosine = 1.0 - cosine

    rotation = torch.stack(
        (
            torch.stack(
                (
                    cosine + x * x * one_minus_cosine,
                    x * y * one_minus_cosine - z * sine,
                    x * z * one_minus_cosine + y * sine,
                )
            ),
            torch.stack(
                (
                    y * x * one_minus_cosine + z * sine,
                    cosine + y * y * one_minus_cosine,
                    y * z * one_minus_cosine - x * sine,
                )
            ),
            torch.stack(
                (
                    z * x * one_minus_cosine - y * sine,
                    z * y * one_minus_cosine + x * sine,
                    cosine + z * z * one_minus_cosine,
                )
            ),
        )
    )

    transform = _identity_transform(dtype=dtype)
    transform[:3, :3] = rotation
    return transform


def _prismatic_transform(
    joint: JointDefinition,
    position: float,
    *,
    dtype: torch.dtype,
) -> Tensor:
    axis = _normalized_axis(joint, dtype=dtype)

    transform = _identity_transform(dtype=dtype)
    transform[:3, 3] = axis * position
    return transform


def _joint_motion_transform(
    joint: JointDefinition,
    position: float,
    *,
    dtype: torch.dtype,
) -> Tensor:
    if joint.joint_type in {"revolute", "continuous"}:
        return _revolute_transform(
            joint,
            position,
            dtype=dtype,
        )

    if joint.joint_type == "prismatic":
        return _prismatic_transform(
            joint,
            position,
            dtype=dtype,
        )

    if joint.joint_type == "fixed":
        return _identity_transform(dtype=dtype)

    raise ValueError(f"Unsupported joint type for {joint.name}: {joint.joint_type}")


def compute_link_transforms(
    model: RobotModel,
    joint_positions: Mapping[str, float] | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Compute root-relative transforms for every link."""
    positions = dict(joint_positions or {})
    known_joint_names = {joint.name for joint in model.joints}
    unknown_joint_names = set(positions) - known_joint_names

    if unknown_joint_names:
        names = ", ".join(sorted(unknown_joint_names))
        raise KeyError(f"Unknown joint positions: {names}")

    joints_by_parent: dict[str, list[JointDefinition]] = defaultdict(list)

    for joint in model.joints:
        joints_by_parent[joint.parent_link].append(joint)

    transforms = {model.root_link: _identity_transform(dtype=dtype)}
    pending_links = deque([model.root_link])

    while pending_links:
        parent_link = pending_links.popleft()
        parent_transform = transforms[parent_link]

        for joint in joints_by_parent[parent_link]:
            if joint.child_link in transforms:
                raise ValueError(
                    f"Link {joint.child_link} has multiple kinematic paths."
                )

            position = float(positions.get(joint.name, 0.0))
            child_transform = (
                parent_transform
                @ _origin_transform(joint, dtype=dtype)
                @ _joint_motion_transform(
                    joint,
                    position,
                    dtype=dtype,
                )
            )

            transforms[joint.child_link] = child_transform
            pending_links.append(joint.child_link)

    missing_links = set(model.links) - set(transforms)

    if missing_links:
        names = ", ".join(sorted(missing_links))
        raise ValueError(f"Robot model contains unreachable links: {names}")

    return transforms


def get_link_position(
    link_transforms: Mapping[str, Tensor],
    link_name: str,
) -> Tensor:
    """Return the three-dimensional position of a link."""
    try:
        transform = link_transforms[link_name]
    except KeyError as error:
        raise KeyError(f"Unknown link: {link_name}") from error

    if transform.shape != (4, 4):
        raise ValueError(f"Transform for {link_name} must have shape (4, 4).")

    return transform[:3, 3].clone()
