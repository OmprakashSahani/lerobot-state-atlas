from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from lerobot_state_atlas.urdf import JointDefinition, RobotModel


_DEFAULT_CHUNK_SIZE = 65_536


@dataclass(frozen=True)
class ToolTrajectory:
    """Three-dimensional tool positions computed from robot states."""

    arm: str
    link_name: str
    positions: Tensor
    episode_indices: Tensor | None = None

    @property
    def num_frames(self) -> int:
        """Return the number of trajectory frames."""
        return int(self.positions.shape[0])

    @property
    def num_episodes(self) -> int:
        """Return the number of represented episodes."""
        if self.episode_indices is None:
            return 1

        return int(torch.unique(self.episode_indices).numel())


def build_trlc_dk1_joint_component_map(
    arm: str,
) -> dict[str, str]:
    """Map TRLC-DK1 URDF joints to LeRobot state components."""
    if arm not in {"left", "right"}:
        raise ValueError("Arm must be either 'left' or 'right'.")

    return {f"joint{index}": f"{arm}_joint_{index}.pos" for index in range(1, 7)}


def _origin_transform(joint: JointDefinition) -> Tensor:
    roll, pitch, yaw = joint.origin_rpy

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
        dtype=torch.float64,
    )
    rotation_y = torch.tensor(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=torch.float64,
    )
    rotation_z = torch.tensor(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = torch.tensor(
        joint.origin_xyz,
        dtype=torch.float64,
    )
    return transform


def _normalized_axis(joint: JointDefinition) -> Tensor:
    if joint.axis is None:
        raise ValueError(f"Joint {joint.name} requires an axis.")

    axis = torch.tensor(
        joint.axis,
        dtype=torch.float64,
    )
    magnitude = torch.linalg.vector_norm(axis)

    if magnitude.item() == 0.0:
        raise ValueError(f"Joint {joint.name} has a zero-length axis.")

    return axis / magnitude


def _batched_revolute_transform(
    joint: JointDefinition,
    positions: Tensor,
) -> Tensor:
    axis = _normalized_axis(joint)
    x, y, z = axis.unbind()

    cosine = torch.cos(positions)
    sine = torch.sin(positions)
    one_minus_cosine = 1.0 - cosine

    transform = torch.zeros(
        (positions.shape[0], 4, 4),
        dtype=torch.float64,
    )

    transform[:, 0, 0] = cosine + x * x * one_minus_cosine
    transform[:, 0, 1] = x * y * one_minus_cosine - z * sine
    transform[:, 0, 2] = x * z * one_minus_cosine + y * sine

    transform[:, 1, 0] = y * x * one_minus_cosine + z * sine
    transform[:, 1, 1] = cosine + y * y * one_minus_cosine
    transform[:, 1, 2] = y * z * one_minus_cosine - x * sine

    transform[:, 2, 0] = z * x * one_minus_cosine - y * sine
    transform[:, 2, 1] = z * y * one_minus_cosine + x * sine
    transform[:, 2, 2] = cosine + z * z * one_minus_cosine

    transform[:, 3, 3] = 1.0
    return transform


def _batched_prismatic_transform(
    joint: JointDefinition,
    positions: Tensor,
) -> Tensor:
    axis = _normalized_axis(joint)

    transform = (
        torch.eye(
            4,
            dtype=torch.float64,
        )
        .expand(positions.shape[0], -1, -1)
        .clone()
    )

    transform[:, :3, 3] = positions.unsqueeze(1) * axis
    return transform


def _batched_joint_motion_transform(
    joint: JointDefinition,
    positions: Tensor,
) -> Tensor:
    if joint.joint_type in {"revolute", "continuous"}:
        return _batched_revolute_transform(
            joint,
            positions,
        )

    if joint.joint_type == "prismatic":
        return _batched_prismatic_transform(
            joint,
            positions,
        )

    raise ValueError(
        f"Unsupported movable joint type for {joint.name}: {joint.joint_type}"
    )


def _find_joint_chain(
    model: RobotModel,
    target_link: str,
) -> tuple[JointDefinition, ...]:
    if target_link not in model.links:
        raise KeyError(f"Unknown target link: {target_link}")

    joint_by_child: dict[str, JointDefinition] = {}

    for joint in model.joints:
        if joint.child_link in joint_by_child:
            raise ValueError(f"Link {joint.child_link} has multiple parent joints.")

        joint_by_child[joint.child_link] = joint

    chain: list[JointDefinition] = []
    current_link = target_link
    visited_links: set[str] = set()

    while current_link != model.root_link:
        if current_link in visited_links:
            raise ValueError(f"Kinematic cycle detected at link: {current_link}")

        visited_links.add(current_link)

        try:
            joint = joint_by_child[current_link]
        except KeyError as error:
            raise ValueError(
                f"Target link {target_link} is not reachable "
                f"from root link {model.root_link}."
            ) from error

        chain.append(joint)
        current_link = joint.parent_link

    chain.reverse()
    return tuple(chain)


def _compute_chunk_positions(
    state_values: Tensor,
    chain: Sequence[JointDefinition],
    component_indices: Mapping[str, int],
    joint_component_map: Mapping[str, str],
) -> Tensor:
    num_frames = int(state_values.shape[0])

    transform = (
        torch.eye(
            4,
            dtype=torch.float64,
        )
        .expand(num_frames, -1, -1)
        .clone()
    )

    for joint in chain:
        transform = transform @ _origin_transform(joint)

        if joint.joint_type == "fixed":
            continue

        component_name = joint_component_map[joint.name]
        joint_positions = state_values[
            :,
            component_indices[component_name],
        ]

        transform = transform @ _batched_joint_motion_transform(
            joint,
            joint_positions,
        )

    return transform[:, :3, 3].clone()


def compute_tool_trajectory(
    states: Tensor,
    component_names: Sequence[str],
    model: RobotModel,
    joint_component_map: Mapping[str, str],
    *,
    arm: str,
    link_name: str = "tool0",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    episode_indices: Tensor | None = None,
) -> ToolTrajectory:
    """Convert batched joint states into three-dimensional tool positions."""
    if states.ndim != 2:
        raise ValueError("States must have shape (num_frames, state_dimension).")

    num_frames, state_dimension = states.shape

    if num_frames == 0:
        raise ValueError("States must contain at least one frame.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    normalized_episode_indices: Tensor | None = None

    if episode_indices is not None:
        if episode_indices.ndim != 1:
            raise ValueError("Episode indices must be one-dimensional.")

        if episode_indices.shape[0] != num_frames:
            raise ValueError(
                "Episode index count must match the number of state frames."
            )

        if (
            episode_indices.dtype == torch.bool
            or episode_indices.is_floating_point()
            or episode_indices.is_complex()
        ):
            raise ValueError("Episode indices must use an integer dtype.")

        normalized_episode_indices = (
            episode_indices.detach()
            .to(
                device="cpu",
                dtype=torch.int64,
            )
            .clone()
        )

    names = tuple(component_names)

    if len(names) != state_dimension:
        raise ValueError(
            "The number of component names must match the state dimension."
        )

    if len(set(names)) != len(names):
        raise ValueError("State component names must be unique.")

    if not arm:
        raise ValueError("Arm name must not be empty.")

    known_joint_names = {joint.name for joint in model.joints}
    unknown_mapped_joints = set(joint_component_map) - known_joint_names

    if unknown_mapped_joints:
        unknown = ", ".join(sorted(unknown_mapped_joints))
        raise KeyError(f"Unknown mapped joints: {unknown}")

    chain = _find_joint_chain(model, link_name)
    movable_joints = tuple(joint for joint in chain if joint.joint_type != "fixed")

    missing_joint_mappings = tuple(
        joint.name for joint in movable_joints if joint.name not in joint_component_map
    )

    if missing_joint_mappings:
        missing = ", ".join(missing_joint_mappings)
        raise ValueError(f"Missing state-component mappings for joints: {missing}")

    component_indices = {name: index for index, name in enumerate(names)}

    missing_components = tuple(
        joint_component_map[joint.name]
        for joint in movable_joints
        if joint_component_map[joint.name] not in component_indices
    )

    if missing_components:
        missing = ", ".join(missing_components)
        raise ValueError(f"Missing state components: {missing}")

    state_values = states.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    position_chunks: list[Tensor] = []

    for start in range(0, num_frames, chunk_size):
        stop = min(start + chunk_size, num_frames)
        position_chunks.append(
            _compute_chunk_positions(
                state_values[start:stop],
                chain,
                component_indices,
                joint_component_map,
            )
        )

    return ToolTrajectory(
        arm=arm,
        link_name=link_name,
        positions=torch.cat(position_chunks, dim=0),
        episode_indices=normalized_episode_indices,
    )
