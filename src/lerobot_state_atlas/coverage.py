from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, prod

import torch
from torch import Tensor

from lerobot_state_atlas.trajectory import ToolTrajectory


Vector3 = tuple[float, float, float]
GridShape = tuple[int, int, int]


@dataclass(frozen=True)
class WorkspaceCoverage:
    """Voxel-based workspace coverage for one tool trajectory."""

    arm: str
    link_name: str
    num_points: int
    num_episodes: int
    voxel_size: float
    minimum_xyz: Vector3
    voxel_origin_xyz: Vector3
    maximum_xyz: Vector3
    span_xyz: Vector3
    centroid_xyz: Vector3
    grid_shape: GridShape
    occupied_voxels: int
    total_voxels: int
    occupancy_ratio: float
    bounding_box_volume: float
    occupied_volume: float
    voxel_indices: Tensor
    visit_counts: Tensor
    episode_counts: Tensor
    episode_frequencies: Tensor
    episode_ids_by_voxel: tuple[tuple[int, ...], ...]

    @property
    def maximum_visit_count(self) -> int:
        """Return the highest number of visits to one voxel."""
        return int(self.visit_counts.max().item())

    @property
    def mean_visits_per_occupied_voxel(self) -> float:
        """Return the average number of points per occupied voxel."""
        return self.num_points / self.occupied_voxels


def _as_vector3(values: Tensor) -> Vector3:
    return (
        float(values[0].item()),
        float(values[1].item()),
        float(values[2].item()),
    )


def _as_grid_shape(values: Tensor) -> GridShape:
    return (
        int(values[0].item()),
        int(values[1].item()),
        int(values[2].item()),
    )


def _trajectory_episode_indices(
    trajectory: ToolTrajectory,
    *,
    num_points: int,
) -> Tensor:
    """Return validated frame-aligned episode indices on the CPU."""
    if trajectory.episode_indices is None:
        return torch.zeros(
            num_points,
            dtype=torch.int64,
        )

    episode_indices = trajectory.episode_indices.detach().to(device="cpu")

    if episode_indices.ndim != 1 or episode_indices.shape[0] != num_points:
        raise ValueError("Trajectory episode indices must have shape (num_points,).")

    if episode_indices.dtype == torch.bool or episode_indices.dtype.is_complex:
        raise ValueError("Trajectory episode indices must contain integers.")

    if episode_indices.dtype.is_floating_point:
        if not torch.isfinite(episode_indices).all().item():
            raise ValueError(
                "Trajectory episode indices must contain only finite values."
            )

        rounded = torch.round(episode_indices)

        if not torch.equal(episode_indices, rounded):
            raise ValueError("Trajectory episode indices must contain integers.")

    return episode_indices.to(dtype=torch.int64)


def compute_workspace_coverage(
    trajectory: ToolTrajectory,
    *,
    voxel_size: float,
    voxel_origin_xyz: Sequence[float] | None = None,
) -> WorkspaceCoverage:
    """Compute voxel occupancy within a trajectory's axis-aligned bounds."""
    positions = trajectory.positions

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Trajectory positions must have shape (num_points, 3).")

    num_points = int(positions.shape[0])

    if num_points == 0:
        raise ValueError("Trajectory must contain at least one point.")

    if not isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("Voxel size must be finite and greater than zero.")

    if not trajectory.arm:
        raise ValueError("Trajectory arm must not be empty.")

    if not trajectory.link_name:
        raise ValueError("Trajectory link name must not be empty.")

    values = positions.detach().to(
        device="cpu",
        dtype=torch.float64,
    )

    if not torch.isfinite(values).all().item():
        raise ValueError("Trajectory positions must contain only finite values.")

    episode_indices = _trajectory_episode_indices(
        trajectory,
        num_points=num_points,
    )
    num_episodes = int(torch.unique(episode_indices).numel())

    minimums = values.min(dim=0).values
    maximums = values.max(dim=0).values
    spans = maximums - minimums
    centroid = values.mean(dim=0)

    if voxel_origin_xyz is None:
        voxel_origin = minimums.clone()
    else:
        if len(voxel_origin_xyz) != 3 or not all(
            isfinite(float(value)) for value in voxel_origin_xyz
        ):
            raise ValueError("Voxel origin must contain three finite values.")

        voxel_origin = torch.tensor(
            tuple(float(value) for value in voxel_origin_xyz),
            dtype=torch.float64,
        )

    point_voxel_indices = torch.floor((values - voxel_origin) / voxel_size).to(
        dtype=torch.int64
    )

    voxel_indices, visit_counts = torch.unique(
        point_voxel_indices,
        dim=0,
        sorted=True,
        return_counts=True,
    )

    voxel_episode_pairs = torch.unique(
        torch.cat(
            (
                point_voxel_indices,
                episode_indices.unsqueeze(dim=1),
            ),
            dim=1,
        ),
        dim=0,
        sorted=True,
    )
    episode_voxel_indices, episode_counts = torch.unique(
        voxel_episode_pairs[:, :3],
        dim=0,
        sorted=True,
        return_counts=True,
    )

    if not torch.equal(episode_voxel_indices, voxel_indices):
        raise RuntimeError("Episode-count voxels must match visited voxels.")

    episode_frequencies = episode_counts.to(dtype=torch.float64) / num_episodes

    episode_ids_lookup: dict[
        tuple[int, int, int],
        list[int],
    ] = {tuple(int(value) for value in index): [] for index in voxel_indices.tolist()}

    for x_index, y_index, z_index, episode_index in voxel_episode_pairs.tolist():
        key = (
            int(x_index),
            int(y_index),
            int(z_index),
        )
        episode_ids_lookup[key].append(int(episode_index))

    episode_ids_by_voxel = tuple(
        tuple(episode_ids_lookup[tuple(int(value) for value in index)])
        for index in voxel_indices.tolist()
    )

    minimum_voxel_indices = voxel_indices.min(dim=0).values
    maximum_voxel_indices = voxel_indices.max(dim=0).values
    grid_shape_values = maximum_voxel_indices - minimum_voxel_indices + 1
    grid_shape = _as_grid_shape(grid_shape_values)

    occupied_voxels = int(voxel_indices.shape[0])
    total_voxels = prod(grid_shape)
    occupancy_ratio = occupied_voxels / total_voxels

    voxel_volume = voxel_size**3
    bounding_box_volume = float(torch.prod(spans).item())
    occupied_volume = occupied_voxels * voxel_volume

    return WorkspaceCoverage(
        arm=trajectory.arm,
        link_name=trajectory.link_name,
        num_points=num_points,
        num_episodes=num_episodes,
        voxel_size=float(voxel_size),
        minimum_xyz=_as_vector3(minimums),
        voxel_origin_xyz=_as_vector3(voxel_origin),
        maximum_xyz=_as_vector3(maximums),
        span_xyz=_as_vector3(spans),
        centroid_xyz=_as_vector3(centroid),
        grid_shape=grid_shape,
        occupied_voxels=occupied_voxels,
        total_voxels=total_voxels,
        occupancy_ratio=occupancy_ratio,
        bounding_box_volume=bounding_box_volume,
        occupied_volume=occupied_volume,
        voxel_indices=voxel_indices,
        visit_counts=visit_counts,
        episode_counts=episode_counts,
        episode_frequencies=episode_frequencies,
        episode_ids_by_voxel=episode_ids_by_voxel,
    )


class WorkspaceCoverageAccumulator:
    """Incrementally aggregate workspace coverage without retaining all points."""

    def __init__(
        self,
        *,
        arm: str,
        link_name: str,
        voxel_size: float,
        voxel_origin_xyz: Sequence[float] | None = None,
    ) -> None:
        if not arm:
            raise ValueError("Accumulator arm must not be empty.")

        if not link_name:
            raise ValueError("Accumulator link name must not be empty.")

        if not isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("Voxel size must be finite and greater than zero.")

        origin_values = (
            (0.0, 0.0, 0.0)
            if voxel_origin_xyz is None
            else tuple(float(value) for value in voxel_origin_xyz)
        )

        if len(origin_values) != 3 or not all(
            isfinite(value) for value in origin_values
        ):
            raise ValueError("Voxel origin must contain three finite values.")

        self.arm = arm
        self.link_name = link_name
        self.voxel_size = float(voxel_size)
        self.voxel_origin_xyz: Vector3 = (
            origin_values[0],
            origin_values[1],
            origin_values[2],
        )

        self._voxel_origin = torch.tensor(
            self.voxel_origin_xyz,
            dtype=torch.float64,
        )
        self._num_points = 0
        self._position_sum = torch.zeros(3, dtype=torch.float64)
        self._minimums: Tensor | None = None
        self._maximums: Tensor | None = None
        self._visit_counts: dict[tuple[int, int, int], int] = {}
        self._episode_indices: set[int] = set()
        self._episode_ids_by_voxel: dict[
            tuple[int, int, int],
            set[int],
        ] = {}

    @property
    def cumulative_occupied_entries(self) -> int:
        """Return the number of occupied entries accumulated so far."""
        return len(self._visit_counts)

    @property
    def cumulative_episode_incidence(self) -> int:
        """Return exact accumulated voxel-episode incidence."""
        return sum(len(values) for values in self._episode_ids_by_voxel.values())

    @property
    def cumulative_raw_visits(self) -> int:
        """Return accumulated raw tool-point visits."""
        return self._num_points

    def update(self, trajectory: ToolTrajectory) -> None:
        """Add one trajectory batch to the aggregate."""
        if trajectory.arm != self.arm:
            raise ValueError("Trajectory arm must match accumulator arm.")

        if trajectory.link_name != self.link_name:
            raise ValueError("Trajectory link name must match accumulator link name.")

        positions = trajectory.positions

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Trajectory positions must have shape (num_points, 3).")

        if positions.shape[0] == 0:
            raise ValueError("Trajectory must contain at least one point.")

        values = positions.detach().to(
            device="cpu",
            dtype=torch.float64,
        )

        if not torch.isfinite(values).all().item():
            raise ValueError("Trajectory positions must contain only finite values.")

        episode_indices = _trajectory_episode_indices(
            trajectory,
            num_points=int(values.shape[0]),
        )
        self._episode_indices.update(int(index) for index in episode_indices.tolist())

        batch_minimums = values.min(dim=0).values
        batch_maximums = values.max(dim=0).values

        if self._minimums is None:
            self._minimums = batch_minimums.clone()
            self._maximums = batch_maximums.clone()
        else:
            self._minimums = torch.minimum(
                self._minimums,
                batch_minimums,
            )
            self._maximums = torch.maximum(
                self._maximums,
                batch_maximums,
            )

        self._num_points += int(values.shape[0])
        self._position_sum += values.sum(dim=0)

        point_voxel_indices = torch.floor(
            (values - self._voxel_origin) / self.voxel_size
        ).to(dtype=torch.int64)

        voxel_indices, visit_counts = torch.unique(
            point_voxel_indices,
            dim=0,
            sorted=True,
            return_counts=True,
        )

        for index, count in zip(
            voxel_indices.tolist(),
            visit_counts.tolist(),
            strict=True,
        ):
            key = (int(index[0]), int(index[1]), int(index[2]))
            self._visit_counts[key] = self._visit_counts.get(key, 0) + int(count)

        voxel_episode_pairs = torch.unique(
            torch.cat(
                (
                    point_voxel_indices,
                    episode_indices.unsqueeze(dim=1),
                ),
                dim=1,
            ),
            dim=0,
            sorted=True,
        )

        for x_index, y_index, z_index, episode_index in voxel_episode_pairs.tolist():
            key = (
                int(x_index),
                int(y_index),
                int(z_index),
            )
            self._episode_ids_by_voxel.setdefault(key, set()).add(int(episode_index))

    def finalize(self) -> WorkspaceCoverage:
        """Build immutable aggregate coverage statistics."""
        if self._num_points == 0 or self._minimums is None or self._maximums is None:
            raise ValueError("Coverage accumulator requires at least one trajectory.")

        ordered_counts = sorted(self._visit_counts.items())

        voxel_indices = torch.tensor(
            [index for index, _ in ordered_counts],
            dtype=torch.int64,
        )
        visit_counts = torch.tensor(
            [count for _, count in ordered_counts],
            dtype=torch.int64,
        )
        episode_counts = torch.tensor(
            [len(self._episode_ids_by_voxel[index]) for index, _ in ordered_counts],
            dtype=torch.int64,
        )
        num_episodes = len(self._episode_indices)
        episode_frequencies = episode_counts.to(dtype=torch.float64) / num_episodes
        episode_ids_by_voxel = tuple(
            tuple(sorted(self._episode_ids_by_voxel[index]))
            for index, _ in ordered_counts
        )

        minimum_voxel_indices = voxel_indices.min(dim=0).values
        maximum_voxel_indices = voxel_indices.max(dim=0).values
        grid_shape_values = maximum_voxel_indices - minimum_voxel_indices + 1
        grid_shape = _as_grid_shape(grid_shape_values)

        spans = self._maximums - self._minimums
        centroid = self._position_sum / self._num_points

        occupied_voxels = int(voxel_indices.shape[0])
        total_voxels = prod(grid_shape)
        occupancy_ratio = occupied_voxels / total_voxels
        bounding_box_volume = float(torch.prod(spans).item())
        occupied_volume = occupied_voxels * self.voxel_size**3

        return WorkspaceCoverage(
            arm=self.arm,
            link_name=self.link_name,
            num_points=self._num_points,
            num_episodes=num_episodes,
            voxel_size=self.voxel_size,
            minimum_xyz=_as_vector3(self._minimums),
            voxel_origin_xyz=self.voxel_origin_xyz,
            maximum_xyz=_as_vector3(self._maximums),
            span_xyz=_as_vector3(spans),
            centroid_xyz=_as_vector3(centroid),
            grid_shape=grid_shape,
            occupied_voxels=occupied_voxels,
            total_voxels=total_voxels,
            occupancy_ratio=occupancy_ratio,
            bounding_box_volume=bounding_box_volume,
            occupied_volume=occupied_volume,
            voxel_indices=voxel_indices,
            visit_counts=visit_counts,
            episode_counts=episode_counts,
            episode_frequencies=episode_frequencies,
            episode_ids_by_voxel=episode_ids_by_voxel,
        )
