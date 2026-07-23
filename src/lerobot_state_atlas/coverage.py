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
    voxel_size: float
    minimum_xyz: Vector3
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


def compute_workspace_coverage(
    trajectory: ToolTrajectory,
    *,
    voxel_size: float,
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

    minimums = values.min(dim=0).values
    maximums = values.max(dim=0).values
    spans = maximums - minimums
    centroid = values.mean(dim=0)

    point_voxel_indices = torch.floor((values - minimums) / voxel_size).to(
        dtype=torch.int64
    )

    voxel_indices, visit_counts = torch.unique(
        point_voxel_indices,
        dim=0,
        sorted=True,
        return_counts=True,
    )

    grid_shape_values = voxel_indices.max(dim=0).values + 1
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
        voxel_size=float(voxel_size),
        minimum_xyz=_as_vector3(minimums),
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
    )
