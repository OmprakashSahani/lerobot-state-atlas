from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from lerobot_state_atlas.coverage import WorkspaceCoverage
from lerobot_state_atlas.trajectory import ToolTrajectory


matplotlib.use("Agg")


@dataclass(frozen=True)
class WorkspacePlot:
    """Metadata describing a saved workspace visualization."""

    output_path: Path
    num_trajectories: int
    num_points: int
    voxel_size: float | None


def _validate_trajectory(trajectory: ToolTrajectory) -> Tensor:
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

    return values


def _validate_coverage(
    trajectory: ToolTrajectory,
    coverage: WorkspaceCoverage,
) -> None:
    if coverage.arm != trajectory.arm:
        raise ValueError("Coverage arm must match trajectory arm.")

    if coverage.link_name != trajectory.link_name:
        raise ValueError("Coverage link name must match trajectory link name.")

    if coverage.num_points != trajectory.num_frames:
        raise ValueError("Coverage point count must match trajectory point count.")


def _set_equal_axes(
    axis: Axes,
    positions: Tensor,
) -> None:
    minimums = positions.min(dim=0).values
    maximums = positions.max(dim=0).values
    centers = (minimums + maximums) / 2.0
    maximum_span = float((maximums - minimums).max().item())

    if maximum_span == 0.0:
        maximum_span = 1.0

    radius = maximum_span / 2.0

    axis.set_xlim(
        float(centers[0].item() - radius),
        float(centers[0].item() + radius),
    )
    axis.set_ylim(
        float(centers[1].item() - radius),
        float(centers[1].item() + radius),
    )
    axis.set_zlim(
        float(centers[2].item() - radius),
        float(centers[2].item() + radius),
    )


def _episode_position_groups(
    trajectory: ToolTrajectory,
    positions: Tensor,
) -> tuple[tuple[int | None, Tensor], ...]:
    """Group positions into contiguous episode segments."""
    episode_indices = trajectory.episode_indices

    if episode_indices is None:
        return ((None, positions),)

    if episode_indices.ndim != 1:
        raise ValueError("Trajectory episode indices must be one-dimensional.")

    if episode_indices.shape[0] != positions.shape[0]:
        raise ValueError(
            "Trajectory episode index count must match the number of points."
        )

    if (
        episode_indices.dtype == torch.bool
        or episode_indices.is_floating_point()
        or episode_indices.is_complex()
    ):
        raise ValueError("Trajectory episode indices must use an integer dtype.")

    normalized_indices = episode_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    transition_indices = (
        torch.nonzero(
            normalized_indices[1:] != normalized_indices[:-1],
            as_tuple=False,
        ).flatten()
        + 1
    )
    boundaries = (
        0,
        *(int(index) for index in transition_indices.tolist()),
        positions.shape[0],
    )

    return tuple(
        (
            int(normalized_indices[start].item()),
            positions[start:stop],
        )
        for start, stop in zip(
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        )
    )


def _plot_trajectory(
    axis: Axes,
    trajectory: ToolTrajectory,
    positions: Tensor,
    coverage: WorkspaceCoverage | None,
) -> None:
    axis_positions = positions
    position_groups = _episode_position_groups(
        trajectory,
        positions,
    )
    seen_episode_labels: set[int] = set()

    for index, (episode, group) in enumerate(position_groups):
        group_values = group.numpy()

        if episode is None:
            path_label = "Tool path"
        elif episode in seen_episode_labels:
            path_label = "_nolegend_"
        else:
            path_label = f"Episode {episode}"
            seen_episode_labels.add(episode)

        line = axis.plot(
            group_values[:, 0],
            group_values[:, 1],
            group_values[:, 2],
            linewidth=1.4,
            alpha=0.9,
            label=path_label,
        )[0]
        path_color = line.get_color()

        axis.scatter(
            group_values[0, 0],
            group_values[0, 1],
            group_values[0, 2],
            marker="o",
            s=36,
            color=path_color,
            label="Start" if index == 0 else "_nolegend_",
        )
        axis.scatter(
            group_values[-1, 0],
            group_values[-1, 1],
            group_values[-1, 2],
            marker="X",
            s=42,
            color=path_color,
            label="End" if index == 0 else "_nolegend_",
        )

    if coverage is not None:
        minimums = torch.tensor(
            coverage.minimum_xyz,
            dtype=torch.float64,
        )
        voxel_minimums = (
            coverage.voxel_indices.to(dtype=torch.float64) * coverage.voxel_size
            + minimums
        )
        voxel_maximums = voxel_minimums + coverage.voxel_size
        voxel_centers = (voxel_minimums + voxel_maximums) / 2.0
        axis_positions = torch.cat(
            (
                positions,
                voxel_minimums,
                voxel_maximums,
            ),
            dim=0,
        )

        centers = voxel_centers.numpy()
        visit_counts = coverage.visit_counts.numpy()

        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            s=12 + 8 * visit_counts**0.5,
            alpha=0.5,
            label="Occupied voxels",
        )

        axis.text2D(
            0.02,
            0.98,
            (
                f"{coverage.occupied_voxels:,}/"
                f"{coverage.total_voxels:,} voxels\n"
                f"{coverage.occupancy_ratio * 100:.2f}% "
                f"occupancy · "
                f"{coverage.voxel_size * 100:.1f} cm"
            ),
            transform=axis.transAxes,
            verticalalignment="top",
        )

    axis.set_title(f"{trajectory.arm.capitalize()} {trajectory.link_name}")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.legend(loc="upper right")
    axis.view_init(elev=24, azim=-58)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    _set_equal_axes(axis, axis_positions)


def save_workspace_plot(
    trajectories: tuple[ToolTrajectory, ...],
    output_path: str | Path,
    *,
    coverages: tuple[WorkspaceCoverage, ...] | None = None,
    title: str = "Tool workspace coverage",
    dpi: int = 160,
) -> WorkspacePlot:
    """Save trajectory and voxel workspace views as a PNG image."""
    if not trajectories:
        raise ValueError("At least one trajectory is required.")

    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    if coverages is not None and (len(coverages) != len(trajectories)):
        raise ValueError("Coverage count must match trajectory count.")

    validated_positions = tuple(
        _validate_trajectory(trajectory) for trajectory in trajectories
    )

    if coverages is not None:
        for trajectory, coverage in zip(
            trajectories,
            coverages,
            strict=True,
        ):
            _validate_coverage(trajectory, coverage)

    destination = Path(output_path)

    if destination.suffix.lower() != ".png":
        raise ValueError("Workspace plot output must use a .png suffix.")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = Figure(
        figsize=(7.0 * len(trajectories), 6.0),
        constrained_layout=True,
    )
    figure.suptitle(title)

    for index, (trajectory, positions) in enumerate(
        zip(
            trajectories,
            validated_positions,
            strict=True,
        ),
        start=1,
    ):
        axis = figure.add_subplot(
            1,
            len(trajectories),
            index,
            projection="3d",
        )
        coverage = None if coverages is None else coverages[index - 1]

        _plot_trajectory(
            axis,
            trajectory,
            positions,
            coverage,
        )

    figure.savefig(
        destination,
        dpi=dpi,
        bbox_inches="tight",
    )

    voxel_size = None if coverages is None else coverages[0].voxel_size

    return WorkspacePlot(
        output_path=destination,
        num_trajectories=len(trajectories),
        num_points=sum(trajectory.num_frames for trajectory in trajectories),
        voxel_size=voxel_size,
    )
