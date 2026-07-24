from dataclasses import dataclass
from pathlib import Path

import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from lerobot_state_atlas.coverage import WorkspaceCoverage
from lerobot_state_atlas.trajectory import ToolTrajectory


@dataclass(frozen=True)
class InteractiveWorkspaceHeatmap:
    """Metadata describing a saved interactive workspace heatmap."""

    output_path: Path
    num_trajectories: int
    num_points: int
    occupied_voxels: int
    voxel_size: float


def _validate_inputs(
    trajectories: tuple[ToolTrajectory, ...],
    coverages: tuple[WorkspaceCoverage, ...],
) -> None:
    if not trajectories:
        raise ValueError("At least one trajectory is required.")

    if len(coverages) != len(trajectories):
        raise ValueError("Coverage count must match trajectory count.")

    voxel_sizes = {coverage.voxel_size for coverage in coverages}

    if len(voxel_sizes) != 1:
        raise ValueError("All coverages must use the same voxel size.")

    for trajectory, coverage in zip(
        trajectories,
        coverages,
        strict=True,
    ):
        positions = trajectory.positions

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Trajectory positions must have shape (num_points, 3).")

        if positions.shape[0] == 0:
            raise ValueError("Trajectory must contain at least one point.")

        if not torch.isfinite(positions).all().item():
            raise ValueError("Trajectory positions must contain only finite values.")

        if coverage.arm != trajectory.arm:
            raise ValueError("Coverage arm must match trajectory arm.")

        if coverage.link_name != trajectory.link_name:
            raise ValueError("Coverage link name must match trajectory link name.")

        if coverage.num_points != trajectory.num_frames:
            raise ValueError("Coverage point count must match trajectory point count.")


def _voxel_centers(
    coverage: WorkspaceCoverage,
) -> torch.Tensor:
    minimums = torch.tensor(
        coverage.minimum_xyz,
        dtype=torch.float64,
    )
    voxel_minimums = (
        coverage.voxel_indices.to(dtype=torch.float64) * coverage.voxel_size + minimums
    )

    return voxel_minimums + coverage.voxel_size / 2.0


def save_interactive_workspace_heatmap(
    trajectories: tuple[ToolTrajectory, ...],
    output_path: str | Path,
    *,
    coverages: tuple[WorkspaceCoverage, ...],
    title: str = "Interactive tool workspace heatmap",
) -> InteractiveWorkspaceHeatmap:
    """Save an offline interactive 3D workspace heatmap as HTML."""
    _validate_inputs(
        trajectories,
        coverages,
    )

    destination = Path(output_path)

    if destination.suffix.lower() != ".html":
        raise ValueError("Interactive workspace output must use an .html suffix.")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subplot_titles = tuple(
        f"{trajectory.arm.capitalize()} {trajectory.link_name}"
        for trajectory in trajectories
    )

    figure = make_subplots(
        rows=1,
        cols=len(trajectories),
        specs=[[{"type": "scene"} for _ in trajectories]],
        subplot_titles=subplot_titles,
    )

    for column, (trajectory, coverage) in enumerate(
        zip(
            trajectories,
            coverages,
            strict=True,
        ),
        start=1,
    ):
        positions = trajectory.positions.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        centers = _voxel_centers(coverage)
        visit_counts = coverage.visit_counts.detach().to(
            device="cpu",
            dtype=torch.int64,
        )

        figure.add_trace(
            go.Scatter3d(
                x=positions[:, 0].tolist(),
                y=positions[:, 1].tolist(),
                z=positions[:, 2].tolist(),
                mode="lines",
                name=f"{trajectory.arm.capitalize()} trajectory",
                line={
                    "width": 4,
                },
                hovertemplate=(
                    "Trajectory"
                    "<br>x=%{x:.4f} m"
                    "<br>y=%{y:.4f} m"
                    "<br>z=%{z:.4f} m"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )

        figure.add_trace(
            go.Scatter3d(
                x=centers[:, 0].tolist(),
                y=centers[:, 1].tolist(),
                z=centers[:, 2].tolist(),
                mode="markers",
                name=f"{trajectory.arm.capitalize()} visited voxels",
                marker={
                    "size": 6,
                    "color": visit_counts.tolist(),
                    "coloraxis": "coloraxis",
                    "opacity": 0.72,
                },
                customdata=visit_counts.tolist(),
                hovertemplate=(
                    "Visited voxel"
                    "<br>x=%{x:.4f} m"
                    "<br>y=%{y:.4f} m"
                    "<br>z=%{z:.4f} m"
                    "<br>visits=%{customdata}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )

    scene_layout = {
        "xaxis_title": "X (m)",
        "yaxis_title": "Y (m)",
        "zaxis_title": "Z (m)",
        "aspectmode": "data",
    }

    layout_updates: dict[str, object] = {
        "title": {
            "text": title,
            "x": 0.5,
        },
        "coloraxis": {
            "colorscale": "Viridis",
            "colorbar": {
                "title": "Visits",
            },
        },
        "hovermode": "closest",
        "template": "plotly_white",
        "height": 650,
        "margin": {
            "l": 20,
            "r": 20,
            "t": 90,
            "b": 20,
        },
    }

    for index in range(1, len(trajectories) + 1):
        scene_name = "scene" if index == 1 else f"scene{index}"
        layout_updates[scene_name] = scene_layout

    figure.update_layout(**layout_updates)

    figure.write_html(
        destination,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

    return InteractiveWorkspaceHeatmap(
        output_path=destination,
        num_trajectories=len(trajectories),
        num_points=sum(trajectory.num_frames for trajectory in trajectories),
        occupied_voxels=sum(coverage.occupied_voxels for coverage in coverages),
        voxel_size=coverages[0].voxel_size,
    )
