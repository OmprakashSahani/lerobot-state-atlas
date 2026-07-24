from dataclasses import dataclass
from math import isfinite
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


def _episode_position_segments(
    trajectory: ToolTrajectory,
) -> tuple[tuple[int | None, torch.Tensor], ...]:
    positions = trajectory.positions.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    episode_indices = trajectory.episode_indices

    if episode_indices is None:
        return ((None, positions),)

    episodes = episode_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    transition_indices = (
        torch.nonzero(
            episodes[1:] != episodes[:-1],
            as_tuple=False,
        ).flatten()
        + 1
    )
    boundaries = (
        0,
        *transition_indices.tolist(),
        positions.shape[0],
    )

    return tuple(
        (
            int(episodes[start].item()),
            positions[start:end],
        )
        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        )
    )


def _trajectory_playback_script(
    playback_fps: float,
) -> str:
    frame_interval_ms = 1000.0 / playback_fps
    interval_text = f"{frame_interval_ms:g}"

    return f"""
(function() {{
    const graph = document.getElementById('{{plot_id}}');

    if (!graph) {{
        return;
    }}

    const frameIntervalMs = {interval_text};
    const trajectories = [];

    graph.data.forEach((trace, traceIndex) => {{
        const meta = trace.meta || {{}};

        if (!Array.isArray(meta.playback_x)) {{
            return;
        }}

        trajectories.push({{
            traceIndex: traceIndex,
            startFrame: meta.playback_start_frame,
            x: meta.playback_x,
            y: meta.playback_y,
            z: meta.playback_z
        }});
    }});

    if (trajectories.length === 0) {{
        return;
    }}

    const controls = document.createElement("div");
    controls.style.display = "flex";
    controls.style.alignItems = "center";
    controls.style.justifyContent = "center";
    controls.style.gap = "10px";
    controls.style.margin = "12px 0 4px";

    const toggleButton = document.createElement("button");
    toggleButton.id = "lerobot-playback-toggle";
    toggleButton.type = "button";
    toggleButton.textContent = "Play";

    const resetButton = document.createElement("button");
    resetButton.id = "lerobot-playback-reset";
    resetButton.type = "button";
    resetButton.textContent = "Reset";

    const frameLabel = document.createElement("span");
    frameLabel.id = "lerobot-playback-frame";
    frameLabel.textContent = "Frame 0";

    controls.appendChild(toggleButton);
    controls.appendChild(resetButton);
    controls.appendChild(frameLabel);
    graph.parentNode.insertBefore(controls, graph);

    const totalFrames = trajectories.reduce(
        (maximum, trajectory) =>
            Math.max(
                maximum,
                trajectory.startFrame + trajectory.x.length
            ),
        0
    );

    let currentFrame = 0;
    let timer = null;

    function stopPlayback() {{
        if (timer !== null) {{
            window.clearInterval(timer);
            timer = null;
        }}

        toggleButton.textContent = "Play";
    }}

    function resetPlayback() {{
        stopPlayback();
        currentFrame = 0;
        frameLabel.textContent = "Frame 0";

        trajectories.forEach((trajectory) => {{
            Plotly.restyle(
                graph,
                {{
                    x: [[]],
                    y: [[]],
                    z: [[]]
                }},
                [trajectory.traceIndex]
            );
        }});
    }}

    function advancePlayback() {{
        trajectories.forEach((trajectory) => {{
            const localFrame =
                currentFrame - trajectory.startFrame;

            if (
                localFrame >= 0 &&
                localFrame < trajectory.x.length
            ) {{
                Plotly.extendTraces(
                    graph,
                    {{
                        x: [[trajectory.x[localFrame]]],
                        y: [[trajectory.y[localFrame]]],
                        z: [[trajectory.z[localFrame]]]
                    }},
                    [trajectory.traceIndex]
                );
            }}
        }});

        currentFrame += 1;
        frameLabel.textContent =
            "Frame " + Math.min(currentFrame, totalFrames);

        if (currentFrame >= totalFrames) {{
            stopPlayback();
        }}
    }}

    toggleButton.addEventListener("click", () => {{
        if (timer !== null) {{
            stopPlayback();
            return;
        }}

        if (currentFrame >= totalFrames) {{
            resetPlayback();
        }}

        toggleButton.textContent = "Pause";
        timer = window.setInterval(
            advancePlayback,
            frameIntervalMs
        );
    }});

    resetButton.addEventListener(
        "click",
        resetPlayback
    );

    resetPlayback();
}})();
"""


def save_interactive_workspace_heatmap(
    trajectories: tuple[ToolTrajectory, ...],
    output_path: str | Path,
    *,
    coverages: tuple[WorkspaceCoverage, ...],
    title: str = "Interactive tool workspace heatmap",
    playback_fps: float | None = None,
) -> InteractiveWorkspaceHeatmap:
    """Save an offline interactive 3D workspace heatmap as HTML."""
    _validate_inputs(
        trajectories,
        coverages,
    )

    if playback_fps is not None and (not isfinite(playback_fps) or playback_fps <= 0.0):
        raise ValueError("Playback FPS must be finite and greater than zero.")

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
        centers = _voxel_centers(coverage)
        visit_counts = coverage.visit_counts.detach().to(
            device="cpu",
            dtype=torch.int64,
        )

        playback_start_frame = 0

        for episode_index, positions in _episode_position_segments(trajectory):
            trace_name = (
                f"{trajectory.arm.capitalize()} trajectory"
                if episode_index is None
                else (f"{trajectory.arm.capitalize()} episode {episode_index}")
            )
            hover_label = (
                "Trajectory" if episode_index is None else f"Episode {episode_index}"
            )

            playback_x = positions[:, 0].tolist()
            playback_y = positions[:, 1].tolist()
            playback_z = positions[:, 2].tolist()
            displayed_positions = (
                positions[:0] if playback_fps is not None else positions
            )

            figure.add_trace(
                go.Scatter3d(
                    x=displayed_positions[:, 0].tolist(),
                    y=displayed_positions[:, 1].tolist(),
                    z=displayed_positions[:, 2].tolist(),
                    mode="lines",
                    name=trace_name,
                    line={
                        "width": 4,
                    },
                    meta={
                        "playback_start_frame": (playback_start_frame),
                        "playback_x": playback_x,
                        "playback_y": playback_y,
                        "playback_z": playback_z,
                    },
                    hovertemplate=(
                        hover_label
                        + "<br>x=%{x:.4f} m"
                        + "<br>y=%{y:.4f} m"
                        + "<br>z=%{z:.4f} m"
                        + "<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )

            playback_start_frame += positions.shape[0]

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
                "x": 1.16,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
                "len": 0.82,
                "thickness": 18,
            },
        },
        "legend": {
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        "hovermode": "closest",
        "template": "plotly_white",
        "height": 650,
        "margin": {
            "l": 20,
            "r": 260,
            "t": 90,
            "b": 70,
        },
    }

    for index in range(1, len(trajectories) + 1):
        scene_name = "scene" if index == 1 else f"scene{index}"
        layout_updates[scene_name] = scene_layout

    figure.update_layout(**layout_updates)
    figure.add_annotation(
        text=("Left and right panels use their respective local base_link frames."),
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={
            "size": 12,
        },
    )

    write_options: dict[str, object] = {
        "include_plotlyjs": True,
        "full_html": True,
        "auto_open": False,
    }

    if playback_fps is not None:
        write_options["post_script"] = _trajectory_playback_script(playback_fps)

    figure.write_html(
        destination,
        **write_options,
    )

    return InteractiveWorkspaceHeatmap(
        output_path=destination,
        num_trajectories=len(trajectories),
        num_points=sum(trajectory.num_frames for trajectory in trajectories),
        occupied_voxels=sum(coverage.occupied_voxels for coverage in coverages),
        voxel_size=coverages[0].voxel_size,
    )
