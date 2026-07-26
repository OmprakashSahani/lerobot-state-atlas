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
        coverage.voxel_origin_xyz,
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
    controls.style.flexWrap = "wrap";
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

    const rotationButton = document.createElement("button");
    rotationButton.id = "lerobot-playback-auto-rotate";
    rotationButton.type = "button";
    rotationButton.textContent = "Auto rotate: On";

    const timeline = document.createElement("input");
    timeline.id = "lerobot-playback-timeline";
    timeline.type = "range";
    timeline.min = "0";
    timeline.step = "1";
    timeline.value = "0";
    timeline.setAttribute("aria-label", "Playback timeline");
    timeline.style.width = "320px";

    const frameLabel = document.createElement("span");
    frameLabel.id = "lerobot-playback-frame";
    frameLabel.textContent = "Frame 0";
    frameLabel.style.minWidth = "230px";

    controls.appendChild(toggleButton);
    controls.appendChild(resetButton);
    controls.appendChild(rotationButton);
    controls.appendChild(timeline);
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

    timeline.max = String(totalFrames);

    let currentFrame = 0;
    let renderedFrame = 0;
    let timer = null;
    let isInteracting = false;
    let isSeeking = false;
    let resumeAfterSeek = false;
    let autoRotateEnabled = true;
    let rotationAnimationFrame = null;
    let previousRotationTime = null;
    let wheelInteractionTimeout = null;

    const sceneNames = Object.keys(graph._fullLayout).filter(
        (name) => /^scene\\d*$/.test(name)
    );

    function sceneCameraState(sceneName) {{
        const scene = graph._fullLayout[sceneName];
        const eye =
            scene.camera && scene.camera.eye
                ? scene.camera.eye
                : {{x: 1.25, y: 1.25, z: 1.25}};

        return {{
            radius: Math.hypot(eye.x, eye.y),
            angle: Math.atan2(eye.y, eye.x),
            z: eye.z
        }};
    }}

    const cameraStates = Object.fromEntries(
        sceneNames.map((sceneName) => [
            sceneName,
            sceneCameraState(sceneName)
        ])
    );

    function visibleFrameCount(trajectory, frame) {{
        return Math.max(
            0,
            Math.min(
                frame - trajectory.startFrame,
                trajectory.x.length
            )
        );
    }}

    function formatPlaybackTime(frame) {{
        return (
            frame * frameIntervalMs / 1000
        ).toFixed(2) + " s";
    }}

    function updatePlaybackStatus() {{
        const displayedFrame = Math.min(
            Math.max(currentFrame, 0),
            totalFrames
        );

        timeline.value = String(displayedFrame);
        frameLabel.textContent =
            "Frame " + displayedFrame +
            " / " + totalFrames +
            " · " + formatPlaybackTime(displayedFrame) +
            " / " + formatPlaybackTime(totalFrames);
    }}

    function seekPlayback(frame) {{
        const numericFrame = Number(frame);

        if (!Number.isFinite(numericFrame)) {{
            return;
        }}

        currentFrame = Math.min(
            totalFrames,
            Math.max(0, Math.round(numericFrame))
        );

        synchronizePlayback();
        updatePlaybackStatus();
    }}

    function synchronizePlayback() {{
        trajectories.forEach((trajectory) => {{
            const visibleFrames = visibleFrameCount(
                trajectory,
                currentFrame
            );

            Plotly.restyle(
                graph,
                {{
                    x: [trajectory.x.slice(0, visibleFrames)],
                    y: [trajectory.y.slice(0, visibleFrames)],
                    z: [trajectory.z.slice(0, visibleFrames)]
                }},
                [trajectory.traceIndex]
            );
        }});

        renderedFrame = currentFrame;
    }}

    function stopRotation() {{
        if (rotationAnimationFrame !== null) {{
            window.cancelAnimationFrame(rotationAnimationFrame);
            rotationAnimationFrame = null;
        }}

        previousRotationTime = null;
    }}

    function rotateScenes(timestamp) {{
        if (
            timer === null ||
            !autoRotateEnabled ||
            isInteracting
        ) {{
            stopRotation();
            return;
        }}

        if (previousRotationTime === null) {{
            previousRotationTime = timestamp;
        }}

        const elapsedSeconds =
            (timestamp - previousRotationTime) / 1000.0;
        previousRotationTime = timestamp;

        const angularSpeed = 0.35;
        const layoutUpdate = {{}};

        sceneNames.forEach((sceneName) => {{
            const cameraState = cameraStates[sceneName];
            cameraState.angle += angularSpeed * elapsedSeconds;

            layoutUpdate[sceneName + ".camera.eye.x"] =
                cameraState.radius * Math.cos(cameraState.angle);
            layoutUpdate[sceneName + ".camera.eye.y"] =
                cameraState.radius * Math.sin(cameraState.angle);
            layoutUpdate[sceneName + ".camera.eye.z"] =
                cameraState.z;
        }});

        Plotly.relayout(graph, layoutUpdate);
        rotationAnimationFrame =
            window.requestAnimationFrame(rotateScenes);
    }}

    function startRotation() {{
        if (
            rotationAnimationFrame !== null ||
            timer === null ||
            !autoRotateEnabled ||
            isInteracting
        ) {{
            return;
        }}

        previousRotationTime = null;
        rotationAnimationFrame =
            window.requestAnimationFrame(rotateScenes);
    }}

    function stopPlayback() {{
        if (timer !== null) {{
            window.clearInterval(timer);
            timer = null;
        }}

        stopRotation();
        toggleButton.textContent = "Play";
    }}

    function resetPlayback() {{
        stopPlayback();
        currentFrame = 0;
        synchronizePlayback();
        updatePlaybackStatus();
    }}

    function advancePlayback() {{
        if (!isInteracting) {{
            if (renderedFrame !== currentFrame) {{
                synchronizePlayback();
            }}

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

            renderedFrame = currentFrame + 1;
        }}

        currentFrame += 1;
        updatePlaybackStatus();

        if (currentFrame >= totalFrames) {{
            stopPlayback();
        }}
    }}

    function startPlayback() {{
        if (
            timer !== null ||
            currentFrame >= totalFrames
        ) {{
            return;
        }}

        toggleButton.textContent = "Pause";
        timer = window.setInterval(
            advancePlayback,
            frameIntervalMs
        );
        startRotation();
    }}

    function beginSeek() {{
        if (isSeeking) {{
            return;
        }}

        isSeeking = true;
        resumeAfterSeek = timer !== null;
        stopPlayback();
    }}

    function finishSeek() {{
        if (!isSeeking) {{
            return;
        }}

        seekPlayback(timeline.value);
        isSeeking = false;

        if (
            resumeAfterSeek &&
            currentFrame < totalFrames
        ) {{
            startPlayback();
        }}

        resumeAfterSeek = false;
    }}

    function refreshCameraStates() {{
        sceneNames.forEach((sceneName) => {{
            cameraStates[sceneName] =
                sceneCameraState(sceneName);
        }});
    }}

    function beginInteraction() {{
        isInteracting = true;
        stopRotation();
    }}

    function finishInteraction() {{
        if (!isInteracting) {{
            return;
        }}

        isInteracting = false;
        refreshCameraStates();
        synchronizePlayback();
        startRotation();
    }}

    graph.addEventListener("pointerdown", beginInteraction);

    graph.addEventListener(
        "wheel",
        () => {{
            beginInteraction();

            if (wheelInteractionTimeout !== null) {{
                window.clearTimeout(wheelInteractionTimeout);
            }}

            wheelInteractionTimeout = window.setTimeout(
                finishInteraction,
                250
            );
        }},
        {{passive: true}}
    );

    window.addEventListener(
        "pointerup",
        finishInteraction
    );
    window.addEventListener(
        "pointercancel",
        finishInteraction
    );

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
        startRotation();
    }});

    resetButton.addEventListener(
        "click",
        resetPlayback
    );

    rotationButton.addEventListener("click", () => {{
        autoRotateEnabled = !autoRotateEnabled;
        rotationButton.textContent = autoRotateEnabled
            ? "Auto rotate: On"
            : "Auto rotate: Off";

        if (autoRotateEnabled) {{
            refreshCameraStates();
            startRotation();
        }} else {{
            stopRotation();
        }}
    }});

    timeline.addEventListener(
        "pointerdown",
        beginSeek
    );

    timeline.addEventListener("input", () => {{
        if (!isSeeking) {{
            beginSeek();
        }}

        seekPlayback(timeline.value);
    }});

    timeline.addEventListener(
        "change",
        finishSeek
    );

    window.addEventListener(
        "pointerup",
        finishSeek
    );

    window.addEventListener(
        "pointercancel",
        finishSeek
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
    shared_space: bool = False,
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

    if shared_space:
        subplot_titles = ("Shared dual-arm workspace",)
        figure = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "scene"}]],
            subplot_titles=subplot_titles,
        )
    else:
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
        target_column = 1 if shared_space else column
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
                col=target_column,
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
            col=target_column,
        )

    scene_layout = {
        "xaxis_title": "X (m)",
        "yaxis_title": "Y (m)",
        "zaxis_title": "Z (m)",
        "aspectmode": "data",
        "uirevision": "lerobot-playback-camera",
    }

    layout_updates: dict[str, object] = {
        "uirevision": "lerobot-playback-layout",
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

    scene_count = 1 if shared_space else len(trajectories)

    for index in range(1, scene_count + 1):
        scene_name = "scene" if index == 1 else f"scene{index}"
        layout_updates[scene_name] = scene_layout

    figure.update_layout(**layout_updates)
    figure.add_annotation(
        text=(
            "Left and right arms are shown in one shared world frame."
            if shared_space
            else "Left and right panels use their respective local base_link frames."
        ),
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
        "config": {
            "scrollZoom": True,
            "responsive": True,
        },
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


def save_interactive_workspace_coverage_heatmap(
    coverages: tuple[WorkspaceCoverage, ...],
    output_path: str | Path,
    *,
    title: str = "Aggregated workspace coverage heatmap",
    shared_space: bool = False,
) -> InteractiveWorkspaceHeatmap:
    """Save aggregated workspace coverage without trajectory data."""
    if not coverages:
        raise ValueError("At least one workspace coverage is required.")

    voxel_sizes = {coverage.voxel_size for coverage in coverages}

    if len(voxel_sizes) != 1:
        raise ValueError("All coverages must use the same voxel size.")

    for coverage in coverages:
        if not coverage.arm:
            raise ValueError("Coverage arm must not be empty.")

        if not coverage.link_name:
            raise ValueError("Coverage link name must not be empty.")

        if coverage.num_points <= 0:
            raise ValueError("Coverage must represent at least one point.")

        if coverage.voxel_indices.ndim != 2 or coverage.voxel_indices.shape[1] != 3:
            raise ValueError(
                "Coverage voxel indices must have shape (occupied_voxels, 3)."
            )

        if coverage.visit_counts.ndim != 1:
            raise ValueError("Coverage visit counts must be one-dimensional.")

        if coverage.visit_counts.shape[0] != coverage.voxel_indices.shape[0]:
            raise ValueError("Coverage visit-count size must match voxel count.")

        if coverage.num_episodes <= 0:
            raise ValueError("Coverage must represent at least one episode.")

        if coverage.episode_counts.ndim != 1:
            raise ValueError("Coverage episode counts must be one-dimensional.")

        if coverage.episode_counts.shape[0] != coverage.voxel_indices.shape[0]:
            raise ValueError("Coverage episode-count size must match voxel count.")

        if coverage.episode_frequencies.ndim != 1:
            raise ValueError("Coverage episode frequencies must be one-dimensional.")

        if coverage.episode_frequencies.shape[0] != coverage.voxel_indices.shape[0]:
            raise ValueError("Coverage episode-frequency size must match voxel count.")

        if not torch.isfinite(coverage.episode_frequencies).all().item():
            raise ValueError(
                "Coverage episode frequencies must contain only finite values."
            )

        if (coverage.episode_frequencies < 0.0).any().item() or (
            coverage.episode_frequencies > 1.0
        ).any().item():
            raise ValueError(
                "Coverage episode frequencies must be between zero and one."
            )

    destination = Path(output_path)

    if destination.suffix.lower() != ".html":
        raise ValueError("Interactive workspace output must use an .html suffix.")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if shared_space:
        subplot_titles = ("Shared dual-arm workspace",)
        figure = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "scene"}]],
            subplot_titles=subplot_titles,
        )
    else:
        subplot_titles = tuple(
            f"{coverage.arm.capitalize()} {coverage.link_name}"
            for coverage in coverages
        )
        figure = make_subplots(
            rows=1,
            cols=len(coverages),
            specs=[[{"type": "scene"} for _ in coverages]],
            subplot_titles=subplot_titles,
        )

    visit_color_values: list[list[int]] = []
    log_visit_color_values: list[list[float]] = []
    episode_count_color_values: list[list[int]] = []

    for column, coverage in enumerate(
        coverages,
        start=1,
    ):
        target_column = 1 if shared_space else column
        centers = _voxel_centers(coverage)
        visit_counts = coverage.visit_counts.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        episode_counts = coverage.episode_counts.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        log_visits = torch.log1p(visit_counts.to(dtype=torch.float64))

        visit_color_values.append(visit_counts.tolist())
        log_visit_color_values.append(log_visits.tolist())
        episode_count_color_values.append(episode_counts.tolist())

        customdata = torch.stack(
            (
                visit_counts.to(dtype=torch.float64),
                log_visits,
                episode_counts.to(dtype=torch.float64),
            ),
            dim=1,
        )

        figure.add_trace(
            go.Scatter3d(
                x=centers[:, 0].tolist(),
                y=centers[:, 1].tolist(),
                z=centers[:, 2].tolist(),
                mode="markers",
                name=(f"{coverage.arm.capitalize()} visited voxels"),
                marker={
                    "size": 6,
                    "color": visit_counts.tolist(),
                    "coloraxis": "coloraxis",
                    "opacity": 0.72,
                },
                customdata=customdata.tolist(),
                hovertemplate=(
                    "Visited voxel"
                    "<br>x=%{x:.4f} m"
                    "<br>y=%{y:.4f} m"
                    "<br>z=%{z:.4f} m"
                    "<br>visits=%{customdata[0]:.0f}"
                    "<br>log visits=%{customdata[1]:.3f}"
                    "<br>episodes=%{customdata[2]:.0f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=target_column,
        )

    metric_buttons = [
        {
            "label": "Visits",
            "method": "update",
            "args": [
                {
                    "marker.color": visit_color_values,
                },
                {
                    "coloraxis.colorbar.title.text": "Visits",
                    "coloraxis.cauto": True,
                    "coloraxis.cmin": None,
                    "coloraxis.cmax": None,
                },
            ],
        },
        {
            "label": "Log visits",
            "method": "update",
            "args": [
                {
                    "marker.color": log_visit_color_values,
                },
                {
                    "coloraxis.colorbar.title.text": "Log visits",
                    "coloraxis.cauto": True,
                    "coloraxis.cmin": None,
                    "coloraxis.cmax": None,
                },
            ],
        },
        {
            "label": "Episode count",
            "method": "update",
            "args": [
                {
                    "marker.color": episode_count_color_values,
                },
                {
                    "coloraxis.colorbar.title.text": "Episodes",
                    "coloraxis.cauto": True,
                    "coloraxis.cmin": None,
                    "coloraxis.cmax": None,
                },
            ],
        },
    ]

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
        "updatemenus": [
            {
                "type": "dropdown",
                "direction": "down",
                "active": 0,
                "buttons": metric_buttons,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.12,
                "yanchor": "top",
                "showactive": True,
            }
        ],
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

    scene_count = 1 if shared_space else len(coverages)

    for index in range(1, scene_count + 1):
        scene_name = "scene" if index == 1 else f"scene{index}"
        layout_updates[scene_name] = scene_layout

    figure.update_layout(**layout_updates)

    coordinate_note = (
        "Left and right arms are shown in one shared world frame."
        if shared_space
        else "Left and right panels use their respective local base_link frames."
    )

    figure.add_annotation(
        text=coordinate_note,
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={
            "size": 12,
        },
    )

    figure.write_html(
        destination,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

    return InteractiveWorkspaceHeatmap(
        output_path=destination,
        num_trajectories=0,
        num_points=sum(coverage.num_points for coverage in coverages),
        occupied_voxels=sum(coverage.occupied_voxels for coverage in coverages),
        voxel_size=coverages[0].voxel_size,
    )
