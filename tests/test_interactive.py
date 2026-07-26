from dataclasses import replace
from pathlib import Path

import pytest
import torch
from plotly import graph_objects as go

from lerobot_state_atlas.coverage import compute_workspace_coverage
from lerobot_state_atlas.interactive import (
    save_interactive_workspace_coverage_heatmap,
    save_interactive_workspace_heatmap,
)
from lerobot_state_atlas.trajectory import ToolTrajectory


def make_trajectory(
    positions: torch.Tensor,
    *,
    arm: str,
    episode_indices: torch.Tensor | None = None,
) -> ToolTrajectory:
    return ToolTrajectory(
        arm=arm,
        link_name="tool0",
        positions=positions,
        episode_indices=episode_indices,
    )


def test_save_interactive_workspace_heatmap(
    tmp_path: Path,
) -> None:
    left = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.2],
                [0.2, 0.1, 0.3],
            ],
            dtype=torch.float64,
        ),
        arm="left",
    )
    right = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, -0.1, 0.1],
                [0.2, -0.2, 0.2],
            ],
            dtype=torch.float64,
        ),
        arm="right",
    )

    left_coverage = compute_workspace_coverage(
        left,
        voxel_size=0.05,
    )
    right_coverage = compute_workspace_coverage(
        right,
        voxel_size=0.05,
    )

    output_path = tmp_path / "workspace-heatmap.html"

    result = save_interactive_workspace_heatmap(
        (left, right),
        output_path,
        coverages=(
            left_coverage,
            right_coverage,
        ),
        title="Interactive workspace heatmap",
    )

    assert result.output_path == output_path
    assert result.num_trajectories == 2
    assert result.num_points == 6
    assert result.occupied_voxels == (
        left_coverage.occupied_voxels + right_coverage.occupied_voxels
    )
    assert result.voxel_size == pytest.approx(0.05)

    html = output_path.read_text(encoding="utf-8")

    assert output_path.is_file()
    assert "<html" in html.lower()
    assert "Interactive workspace heatmap" in html
    assert "Left tool0" in html
    assert "Right tool0" in html
    assert '<script src="https://cdn.plot.ly' not in html
    assert "Plotly.newPlot" in html


def test_interactive_heatmap_rejects_no_trajectories(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one trajectory is required",
    ):
        save_interactive_workspace_heatmap(
            (),
            tmp_path / "workspace.html",
            coverages=(),
        )


def test_interactive_heatmap_rejects_coverage_count(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory(
        torch.zeros((2, 3)),
        arm="left",
    )

    with pytest.raises(
        ValueError,
        match="Coverage count must match trajectory count",
    ):
        save_interactive_workspace_heatmap(
            (trajectory,),
            tmp_path / "workspace.html",
            coverages=(),
        )


def test_interactive_heatmap_rejects_mixed_voxel_sizes(
    tmp_path: Path,
) -> None:
    left = make_trajectory(
        torch.zeros((2, 3)),
        arm="left",
    )
    right = make_trajectory(
        torch.zeros((2, 3)),
        arm="right",
    )

    with pytest.raises(
        ValueError,
        match="same voxel size",
    ):
        save_interactive_workspace_heatmap(
            (left, right),
            tmp_path / "workspace.html",
            coverages=(
                compute_workspace_coverage(
                    left,
                    voxel_size=0.05,
                ),
                compute_workspace_coverage(
                    right,
                    voxel_size=0.10,
                ),
            ),
        )


def test_interactive_heatmap_rejects_non_html_path(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory(
        torch.zeros((2, 3)),
        arm="left",
    )
    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.05,
    )

    with pytest.raises(
        ValueError,
        match=r"must use an \.html suffix",
    ):
        save_interactive_workspace_heatmap(
            (trajectory,),
            tmp_path / "workspace.png",
            coverages=(coverage,),
        )


def test_interactive_heatmap_rejects_coverage_arm(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory(
        torch.zeros((2, 3)),
        arm="left",
    )
    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.05,
    )

    with pytest.raises(
        ValueError,
        match="Coverage arm must match trajectory arm",
    ):
        save_interactive_workspace_heatmap(
            (trajectory,),
            tmp_path / "workspace.html",
            coverages=(
                replace(
                    coverage,
                    arm="right",
                ),
            ),
        )


def test_interactive_heatmap_splits_episode_trajectory_lines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        arm="left",
        episode_indices=torch.tensor(
            [2, 2, 5, 5],
            dtype=torch.int64,
        ),
    )
    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.05,
    )
    captured: dict[str, go.Figure] = {}

    def fake_write_html(
        figure: go.Figure,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(
        go.Figure,
        "write_html",
        fake_write_html,
    )

    save_interactive_workspace_heatmap(
        (trajectory,),
        tmp_path / "workspace.html",
        coverages=(coverage,),
    )

    line_traces = [trace for trace in captured["figure"].data if trace.mode == "lines"]

    assert [trace.name for trace in line_traces] == [
        "Left episode 2",
        "Left episode 5",
    ]
    assert [tuple(trace.x) for trace in line_traces] == [
        (0.0, 0.1),
        (1.0, 1.1),
    ]


def test_interactive_heatmap_separates_legend_and_colorbar(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
            ],
            dtype=torch.float64,
        ),
        arm="left",
    )
    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.05,
    )
    captured: dict[str, go.Figure] = {}

    def fake_write_html(
        figure: go.Figure,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(
        go.Figure,
        "write_html",
        fake_write_html,
    )

    save_interactive_workspace_heatmap(
        (trajectory,),
        tmp_path / "workspace.html",
        coverages=(coverage,),
    )

    layout = captured["figure"].layout

    assert layout.legend.x == pytest.approx(1.02)
    assert layout.coloraxis.colorbar.x == pytest.approx(1.16)
    assert layout.margin.r >= 180


def test_interactive_heatmap_displays_coordinate_frame_note(
    monkeypatch,
    tmp_path: Path,
) -> None:
    left = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
            ],
            dtype=torch.float64,
        ),
        arm="left",
    )
    right = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, -0.1, 0.1],
            ],
            dtype=torch.float64,
        ),
        arm="right",
    )
    captured: dict[str, go.Figure] = {}

    def fake_write_html(
        figure: go.Figure,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(
        go.Figure,
        "write_html",
        fake_write_html,
    )

    save_interactive_workspace_heatmap(
        (left, right),
        tmp_path / "workspace.html",
        coverages=(
            compute_workspace_coverage(
                left,
                voxel_size=0.05,
            ),
            compute_workspace_coverage(
                right,
                voxel_size=0.05,
            ),
        ),
    )

    annotation_texts = [
        annotation.text for annotation in captured["figure"].layout.annotations
    ]

    assert (
        "Left and right panels use their respective local base_link frames."
    ) in annotation_texts
    assert captured["figure"].layout.margin.b >= 60


def test_interactive_heatmap_embeds_trajectory_playback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        arm="left",
        episode_indices=torch.tensor(
            [2, 2, 5, 5],
            dtype=torch.int64,
        ),
    )
    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.05,
    )
    captured: dict[str, object] = {}

    def fake_write_html(
        figure: go.Figure,
        *_args: object,
        **kwargs: object,
    ) -> None:
        captured["figure"] = figure
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        go.Figure,
        "write_html",
        fake_write_html,
    )

    save_interactive_workspace_heatmap(
        (trajectory,),
        tmp_path / "workspace.html",
        coverages=(coverage,),
        playback_fps=50.0,
    )

    figure = captured["figure"]
    assert isinstance(figure, go.Figure)

    line_traces = [trace for trace in figure.data if trace.mode == "lines"]

    assert [trace.meta["playback_start_frame"] for trace in line_traces] == [0, 2]

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)

    script = kwargs["post_script"]
    assert isinstance(script, str)
    assert "lerobot-playback-toggle" in script
    assert "lerobot-playback-reset" in script
    assert "Plotly.extendTraces" in script
    assert "frameIntervalMs = 20" in script


def test_save_interactive_workspace_coverage_heatmap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    left = make_trajectory(
        torch.tensor(
            [
                [-0.2, 0.0, 0.1],
                [0.0, 0.1, 0.2],
                [0.2, 0.2, 0.3],
            ],
            dtype=torch.float64,
        ),
        arm="left",
    )
    right = make_trajectory(
        torch.tensor(
            [
                [-0.1, 0.0, 0.1],
                [0.1, -0.1, 0.2],
            ],
            dtype=torch.float64,
        ),
        arm="right",
    )

    coverages = (
        compute_workspace_coverage(
            left,
            voxel_size=0.05,
            voxel_origin_xyz=(0.0, 0.0, 0.0),
        ),
        compute_workspace_coverage(
            right,
            voxel_size=0.05,
            voxel_origin_xyz=(0.0, 0.0, 0.0),
        ),
    )

    captured: dict[str, object] = {}

    def fake_write_html(
        figure: go.Figure,
        *_args: object,
        **kwargs: object,
    ) -> None:
        captured["figure"] = figure
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        go.Figure,
        "write_html",
        fake_write_html,
    )

    output_path = tmp_path / "aggregated-workspace.html"

    result = save_interactive_workspace_coverage_heatmap(
        coverages,
        output_path,
        title="Aggregated workspace coverage",
    )

    assert result.output_path == output_path
    assert result.num_trajectories == 0
    assert result.num_points == 5
    assert result.occupied_voxels == sum(
        coverage.occupied_voxels for coverage in coverages
    )
    assert result.voxel_size == pytest.approx(0.05)

    figure = captured["figure"]
    assert isinstance(figure, go.Figure)

    assert not [trace for trace in figure.data if trace.mode == "lines"]

    marker_traces = [trace for trace in figure.data if trace.mode == "markers"]

    assert [trace.name for trace in marker_traces] == [
        "Left visited voxels",
        "Right visited voxels",
    ]

    annotation_texts = [annotation.text for annotation in figure.layout.annotations]

    assert "Left tool0" in annotation_texts
    assert "Right tool0" in annotation_texts
    assert (
        "Left and right panels use their respective local base_link frames."
        in annotation_texts
    )

    assert captured["kwargs"]["include_plotlyjs"] is True
    assert captured["kwargs"]["full_html"] is True
    assert captured["kwargs"]["auto_open"] is False


def test_coverage_heatmap_adds_log_visit_metric_selector(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        arm="left",
        episode_indices=torch.tensor(
            [0, 1, 1, 1, 1],
            dtype=torch.int64,
        ),
    )
    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.25,
        voxel_origin_xyz=(0.0, 0.0, 0.0),
    )
    captured: dict[str, go.Figure] = {}

    def fake_write_html(
        figure: go.Figure,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(
        go.Figure,
        "write_html",
        fake_write_html,
    )

    save_interactive_workspace_coverage_heatmap(
        (coverage,),
        tmp_path / "log-visit-metrics.html",
    )

    figure = captured["figure"]
    marker_trace = figure.data[0]

    log_visits = torch.log1p(coverage.visit_counts.to(dtype=torch.float64))

    assert tuple(marker_trace.marker.color) == tuple(coverage.visit_counts.tolist())

    expected_customdata = torch.stack(
        (
            coverage.visit_counts.to(dtype=torch.float64),
            log_visits,
            coverage.episode_counts.to(dtype=torch.float64),
        ),
        dim=1,
    )
    torch.testing.assert_close(
        torch.tensor(
            marker_trace.customdata,
            dtype=torch.float64,
        ),
        expected_customdata,
    )

    assert "visits=%{customdata[0]:.0f}" in marker_trace.hovertemplate
    assert "log visits=%{customdata[1]:.3f}" in marker_trace.hovertemplate
    assert "episodes=%{customdata[2]:.0f}" in marker_trace.hovertemplate

    menus = figure.layout.updatemenus
    assert len(menus) == 1

    buttons = menus[0].buttons
    assert [button.label for button in buttons] == [
        "Visits",
        "Log visits",
        "Episode count",
    ]

    visits_update = buttons[0].args[0]
    log_visits_update = buttons[1].args[0]
    episode_count_update = buttons[2].args[0]

    assert visits_update["marker.color"] == [coverage.visit_counts.tolist()]
    assert log_visits_update["marker.color"] == [log_visits.tolist()]
    assert episode_count_update["marker.color"] == [coverage.episode_counts.tolist()]

    assert buttons[0].args[1]["coloraxis.colorbar.title.text"] == "Visits"
    assert buttons[0].args[1]["coloraxis.cauto"] is True

    assert buttons[1].args[1]["coloraxis.colorbar.title.text"] == "Log visits"
    assert buttons[1].args[1]["coloraxis.cauto"] is True

    assert buttons[2].args[1]["coloraxis.colorbar.title.text"] == "Episodes"
    assert buttons[2].args[1]["coloraxis.cauto"] is True


def test_coverage_heatmap_renders_shared_dual_arm_scene(
    monkeypatch,
    tmp_path: Path,
) -> None:
    left = make_trajectory(
        torch.tensor(
            [
                [-0.4, 0.0, 0.1],
                [-0.3, 0.1, 0.2],
            ],
            dtype=torch.float64,
        ),
        arm="left",
    )
    right = make_trajectory(
        torch.tensor(
            [
                [0.4, 0.0, 0.1],
                [0.3, 0.1, 0.2],
            ],
            dtype=torch.float64,
        ),
        arm="right",
    )
    coverages = (
        compute_workspace_coverage(
            left,
            voxel_size=0.05,
            voxel_origin_xyz=(0.0, 0.0, 0.0),
        ),
        compute_workspace_coverage(
            right,
            voxel_size=0.05,
            voxel_origin_xyz=(0.0, 0.0, 0.0),
        ),
    )
    captured: dict[str, go.Figure] = {}

    def fake_write_html(
        figure: go.Figure,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(
        go.Figure,
        "write_html",
        fake_write_html,
    )

    save_interactive_workspace_coverage_heatmap(
        coverages,
        tmp_path / "shared-workspace.html",
        shared_space=True,
    )

    figure = captured["figure"]
    marker_traces = [trace for trace in figure.data if trace.mode == "markers"]

    assert [trace.name for trace in marker_traces] == [
        "Left visited voxels",
        "Right visited voxels",
    ]
    assert [trace.scene for trace in marker_traces] == [
        "scene",
        "scene",
    ]

    layout = figure.layout.to_plotly_json()

    assert "scene" in layout
    assert "scene2" not in layout

    annotation_texts = [annotation.text for annotation in figure.layout.annotations]

    assert "Shared dual-arm workspace" in annotation_texts
    assert (
        "Left and right arms are shown in one shared world frame." in annotation_texts
    )
    assert (
        "Left and right panels use their respective local base_link frames."
        not in annotation_texts
    )
