from dataclasses import replace
from pathlib import Path

import pytest
import torch

from lerobot_state_atlas.coverage import compute_workspace_coverage
from lerobot_state_atlas.interactive import (
    save_interactive_workspace_heatmap,
)
from lerobot_state_atlas.trajectory import ToolTrajectory


def make_trajectory(
    positions: torch.Tensor,
    *,
    arm: str,
) -> ToolTrajectory:
    return ToolTrajectory(
        arm=arm,
        link_name="tool0",
        positions=positions,
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
