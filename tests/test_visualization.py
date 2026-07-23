from dataclasses import replace
from pathlib import Path

import pytest
import torch

from lerobot_state_atlas.coverage import (
    WorkspaceCoverage,
    compute_workspace_coverage,
)
from lerobot_state_atlas.trajectory import ToolTrajectory
from lerobot_state_atlas.visualization import (
    save_workspace_plot,
)


def make_trajectory(
    positions: torch.Tensor | None = None,
    *,
    arm: str = "left",
    link_name: str = "tool0",
) -> ToolTrajectory:
    if positions is None:
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.3],
                [0.2, 0.1, 0.4],
            ],
            dtype=torch.float64,
        )

    return ToolTrajectory(
        arm=arm,
        link_name=link_name,
        positions=positions,
    )


def make_coverage(
    trajectory: ToolTrajectory,
    *,
    voxel_size: float = 0.1,
) -> WorkspaceCoverage:
    return compute_workspace_coverage(
        trajectory,
        voxel_size=voxel_size,
    )


def assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == (b"\x89PNG\r\n\x1a\n")


def test_save_workspace_plot_for_one_trajectory(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory()
    output_path = tmp_path / "workspace.png"

    result = save_workspace_plot(
        (trajectory,),
        output_path,
    )

    assert result.output_path == output_path
    assert result.num_trajectories == 1
    assert result.num_points == 3
    assert result.voxel_size is None
    assert_png(output_path)


def test_save_workspace_plot_with_coverage(
    tmp_path: Path,
) -> None:
    left = make_trajectory()
    right = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.1, 0.1],
            ],
            dtype=torch.float32,
        ),
        arm="right",
    )

    left_coverage = make_coverage(
        left,
        voxel_size=0.05,
    )
    right_coverage = make_coverage(
        right,
        voxel_size=0.05,
    )

    output_path = tmp_path / "nested" / "dual-workspace.png"

    result = save_workspace_plot(
        (left, right),
        output_path,
        coverages=(
            left_coverage,
            right_coverage,
        ),
        title="Bimanual workspace",
        dpi=100,
    )

    assert result.output_path == output_path
    assert result.num_trajectories == 2
    assert result.num_points == 5
    assert result.voxel_size == pytest.approx(0.05)
    assert_png(output_path)


def test_save_workspace_plot_for_static_point(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
    )
    output_path = tmp_path / "static.png"

    result = save_workspace_plot(
        (trajectory,),
        output_path,
    )

    assert result.num_points == 2
    assert_png(output_path)


def test_save_workspace_plot_rejects_no_trajectories(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one trajectory",
    ):
        save_workspace_plot(
            (),
            tmp_path / "workspace.png",
        )


@pytest.mark.parametrize("dpi", (0, -1))
def test_save_workspace_plot_rejects_dpi(
    tmp_path: Path,
    dpi: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="DPI must be greater than zero",
    ):
        save_workspace_plot(
            (make_trajectory(),),
            tmp_path / "workspace.png",
            dpi=dpi,
        )


def test_save_workspace_plot_rejects_coverage_count(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory()

    with pytest.raises(
        ValueError,
        match="Coverage count must match",
    ):
        save_workspace_plot(
            (trajectory,),
            tmp_path / "workspace.png",
            coverages=(),
        )


@pytest.mark.parametrize(
    "positions",
    (
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
    ),
)
def test_save_workspace_plot_rejects_shape(
    tmp_path: Path,
    positions: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        save_workspace_plot(
            (make_trajectory(positions),),
            tmp_path / "workspace.png",
        )


def test_save_workspace_plot_rejects_empty_trajectory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least one point",
    ):
        save_workspace_plot(
            (make_trajectory(torch.empty((0, 3))),),
            tmp_path / "workspace.png",
        )


@pytest.mark.parametrize(
    "invalid_value",
    (float("nan"), float("inf"), float("-inf")),
)
def test_save_workspace_plot_rejects_nonfinite_positions(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [[invalid_value, 0.0, 0.0]],
            dtype=torch.float64,
        )
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        save_workspace_plot(
            (trajectory,),
            tmp_path / "workspace.png",
        )


def test_save_workspace_plot_rejects_non_png_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"\.png suffix",
    ):
        save_workspace_plot(
            (make_trajectory(),),
            tmp_path / "workspace.svg",
        )


def test_save_workspace_plot_rejects_coverage_arm(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory()
    coverage = replace(
        make_coverage(trajectory),
        arm="right",
    )

    with pytest.raises(
        ValueError,
        match="Coverage arm must match",
    ):
        save_workspace_plot(
            (trajectory,),
            tmp_path / "workspace.png",
            coverages=(coverage,),
        )


def test_save_workspace_plot_rejects_coverage_link(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory()
    coverage = replace(
        make_coverage(trajectory),
        link_name="other_link",
    )

    with pytest.raises(
        ValueError,
        match="Coverage link name must match",
    ):
        save_workspace_plot(
            (trajectory,),
            tmp_path / "workspace.png",
            coverages=(coverage,),
        )


def test_save_workspace_plot_rejects_coverage_points(
    tmp_path: Path,
) -> None:
    trajectory = make_trajectory()
    coverage = replace(
        make_coverage(trajectory),
        num_points=trajectory.num_frames + 1,
    )

    with pytest.raises(
        ValueError,
        match="Coverage point count must match",
    ):
        save_workspace_plot(
            (trajectory,),
            tmp_path / "workspace.png",
            coverages=(coverage,),
        )


def test_plot_includes_full_voxel_extents() -> None:
    from matplotlib.figure import Figure

    from lerobot_state_atlas.visualization import (
        _plot_trajectory,
    )

    trajectory = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
    )
    coverage = make_coverage(
        trajectory,
        voxel_size=1.0,
    )

    figure = Figure()
    axis = figure.add_subplot(
        1,
        1,
        1,
        projection="3d",
    )

    _plot_trajectory(
        axis,
        trajectory,
        trajectory.positions,
        coverage,
    )

    assert axis.get_xlim() == pytest.approx((0.0, 1.0))
    assert axis.get_ylim() == pytest.approx((0.0, 1.0))
    assert axis.get_zlim() == pytest.approx((0.0, 1.0))
