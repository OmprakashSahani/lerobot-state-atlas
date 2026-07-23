import pytest
import torch

from lerobot_state_atlas.coverage import (
    compute_workspace_coverage,
)
from lerobot_state_atlas.trajectory import ToolTrajectory


def make_trajectory(
    positions: torch.Tensor,
    *,
    arm: str = "left",
    link_name: str = "tool0",
) -> ToolTrajectory:
    return ToolTrajectory(
        arm=arm,
        link_name=link_name,
        positions=positions,
    )


def test_compute_workspace_coverage() -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.6, 0.6],
            ],
            dtype=torch.float32,
        )
    )

    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.5,
    )

    assert coverage.arm == "left"
    assert coverage.link_name == "tool0"
    assert coverage.num_points == 2
    assert coverage.voxel_size == pytest.approx(0.5)

    assert coverage.minimum_xyz == pytest.approx((0.0, 0.0, 0.0))
    assert coverage.maximum_xyz == pytest.approx((0.6, 0.6, 0.6))
    assert coverage.span_xyz == pytest.approx((0.6, 0.6, 0.6))
    assert coverage.centroid_xyz == pytest.approx((0.3, 0.3, 0.3))

    assert coverage.grid_shape == (2, 2, 2)
    assert coverage.occupied_voxels == 2
    assert coverage.total_voxels == 8
    assert coverage.occupancy_ratio == pytest.approx(0.25)
    assert coverage.bounding_box_volume == pytest.approx(0.216)
    assert coverage.occupied_volume == pytest.approx(0.25)

    torch.testing.assert_close(
        coverage.voxel_indices,
        torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=torch.int64,
        ),
    )
    torch.testing.assert_close(
        coverage.visit_counts,
        torch.tensor([1, 1]),
    )


def test_compute_workspace_coverage_counts_revisits() -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
    )

    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.25,
    )

    assert coverage.grid_shape == (2, 1, 1)
    assert coverage.occupied_voxels == 2
    assert coverage.total_voxels == 2
    assert coverage.occupancy_ratio == pytest.approx(1.0)
    assert coverage.maximum_visit_count == 2
    assert coverage.mean_visits_per_occupied_voxel == pytest.approx(2.0)

    torch.testing.assert_close(
        coverage.visit_counts,
        torch.tensor([2, 2]),
    )


def test_compute_workspace_coverage_for_single_point() -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [[1.0, 2.0, 3.0]],
            dtype=torch.float32,
        )
    )

    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.1,
    )

    assert coverage.grid_shape == (1, 1, 1)
    assert coverage.occupied_voxels == 1
    assert coverage.total_voxels == 1
    assert coverage.occupancy_ratio == pytest.approx(1.0)
    assert coverage.bounding_box_volume == pytest.approx(0.0)
    assert coverage.occupied_volume == pytest.approx(0.001)
    assert coverage.maximum_visit_count == 1
    assert coverage.mean_visits_per_occupied_voxel == pytest.approx(1.0)


@pytest.mark.parametrize(
    "positions",
    (
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
    ),
)
def test_compute_workspace_coverage_rejects_shape(
    positions: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        compute_workspace_coverage(
            make_trajectory(positions),
            voxel_size=0.1,
        )


def test_compute_workspace_coverage_rejects_empty() -> None:
    with pytest.raises(
        ValueError,
        match="at least one point",
    ):
        compute_workspace_coverage(
            make_trajectory(torch.empty((0, 3))),
            voxel_size=0.1,
        )


@pytest.mark.parametrize("voxel_size", (0.0, -0.1))
def test_compute_workspace_coverage_rejects_voxel_size(
    voxel_size: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        compute_workspace_coverage(
            make_trajectory(torch.tensor([[0.0, 0.0, 0.0]])),
            voxel_size=voxel_size,
        )


@pytest.mark.parametrize(
    "invalid_value",
    (float("nan"), float("inf"), float("-inf")),
)
def test_compute_workspace_coverage_rejects_nonfinite_values(
    invalid_value: float,
) -> None:
    positions = torch.tensor(
        [[invalid_value, 0.0, 0.0]],
        dtype=torch.float64,
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        compute_workspace_coverage(
            make_trajectory(positions),
            voxel_size=0.1,
        )


def test_compute_workspace_coverage_rejects_empty_arm() -> None:
    with pytest.raises(
        ValueError,
        match="arm must not be empty",
    ):
        compute_workspace_coverage(
            make_trajectory(
                torch.tensor([[0.0, 0.0, 0.0]]),
                arm="",
            ),
            voxel_size=0.1,
        )


def test_compute_workspace_coverage_rejects_empty_link() -> None:
    with pytest.raises(
        ValueError,
        match="link name must not be empty",
    ):
        compute_workspace_coverage(
            make_trajectory(
                torch.tensor([[0.0, 0.0, 0.0]]),
                link_name="",
            ),
            voxel_size=0.1,
        )
