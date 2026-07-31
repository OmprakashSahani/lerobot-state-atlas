import pytest
import torch

from lerobot_state_atlas.coverage import (
    WorkspaceCoverageAccumulator,
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
        rotation_matrices=torch.eye(3, dtype=torch.float64)
        .expand(positions.shape[0], -1, -1)
        .clone(),
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
    assert coverage.voxel_origin_xyz == pytest.approx((0.0, 0.0, 0.0))
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


@pytest.mark.parametrize(
    "voxel_size",
    (
        0.0,
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_compute_workspace_coverage_rejects_voxel_size(
    voxel_size: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite and greater than zero",
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


def test_compute_workspace_coverage_uses_explicit_voxel_origin() -> None:
    trajectory = make_trajectory(
        torch.tensor(
            [
                [-0.6, 0.1, 0.1],
                [0.6, 0.1, 0.1],
            ],
            dtype=torch.float64,
        )
    )

    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.5,
        voxel_origin_xyz=(0.0, 0.0, 0.0),
    )

    assert coverage.minimum_xyz == pytest.approx((-0.6, 0.1, 0.1))
    assert coverage.voxel_origin_xyz == pytest.approx((0.0, 0.0, 0.0))
    assert coverage.grid_shape == (4, 1, 1)
    assert coverage.total_voxels == 4

    torch.testing.assert_close(
        coverage.voxel_indices,
        torch.tensor(
            [
                [-2, 0, 0],
                [1, 0, 0],
            ],
            dtype=torch.int64,
        ),
    )


@pytest.mark.parametrize(
    "voxel_origin_xyz",
    [
        (0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
    ],
)
def test_compute_workspace_coverage_rejects_invalid_voxel_origin(
    voxel_origin_xyz: tuple[float, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="Voxel origin must contain three finite values",
    ):
        compute_workspace_coverage(
            make_trajectory(
                torch.tensor(
                    [[0.0, 0.0, 0.0]],
                    dtype=torch.float64,
                )
            ),
            voxel_size=0.1,
            voxel_origin_xyz=voxel_origin_xyz,
        )


def test_workspace_coverage_accumulator_matches_combined_trajectory() -> None:
    first = make_trajectory(
        torch.tensor(
            [
                [-0.6, 0.1, 0.0],
                [-0.4, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=torch.float64,
        )
    )
    second = make_trajectory(
        torch.tensor(
            [
                [0.1, 0.1, 0.0],
                [0.6, 0.1, 0.0],
            ],
            dtype=torch.float64,
        )
    )

    accumulator = WorkspaceCoverageAccumulator(
        arm="left",
        link_name="tool0",
        voxel_size=0.5,
        voxel_origin_xyz=(0.0, 0.0, 0.0),
    )
    accumulator.update(first)
    accumulator.update(second)

    aggregated = accumulator.finalize()

    combined = make_trajectory(
        torch.cat(
            [first.positions, second.positions],
            dim=0,
        )
    )
    expected = compute_workspace_coverage(
        combined,
        voxel_size=0.5,
        voxel_origin_xyz=(0.0, 0.0, 0.0),
    )

    assert aggregated.arm == expected.arm
    assert aggregated.link_name == expected.link_name
    assert aggregated.num_points == expected.num_points
    assert aggregated.voxel_size == pytest.approx(expected.voxel_size)
    assert aggregated.minimum_xyz == pytest.approx(expected.minimum_xyz)
    assert aggregated.voxel_origin_xyz == pytest.approx(expected.voxel_origin_xyz)
    assert aggregated.maximum_xyz == pytest.approx(expected.maximum_xyz)
    assert aggregated.span_xyz == pytest.approx(expected.span_xyz)
    assert aggregated.centroid_xyz == pytest.approx(expected.centroid_xyz)
    assert aggregated.grid_shape == expected.grid_shape
    assert aggregated.occupied_voxels == expected.occupied_voxels
    assert aggregated.total_voxels == expected.total_voxels
    assert aggregated.occupancy_ratio == pytest.approx(expected.occupancy_ratio)
    assert aggregated.bounding_box_volume == pytest.approx(expected.bounding_box_volume)
    assert aggregated.occupied_volume == pytest.approx(expected.occupied_volume)

    torch.testing.assert_close(
        aggregated.voxel_indices,
        expected.voxel_indices,
    )
    torch.testing.assert_close(
        aggregated.visit_counts,
        expected.visit_counts,
    )
    assert aggregated.episode_ids_by_voxel == expected.episode_ids_by_voxel


def test_workspace_coverage_accumulator_requires_data() -> None:
    accumulator = WorkspaceCoverageAccumulator(
        arm="left",
        link_name="tool0",
        voxel_size=0.1,
    )

    with pytest.raises(
        ValueError,
        match="at least one trajectory",
    ):
        accumulator.finalize()


def test_workspace_coverage_accumulator_rejects_mismatched_arm() -> None:
    accumulator = WorkspaceCoverageAccumulator(
        arm="left",
        link_name="tool0",
        voxel_size=0.1,
    )

    with pytest.raises(
        ValueError,
        match="Trajectory arm must match accumulator arm",
    ):
        accumulator.update(
            make_trajectory(
                torch.tensor([[0.0, 0.0, 0.0]]),
                arm="right",
            )
        )


def test_compute_workspace_coverage_counts_distinct_episodes_per_voxel() -> None:
    trajectory = ToolTrajectory(
        arm="left",
        link_name="tool0",
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        rotation_matrices=torch.eye(3, dtype=torch.float64).expand(5, -1, -1).clone(),
        episode_indices=torch.tensor(
            [0, 0, 1, 1, 1],
            dtype=torch.int64,
        ),
    )

    coverage = compute_workspace_coverage(
        trajectory,
        voxel_size=0.25,
    )

    assert coverage.num_episodes == 2

    torch.testing.assert_close(
        coverage.visit_counts,
        torch.tensor([3, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        coverage.episode_counts,
        torch.tensor([2, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        coverage.episode_frequencies,
        torch.tensor([1.0, 0.5], dtype=torch.float64),
    )
    assert coverage.episode_ids_by_voxel == (
        (0, 1),
        (1,),
    )


def test_workspace_coverage_accumulator_counts_episodes_across_batches() -> None:
    first = ToolTrajectory(
        arm="left",
        link_name="tool0",
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        rotation_matrices=torch.eye(3, dtype=torch.float64).expand(3, -1, -1).clone(),
        episode_indices=torch.tensor(
            [0, 0, 1],
            dtype=torch.int64,
        ),
    )
    second = ToolTrajectory(
        arm="left",
        link_name="tool0",
        positions=torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        rotation_matrices=torch.eye(3, dtype=torch.float64).expand(2, -1, -1).clone(),
        episode_indices=torch.tensor(
            [2, 2],
            dtype=torch.int64,
        ),
    )

    accumulator = WorkspaceCoverageAccumulator(
        arm="left",
        link_name="tool0",
        voxel_size=0.25,
        voxel_origin_xyz=(0.0, 0.0, 0.0),
    )
    accumulator.update(first)
    accumulator.update(second)

    coverage = accumulator.finalize()

    assert coverage.num_episodes == 3

    torch.testing.assert_close(
        coverage.voxel_indices,
        torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
            ],
            dtype=torch.int64,
        ),
    )
    torch.testing.assert_close(
        coverage.visit_counts,
        torch.tensor([3, 1, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        coverage.episode_counts,
        torch.tensor([2, 1, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        coverage.episode_frequencies,
        torch.tensor(
            [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            dtype=torch.float64,
        ),
    )
    assert coverage.episode_ids_by_voxel == (
        (0, 2),
        (1,),
        (2,),
    )
