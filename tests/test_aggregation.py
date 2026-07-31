from types import SimpleNamespace

import pytest
import torch

from lerobot_state_atlas.aggregation import (
    aggregate_workspace_coverages,
)
from lerobot_state_atlas.coverage import WorkspaceCoverage
from lerobot_state_atlas.export_measurement import ExportMeasurementSession
from lerobot_state_atlas.trajectory import ToolTrajectory
from lerobot_state_atlas.transforms import RigidTransform


def test_aggregate_workspace_coverages_processes_bounded_batches(
    monkeypatch,
) -> None:
    batches = (
        SimpleNamespace(
            states=torch.zeros((3, 14), dtype=torch.float32),
            episode_indices=torch.tensor([0, 0, 1], dtype=torch.int64),
        ),
        SimpleNamespace(
            states=torch.zeros((2, 14), dtype=torch.float32),
            episode_indices=torch.tensor([2, 2], dtype=torch.int64),
        ),
    )
    calls: dict[str, object] = {
        "arms": [],
    }

    def fake_iter_state_batches(
        repo_id: str,
        episodes: list[int],
        *,
        episode_batch_size: int,
        revision: str,
    ):
        calls["repo_id"] = repo_id
        calls["episodes"] = episodes
        calls["episode_batch_size"] = episode_batch_size
        calls["revision"] = revision
        yield from batches

    monkeypatch.setattr(
        "lerobot_state_atlas.aggregation.iter_state_batches",
        fake_iter_state_batches,
    )

    def fake_compute_tool_trajectory(
        states: torch.Tensor,
        component_names: tuple[str, ...],
        model: object,
        joint_component_map: dict[str, str],
        *,
        arm: str,
        episode_indices: torch.Tensor,
    ) -> ToolTrajectory:
        calls["arms"].append(arm)
        offset = 0.0 if arm == "left" else 1.0
        positions = torch.stack(
            (
                torch.full(
                    (states.shape[0],),
                    offset,
                    dtype=torch.float64,
                ),
                torch.arange(
                    states.shape[0],
                    dtype=torch.float64,
                ),
                torch.zeros(
                    states.shape[0],
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )

        return ToolTrajectory(
            arm=arm,
            link_name="tool0",
            positions=positions,
            rotation_matrices=torch.eye(3, dtype=torch.float64)
            .expand(positions.shape[0], -1, -1)
            .clone(),
            episode_indices=episode_indices,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.aggregation.compute_tool_trajectory",
        fake_compute_tool_trajectory,
    )

    result = aggregate_workspace_coverages(
        "organization/dataset",
        [0, 1, 2],
        component_names=("component",) * 14,
        model=object(),
        voxel_size=0.5,
        episode_batch_size=2,
        revision="a" * 40,
    )

    assert calls["repo_id"] == "organization/dataset"
    assert calls["episodes"] == (0, 1, 2)
    assert calls["episode_batch_size"] == 2
    assert calls["revision"] == "a" * 40
    assert calls["arms"] == ["left", "right", "left", "right"]

    assert result.num_batches == 2
    assert result.num_episodes == 3
    assert result.num_frames == 5
    assert len(result.coverages) == 2
    assert all(isinstance(coverage, WorkspaceCoverage) for coverage in result.coverages)
    assert [coverage.arm for coverage in result.coverages] == [
        "left",
        "right",
    ]
    assert all(coverage.num_points == 5 for coverage in result.coverages)


def test_aggregate_workspace_coverages_reports_cumulative_batch_counts(
    monkeypatch,
) -> None:
    batches = (
        SimpleNamespace(
            states=torch.zeros((3, 14), dtype=torch.float32),
            episode_indices=torch.tensor([0, 0, 1], dtype=torch.int64),
        ),
        SimpleNamespace(
            states=torch.zeros((2, 14), dtype=torch.float32),
            episode_indices=torch.tensor([2, 2], dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.aggregation.iter_state_batches",
        lambda *args, **kwargs: iter(batches),
    )

    def fake_compute(
        states: torch.Tensor,
        *args,
        arm: str,
        episode_indices: torch.Tensor,
        **kwargs,
    ) -> ToolTrajectory:
        del args, kwargs
        offset = 0.0 if arm == "left" else 1.0
        positions = torch.stack(
            (
                torch.full((states.shape[0],), offset, dtype=torch.float64),
                torch.arange(states.shape[0], dtype=torch.float64),
                torch.zeros(states.shape[0], dtype=torch.float64),
            ),
            dim=1,
        )
        return ToolTrajectory(
            arm=arm,
            link_name="tool0",
            positions=positions,
            rotation_matrices=torch.eye(3, dtype=torch.float64).repeat(
                states.shape[0], 1, 1
            ),
            episode_indices=episode_indices,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.aggregation.compute_tool_trajectory",
        fake_compute,
    )
    completed = []
    clock = iter(float(value) for value in range(5))
    measurement = ExportMeasurementSession(
        monotonic_clock=lambda: next(clock),
        progress_callback=completed.append,
    )

    result = aggregate_workspace_coverages(
        "organization/dataset",
        (0, 1, 2),
        component_names=("component",) * 14,
        model=object(),
        voxel_size=0.5,
        episode_batch_size=2,
        measurement=measurement,
    )

    assert result.num_frames == 5
    assert [batch.episode_ids for batch in completed] == [(0, 1), (2,)]
    assert [batch.frame_count for batch in completed] == [3, 2]
    assert [batch.cumulative_frame_count for batch in completed] == [3, 5]
    assert [batch.elapsed_seconds for batch in completed] == [1.0, 1.0]
    for arm in ("left", "right"):
        assert completed[0].arms[arm].occupied_entries == 3
        assert completed[0].arms[arm].csr_incidence == 3
        assert completed[0].arms[arm].raw_visits == 3
        assert completed[1].arms[arm].occupied_entries == 3
        assert completed[1].arms[arm].csr_incidence == 5
        assert completed[1].arms[arm].raw_visits == 5


@pytest.mark.parametrize("episode_batch_size", [0, -1])
def test_aggregate_workspace_coverages_rejects_invalid_batch_size(
    episode_batch_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Episode batch size must be greater than zero",
    ):
        aggregate_workspace_coverages(
            "organization/dataset",
            [0],
            component_names=("component",),
            model=object(),
            voxel_size=0.1,
            episode_batch_size=episode_batch_size,
        )


def test_aggregate_workspace_coverages_applies_arm_transforms_before_voxelization(
    monkeypatch,
) -> None:
    batch = SimpleNamespace(
        states=torch.zeros((2, 14), dtype=torch.float32),
        episode_indices=torch.tensor([3, 7], dtype=torch.int64),
    )

    def fake_iter_state_batches(
        repo_id: str,
        episodes: tuple[int, ...],
        *,
        episode_batch_size: int,
    ):
        del repo_id, episodes, episode_batch_size
        yield batch

    monkeypatch.setattr(
        "lerobot_state_atlas.aggregation.iter_state_batches",
        fake_iter_state_batches,
    )

    def fake_compute_tool_trajectory(
        states: torch.Tensor,
        component_names: tuple[str, ...],
        model: object,
        joint_component_map: dict[str, str],
        *,
        arm: str,
        episode_indices: torch.Tensor,
    ) -> ToolTrajectory:
        del states, component_names, model, joint_component_map

        positions = (
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                ],
                dtype=torch.float64,
            )
            if arm == "left"
            else torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                ],
                dtype=torch.float64,
            )
        )

        return ToolTrajectory(
            arm=arm,
            link_name="tool0",
            positions=positions,
            rotation_matrices=torch.eye(3, dtype=torch.float64)
            .expand(positions.shape[0], -1, -1)
            .clone(),
            episode_indices=episode_indices,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.aggregation.compute_tool_trajectory",
        fake_compute_tool_trajectory,
    )

    result = aggregate_workspace_coverages(
        "organization/dataset",
        (3, 7),
        component_names=("component",) * 14,
        model=object(),
        voxel_size=0.05,
        episode_batch_size=2,
        arm_transforms={
            "left": RigidTransform(
                translation_xyz=(-0.4, 0.0, 0.0),
            ),
            "right": RigidTransform(
                translation_xyz=(0.4, 0.0, 0.0),
            ),
        },
    )

    left, right = result.coverages

    assert left.voxel_origin_xyz == pytest.approx((0.0, 0.0, 0.0))
    assert right.voxel_origin_xyz == pytest.approx((0.0, 0.0, 0.0))

    assert left.minimum_xyz == pytest.approx((-0.4, 0.0, 0.0))
    assert left.maximum_xyz == pytest.approx((-0.3, 0.0, 0.0))

    assert right.minimum_xyz == pytest.approx((0.4, 0.0, 0.0))
    assert right.maximum_xyz == pytest.approx((0.4, 0.1, 0.0))

    assert left.num_episodes == 2
    assert right.num_episodes == 2
