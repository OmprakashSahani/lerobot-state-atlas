from types import SimpleNamespace

import pytest
import torch

from lerobot_state_atlas.aggregation import (
    aggregate_workspace_coverages,
)
from lerobot_state_atlas.coverage import WorkspaceCoverage
from lerobot_state_atlas.trajectory import ToolTrajectory


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
    ):
        calls["repo_id"] = repo_id
        calls["episodes"] = episodes
        calls["episode_batch_size"] = episode_batch_size
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
    )

    assert calls["repo_id"] == "organization/dataset"
    assert calls["episodes"] == (0, 1, 2)
    assert calls["episode_batch_size"] == 2
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
