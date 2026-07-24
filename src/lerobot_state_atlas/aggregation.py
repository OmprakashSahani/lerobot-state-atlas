from collections.abc import Sequence
from dataclasses import dataclass

from lerobot_state_atlas.coverage import (
    WorkspaceCoverage,
    WorkspaceCoverageAccumulator,
)
from lerobot_state_atlas.state import iter_state_batches
from lerobot_state_atlas.trajectory import (
    build_trlc_dk1_joint_component_map,
    compute_tool_trajectory,
)
from lerobot_state_atlas.urdf import RobotModel


@dataclass(frozen=True)
class WorkspaceAggregation:
    """Summary of workspace coverage aggregated across episode batches."""

    coverages: tuple[WorkspaceCoverage, ...]
    num_batches: int
    num_episodes: int
    num_frames: int


def aggregate_workspace_coverages(
    repo_id: str,
    episodes: Sequence[int],
    *,
    component_names: Sequence[str],
    model: RobotModel,
    voxel_size: float,
    episode_batch_size: int,
) -> WorkspaceAggregation:
    """Aggregate dual-arm workspace coverage in bounded episode batches."""
    if not episodes:
        raise ValueError("At least one episode must be selected.")

    if episode_batch_size <= 0:
        raise ValueError("Episode batch size must be greater than zero.")

    normalized_episodes = tuple(episodes)

    accumulators = {
        arm: WorkspaceCoverageAccumulator(
            arm=arm,
            link_name="tool0",
            voxel_size=voxel_size,
        )
        for arm in ("left", "right")
    }

    num_batches = 0
    num_frames = 0

    for batch in iter_state_batches(
        repo_id,
        normalized_episodes,
        episode_batch_size=episode_batch_size,
    ):
        num_batches += 1
        num_frames += int(batch.states.shape[0])

        for arm, accumulator in accumulators.items():
            trajectory = compute_tool_trajectory(
                batch.states,
                component_names,
                model,
                build_trlc_dk1_joint_component_map(arm),
                arm=arm,
                episode_indices=batch.episode_indices,
            )
            accumulator.update(trajectory)

    if num_batches == 0:
        raise ValueError("Episode loading produced no state batches.")

    return WorkspaceAggregation(
        coverages=tuple(accumulators[arm].finalize() for arm in ("left", "right")),
        num_batches=num_batches,
        num_episodes=len(normalized_episodes),
        num_frames=num_frames,
    )
