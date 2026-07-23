from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor


_STATE_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "observation.state",
    "action",
)


@dataclass(frozen=True)
class StateBatch:
    """Batched state and action data from selected LeRobot episodes."""

    timestamps: Tensor
    frame_indices: Tensor
    episode_indices: Tensor
    states: Tensor
    actions: Tensor

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the batch."""
        return int(self.states.shape[0])

    @property
    def state_dimension(self) -> int:
        """Return the number of state components per frame."""
        return int(self.states.shape[1])

    @property
    def action_dimension(self) -> int:
        """Return the number of action components per frame."""
        return int(self.actions.shape[1])


def build_state_batch(dataset: Any) -> StateBatch:
    """Build a state batch from an initialized LeRobot dataset."""
    tabular_dataset = dataset.hf_dataset.with_format(
        "torch",
        columns=list(_STATE_COLUMNS),
    )
    batch = tabular_dataset[:]

    return StateBatch(
        timestamps=batch["timestamp"],
        frame_indices=batch["frame_index"],
        episode_indices=batch["episode_index"],
        states=batch["observation.state"],
        actions=batch["action"],
    )


def split_state_batch_by_episode(
    batch: StateBatch,
) -> dict[int, StateBatch]:
    """Split a state batch while preserving first-seen episode order."""
    if batch.episode_indices.ndim != 1:
        raise ValueError("Episode indices must be one-dimensional.")

    num_frames = batch.num_frames
    frame_tensors = (
        batch.timestamps,
        batch.frame_indices,
        batch.episode_indices,
        batch.states,
        batch.actions,
    )

    if any(tensor.shape[0] != num_frames for tensor in frame_tensors):
        raise ValueError(
            "All state-batch tensors must contain the same number of frames."
        )

    episode_order: list[int] = []
    seen_episodes: set[int] = set()

    for value in batch.episode_indices.detach().cpu().tolist():
        episode = int(value)

        if episode not in seen_episodes:
            seen_episodes.add(episode)
            episode_order.append(episode)

    episode_batches: dict[int, StateBatch] = {}

    for episode in episode_order:
        mask = batch.episode_indices == episode
        episode_batches[episode] = StateBatch(
            timestamps=batch.timestamps[mask],
            frame_indices=batch.frame_indices[mask],
            episode_indices=batch.episode_indices[mask],
            states=batch.states[mask],
            actions=batch.actions[mask],
        )

    return episode_batches


def load_state_batch(
    repo_id: str,
    episodes: Sequence[int],
) -> StateBatch:
    """Load batched state and action data without decoding videos."""
    if not episodes:
        raise ValueError("At least one episode must be selected.")

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id,
        episodes=list(episodes),
        download_videos=False,
    )

    return build_state_batch(dataset)
