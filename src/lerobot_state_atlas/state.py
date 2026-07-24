from collections.abc import Iterator, Sequence
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


def iter_state_batches(
    repo_id: str,
    episodes: Sequence[int],
    *,
    episode_batch_size: int,
) -> Iterator[StateBatch]:
    """Load selected episodes in bounded batches."""
    if not episodes:
        raise ValueError("At least one episode must be selected.")

    if episode_batch_size <= 0:
        raise ValueError("Episode batch size must be greater than zero.")

    normalized_episodes = tuple(episodes)

    for start in range(0, len(normalized_episodes), episode_batch_size):
        episode_chunk = normalized_episodes[start : start + episode_batch_size]
        yield load_state_batch(
            repo_id,
            episodes=episode_chunk,
        )
