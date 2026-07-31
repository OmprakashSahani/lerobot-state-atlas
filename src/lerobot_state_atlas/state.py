from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from torch import Tensor


_STATE_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "observation.state",
    "action",
)
_COVERAGE_STATE_COLUMNS = (
    "episode_index",
    "observation.state",
)

BatchType = TypeVar("BatchType")


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


@dataclass(frozen=True)
class CoverageStateBatch:
    """State and episode identity required for workspace aggregation."""

    episode_indices: Tensor
    states: Tensor

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the batch."""
        return int(self.states.shape[0])

    @property
    def state_dimension(self) -> int:
        """Return the number of state components per frame."""
        return int(self.states.shape[1])


def _materialize_columns(dataset: Any, columns: Sequence[str]) -> Mapping[str, Tensor]:
    tabular_dataset = dataset.hf_dataset.with_format(
        "torch",
        columns=list(columns),
    )
    return tabular_dataset[:]


def build_state_batch(dataset: Any) -> StateBatch:
    """Build a state batch from an initialized LeRobot dataset."""
    batch = _materialize_columns(dataset, _STATE_COLUMNS)

    return StateBatch(
        timestamps=batch["timestamp"],
        frame_indices=batch["frame_index"],
        episode_indices=batch["episode_index"],
        states=batch["observation.state"],
        actions=batch["action"],
    )


def build_coverage_state_batch(dataset: Any) -> CoverageStateBatch:
    """Build a coverage batch without unused trajectory fields."""
    batch = _materialize_columns(dataset, _COVERAGE_STATE_COLUMNS)
    return CoverageStateBatch(
        episode_indices=batch["episode_index"],
        states=batch["observation.state"],
    )


def _load_batch(
    repo_id: str,
    episodes: Sequence[int],
    *,
    revision: str | None,
    builder: Callable[[Any], BatchType],
) -> BatchType:
    if not episodes:
        raise ValueError("At least one episode must be selected.")

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id,
        episodes=list(episodes),
        revision=revision,
        download_videos=False,
    )
    return builder(dataset)


def load_state_batch(
    repo_id: str,
    episodes: Sequence[int],
    *,
    revision: str | None = None,
) -> StateBatch:
    """Load batched state and action data without decoding videos."""
    return _load_batch(
        repo_id,
        episodes,
        revision=revision,
        builder=build_state_batch,
    )


def load_coverage_state_batch(
    repo_id: str,
    episodes: Sequence[int],
    *,
    revision: str | None = None,
) -> CoverageStateBatch:
    """Load only state and episode identity needed for coverage."""
    return _load_batch(
        repo_id,
        episodes,
        revision=revision,
        builder=build_coverage_state_batch,
    )


def _iter_batches(
    repo_id: str,
    episodes: Sequence[int],
    *,
    episode_batch_size: int,
    revision: str | None,
    loader: Callable[..., BatchType],
) -> Iterator[BatchType]:
    if not episodes:
        raise ValueError("At least one episode must be selected.")

    if episode_batch_size <= 0:
        raise ValueError("Episode batch size must be greater than zero.")

    normalized_episodes = tuple(episodes)

    for start in range(0, len(normalized_episodes), episode_batch_size):
        episode_chunk = normalized_episodes[start : start + episode_batch_size]
        if revision is None:
            yield loader(repo_id, episodes=episode_chunk)
        else:
            yield loader(repo_id, episodes=episode_chunk, revision=revision)


def iter_state_batches(
    repo_id: str,
    episodes: Sequence[int],
    *,
    episode_batch_size: int,
    revision: str | None = None,
) -> Iterator[StateBatch]:
    """Load selected episodes in bounded batches."""
    yield from _iter_batches(
        repo_id,
        episodes,
        episode_batch_size=episode_batch_size,
        revision=revision,
        loader=load_state_batch,
    )


def iter_coverage_state_batches(
    repo_id: str,
    episodes: Sequence[int],
    *,
    episode_batch_size: int,
    revision: str | None = None,
) -> Iterator[CoverageStateBatch]:
    """Load coverage-only state tensors in bounded episode batches."""
    yield from _iter_batches(
        repo_id,
        episodes,
        episode_batch_size=episode_batch_size,
        revision=revision,
        loader=load_coverage_state_batch,
    )
