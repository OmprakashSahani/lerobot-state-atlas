from types import SimpleNamespace

import pytest
import torch

from lerobot_state_atlas.state import (
    StateBatch,
    build_state_batch,
    load_state_batch,
    split_state_batch_by_episode,
)


class FakeTabularDataset:
    def __init__(self) -> None:
        self.requested_format = None
        self.requested_columns = None

    def with_format(self, format_name: str, columns: list[str]):
        self.requested_format = format_name
        self.requested_columns = columns
        return self

    def __getitem__(self, item):
        assert item == slice(None)

        return {
            "timestamp": torch.tensor([0.0, 0.02], dtype=torch.float32),
            "frame_index": torch.tensor([0, 1], dtype=torch.int64),
            "episode_index": torch.tensor([3, 3], dtype=torch.int64),
            "observation.state": torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                dtype=torch.float32,
            ),
            "action": torch.tensor(
                [
                    [0.2, 0.3],
                    [0.5, 0.6],
                ],
                dtype=torch.float32,
            ),
        }


def test_build_state_batch() -> None:
    tabular_dataset = FakeTabularDataset()
    dataset = SimpleNamespace(hf_dataset=tabular_dataset)

    batch = build_state_batch(dataset)

    assert tabular_dataset.requested_format == "torch"
    assert tabular_dataset.requested_columns == [
        "timestamp",
        "frame_index",
        "episode_index",
        "observation.state",
        "action",
    ]

    assert batch.num_frames == 2
    assert batch.state_dimension == 3
    assert batch.action_dimension == 2

    assert torch.equal(
        batch.episode_indices,
        torch.tensor([3, 3], dtype=torch.int64),
    )
    assert torch.equal(
        batch.states[0],
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    )


def test_load_state_batch_requires_an_episode() -> None:
    with pytest.raises(
        ValueError,
        match="At least one episode must be selected",
    ):
        load_state_batch("organization/dataset", [])


def test_split_state_batch_by_episode() -> None:
    batch = StateBatch(
        timestamps=torch.tensor(
            [0.0, 0.02, 0.0, 0.02],
            dtype=torch.float32,
        ),
        frame_indices=torch.tensor(
            [0, 1, 0, 1],
            dtype=torch.int64,
        ),
        episode_indices=torch.tensor(
            [7, 7, 3, 3],
            dtype=torch.int64,
        ),
        states=torch.tensor(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
            ],
            dtype=torch.float32,
        ),
        actions=torch.tensor(
            [
                [1.1],
                [1.2],
                [1.3],
                [1.4],
            ],
            dtype=torch.float32,
        ),
    )

    episode_batches = split_state_batch_by_episode(batch)

    assert tuple(episode_batches) == (7, 3)

    episode_7 = episode_batches[7]
    assert episode_7.num_frames == 2
    assert torch.equal(
        episode_7.frame_indices,
        torch.tensor([0, 1], dtype=torch.int64),
    )
    assert torch.equal(
        episode_7.episode_indices,
        torch.tensor([7, 7], dtype=torch.int64),
    )
    assert torch.equal(
        episode_7.states,
        torch.tensor(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            dtype=torch.float32,
        ),
    )

    episode_3 = episode_batches[3]
    assert episode_3.num_frames == 2
    assert torch.equal(
        episode_3.episode_indices,
        torch.tensor([3, 3], dtype=torch.int64),
    )
    assert torch.equal(
        episode_3.actions,
        torch.tensor(
            [
                [1.3],
                [1.4],
            ],
            dtype=torch.float32,
        ),
    )
