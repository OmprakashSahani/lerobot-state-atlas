from types import SimpleNamespace

import pytest
import torch

from lerobot_state_atlas.state import (
    build_state_batch,
    load_state_batch,
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
