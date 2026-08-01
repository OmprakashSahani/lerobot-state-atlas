from types import SimpleNamespace

import pytest
import torch

from lerobot_state_atlas.state import (
    build_coverage_state_batch,
    build_state_batch,
    iter_coverage_state_batches,
    iter_state_batches,
    load_coverage_state_batch,
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


def test_build_coverage_state_batch_materializes_only_required_columns() -> None:
    tabular_dataset = FakeTabularDataset()
    dataset = SimpleNamespace(hf_dataset=tabular_dataset)

    batch = build_coverage_state_batch(dataset)

    assert tabular_dataset.requested_format == "torch"
    assert tabular_dataset.requested_columns == [
        "episode_index",
        "observation.state",
    ]
    assert batch.num_frames == 2
    assert batch.state_dimension == 3
    assert torch.equal(
        batch.episode_indices,
        torch.tensor([3, 3], dtype=torch.int64),
    )
    assert torch.equal(
        batch.states,
        torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        ),
    )
    assert not hasattr(batch, "timestamps")
    assert not hasattr(batch, "frame_indices")
    assert not hasattr(batch, "actions")


def test_load_state_batch_requires_an_episode() -> None:
    with pytest.raises(
        ValueError,
        match="At least one episode must be selected",
    ):
        load_state_batch("organization/dataset", [])


def test_load_state_batch_uses_resolved_revision(monkeypatch) -> None:
    calls: list[tuple[str, list[int], str | None, bool]] = []

    class FakeDataset:
        def __init__(
            self,
            repo_id: str,
            *,
            episodes: list[int],
            revision: str | None,
            download_videos: bool,
        ) -> None:
            calls.append((repo_id, episodes, revision, download_videos))
            self.hf_dataset = FakeTabularDataset()

    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", FakeDataset)

    load_state_batch(
        "organization/dataset",
        [0],
        revision="a" * 40,
    )

    assert calls == [
        ("organization/dataset", [0], "a" * 40, False),
    ]


def test_load_coverage_state_batch_forwards_episodes_and_revision(monkeypatch) -> None:
    calls: list[tuple[str, list[int], str | None, bool]] = []

    class FakeDataset:
        def __init__(
            self,
            repo_id: str,
            *,
            episodes: list[int],
            revision: str | None,
            download_videos: bool,
        ) -> None:
            calls.append((repo_id, episodes, revision, download_videos))
            self.hf_dataset = FakeTabularDataset()

    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", FakeDataset)

    batch = load_coverage_state_batch(
        "organization/dataset",
        [7, 2],
        revision="b" * 40,
    )

    assert calls == [
        ("organization/dataset", [7, 2], "b" * 40, False),
    ]
    assert batch.num_frames == 2


def test_iter_state_batches_loads_bounded_episode_chunks(
    monkeypatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_load_state_batch(
        repo_id: str,
        episodes: tuple[int, ...],
    ) -> SimpleNamespace:
        normalized_episodes = tuple(episodes)
        calls.append((repo_id, normalized_episodes))
        return SimpleNamespace(episodes=normalized_episodes)

    monkeypatch.setattr(
        "lerobot_state_atlas.state.load_state_batch",
        fake_load_state_batch,
    )

    batches = tuple(
        iter_state_batches(
            "organization/dataset",
            [2, 5, 8, 11, 14],
            episode_batch_size=2,
        )
    )

    assert calls == [
        ("organization/dataset", (2, 5)),
        ("organization/dataset", (8, 11)),
        ("organization/dataset", (14,)),
    ]
    assert [batch.episodes for batch in batches] == [
        (2, 5),
        (8, 11),
        (14,),
    ]


def test_iter_coverage_state_batches_loads_bounded_chunks_with_revision(
    monkeypatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], str]] = []

    def fake_load_coverage_state_batch(
        repo_id: str,
        episodes: tuple[int, ...],
        *,
        revision: str,
    ) -> SimpleNamespace:
        normalized_episodes = tuple(episodes)
        calls.append((repo_id, normalized_episodes, revision))
        return SimpleNamespace(episodes=normalized_episodes)

    monkeypatch.setattr(
        "lerobot_state_atlas.state.load_coverage_state_batch",
        fake_load_coverage_state_batch,
    )

    batches = tuple(
        iter_coverage_state_batches(
            "organization/dataset",
            [4, 8, 12],
            episode_batch_size=2,
            revision="c" * 40,
        )
    )

    assert calls == [
        ("organization/dataset", (4, 8), "c" * 40),
        ("organization/dataset", (12,), "c" * 40),
    ]
    assert [batch.episodes for batch in batches] == [(4, 8), (12,)]


def test_iter_state_batches_requires_an_episode() -> None:
    with pytest.raises(
        ValueError,
        match="At least one episode must be selected",
    ):
        tuple(
            iter_state_batches(
                "organization/dataset",
                [],
                episode_batch_size=4,
            )
        )


def test_iter_state_batches_threads_resolved_revision(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...], str]] = []

    def fake_load_state_batch(
        repo_id: str,
        episodes: tuple[int, ...],
        *,
        revision: str,
    ) -> SimpleNamespace:
        calls.append((repo_id, tuple(episodes), revision))
        return SimpleNamespace(episodes=tuple(episodes))

    monkeypatch.setattr(
        "lerobot_state_atlas.state.load_state_batch",
        fake_load_state_batch,
    )

    tuple(
        iter_state_batches(
            "organization/dataset",
            [0, 1],
            episode_batch_size=1,
            revision="a" * 40,
        )
    )

    assert calls == [
        ("organization/dataset", (0,), "a" * 40),
        ("organization/dataset", (1,), "a" * 40),
    ]


@pytest.mark.parametrize("episode_batch_size", [0, -1])
def test_iter_state_batches_rejects_invalid_batch_size(
    episode_batch_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Episode batch size must be greater than zero",
    ):
        tuple(
            iter_state_batches(
                "organization/dataset",
                [0, 1],
                episode_batch_size=episode_batch_size,
            )
        )
