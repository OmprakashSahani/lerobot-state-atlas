from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_state_atlas.dataset import (
    build_dataset_summary,
    load_dataset_summary,
    resolve_dataset_revision,
)
from lerobot_state_atlas.schema import FeatureSummary


class HashableNamespace(SimpleNamespace):
    __hash__ = object.__hash__


def test_build_dataset_summary() -> None:
    metadata = SimpleNamespace(
        repo_id="DreamMachines/example",
        revision="a" * 40,
        info=SimpleNamespace(
            codebase_version="v3.0",
            robot_type="bi_dk1_follower",
            fps=50,
            total_episodes=2,
            total_frames=750,
            total_tasks=1,
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["joint"] * 14,
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": None,
                },
            },
        ),
    )

    summary = build_dataset_summary(
        metadata,
        requested_revision="v3.0",
        resolved_revision="a" * 40,
    )

    assert summary.repo_id == "DreamMachines/example"
    assert summary.requested_revision == "v3.0"
    assert summary.resolved_revision == "a" * 40
    assert summary.lerobot_codebase_version == "v3.0"
    assert summary.robot_type == "bi_dk1_follower"
    assert summary.fps == 50.0
    assert summary.total_episodes == 2
    assert summary.total_frames == 750
    assert summary.total_duration_seconds == 15.0

    assert summary.features[0] == FeatureSummary(
        name="observation.state",
        dtype="float32",
        shape=(14,),
        component_names=("joint",) * 14,
    )
    assert summary.features[1].component_names is None


def _cached_revision(
    commit: str,
    snapshot_path: Path,
    *,
    refs: frozenset[str] = frozenset(),
) -> SimpleNamespace:
    return HashableNamespace(
        commit_hash=commit,
        refs=refs,
        snapshot_path=snapshot_path,
    )


def _cached_repo(
    repo_id: str,
    *revisions: SimpleNamespace,
    repo_type: str = "dataset",
) -> SimpleNamespace:
    return HashableNamespace(
        repo_id=repo_id,
        repo_type=repo_type,
        revisions=frozenset(revisions),
    )


def _failing_api() -> SimpleNamespace:
    def fail(repo_id: str, revision: str) -> None:
        raise RuntimeError(f"Unknown revision: {revision}")

    return SimpleNamespace(dataset_info=fail)


def test_resolve_dataset_revision_preserves_requested_and_resolved_values(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir",
        lambda _: pytest.fail("Online success must not scan the cache."),
    )
    api = SimpleNamespace(
        dataset_info=lambda repo_id, revision: SimpleNamespace(sha="A" * 40)
    )

    revision = resolve_dataset_revision(
        "DreamMachines/example",
        "v3.0",
        api=api,
    )

    assert revision.requested == "v3.0"
    assert revision.resolved == "a" * 40


def test_resolve_dataset_revision_rejects_invalid_resolved_sha(monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir",
        lambda _: pytest.fail("Invalid online SHA must not scan the cache."),
    )
    api = SimpleNamespace(
        dataset_info=lambda repo_id, revision: SimpleNamespace(sha="v3.0")
    )

    with pytest.raises(ValueError, match="full 40-character hexadecimal"):
        resolve_dataset_revision(
            "DreamMachines/example",
            "v3.0",
            api=api,
        )


def test_resolve_dataset_revision_reports_unresolved_ref(monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir",
        lambda _: SimpleNamespace(repos=frozenset()),
    )
    with pytest.raises(ValueError, match="Unable to resolve dataset revision"):
        resolve_dataset_revision(
            "DreamMachines/example",
            "missing-ref",
            api=_failing_api(),
        )


def test_resolve_dataset_revision_falls_back_to_cached_named_ref(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    commit = "E973DF866C80F52884CC68355579043CAB828E78"
    cache = SimpleNamespace(
        repos=frozenset(
            {
                _cached_repo(
                    "DreamMachines/actuator_unboxing_4h_diverse",
                    _cached_revision(
                        commit,
                        snapshot,
                        refs=frozenset({"v3.0"}),
                    ),
                )
            }
        )
    )
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda _: cache)

    revision = resolve_dataset_revision(
        "DreamMachines/actuator_unboxing_4h_diverse",
        "v3.0",
        api=_failing_api(),
    )

    assert revision.requested == "v3.0"
    assert revision.resolved == commit.lower()


def test_resolve_dataset_revision_falls_back_to_exact_cached_commit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    commit = "e973df866c80f52884cc68355579043cab828e78"
    cache = SimpleNamespace(
        repos=frozenset(
            {
                _cached_repo(
                    "DreamMachines/actuator_unboxing_4h_diverse",
                    _cached_revision(commit, snapshot),
                )
            }
        )
    )
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda _: cache)

    revision = resolve_dataset_revision(
        "DreamMachines/actuator_unboxing_4h_diverse",
        commit,
        api=_failing_api(),
    )

    assert revision.resolved == commit


def test_resolve_dataset_revision_rejects_ambiguous_cached_matches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_snapshot = tmp_path / "first"
    second_snapshot = tmp_path / "second"
    first_snapshot.mkdir()
    second_snapshot.mkdir()
    cache = SimpleNamespace(
        repos=frozenset(
            {
                _cached_repo(
                    "DreamMachines/example",
                    _cached_revision(
                        "a" * 40,
                        first_snapshot,
                        refs=frozenset({"v3.0"}),
                    ),
                    _cached_revision(
                        "b" * 40,
                        second_snapshot,
                        refs=frozenset({"v3.0"}),
                    ),
                )
            }
        )
    )
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda _: cache)

    with pytest.raises(ValueError, match="Unable to resolve dataset revision"):
        resolve_dataset_revision(
            "DreamMachines/example",
            "v3.0",
            api=_failing_api(),
        )


def test_resolve_dataset_revision_rejects_missing_cached_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache = SimpleNamespace(
        repos=frozenset(
            {
                _cached_repo(
                    "DreamMachines/example",
                    _cached_revision(
                        "a" * 40,
                        tmp_path / "missing",
                        refs=frozenset({"v3.0"}),
                    ),
                )
            }
        )
    )
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda _: cache)

    with pytest.raises(ValueError, match="Unable to resolve dataset revision"):
        resolve_dataset_revision(
            "DreamMachines/example",
            "v3.0",
            api=_failing_api(),
        )


def test_load_dataset_summary_uses_resolved_revision(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeMetadata:
        def __init__(self, repo_id: str, revision: str) -> None:
            calls.append((repo_id, revision))
            self.repo_id = repo_id
            self.revision = revision
            self.info = SimpleNamespace(
                codebase_version="v3.0",
                robot_type="robot",
                fps=50,
                total_episodes=1,
                total_frames=2,
                total_tasks=1,
                features={},
            )

    monkeypatch.setattr(
        "lerobot.datasets.LeRobotDatasetMetadata",
        FakeMetadata,
    )

    summary = load_dataset_summary(
        "DreamMachines/example",
        requested_revision="v3.0",
        resolved_revision="a" * 40,
    )

    assert calls == [("DreamMachines/example", "a" * 40)]
    assert summary.requested_revision == "v3.0"
    assert summary.resolved_revision == "a" * 40
