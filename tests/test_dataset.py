from types import SimpleNamespace

import pytest

from lerobot_state_atlas.dataset import (
    build_dataset_summary,
    load_dataset_summary,
    resolve_dataset_revision,
)
from lerobot_state_atlas.schema import FeatureSummary


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


def test_resolve_dataset_revision_preserves_requested_and_resolved_values() -> None:
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


def test_resolve_dataset_revision_rejects_invalid_resolved_sha() -> None:
    api = SimpleNamespace(
        dataset_info=lambda repo_id, revision: SimpleNamespace(sha="v3.0")
    )

    with pytest.raises(ValueError, match="full 40-character hexadecimal"):
        resolve_dataset_revision(
            "DreamMachines/example",
            "v3.0",
            api=api,
        )


def test_resolve_dataset_revision_reports_unresolved_ref() -> None:
    def fail(repo_id: str, revision: str) -> None:
        raise RuntimeError(f"Unknown revision: {revision}")

    with pytest.raises(ValueError, match="Unable to resolve dataset revision"):
        resolve_dataset_revision(
            "DreamMachines/example",
            "missing-ref",
            api=SimpleNamespace(dataset_info=fail),
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
