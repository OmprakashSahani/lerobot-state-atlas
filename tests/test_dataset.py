from types import SimpleNamespace

from lerobot_state_atlas.dataset import build_dataset_summary
from lerobot_state_atlas.schema import FeatureSummary


def test_build_dataset_summary() -> None:
    metadata = SimpleNamespace(
        repo_id="DreamMachines/example",
        revision="v3.0",
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

    summary = build_dataset_summary(metadata)

    assert summary.repo_id == "DreamMachines/example"
    assert summary.revision == "v3.0"
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
