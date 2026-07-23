from lerobot_state_atlas.schema import DatasetSummary, FeatureSummary


def test_feature_summary_from_feature_with_component_names() -> None:
    feature = {
        "dtype": "float32",
        "shape": (3,),
        "names": ["joint_1", "joint_2", "joint_3"],
    }

    summary = FeatureSummary.from_feature("observation.state", feature)

    assert summary == FeatureSummary(
        name="observation.state",
        dtype="float32",
        shape=(3,),
        component_names=("joint_1", "joint_2", "joint_3"),
    )


def test_feature_summary_from_feature_without_component_names() -> None:
    feature = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }

    summary = FeatureSummary.from_feature("episode_index", feature)

    assert summary.component_names is None


def test_dataset_summary_is_immutable() -> None:
    summary = DatasetSummary(
        repo_id="organization/dataset",
        revision="v3.0",
        codebase_version="v3.0",
        robot_type="test_robot",
        fps=50.0,
        total_episodes=10,
        total_frames=500,
        total_tasks=1,
        total_duration_seconds=10.0,
        features=(),
    )

    assert summary.repo_id == "organization/dataset"
    assert summary.total_duration_seconds == 10.0
