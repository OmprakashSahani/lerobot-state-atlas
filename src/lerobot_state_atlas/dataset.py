from typing import Any

from lerobot_state_atlas.schema import DatasetSummary, FeatureSummary


def build_dataset_summary(metadata: Any) -> DatasetSummary:
    """Convert LeRobot metadata into a stable project summary."""
    info = metadata.info

    features = tuple(
        FeatureSummary.from_feature(name, feature)
        for name, feature in info.features.items()
    )

    return DatasetSummary(
        repo_id=metadata.repo_id,
        revision=str(metadata.revision),
        codebase_version=info.codebase_version,
        robot_type=info.robot_type,
        fps=float(info.fps),
        total_episodes=int(info.total_episodes),
        total_frames=int(info.total_frames),
        total_tasks=int(info.total_tasks),
        total_duration_seconds=float(info.total_frames / info.fps),
        features=features,
    )


def load_dataset_summary(repo_id: str) -> DatasetSummary:
    """Load metadata from the Hugging Face Hub and summarize the dataset."""
    from lerobot.datasets import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id)
    return build_dataset_summary(metadata)
