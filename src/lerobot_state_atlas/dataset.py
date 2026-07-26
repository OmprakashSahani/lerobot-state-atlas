from dataclasses import dataclass
import re
from typing import Any

from lerobot_state_atlas.schema import DatasetSummary, FeatureSummary


_FULL_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class DatasetRevision:
    """Requested dataset ref and its immutable Hugging Face commit."""

    requested: str
    resolved: str


def default_dataset_revision() -> str:
    """Return the revision LeRobot uses when no explicit ref is supplied."""
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

    return str(CODEBASE_VERSION)


def resolve_dataset_revision(
    repo_id: str,
    requested_revision: str | None = None,
    *,
    api: Any | None = None,
) -> DatasetRevision:
    """Resolve a dataset ref to a full immutable Hugging Face commit SHA."""
    requested = requested_revision or default_dataset_revision()

    if not requested:
        raise ValueError("Requested dataset revision must not be empty.")

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    try:
        info = api.dataset_info(repo_id, revision=requested)
    except Exception as error:
        raise ValueError(
            f"Unable to resolve dataset revision {requested!r} for {repo_id}."
        ) from error

    resolved = str(getattr(info, "sha", ""))

    if not _FULL_COMMIT_SHA.fullmatch(resolved):
        raise ValueError(
            "Resolved dataset revision must be a full 40-character "
            "hexadecimal Hugging Face commit SHA."
        )

    return DatasetRevision(
        requested=requested,
        resolved=resolved.lower(),
    )


def build_dataset_summary(
    metadata: Any,
    *,
    requested_revision: str,
    resolved_revision: str,
) -> DatasetSummary:
    """Convert LeRobot metadata into a stable project summary."""
    info = metadata.info

    if not _FULL_COMMIT_SHA.fullmatch(resolved_revision):
        raise ValueError(
            "Resolved dataset revision must be a full 40-character "
            "hexadecimal commit SHA."
        )

    metadata_revision = str(metadata.revision).lower()

    if metadata_revision != resolved_revision.lower():
        raise ValueError(
            "LeRobot metadata was not loaded from the resolved dataset revision."
        )

    features = tuple(
        FeatureSummary.from_feature(name, feature)
        for name, feature in info.features.items()
    )

    return DatasetSummary(
        repo_id=metadata.repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision.lower(),
        lerobot_codebase_version=info.codebase_version,
        robot_type=info.robot_type,
        fps=float(info.fps),
        total_episodes=int(info.total_episodes),
        total_frames=int(info.total_frames),
        total_tasks=int(info.total_tasks),
        total_duration_seconds=float(info.total_frames / info.fps),
        features=features,
    )


def load_dataset_summary(
    repo_id: str,
    *,
    requested_revision: str | None = None,
    resolved_revision: str | None = None,
) -> DatasetSummary:
    """Resolve and load immutable dataset metadata from the Hugging Face Hub."""
    from lerobot.datasets import LeRobotDatasetMetadata

    revision = (
        resolve_dataset_revision(repo_id, requested_revision)
        if resolved_revision is None
        else DatasetRevision(
            requested=requested_revision or default_dataset_revision(),
            resolved=resolved_revision,
        )
    )
    metadata = LeRobotDatasetMetadata(repo_id, revision=revision.resolved)
    return build_dataset_summary(
        metadata,
        requested_revision=revision.requested,
        resolved_revision=revision.resolved,
    )
