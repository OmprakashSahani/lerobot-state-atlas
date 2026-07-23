from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSummary:
    """Summary of one feature in a LeRobot dataset."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    component_names: tuple[str, ...] | None

    @classmethod
    def from_feature(cls, name: str, feature: dict[str, Any]) -> "FeatureSummary":
        """Create a feature summary from LeRobot metadata."""
        component_names = feature.get("names")

        return cls(
            name=name,
            dtype=str(feature["dtype"]),
            shape=tuple(feature["shape"]),
            component_names=(
                tuple(str(component) for component in component_names)
                if component_names is not None
                else None
            ),
        )


@dataclass(frozen=True)
class DatasetSummary:
    """High-level summary of a LeRobot dataset."""

    repo_id: str
    revision: str
    codebase_version: str
    robot_type: str | None
    fps: float
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_duration_seconds: float
    features: tuple[FeatureSummary, ...]
