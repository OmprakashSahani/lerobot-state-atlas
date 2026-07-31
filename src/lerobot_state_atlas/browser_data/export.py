"""Deterministic, atomic browser-data v1 exporter."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any
from uuid import uuid4

import torch

from lerobot_state_atlas.aggregation import aggregate_workspace_coverages
from lerobot_state_atlas.browser_data.models import BrowserDataExport
from lerobot_state_atlas.browser_data.schema import (
    COVERAGE_FILENAME,
    EPISODE_VIDEO_FILENAME,
    MANIFEST_FILENAME,
    SCHEMA_MAJOR,
    SCHEMA_MINOR,
    SCHEMA_NAME,
    TRAJECTORY_FILENAME,
)
from lerobot_state_atlas.browser_data.serialize import (
    deterministic_json_bytes,
    sha256_bytes,
    sha256_file,
)
from lerobot_state_atlas.browser_data.validate import validate_browser_data
from lerobot_state_atlas.dataset import (
    load_dataset_summary,
    resolve_dataset_revision,
)
from lerobot_state_atlas.schema import DatasetSummary
from lerobot_state_atlas.orientation import (
    rotation_matrices_to_quaternions_xyzw,
)
from lerobot_state_atlas.state import load_state_batch
from lerobot_state_atlas.trajectory import (
    ToolTrajectory,
    build_trlc_dk1_gripper_component_name,
    build_trlc_dk1_joint_component_map,
    compute_tool_trajectory,
)
from lerobot_state_atlas.transforms import RigidTransform, transform_tool_trajectory
from lerobot_state_atlas.urdf import RobotModel, load_robot_model


_SPACING_DISCLOSURE = (
    "The 0.8 metre default arm spacing is provisional, configurable, "
    "and is not calibrated physical geometry."
)
_FULL_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class SourceProvenance:
    """Truthful repository source state at export time."""

    repository_head_commit: str
    working_tree_dirty: bool


def _schema_reference() -> dict[str, object]:
    return {
        "name": SCHEMA_NAME,
        "major": SCHEMA_MAJOR,
        "minor": SCHEMA_MINOR,
    }


def _package_version(name: str, fallback: str = "unknown") -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _git_source_provenance(repository_path: Path) -> SourceProvenance:
    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Unable to determine the repository source state.") from error

    commit = head_result.stdout.strip()
    if not _FULL_COMMIT_SHA.fullmatch(commit):
        raise ValueError(
            "Repository HEAD is not a full 40-character hexadecimal commit hash."
        )

    return SourceProvenance(
        repository_head_commit=commit.lower(),
        working_tree_dirty=bool(status_result.stdout.strip()),
    )


def _state_component_names(summary: DatasetSummary) -> tuple[str, ...]:
    for feature in summary.features:
        if feature.name == "observation.state":
            if feature.component_names is None:
                raise ValueError("observation.state does not define component names.")
            return feature.component_names
    raise ValueError("Dataset does not define observation.state.")


def _arm_transforms(arm_spacing: float) -> dict[str, RigidTransform]:
    half_spacing = arm_spacing / 2.0
    return {
        "left": RigidTransform(translation_xyz=(0.0, half_spacing, 0.0)),
        "right": RigidTransform(translation_xyz=(0.0, -half_spacing, 0.0)),
    }


def _coverage_payload(coverages: Sequence[Any]) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    for coverage in sorted(coverages, key=lambda value: value.arm):
        episode_ids: list[int] = []
        offsets = [0]
        for voxel_episode_ids in coverage.episode_ids_by_voxel:
            episode_ids.extend(int(value) for value in voxel_episode_ids)
            offsets.append(len(episode_ids))

        visit_counts = [int(value) for value in coverage.visit_counts.tolist()]
        episode_counts = [int(value) for value in coverage.episode_counts.tolist()]
        voxel_indices = [
            [int(component) for component in index]
            for index in coverage.voxel_indices.tolist()
        ]
        arms.append(
            {
                "arm": coverage.arm,
                "toolLink": coverage.link_name,
                "voxelIndices": voxel_indices,
                "visitCounts": visit_counts,
                "episodeCounts": episode_counts,
                "episodeIdOffsets": offsets,
                "episodeIds": episode_ids,
                "statistics": {
                    "voxelEntryCount": len(voxel_indices),
                    "minimumVisitCount": min(visit_counts),
                    "maximumVisitCount": max(visit_counts),
                    "minimumEpisodeCount": min(episode_counts),
                    "maximumEpisodeCount": max(episode_counts),
                },
            }
        )

    return {"schema": _schema_reference(), "arms": arms}


def _trajectory_payload(
    *,
    trajectories: Mapping[str, ToolTrajectory],
    episode_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    timestamps: torch.Tensor,
    selected_episodes: Sequence[int],
) -> dict[str, Any]:
    normalized_episode_indices = episode_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    normalized_frame_indices = frame_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    normalized_timestamps = timestamps.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    arm_positions = {
        arm: trajectory.positions.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        for arm, trajectory in trajectories.items()
    }
    arm_orientations = {
        arm: rotation_matrices_to_quaternions_xyzw(
            trajectory.rotation_matrices,
            normalized_episode_indices,
        )
        for arm, trajectory in trajectories.items()
    }
    arm_gripper_values: dict[str, torch.Tensor] = {}
    for arm, trajectory in trajectories.items():
        if trajectory.recorded_gripper_values is None:
            raise ValueError(
                f"{arm.capitalize()} trajectory is missing recorded gripper values."
            )
        arm_gripper_values[arm] = trajectory.recorded_gripper_values.detach().to(
            device="cpu",
            dtype=torch.float64,
        )

    episodes: list[dict[str, Any]] = []
    for episode_id in sorted(selected_episodes):
        mask = normalized_episode_indices == episode_id
        if not mask.any().item():
            raise ValueError(
                f"Trajectory episode {episode_id} contains no dataset frames."
            )
        episodes.append(
            {
                "episodeId": episode_id,
                "frameIndices": normalized_frame_indices[mask].tolist(),
                "timestampsSeconds": normalized_timestamps[mask].tolist(),
                "leftPositionsXyz": arm_positions["left"][mask].tolist(),
                "rightPositionsXyz": arm_positions["right"][mask].tolist(),
                "leftOrientationsXyzw": arm_orientations["left"][mask].tolist(),
                "rightOrientationsXyzw": arm_orientations["right"][mask].tolist(),
                "leftRecordedGripperValues": arm_gripper_values["left"][mask].tolist(),
                "rightRecordedGripperValues": arm_gripper_values["right"][
                    mask
                ].tolist(),
            }
        )

    return {"schema": _schema_reference(), "episodes": episodes}


def _scene_bounds(coverage_payload: Mapping[str, Any], voxel_size: float) -> dict:
    centers: list[list[float]] = []
    for arm in coverage_payload["arms"]:
        centers.extend(
            [
                [
                    index[0] * voxel_size + voxel_size / 2.0,
                    index[1] * voxel_size + voxel_size / 2.0,
                    index[2] * voxel_size + voxel_size / 2.0,
                ]
                for index in arm["voxelIndices"]
            ]
        )
    if not centers:
        raise ValueError("Coverage payload contains no voxel centers.")
    return {
        "minimumXyz": [min(point[axis] for point in centers) for axis in range(3)],
        "maximumXyz": [max(point[axis] for point in centers) for axis in range(3)],
    }


def _payload_reference(
    *,
    kind: str,
    filename: str,
    required: bool,
    content: bytes,
) -> dict[str, object]:
    return {
        "kind": kind,
        "filename": filename,
        "required": required,
        "encoding": "json",
        "byteSize": len(content),
        "sha256": sha256_bytes(content),
    }


def build_browser_data_documents(
    *,
    bundle_id: str,
    summary: DatasetSummary,
    selected_episodes: Sequence[int],
    model: RobotModel,
    urdf_sha256: str,
    urdf_upstream_identity: str,
    voxel_size: float,
    arm_spacing: float,
    coverages: Sequence[Any],
    source_provenance: SourceProvenance,
    trajectory_payload: Mapping[str, Any] | None = None,
    episode_video_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes, bytes | None, bytes | None]:
    """Build deterministic validated document bytes from computed domain models."""
    normalized_episodes = sorted(set(int(value) for value in selected_episodes))
    if not bundle_id:
        raise ValueError("Bundle ID must not be empty.")
    if normalized_episodes != list(selected_episodes):
        raise ValueError("Selected episodes must be sorted and distinct.")
    if len(coverages) != 2:
        raise ValueError("Exactly two arm coverages are required.")

    coverage = _coverage_payload(coverages)
    coverage_bytes = deterministic_json_bytes(coverage)
    trajectory_bytes = (
        None
        if trajectory_payload is None
        else deterministic_json_bytes(trajectory_payload)
    )
    episode_video_bytes = (
        None
        if episode_video_payload is None
        else deterministic_json_bytes(episode_video_payload)
    )

    tool_point_visits = sum(sum(arm["visitCounts"]) for arm in coverage["arms"])
    arm_voxel_entries = sum(len(arm["voxelIndices"]) for arm in coverage["arms"])
    unique_grid_cells = len(
        {tuple(index) for arm in coverage["arms"] for index in arm["voxelIndices"]}
    )
    transforms = _arm_transforms(arm_spacing)
    payloads = [
        _payload_reference(
            kind="coverage",
            filename=COVERAGE_FILENAME,
            required=True,
            content=coverage_bytes,
        )
    ]
    if trajectory_bytes is not None:
        payloads.append(
            _payload_reference(
                kind="trajectories",
                filename=TRAJECTORY_FILENAME,
                required=False,
                content=trajectory_bytes,
            )
        )
    if episode_video_bytes is not None:
        payloads.append(
            _payload_reference(
                kind="episode-videos",
                filename=EPISODE_VIDEO_FILENAME,
                required=False,
                content=episode_video_bytes,
            )
        )

    manifest = {
        "schema": _schema_reference(),
        "bundleId": bundle_id,
        "exporter": {
            "packageVersion": _package_version(
                "lerobot-state-atlas",
                fallback="0.1.0",
            ),
            "repositoryHeadCommit": source_provenance.repository_head_commit,
            "workingTreeDirty": source_provenance.working_tree_dirty,
            "sourceDescription": (
                "repositoryHeadCommit identifies Git HEAD; uncommitted tracked, "
                "staged, or untracked source changes are present."
                if source_provenance.working_tree_dirty
                else "repositoryHeadCommit identifies the complete clean "
                "repository source state."
            ),
            "determinism": (
                "Coverage, trajectory, and episode-video metadata payloads are "
                "deterministic for identical "
                "pinned dataset, URDF, parameters, and exporter source. Manifest "
                "provenance may reflect repository state. No generation timestamp "
                "is stored."
            ),
        },
        "dataset": {
            "repositoryId": summary.repo_id,
            "requestedRevision": summary.requested_revision,
            "resolvedRevision": summary.resolved_revision,
            "lerobotCodebaseVersion": summary.lerobot_codebase_version,
            "lerobotPackageVersion": _package_version(
                "lerobot",
                fallback="unknown",
            ),
            "robotType": summary.robot_type,
            "fps": summary.fps,
            "episodeIds": normalized_episodes,
            "episodeCount": len(normalized_episodes),
            "datasetFrameCount": sum(
                int(coverage_value.num_points) for coverage_value in coverages
            )
            // 2,
        },
        "robot": {
            "modelName": model.name,
            "rootLink": model.root_link,
            "toolLinks": {"left": "tool0", "right": "tool0"},
            "urdfSha256": urdf_sha256,
            "urdfUpstreamIdentity": urdf_upstream_identity,
        },
        "coverage": {
            "voxelSize": voxel_size,
            "voxelOrigin": [0.0, 0.0, 0.0],
            "canonicalTransforms": {
                arm: {
                    "translationXyz": list(transform.translation_xyz),
                    "rotationRpy": list(transform.rotation_rpy),
                }
                for arm, transform in sorted(transforms.items())
            },
            "armSpacing": arm_spacing,
            "spacingCalibrated": False,
            "spacingDisclosure": _SPACING_DISCLOSURE,
        },
        "coordinates": {
            "lengthUnit": "metre",
            "angleUnit": "radian",
            "handedness": "right-handed",
            "positionFrame": "canonical-shared-world",
            "voxelIndexFormula": ("floor((position - voxelOrigin) / voxelSize)"),
            "rotationConvention": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "pointTransform": "p_world = R @ p_local + translation",
        },
        "totals": {
            "datasetFrameCount": tool_point_visits // 2,
            "toolPointVisitCount": tool_point_visits,
            "armVoxelEntryCount": arm_voxel_entries,
            "uniqueSharedGridCellCount": unique_grid_cells,
        },
        "sceneBounds": _scene_bounds(coverage, voxel_size),
        "payloads": payloads,
    }
    if trajectory_payload is not None:
        episodes = trajectory_payload.get("episodes", [])
        orientation_available = bool(episodes) and all(
            "leftOrientationsXyzw" in episode and "rightOrientationsXyzw" in episode
            for episode in episodes
        )
        gripper_available = bool(episodes) and all(
            "leftRecordedGripperValues" in episode
            and "rightRecordedGripperValues" in episode
            for episode in episodes
        )
        manifest["trajectoryState"] = {
            "orientation": {
                "available": orientation_available,
                "representation": "unit-quaternion",
                "componentOrder": ["x", "y", "z", "w"],
                "frame": "canonical-shared-world",
                "samplePolicy": "recorded-sample",
            },
            "gripper": {
                "available": gripper_available,
                "leftSourceComponent": "left_gripper.pos",
                "rightSourceComponent": "right_gripper.pos",
                "valueSemantics": "raw-device-specific-unproven",
                "physicalJawWidthCalibrated": False,
                "polarityEstablished": False,
                "visualizationGeometryCalibrated": False,
            },
        }
    return (
        manifest,
        coverage_bytes,
        trajectory_bytes,
        episode_video_bytes,
    )


def _episode_video_media_filenames(content: bytes) -> tuple[str, ...]:
    try:
        payload = json.loads(content)
        return tuple(
            str(source["filename"])
            for episode in payload["episodes"]
            for source in episode["videos"]
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Episode-video payload does not contain valid media references."
        ) from error


def _bundle_media_target(root: Path, filename: str) -> Path:
    relative = PurePosixPath(filename)
    if (
        "\\" in filename
        or "://" in filename
        or filename.startswith("//")
        or re.match(r"^[A-Za-z]:", filename)
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != filename
    ):
        raise ValueError(
            "Episode-video media filenames must be safe bundle-relative POSIX paths."
        )
    return root.joinpath(*relative.parts)


def _write_bundle(
    destination: Path,
    manifest: Mapping[str, Any],
    coverage_bytes: bytes,
    trajectory_bytes: bytes | None,
    episode_video_bytes: bytes | None = None,
    episode_video_media: Mapping[str, str | Path] | None = None,
) -> BrowserDataExport:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    backup_path: Path | None = None

    try:
        (temporary_path / COVERAGE_FILENAME).write_bytes(coverage_bytes)
        if trajectory_bytes is not None:
            (temporary_path / TRAJECTORY_FILENAME).write_bytes(trajectory_bytes)

        media_sources = dict(episode_video_media or {})
        if episode_video_bytes is None:
            if media_sources:
                raise ValueError(
                    "Episode-video media was provided without a metadata payload."
                )
        else:
            (temporary_path / EPISODE_VIDEO_FILENAME).write_bytes(episode_video_bytes)
            declared_media = _episode_video_media_filenames(episode_video_bytes)
            if len(declared_media) != len(set(declared_media)):
                raise ValueError("Episode-video media filenames must be unique.")
            if set(media_sources) != set(declared_media):
                raise ValueError(
                    "Episode-video media files must exactly match the metadata payload."
                )

            for filename in sorted(declared_media):
                source = Path(media_sources[filename]).resolve()
                if not source.is_file():
                    raise ValueError(
                        f"Episode-video source file does not exist: {source}."
                    )
                target = _bundle_media_target(temporary_path, filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

        (temporary_path / MANIFEST_FILENAME).write_bytes(
            deterministic_json_bytes(manifest)
        )
        validate_browser_data(temporary_path)

        if destination.exists():
            backup_path = destination.with_name(
                f".{destination.name}.previous-{uuid4().hex}"
            )
            os.replace(destination, backup_path)
        try:
            os.replace(temporary_path, destination)
        except BaseException:
            if backup_path is not None:
                os.replace(backup_path, destination)
                backup_path = None
            raise
        if backup_path is not None:
            shutil.rmtree(backup_path)
            backup_path = None
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        if backup_path is not None and not destination.exists():
            os.replace(backup_path, destination)
        raise

    totals = manifest["totals"]
    return BrowserDataExport(
        output_path=destination,
        bundle_id=str(manifest["bundleId"]),
        dataset_frame_count=int(totals["datasetFrameCount"]),
        tool_point_visit_count=int(totals["toolPointVisitCount"]),
        arm_voxel_entry_count=int(totals["armVoxelEntryCount"]),
        unique_shared_grid_cell_count=int(totals["uniqueSharedGridCellCount"]),
        payload_byte_count=sum(
            int(payload["byteSize"]) for payload in manifest["payloads"]
        ),
    )


def export_browser_data(
    repo_id: str,
    *,
    urdf_path: str | Path,
    episodes: Sequence[int],
    trajectory_episodes: Sequence[int],
    episode_batch_size: int,
    voxel_size: float,
    arm_spacing: float,
    output_path: str | Path,
    bundle_id: str,
    urdf_upstream_identity: str,
    repository_path: str | Path,
    dataset_revision: str | None = None,
    episode_video_payload: Mapping[str, Any] | None = None,
    episode_video_media: Mapping[str, str | Path] | None = None,
) -> BrowserDataExport:
    """Generate, validate, and atomically install a browser-data bundle."""
    normalized_episodes = tuple(episodes)
    normalized_trajectory_episodes = tuple(trajectory_episodes)
    if not normalized_episodes:
        raise ValueError("At least one coverage episode must be selected.")
    if tuple(sorted(set(normalized_episodes))) != normalized_episodes:
        raise ValueError("Coverage episodes must be sorted and distinct.")
    if tuple(sorted(set(normalized_trajectory_episodes))) != (
        normalized_trajectory_episodes
    ):
        raise ValueError("Trajectory episodes must be sorted and distinct.")
    if not set(normalized_trajectory_episodes).issubset(normalized_episodes):
        raise ValueError("Trajectory episodes must be included in coverage episodes.")

    resolved_dataset_revision = resolve_dataset_revision(
        repo_id,
        dataset_revision,
    )
    summary = load_dataset_summary(
        repo_id,
        requested_revision=resolved_dataset_revision.requested,
        resolved_revision=resolved_dataset_revision.resolved,
    )
    if normalized_episodes[-1] >= summary.total_episodes:
        raise ValueError("Selected episode exceeds the dataset episode count.")
    component_names = _state_component_names(summary)
    model = load_robot_model(urdf_path)
    transforms = _arm_transforms(arm_spacing)
    aggregation = aggregate_workspace_coverages(
        repo_id,
        normalized_episodes,
        component_names=component_names,
        model=model,
        voxel_size=voxel_size,
        episode_batch_size=episode_batch_size,
        arm_transforms=transforms,
        revision=resolved_dataset_revision.resolved,
    )

    trajectory_document: Mapping[str, Any] | None = None
    if normalized_trajectory_episodes:
        batch = load_state_batch(
            repo_id,
            normalized_trajectory_episodes,
            revision=resolved_dataset_revision.resolved,
        )
        trajectories = {}
        for arm in ("left", "right"):
            local = compute_tool_trajectory(
                batch.states,
                component_names,
                model,
                build_trlc_dk1_joint_component_map(arm),
                arm=arm,
                episode_indices=batch.episode_indices,
                gripper_component_name=build_trlc_dk1_gripper_component_name(arm),
            )
            trajectories[arm] = transform_tool_trajectory(local, transforms[arm])
        trajectory_document = _trajectory_payload(
            trajectories=trajectories,
            episode_indices=batch.episode_indices,
            frame_indices=batch.frame_indices,
            timestamps=batch.timestamps,
            selected_episodes=normalized_trajectory_episodes,
        )

    (
        manifest,
        coverage_bytes,
        trajectory_bytes,
        episode_video_bytes,
    ) = build_browser_data_documents(
        bundle_id=bundle_id,
        summary=summary,
        selected_episodes=normalized_episodes,
        model=model,
        urdf_sha256=sha256_file(urdf_path),
        urdf_upstream_identity=urdf_upstream_identity,
        voxel_size=voxel_size,
        arm_spacing=arm_spacing,
        coverages=aggregation.coverages,
        source_provenance=_git_source_provenance(Path(repository_path)),
        trajectory_payload=trajectory_document,
        episode_video_payload=episode_video_payload,
    )
    return _write_bundle(
        Path(output_path),
        manifest,
        coverage_bytes,
        trajectory_bytes,
        episode_video_bytes,
        episode_video_media,
    )
