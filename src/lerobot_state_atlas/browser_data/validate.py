"""Strict structural and semantic validation for browser-data v1 bundles."""

from collections.abc import Mapping
import json
from math import isfinite
from pathlib import Path, PurePath
import re
from typing import Any

from lerobot_state_atlas.browser_data.schema import (
    BOUNDS_FIELDS,
    COORDINATE_FIELDS,
    COVERAGE_ARM_FIELDS,
    COVERAGE_CONFIG_FIELDS,
    COVERAGE_PAYLOAD_FIELDS,
    COVERAGE_STATISTIC_FIELDS,
    DATASET_FIELDS,
    EXPORTER_FIELDS,
    MANIFEST_FIELDS,
    MANIFEST_FILENAME,
    PAYLOAD_REFERENCE_FIELDS,
    ROBOT_FIELDS,
    SCHEMA_FIELDS,
    SCHEMA_MAJOR,
    SCHEMA_NAME,
    TOTAL_FIELDS,
    TRAJECTORY_EPISODE_FIELDS,
    TRAJECTORY_PAYLOAD_FIELDS,
    TRANSFORM_FIELDS,
)
from lerobot_state_atlas.browser_data.serialize import sha256_file


class BrowserDataValidationError(ValueError):
    """Raised when a browser-data bundle violates the v1 contract."""


def _fail(message: str) -> None:
    raise BrowserDataValidationError(message)


def _object(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object.")

    actual = set(value)
    missing = fields - actual
    unsupported = actual - fields

    if missing:
        _fail(f"{label} is missing fields: {', '.join(sorted(missing))}.")

    if unsupported:
        _fail(f"{label} has unsupported fields: {', '.join(sorted(unsupported))}.")

    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a number.")

    normalized = float(value)

    if not isfinite(normalized):
        _fail(f"{label} must be finite.")

    if positive and normalized <= 0:
        _fail(f"{label} must be greater than zero.")

    return normalized


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer greater than or equal to {minimum}.")

    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string.")

    return value


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) for item in value)
    ):
        _fail(f"{label} must contain three numbers.")

    values = tuple(_number(item, label) for item in value)
    return (values[0], values[1], values[2])


def _schema(value: Any, label: str) -> None:
    schema = _object(value, SCHEMA_FIELDS, f"{label}.schema")

    if schema["name"] != SCHEMA_NAME:
        _fail(f"{label} uses an unsupported schema name.")

    if schema["major"] != SCHEMA_MAJOR:
        _fail(f"{label} uses an unsupported schema major version.")

    _integer(schema["minor"], f"{label}.schema.minor")


def _reject_absolute_paths(value: Any, label: str = "bundle") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_paths(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_paths(item, f"{label}[{index}]")
    elif isinstance(value, str) and PurePath(value).is_absolute():
        _fail(f"{label} must not contain an absolute filesystem path.")


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BrowserDataValidationError(f"Missing {label}: {path.name}.") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BrowserDataValidationError(f"Invalid {label}: {path.name}.") from error


def _validate_manifest(manifest: Any) -> Mapping[str, Any]:
    manifest = _object(manifest, MANIFEST_FIELDS, "manifest")
    _schema(manifest["schema"], "manifest")
    _string(manifest["bundleId"], "manifest.bundleId")

    exporter = _object(manifest["exporter"], EXPORTER_FIELDS, "manifest.exporter")
    _string(exporter["packageVersion"], "manifest.exporter.packageVersion")
    repository_head_commit = _string(
        exporter["repositoryHeadCommit"],
        "manifest.exporter.repositoryHeadCommit",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", repository_head_commit):
        _fail(
            "manifest.exporter.repositoryHeadCommit must be a full lowercase "
            "hexadecimal Git commit hash."
        )
    if not isinstance(exporter["workingTreeDirty"], bool):
        _fail("manifest.exporter.workingTreeDirty must be boolean.")
    _string(exporter["sourceDescription"], "manifest.exporter.sourceDescription")
    _string(exporter["determinism"], "manifest.exporter.determinism")

    dataset = _object(manifest["dataset"], DATASET_FIELDS, "manifest.dataset")
    _string(dataset["repositoryId"], "manifest.dataset.repositoryId")
    _string(dataset["requestedRevision"], "manifest.dataset.requestedRevision")
    resolved_revision = _string(
        dataset["resolvedRevision"],
        "manifest.dataset.resolvedRevision",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
        _fail(
            "manifest.dataset.resolvedRevision must be a full lowercase "
            "hexadecimal Hugging Face commit SHA."
        )
    _string(
        dataset["lerobotCodebaseVersion"],
        "manifest.dataset.lerobotCodebaseVersion",
    )
    _string(
        dataset["lerobotPackageVersion"],
        "manifest.dataset.lerobotPackageVersion",
    )
    if dataset["robotType"] is not None:
        _string(dataset["robotType"], "manifest.dataset.robotType")
    _number(dataset["fps"], "manifest.dataset.fps", positive=True)
    episode_ids = dataset["episodeIds"]
    if (
        not isinstance(episode_ids, list)
        or not episode_ids
        or episode_ids != sorted(set(episode_ids))
    ):
        _fail("manifest.dataset.episodeIds must be sorted, distinct, and non-empty.")
    for value in episode_ids:
        _integer(value, "manifest.dataset.episodeIds[]")
    if dataset["episodeCount"] != len(episode_ids):
        _fail("manifest.dataset.episodeCount does not match episodeIds.")
    _integer(
        dataset["datasetFrameCount"], "manifest.dataset.datasetFrameCount", minimum=1
    )

    robot = _object(manifest["robot"], ROBOT_FIELDS, "manifest.robot")
    _string(robot["modelName"], "manifest.robot.modelName")
    _string(robot["rootLink"], "manifest.robot.rootLink")
    tool_links = robot["toolLinks"]
    if set(tool_links) != {"left", "right"}:
        _fail("manifest.robot.toolLinks must define left and right.")
    _string(tool_links["left"], "manifest.robot.toolLinks.left")
    _string(tool_links["right"], "manifest.robot.toolLinks.right")
    if len(_string(robot["urdfSha256"], "manifest.robot.urdfSha256")) != 64:
        _fail("manifest.robot.urdfSha256 must be a SHA-256 digest.")
    _string(robot["urdfUpstreamIdentity"], "manifest.robot.urdfUpstreamIdentity")

    coverage = _object(
        manifest["coverage"],
        COVERAGE_CONFIG_FIELDS,
        "manifest.coverage",
    )
    _number(coverage["voxelSize"], "manifest.coverage.voxelSize", positive=True)
    _vector3(coverage["voxelOrigin"], "manifest.coverage.voxelOrigin")
    transforms = coverage["canonicalTransforms"]
    if not isinstance(transforms, dict) or set(transforms) != {"left", "right"}:
        _fail("manifest.coverage.canonicalTransforms must define left and right.")
    for arm in ("left", "right"):
        transform = _object(
            transforms[arm],
            TRANSFORM_FIELDS,
            f"manifest.coverage.canonicalTransforms.{arm}",
        )
        _vector3(transform["translationXyz"], f"{arm} translation")
        _vector3(transform["rotationRpy"], f"{arm} rotation")
    spacing = _number(
        coverage["armSpacing"],
        "manifest.coverage.armSpacing",
        positive=True,
    )
    if coverage["spacingCalibrated"] is not False:
        _fail("manifest.coverage.spacingCalibrated must be false.")
    _string(coverage["spacingDisclosure"], "manifest.coverage.spacingDisclosure")
    if transforms["left"]["translationXyz"] != [0.0, spacing / 2.0, 0.0]:
        _fail("Left canonical transform does not match arm spacing.")
    if transforms["right"]["translationXyz"] != [0.0, -spacing / 2.0, 0.0]:
        _fail("Right canonical transform does not match arm spacing.")

    coordinates = _object(
        manifest["coordinates"],
        COORDINATE_FIELDS,
        "manifest.coordinates",
    )
    expected_coordinates = {
        "lengthUnit": "metre",
        "angleUnit": "radian",
        "handedness": "right-handed",
        "positionFrame": "canonical-shared-world",
    }
    for key, expected in expected_coordinates.items():
        if coordinates[key] != expected:
            _fail(f"manifest.coordinates.{key} must be {expected!r}.")
    for key in ("voxelIndexFormula", "rotationConvention", "pointTransform"):
        _string(coordinates[key], f"manifest.coordinates.{key}")

    totals = _object(manifest["totals"], TOTAL_FIELDS, "manifest.totals")
    for key in TOTAL_FIELDS:
        _integer(totals[key], f"manifest.totals.{key}", minimum=1)
    if totals["datasetFrameCount"] != dataset["datasetFrameCount"]:
        _fail("Dataset-frame totals do not agree.")

    bounds = _object(manifest["sceneBounds"], BOUNDS_FIELDS, "manifest.sceneBounds")
    minimum = _vector3(bounds["minimumXyz"], "manifest.sceneBounds.minimumXyz")
    maximum = _vector3(bounds["maximumXyz"], "manifest.sceneBounds.maximumXyz")
    if any(low > high for low, high in zip(minimum, maximum, strict=True)):
        _fail("Scene bounds minimum must not exceed maximum.")

    payloads = manifest["payloads"]
    if not isinstance(payloads, list) or not payloads:
        _fail("manifest.payloads must be a non-empty array.")
    kinds: set[str] = set()
    filenames: set[str] = set()
    for index, value in enumerate(payloads):
        payload = _object(
            value,
            PAYLOAD_REFERENCE_FIELDS,
            f"manifest.payloads[{index}]",
        )
        kind = _string(payload["kind"], f"manifest.payloads[{index}].kind")
        filename = _string(
            payload["filename"],
            f"manifest.payloads[{index}].filename",
        )
        if kind not in {"coverage", "trajectories"}:
            _fail(f"Unsupported payload kind: {kind}.")
        if kind in kinds or filename in filenames:
            _fail("Payload kinds and filenames must be unique.")
        if PurePath(filename).name != filename:
            _fail("Payload filenames must not include directories.")
        if payload["encoding"] != "json":
            _fail("Schema v1.0 supports only json payload encoding.")
        if not isinstance(payload["required"], bool):
            _fail("Payload required must be boolean.")
        _integer(payload["byteSize"], "payload.byteSize", minimum=1)
        if len(_string(payload["sha256"], "payload.sha256")) != 64:
            _fail("Payload sha256 must be a SHA-256 digest.")
        kinds.add(kind)
        filenames.add(filename)
    if "coverage" not in kinds:
        _fail("A coverage payload is required.")

    _reject_absolute_paths(manifest)
    return manifest


def _validate_coverage(value: Any) -> dict[str, int]:
    payload = _object(value, COVERAGE_PAYLOAD_FIELDS, "coverage payload")
    _schema(payload["schema"], "coverage payload")
    arms = payload["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        _fail("coverage payload must contain exactly two arms.")

    total_visits = 0
    total_entries = 0
    grid_cells: set[tuple[int, int, int]] = set()
    found_arms: list[str] = []

    for arm_index, arm_value in enumerate(arms):
        label = f"coverage payload.arms[{arm_index}]"
        arm = _object(arm_value, COVERAGE_ARM_FIELDS, label)
        arm_id = _string(arm["arm"], f"{label}.arm")
        found_arms.append(arm_id)
        _string(arm["toolLink"], f"{label}.toolLink")
        indices = arm["voxelIndices"]
        visits = arm["visitCounts"]
        episode_counts = arm["episodeCounts"]
        offsets = arm["episodeIdOffsets"]
        episode_ids = arm["episodeIds"]
        if not isinstance(indices, list) or not indices:
            _fail(f"{label}.voxelIndices must be a non-empty array.")
        size = len(indices)
        if not all(isinstance(values, list) and len(values) == 3 for values in indices):
            _fail(f"{label}.voxelIndices rows must contain three integers.")
        normalized_indices: list[tuple[int, int, int]] = []
        for row in indices:
            normalized_indices.append(
                tuple(
                    _integer(value, f"{label}.voxelIndices", minimum=-(2**31))
                    for value in row
                )  # type: ignore[arg-type]
            )
        if normalized_indices != sorted(normalized_indices):
            _fail(f"{label}.voxelIndices must be sorted.")
        if len(set(normalized_indices)) != size:
            _fail(f"{label}.voxelIndices must be distinct.")
        if not all(
            isinstance(values, list)
            for values in (visits, episode_counts, offsets, episode_ids)
        ):
            _fail(f"{label} count and CSR fields must be arrays.")
        if len(visits) != size or len(episode_counts) != size:
            _fail(f"{label} count arrays must match voxel count.")
        normalized_visits = [
            _integer(value, f"{label}.visitCounts[]", minimum=1) for value in visits
        ]
        normalized_episode_counts = [
            _integer(value, f"{label}.episodeCounts[]", minimum=1)
            for value in episode_counts
        ]
        if len(offsets) != size + 1 or offsets[0] != 0:
            _fail(
                f"{label}.episodeIdOffsets must have voxel count + 1 entries and start at zero."
            )
        normalized_offsets = [
            _integer(value, f"{label}.episodeIdOffsets[]") for value in offsets
        ]
        if normalized_offsets != sorted(normalized_offsets):
            _fail(f"{label}.episodeIdOffsets must be monotonic.")
        if normalized_offsets[-1] != len(episode_ids):
            _fail(f"{label}.episodeIdOffsets must end at the episodeIds length.")
        normalized_episode_ids = [
            _integer(value, f"{label}.episodeIds[]") for value in episode_ids
        ]
        for index in range(size):
            voxel_episode_ids = normalized_episode_ids[
                normalized_offsets[index] : normalized_offsets[index + 1]
            ]
            if voxel_episode_ids != sorted(set(voxel_episode_ids)):
                _fail(
                    f"{label} episode IDs for each voxel must be sorted and distinct."
                )
            if len(voxel_episode_ids) != normalized_episode_counts[index]:
                _fail(f"{label} episode-count and CSR data disagree.")

        statistics = _object(
            arm["statistics"],
            COVERAGE_STATISTIC_FIELDS,
            f"{label}.statistics",
        )
        expected_statistics = {
            "voxelEntryCount": size,
            "minimumVisitCount": min(normalized_visits),
            "maximumVisitCount": max(normalized_visits),
            "minimumEpisodeCount": min(normalized_episode_counts),
            "maximumEpisodeCount": max(normalized_episode_counts),
        }
        if statistics != expected_statistics:
            _fail(f"{label}.statistics do not match the payload data.")

        total_visits += sum(normalized_visits)
        total_entries += size
        grid_cells.update(normalized_indices)

    if found_arms != ["left", "right"]:
        _fail("coverage payload arms must be sorted left, then right.")

    _reject_absolute_paths(payload)
    return {
        "toolPointVisitCount": total_visits,
        "armVoxelEntryCount": total_entries,
        "uniqueSharedGridCellCount": len(grid_cells),
    }


def _validate_trajectories(value: Any, selected_episodes: set[int]) -> None:
    payload = _object(value, TRAJECTORY_PAYLOAD_FIELDS, "trajectory payload")
    _schema(payload["schema"], "trajectory payload")
    episodes = payload["episodes"]
    if not isinstance(episodes, list) or not episodes:
        _fail("trajectory payload episodes must be a non-empty array.")
    episode_ids: list[int] = []
    for index, episode_value in enumerate(episodes):
        label = f"trajectory payload.episodes[{index}]"
        episode = _object(episode_value, TRAJECTORY_EPISODE_FIELDS, label)
        episode_id = _integer(episode["episodeId"], f"{label}.episodeId")
        episode_ids.append(episode_id)
        arrays = [
            episode["frameIndices"],
            episode["timestampsSeconds"],
            episode["leftPositionsXyz"],
            episode["rightPositionsXyz"],
        ]
        if any(not isinstance(array, list) for array in arrays):
            _fail(f"{label} trajectory fields must be arrays.")
        size = len(arrays[0])
        if size == 0 or any(len(array) != size for array in arrays[1:]):
            _fail(f"{label} trajectory arrays must have one equal, non-zero length.")
        for frame in episode["frameIndices"]:
            _integer(frame, f"{label}.frameIndices[]")
        for timestamp in episode["timestampsSeconds"]:
            _number(timestamp, f"{label}.timestampsSeconds[]")
        for arm in ("leftPositionsXyz", "rightPositionsXyz"):
            for position in episode[arm]:
                _vector3(position, f"{label}.{arm}[]")
    if episode_ids != sorted(set(episode_ids)):
        _fail("trajectory episodes must be sorted and distinct.")
    if not set(episode_ids).issubset(selected_episodes):
        _fail("trajectory episodes must be included in the coverage episode selection.")
    _reject_absolute_paths(payload)


def validate_browser_data(path: str | Path) -> Mapping[str, Any]:
    """Validate a browser-data directory and return its manifest."""
    bundle_path = Path(path)
    if not bundle_path.is_dir():
        _fail("Browser-data path must be a directory.")

    manifest = _validate_manifest(
        _load_json(bundle_path / MANIFEST_FILENAME, "manifest")
    )

    payload_values: dict[str, Any] = {}
    for payload in manifest["payloads"]:
        payload_path = bundle_path / payload["filename"]
        if not payload_path.is_file():
            _fail(f"Missing payload: {payload['filename']}.")
        if payload_path.stat().st_size != payload["byteSize"]:
            _fail(f"Payload byte size does not match: {payload['filename']}.")
        if sha256_file(payload_path) != payload["sha256"]:
            _fail(f"Payload checksum does not match: {payload['filename']}.")
        payload_values[payload["kind"]] = _load_json(
            payload_path,
            f"{payload['kind']} payload",
        )

    coverage_totals = _validate_coverage(payload_values["coverage"])
    for key, expected in coverage_totals.items():
        if manifest["totals"][key] != expected:
            _fail(f"Manifest total does not match coverage payload: {key}.")

    if "trajectories" in payload_values:
        _validate_trajectories(
            payload_values["trajectories"],
            set(manifest["dataset"]["episodeIds"]),
        )

    return manifest
