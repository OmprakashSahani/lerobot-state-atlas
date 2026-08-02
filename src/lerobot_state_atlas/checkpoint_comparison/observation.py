"""Strict local policy-comparison observation parsing and validation."""

from collections.abc import Mapping
from io import BytesIO
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from PIL import Image, UnidentifiedImageError

from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
    read_stable_file_snapshot,
)
from lerobot_state_atlas.checkpoint_comparison.models import (
    ObservationCamera,
    ObservationDatasetIdentity,
    ObservationRecordedGroundTruth,
    ObservationState,
    PolicyComparisonObservation,
)
from lerobot_state_atlas.checkpoint_comparison.serialize import sha256_bytes


SCHEMA_NAME = "lerobot-state-atlas.checkpoint-observation"
SCHEMA_MAJOR = 1
SCHEMA_MINOR = 0
STATE_FEATURE_NAME = "observation.state"
STATE_DTYPE = "float32"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
ACTION_CHUNK_LENGTH = 50
COMPONENT_NAMES = (
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_gripper.pos",
)
CAMERA_FEATURE_NAMES = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top",
)
CADENCE_RELATIVE_TOLERANCE = 1e-6
CADENCE_ABSOLUTE_TOLERANCE_SECONDS = 1e-9


class CheckpointObservationValidationError(ValueError):
    """Raised when a local checkpoint observation violates its contract."""


def _fail(label: str, message: str) -> None:
    raise CheckpointObservationValidationError(f"{label} {message}")


def _object(
    value: Any,
    fields: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(label, "must be an object.")
    missing = fields - set(value)
    unsupported = set(value) - fields - (optional or set())
    if missing:
        _fail(label, f"is missing fields: {', '.join(sorted(missing))}.")
    if unsupported:
        _fail(label, f"has unsupported fields: {', '.join(sorted(unsupported))}.")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(label, "must be a non-empty string.")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(label, "must be an integer.")
    if value < minimum:
        _fail(label, f"must be greater than or equal to {minimum}.")
    return value


def _number(
    value: Any, label: str, *, nonnegative: bool = False, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(label, "must be a number.")
    normalized = float(value)
    if not isfinite(normalized):
        _fail(label, "must be finite.")
    if nonnegative and normalized < 0:
        _fail(label, "must be nonnegative.")
    if positive and normalized <= 0:
        _fail(label, "must be greater than zero.")
    return normalized


def _sha256(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        _fail(label, "must be a lowercase SHA-256 digest.")
    return digest


def _schema(value: Any) -> None:
    schema = _object(value, {"name", "major", "minor"}, "schema")
    if schema["name"] != SCHEMA_NAME:
        _fail("schema.name", f"must be {SCHEMA_NAME!r}.")
    major = _integer(schema["major"], "schema.major")
    minor = _integer(schema["minor"], "schema.minor")
    if major != SCHEMA_MAJOR:
        _fail("schema.major", f"must be {SCHEMA_MAJOR}.")
    if minor != SCHEMA_MINOR:
        _fail("schema.minor", f"must be {SCHEMA_MINOR}.")


def _component_names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(label, "must be an array.")
    for index, item in enumerate(value):
        _string(item, f"{label}[{index}]")
    normalized = tuple(value)
    if normalized != COMPONENT_NAMES:
        _fail(label, "must match the canonical ordered 14-component contract.")
    return normalized


def _action_rows(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != ACTION_CHUNK_LENGTH:
        _fail(label, f"must contain exactly {ACTION_CHUNK_LENGTH} rows.")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        row_label = f"{label}[{row_index}]"
        if not isinstance(row, list) or len(row) != len(COMPONENT_NAMES):
            _fail(row_label, f"must contain exactly {len(COMPONENT_NAMES)} values.")
        rows.append(
            tuple(
                _number(item, f"{row_label}[{index}]") for index, item in enumerate(row)
            )
        )
    return tuple(rows)


def _safe_camera_path(
    filename: Any, label: str, manifest_directory: Path
) -> tuple[str, Path]:
    normalized = _string(filename, label)
    relative = PurePosixPath(normalized)
    if (
        "\\" in normalized
        or "://" in normalized
        or normalized.startswith("//")
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != normalized
    ):
        _fail(label, "must be a safe manifest-relative POSIX path.")
    lexical_path = manifest_directory.joinpath(*relative.parts)
    resolved = lexical_path.resolve()
    try:
        resolved.relative_to(manifest_directory)
    except ValueError:
        _fail(label, "must resolve inside the manifest directory.")
    if lexical_path.is_symlink():
        _fail(label, "must reference a regular file, not a symbolic link.")
    return normalized, resolved


def _validate_camera_file(path: Path, label: str, byte_count: int, sha256: str) -> None:
    if not path.exists():
        _fail(f"{label}.filename", "does not exist.")
    if not path.is_file():
        _fail(f"{label}.filename", "must reference a regular file.")
    if not os.access(path, os.R_OK):
        _fail(f"{label}.filename", "must reference a readable file.")
    try:
        snapshot = read_stable_file_snapshot(path)
    except StableFileSnapshotError as error:
        raise CheckpointObservationValidationError(
            f"{label}.filename could not be acquired as a stable file snapshot: {error}."
        ) from error
    actual_byte_count = len(snapshot)
    actual_sha256 = sha256_bytes(snapshot)
    if actual_byte_count != byte_count:
        _fail(
            f"{label}.byteCount",
            f"declares {byte_count} bytes but the file has {actual_byte_count}.",
        )
    if actual_sha256 != sha256:
        _fail(f"{label}.sha256", "does not match the referenced file.")
    try:
        with Image.open(BytesIO(snapshot)) as image:
            if getattr(image, "n_frames", 1) != 1:
                _fail(f"{label}.filename", "must contain exactly one image frame.")
            if image.width != IMAGE_WIDTH:
                _fail(f"{label}.width", f"decoded image width must be {IMAGE_WIDTH}.")
            if image.height != IMAGE_HEIGHT:
                _fail(
                    f"{label}.height", f"decoded image height must be {IMAGE_HEIGHT}."
                )
            if image.mode != "RGB" or len(image.getbands()) != IMAGE_CHANNELS:
                _fail(
                    f"{label}.channels",
                    f"decoded image must contain exactly {IMAGE_CHANNELS} RGB channels.",
                )
            image.verify()
    except CheckpointObservationValidationError:
        raise
    except (OSError, UnidentifiedImageError) as error:
        raise CheckpointObservationValidationError(
            f"{label}.filename must reference a valid decodable image."
        ) from error


def _recorded_ground_truth(
    value: Any, *, fps: float, observation_frame_index: int
) -> ObservationRecordedGroundTruth:
    label = "recordedGroundTruth"
    recorded = _object(
        value,
        {"available"},
        label,
        optional={
            "reason",
            "componentNames",
            "actions",
            "relativeTimesSeconds",
            "frameIndices",
        },
    )
    available = recorded["available"]
    if not isinstance(available, bool):
        _fail(f"{label}.available", "must be boolean.")
    data_fields = {
        "componentNames",
        "actions",
        "relativeTimesSeconds",
        "frameIndices",
    }
    if not available:
        reason = _string(recorded.get("reason"), f"{label}.reason")
        forbidden = data_fields & set(recorded)
        if forbidden:
            _fail(
                label,
                f"must not include fields when unavailable: {', '.join(sorted(forbidden))}.",
            )
        return ObservationRecordedGroundTruth(available=False, reason=reason)
    if "reason" in recorded:
        _fail(f"{label}.reason", "must be absent when recorded actions are available.")
    missing = data_fields - set(recorded)
    if missing:
        _fail(label, f"is missing fields: {', '.join(sorted(missing))}.")
    component_names = _component_names(
        recorded["componentNames"], f"{label}.componentNames"
    )
    actions = _action_rows(recorded["actions"], f"{label}.actions")
    raw_times = recorded["relativeTimesSeconds"]
    if not isinstance(raw_times, list) or len(raw_times) != ACTION_CHUNK_LENGTH:
        _fail(
            f"{label}.relativeTimesSeconds",
            f"must contain exactly {ACTION_CHUNK_LENGTH} values.",
        )
    times = tuple(
        _number(item, f"{label}.relativeTimesSeconds[{index}]", nonnegative=True)
        for index, item in enumerate(raw_times)
    )
    if times[0] != 0.0:
        _fail(f"{label}.relativeTimesSeconds[0]", "must be 0.0.")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        _fail(f"{label}.relativeTimesSeconds", "must be strictly increasing.")
    expected_interval = 1.0 / fps
    tolerance = max(
        CADENCE_ABSOLUTE_TOLERANCE_SECONDS,
        expected_interval * CADENCE_RELATIVE_TOLERANCE,
    )
    for index, (previous, current) in enumerate(zip(times, times[1:]), start=1):
        if abs((current - previous) - expected_interval) > tolerance:
            _fail(
                f"{label}.relativeTimesSeconds[{index}]",
                f"must be spaced by 1 / fps within {tolerance:g} seconds.",
            )
    raw_frames = recorded["frameIndices"]
    if not isinstance(raw_frames, list) or len(raw_frames) != ACTION_CHUNK_LENGTH:
        _fail(
            f"{label}.frameIndices",
            f"must contain exactly {ACTION_CHUNK_LENGTH} values.",
        )
    frames = tuple(
        _integer(item, f"{label}.frameIndices[{index}]")
        for index, item in enumerate(raw_frames)
    )
    if frames[0] != observation_frame_index:
        _fail(
            f"{label}.frameIndices[0]",
            "must equal dataset.frameIndex.",
        )
    for index, (previous, current) in enumerate(zip(frames, frames[1:]), start=1):
        if current != previous + 1:
            _fail(
                f"{label}.frameIndices[{index}]",
                "must equal the previous frame index plus one.",
            )
    return ObservationRecordedGroundTruth(
        available=True,
        reason=None,
        component_names=component_names,
        actions=actions,
        relative_times_seconds=times,
        frame_indices=frames,
    )


def load_checkpoint_observation(
    manifest_path: str | Path,
) -> PolicyComparisonObservation:
    """Load and strictly validate one local policy-comparison observation."""
    supplied_path = Path(manifest_path)
    if supplied_path.is_symlink():
        raise CheckpointObservationValidationError(
            f"Observation manifest must not be a symbolic link: {supplied_path.name}."
        )
    path = supplied_path.parent.resolve() / supplied_path.name
    if path.is_symlink():
        raise CheckpointObservationValidationError(
            f"Observation manifest must not be a symbolic link: {path.name}."
        )
    if not path.exists():
        raise CheckpointObservationValidationError(
            f"Observation manifest does not exist: {path.name}."
        )
    if not path.is_file():
        raise CheckpointObservationValidationError(
            f"Observation manifest must be a regular file: {path.name}."
        )
    try:
        manifest_snapshot = read_stable_file_snapshot(path)
    except StableFileSnapshotError as error:
        raise CheckpointObservationValidationError(
            f"Observation manifest could not be acquired as a stable file snapshot: "
            f"{path.name}: {error}."
        ) from error
    try:
        manifest_text = manifest_snapshot.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointObservationValidationError(
            f"Observation manifest is not valid UTF-8: {path.name}."
        ) from error
    try:
        document = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise CheckpointObservationValidationError(
            f"Observation manifest is not valid JSON: {path.name}."
        ) from error
    manifest = _object(
        document,
        {
            "schema",
            "observationId",
            "dataset",
            "prompt",
            "state",
            "cameras",
            "recordedGroundTruth",
        },
        "manifest",
    )
    _schema(manifest["schema"])
    observation_id = _string(manifest["observationId"], "observationId")
    prompt = _string(manifest["prompt"], "prompt")
    dataset = _object(
        manifest["dataset"],
        {
            "repoId",
            "revision",
            "episodeIndex",
            "frameIndex",
            "timestampSeconds",
            "fps",
            "task",
        },
        "dataset",
    )
    repository_id = _string(dataset["repoId"], "dataset.repoId")
    revision = _string(dataset["revision"], "dataset.revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        _fail("dataset.revision", "must be a full lowercase 40-character commit SHA.")
    episode_index = _integer(dataset["episodeIndex"], "dataset.episodeIndex")
    frame_index = _integer(dataset["frameIndex"], "dataset.frameIndex")
    timestamp_seconds = _number(
        dataset["timestampSeconds"], "dataset.timestampSeconds", nonnegative=True
    )
    fps = _number(dataset["fps"], "dataset.fps", positive=True)
    task = _string(dataset["task"], "dataset.task")
    state = _object(
        manifest["state"],
        {"featureName", "dtype", "componentNames", "values"},
        "state",
    )
    if state["featureName"] != STATE_FEATURE_NAME:
        _fail("state.featureName", f"must be {STATE_FEATURE_NAME!r}.")
    if state["dtype"] != STATE_DTYPE:
        _fail("state.dtype", f"must be {STATE_DTYPE!r}.")
    state_components = _component_names(state["componentNames"], "state.componentNames")
    raw_state_values = state["values"]
    if not isinstance(raw_state_values, list) or len(raw_state_values) != len(
        COMPONENT_NAMES
    ):
        _fail("state.values", f"must contain exactly {len(COMPONENT_NAMES)} values.")
    state_values = tuple(
        _number(item, f"state.values[{index}]")
        for index, item in enumerate(raw_state_values)
    )
    raw_cameras = manifest["cameras"]
    if not isinstance(raw_cameras, list) or len(raw_cameras) != len(
        CAMERA_FEATURE_NAMES
    ):
        _fail("cameras", "must contain exactly three ordered camera entries.")
    cameras: list[ObservationCamera] = []
    for index, expected_feature in enumerate(CAMERA_FEATURE_NAMES):
        label = f"cameras[{index}]"
        camera = _object(
            raw_cameras[index],
            {
                "featureName",
                "filename",
                "width",
                "height",
                "channels",
                "byteCount",
                "sha256",
            },
            label,
        )
        if camera["featureName"] != expected_feature:
            _fail(f"{label}.featureName", f"must be {expected_feature!r}.")
        width = _integer(camera["width"], f"{label}.width", minimum=1)
        height = _integer(camera["height"], f"{label}.height", minimum=1)
        channels = _integer(camera["channels"], f"{label}.channels", minimum=1)
        if width != IMAGE_WIDTH:
            _fail(f"{label}.width", f"must be {IMAGE_WIDTH}.")
        if height != IMAGE_HEIGHT:
            _fail(f"{label}.height", f"must be {IMAGE_HEIGHT}.")
        if channels != IMAGE_CHANNELS:
            _fail(f"{label}.channels", f"must be {IMAGE_CHANNELS}.")
        byte_count = _integer(camera["byteCount"], f"{label}.byteCount", minimum=1)
        sha256 = _sha256(camera["sha256"], f"{label}.sha256")
        filename, camera_path = _safe_camera_path(
            camera["filename"], f"{label}.filename", path.parent
        )
        _validate_camera_file(camera_path, label, byte_count, sha256)
        cameras.append(
            ObservationCamera(
                feature_name=expected_feature,
                filename=filename,
                path=camera_path,
                width=width,
                height=height,
                channels=channels,
                byte_count=byte_count,
                sha256=sha256,
            )
        )
    recorded = _recorded_ground_truth(
        manifest["recordedGroundTruth"],
        fps=fps,
        observation_frame_index=frame_index,
    )
    return PolicyComparisonObservation(
        manifest_path=path,
        manifest_sha256=sha256_bytes(manifest_snapshot),
        manifest_byte_count=len(manifest_snapshot),
        observation_id=observation_id,
        dataset=ObservationDatasetIdentity(
            repository_id=repository_id,
            revision=revision,
            episode_index=episode_index,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            fps=fps,
            task=task,
        ),
        prompt=prompt,
        state=ObservationState(
            feature_name=STATE_FEATURE_NAME,
            dtype=STATE_DTYPE,
            component_names=state_components,
            values=state_values,
        ),
        cameras=tuple(cameras),
        recorded_ground_truth=recorded,
    )
