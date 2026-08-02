"""Strict local-only checkpoint-comparison runner manifest parsing and resolution."""

from collections.abc import Mapping
import hashlib
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
    read_stable_file_snapshot,
)
from lerobot_state_atlas.checkpoint_comparison.runner_models import (
    CheckpointComparisonRunnerManifest,
    InputInventoryEntry,
    ResolvedRunnerInputs,
    RunnerCheckpointInput,
    RunnerDatasetIdentity,
    RunnerFileInput,
    RunnerOutputConfiguration,
    RunnerProcessorInputs,
    RunnerProjectionConfiguration,
    RunnerRigidTransform,
    RunnerRobotConfiguration,
    RunnerRuntimeConfiguration,
    RunnerSchemaVersion,
    RunnerTokenizerInput,
)


SCHEMA_NAME = "lerobot-state-atlas.checkpoint-comparison-runner"
SCHEMA_VERSION = (1, 0)
DATASET_REPOSITORY_ID = "DreamMachines/actuator_unboxing_4h_diverse"
DATASET_REVISION = "e973df866c80f52884cc68355579043cab828e78"
BASE_REPOSITORY_ID = "lerobot/pi05_base"
BASE_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
FINE_TUNED_REPOSITORY_ID = "DreamMachines/actuator_unboxing_4h_diverse_fullft_bs256"
FINE_TUNED_REVISION = "6c50dbbccd576e4e384ed51a8244272aab5f3c62"
MAX_NOISE_SEED = 2**63 - 1


class CheckpointComparisonRunnerManifestError(ValueError):
    """Raised when a runner manifest or its declared paths are invalid."""


def _fail(path: str, message: str) -> None:
    raise CheckpointComparisonRunnerManifestError(f"{path} {message}")


def _object(value: Any, required: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object.")
    missing = required - set(value)
    extra = set(value) - required
    if missing:
        _fail(path, f"is missing fields: {', '.join(sorted(missing))}.")
    if extra:
        _fail(path, f"has unsupported fields: {', '.join(sorted(extra))}.")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string.")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be boolean.")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer.")
    if value < minimum:
        _fail(path, f"must be greater than or equal to {minimum}.")
    return value


def _sha(value: Any, path: str, length: int = 64) -> str:
    digest = _string(value, path)
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", digest):
        _fail(path, f"must be a lowercase {length}-character hexadecimal digest.")
    return digest


def _exact(value: Any, expected: str, path: str) -> str:
    if value != expected:
        _fail(path, f"must be {expected!r}.")
    return expected


def _file_input(value: Any, path: str) -> RunnerFileInput:
    item = _object(value, {"path", "byteCount", "sha256"}, path)
    byte_count = _integer(item["byteCount"], f"{path}.byteCount", minimum=1)
    return RunnerFileInput(
        path=_relative_path_text(item["path"], f"{path}.path"),
        byte_count=byte_count,
        sha256=_sha(item["sha256"], f"{path}.sha256"),
    )


def _checkpoint_input(
    value: Any,
    path: str,
    expected_repository: str,
    expected_revision: str,
) -> RunnerCheckpointInput:
    item = _object(
        value, {"repositoryId", "revision", "path", "byteCount", "sha256"}, path
    )
    return RunnerCheckpointInput(
        path=_relative_path_text(item["path"], f"{path}.path"),
        byte_count=_integer(item["byteCount"], f"{path}.byteCount", minimum=1),
        sha256=_sha(item["sha256"], f"{path}.sha256"),
        repository_id=_exact(
            item["repositoryId"], expected_repository, f"{path}.repositoryId"
        ),
        revision=_exact(
            _sha(item["revision"], f"{path}.revision", 40),
            expected_revision,
            f"{path}.revision",
        ),
    )


def _relative_path_text(value: Any, path: str) -> str:
    text = _string(value, path)
    relative = PurePosixPath(text)
    if (
        "\\" in text
        or "://" in text
        or text.startswith("//")
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != text
    ):
        _fail(path, "must be a safe manifest-relative POSIX path.")
    return text


def _transform(value: Any, path: str) -> RunnerRigidTransform:
    item = _object(value, {"translationXyz", "rotationRpy"}, path)

    def vector(field: str) -> tuple[float, float, float]:
        raw = item[field]
        field_path = f"{path}.{field}"
        if not isinstance(raw, list) or len(raw) != 3:
            _fail(field_path, "must contain exactly three numbers.")
        normalized: list[float] = []
        for index, number in enumerate(raw):
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                _fail(f"{field_path}[{index}]", "must be a number.")
            result = float(number)
            if not isfinite(result):
                _fail(f"{field_path}[{index}]", "must be finite.")
            normalized.append(result)
        return normalized[0], normalized[1], normalized[2]

    return RunnerRigidTransform(vector("translationXyz"), vector("rotationRpy"))


def _parse_document(
    value: Any, manifest_path: Path, digest: str
) -> CheckpointComparisonRunnerManifest:
    root = _object(
        value,
        {
            "schema",
            "dataset",
            "observationManifest",
            "checkpoints",
            "configuration",
            "processors",
            "robot",
            "runtime",
            "projection",
            "output",
        },
        "manifest",
    )
    schema_value = _object(root["schema"], {"name", "major", "minor"}, "schema")
    name = _exact(schema_value["name"], SCHEMA_NAME, "schema.name")
    major = _integer(schema_value["major"], "schema.major")
    minor = _integer(schema_value["minor"], "schema.minor")
    if (major, minor) != SCHEMA_VERSION:
        _fail("schema", f"version {major}.{minor} is unsupported.")
    dataset = _object(root["dataset"], {"repositoryId", "revision"}, "dataset")
    dataset_identity = RunnerDatasetIdentity(
        _exact(dataset["repositoryId"], DATASET_REPOSITORY_ID, "dataset.repositoryId"),
        _exact(
            _sha(dataset["revision"], "dataset.revision", 40),
            DATASET_REVISION,
            "dataset.revision",
        ),
    )
    checkpoints = _object(root["checkpoints"], {"base", "fineTuned"}, "checkpoints")
    processors = _object(
        root["processors"],
        {
            "preprocessorConfig",
            "preprocessorState",
            "postprocessorConfig",
            "postprocessorState",
            "tokenizerDirectory",
        },
        "processors",
    )
    tokenizer = _object(
        processors["tokenizerDirectory"],
        {"path", "identitySha256"},
        "processors.tokenizerDirectory",
    )
    processor_inputs = RunnerProcessorInputs(
        _file_input(processors["preprocessorConfig"], "processors.preprocessorConfig"),
        _file_input(processors["preprocessorState"], "processors.preprocessorState"),
        _file_input(
            processors["postprocessorConfig"], "processors.postprocessorConfig"
        ),
        _file_input(processors["postprocessorState"], "processors.postprocessorState"),
        RunnerTokenizerInput(
            _relative_path_text(
                tokenizer["path"], "processors.tokenizerDirectory.path"
            ),
            _sha(
                tokenizer["identitySha256"],
                "processors.tokenizerDirectory.identitySha256",
            ),
        ),
    )
    robot = _object(
        root["robot"],
        {
            "urdfPath",
            "urdfByteCount",
            "urdfSha256",
            "upstreamRevisionIdentity",
            "leftArmTransform",
            "rightArmTransform",
            "calibratedArmTransforms",
        },
        "robot",
    )
    robot_config = RunnerRobotConfiguration(
        RunnerFileInput(
            _relative_path_text(robot["urdfPath"], "robot.urdfPath"),
            _integer(robot["urdfByteCount"], "robot.urdfByteCount", minimum=1),
            _sha(robot["urdfSha256"], "robot.urdfSha256"),
        ),
        _string(robot["upstreamRevisionIdentity"], "robot.upstreamRevisionIdentity"),
        _transform(robot["leftArmTransform"], "robot.leftArmTransform"),
        _transform(robot["rightArmTransform"], "robot.rightArmTransform"),
        _boolean(robot["calibratedArmTransforms"], "robot.calibratedArmTransforms"),
    )
    runtime = _object(
        root["runtime"],
        {
            "device",
            "modelDtype",
            "noiseDtype",
            "noiseSeed",
            "numInferenceSteps",
            "minimumFreeVramBytes",
            "minimumAvailableRamBytes",
            "minimumFreeDiskBytes",
        },
        "runtime",
    )
    device = _string(runtime["device"], "runtime.device")
    if not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device):
        _fail("runtime.device", "must use syntax 'cuda:<nonnegative index>'.")
    model_dtype = runtime["modelDtype"]
    if model_dtype not in {"bfloat16", "float32"}:
        _fail("runtime.modelDtype", "must be 'bfloat16' or 'float32'.")
    if runtime["noiseDtype"] != "float32":
        _fail("runtime.noiseDtype", "must be 'float32'.")
    noise_seed = _integer(runtime["noiseSeed"], "runtime.noiseSeed")
    if noise_seed > MAX_NOISE_SEED:
        _fail("runtime.noiseSeed", f"must not exceed {MAX_NOISE_SEED}.")
    runtime_config = RunnerRuntimeConfiguration(
        device=device,
        model_dtype=model_dtype,
        noise_dtype="float32",
        noise_seed=noise_seed,
        num_inference_steps=_integer(
            runtime["numInferenceSteps"], "runtime.numInferenceSteps", minimum=1
        ),
        minimum_free_vram_bytes=_integer(
            runtime["minimumFreeVramBytes"], "runtime.minimumFreeVramBytes"
        ),
        minimum_available_ram_bytes=_integer(
            runtime["minimumAvailableRamBytes"], "runtime.minimumAvailableRamBytes"
        ),
        minimum_free_disk_bytes=_integer(
            runtime["minimumFreeDiskBytes"], "runtime.minimumFreeDiskBytes"
        ),
    )
    projection = _object(
        root["projection"],
        {
            "mode",
            "jointLimitPolicy",
            "unavailableReason",
            "acknowledgeUncalibratedArmTransforms",
            "acknowledgeRecordedLimitViolations",
        },
        "projection",
    )
    mode = projection["mode"]
    if mode not in {"available", "unavailable"}:
        _fail("projection.mode", "must be 'available' or 'unavailable'.")
    limit_policy = projection["jointLimitPolicy"]
    if limit_policy not in {"reject", "allow-with-recorded-violations"}:
        _fail(
            "projection.jointLimitPolicy",
            "must be 'reject' or 'allow-with-recorded-violations'.",
        )
    uncalibrated_ack = _boolean(
        projection["acknowledgeUncalibratedArmTransforms"],
        "projection.acknowledgeUncalibratedArmTransforms",
    )
    violations_ack = _boolean(
        projection["acknowledgeRecordedLimitViolations"],
        "projection.acknowledgeRecordedLimitViolations",
    )
    reason = projection["unavailableReason"]
    if mode == "available":
        if reason is not None:
            _fail("projection.unavailableReason", "must be null in available mode.")
        if not robot_config.calibrated_arm_transforms and not uncalibrated_ack:
            _fail(
                "projection.acknowledgeUncalibratedArmTransforms",
                "must be true for uncalibrated arm transforms.",
            )
        if robot_config.calibrated_arm_transforms and uncalibrated_ack:
            _fail(
                "projection.acknowledgeUncalibratedArmTransforms",
                "must be false for calibrated arm transforms.",
            )
        expected_ack = limit_policy == "allow-with-recorded-violations"
        if violations_ack is not expected_ack:
            _fail(
                "projection.acknowledgeRecordedLimitViolations",
                f"must be {str(expected_ack).lower()} for {limit_policy!r}.",
            )
    else:
        if not isinstance(reason, str) or not reason.strip():
            _fail("projection.unavailableReason", "must be a non-empty string.")
        if reason != reason.strip():
            _fail(
                "projection.unavailableReason",
                "must not contain surrounding whitespace.",
            )
        if uncalibrated_ack or violations_ack:
            _fail("projection", "acknowledgements must be false in unavailable mode.")
    output = _object(
        root["output"], {"runDirectory", "bundleId", "replaceExisting"}, "output"
    )
    output_config = RunnerOutputConfiguration(
        _relative_path_text(output["runDirectory"], "output.runDirectory"),
        _string(output["bundleId"], "output.bundleId"),
        _boolean(output["replaceExisting"], "output.replaceExisting"),
    )
    return CheckpointComparisonRunnerManifest(
        manifest_path=manifest_path,
        manifest_sha256=digest,
        schema=RunnerSchemaVersion(name, major, minor),
        dataset=dataset_identity,
        observation_manifest=_file_input(
            root["observationManifest"], "observationManifest"
        ),
        base_checkpoint=_checkpoint_input(
            checkpoints["base"], "checkpoints.base", BASE_REPOSITORY_ID, BASE_REVISION
        ),
        fine_tuned_checkpoint=_checkpoint_input(
            checkpoints["fineTuned"],
            "checkpoints.fineTuned",
            FINE_TUNED_REPOSITORY_ID,
            FINE_TUNED_REVISION,
        ),
        configuration=_file_input(root["configuration"], "configuration"),
        processors=processor_inputs,
        robot=robot_config,
        runtime=runtime_config,
        projection=RunnerProjectionConfiguration(
            mode, limit_policy, reason, uncalibrated_ack, violations_ack
        ),
        output=output_config,
    )


def load_checkpoint_comparison_runner_manifest(
    path: str | Path,
) -> CheckpointComparisonRunnerManifest:
    """Load one stable, non-symlink local runner manifest snapshot."""
    lexical = Path(path)
    if lexical.is_symlink():
        _fail("manifestPath", "must not be a symbolic link.")
    absolute = Path(os.path.abspath(lexical))
    if not absolute.exists():
        _fail("manifestPath", "does not exist.")
    if not absolute.is_file():
        _fail("manifestPath", "must be a regular file.")
    accepted = absolute.parent.resolve() / absolute.name
    try:
        snapshot = read_stable_file_snapshot(accepted)
    except StableFileSnapshotError as error:
        raise CheckpointComparisonRunnerManifestError(
            f"manifestPath could not be acquired as a stable file snapshot: {error}."
        ) from error
    try:
        text = snapshot.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointComparisonRunnerManifestError(
            "manifestPath contains malformed UTF-8."
        ) from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise CheckpointComparisonRunnerManifestError(
            "manifestPath contains malformed JSON."
        ) from error
    return _parse_document(document, accepted, hashlib.sha256(snapshot).hexdigest())


def _resolved_relative(base: Path, declared: str, field: str) -> Path:
    current = base
    for part in PurePosixPath(declared).parts:
        current = current / part
        if current.is_symlink():
            _fail(field, f"contains symbolic-link path component {current}.")
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError:
        _fail(field, f"resolves outside runner manifest directory {base}.")
    return resolved


def _inventory_file(
    base: Path, logical_id: str, declared: RunnerFileInput
) -> InputInventoryEntry:
    path = _resolved_relative(base, declared.path, f"{logical_id}.path")
    if not path.exists():
        _fail(logical_id, f"references missing file {path}.")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        _fail(logical_id, f"must reference an ordinary regular file; received {path}.")
    if info.st_size != declared.byte_count:
        _fail(
            f"{logical_id}.byteCount",
            f"expected {declared.byte_count} bytes but {path} has {info.st_size}.",
        )
    return InputInventoryEntry(
        logical_id,
        path,
        declared.byte_count,
        info.st_size,
        declared.sha256,
        "file",
        True,
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _inventory_directory(
    base: Path, logical_id: str, declared: RunnerTokenizerInput
) -> InputInventoryEntry:
    path = _resolved_relative(base, declared.path, f"{logical_id}.path")
    if not path.exists():
        _fail(logical_id, f"references missing directory {path}.")
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode):
        _fail(logical_id, f"must reference an ordinary directory; received {path}.")
    return InputInventoryEntry(
        logical_id,
        path,
        None,
        None,
        declared.identity_sha256,
        "directory",
        True,
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _protected_demo(path: Path) -> bool:
    suffixes = (
        ("apps", "web", "public", "atlas-data", "demo-v1"),
        ("apps", "web", "public", "atlas-data", "demo-v2"),
    )
    return any(tuple(path.parts[-len(suffix) :]) == suffix for suffix in suffixes)


def _existing_ancestor(path: Path, base: Path) -> Path:
    current = path
    while not current.exists():
        if current == base:
            break
        current = current.parent
    if current.is_symlink():
        _fail(
            "output.runDirectory", f"contains symbolic-link path component {current}."
        )
    if not current.is_dir():
        _fail(
            "output.runDirectory", f"existing ancestor {current} must be a directory."
        )
    return current


def _path_overlap_relationship(output: Path, source: Path) -> str | None:
    if output == source:
        return "equal"
    try:
        source.relative_to(output)
    except ValueError:
        pass
    else:
        return "output-contains-input"
    try:
        output.relative_to(source)
    except ValueError:
        return None
    return "output-inside-input"


def validate_output_input_disjointness(
    *,
    output_path: Path,
    manifest_path: Path,
    inventory_entries: tuple[InputInventoryEntry, ...],
    additional_inputs: tuple[tuple[str, Path], ...] = (),
) -> None:
    """Reject equality or ancestry overlap between output and immutable sources."""
    sources = (
        ("runnerManifest", manifest_path),
        *(
            (entry.logical_input_id, entry.canonical_path)
            for entry in inventory_entries
        ),
        *additional_inputs,
    )
    for logical_id, source_path in sources:
        relationship = _path_overlap_relationship(output_path, source_path)
        if relationship is not None:
            _fail(
                "output.runDirectory",
                "must be disjoint from runner input "
                f"{logical_id!r}; output {output_path} has relationship "
                f"{relationship} with input {source_path}. Replacement could delete "
                "or mutate required source data.",
            )


def resolve_checkpoint_comparison_runner_inputs(
    manifest: CheckpointComparisonRunnerManifest,
) -> ResolvedRunnerInputs:
    """Resolve and stat all inputs without reading or hashing their contents."""
    if not isinstance(manifest, CheckpointComparisonRunnerManifest):
        _fail("manifest", "must be a CheckpointComparisonRunnerManifest.")
    base = manifest.manifest_path.parent.resolve()
    entries = (
        _inventory_file(base, "observationManifest", manifest.observation_manifest),
        _inventory_file(base, "checkpoints.base", manifest.base_checkpoint),
        _inventory_file(base, "checkpoints.fineTuned", manifest.fine_tuned_checkpoint),
        _inventory_file(base, "configuration", manifest.configuration),
        _inventory_file(
            base,
            "processors.preprocessorConfig",
            manifest.processors.preprocessor_config,
        ),
        _inventory_file(
            base, "processors.preprocessorState", manifest.processors.preprocessor_state
        ),
        _inventory_file(
            base,
            "processors.postprocessorConfig",
            manifest.processors.postprocessor_config,
        ),
        _inventory_file(
            base,
            "processors.postprocessorState",
            manifest.processors.postprocessor_state,
        ),
        _inventory_file(base, "robot.urdf", manifest.robot.urdf),
        _inventory_directory(
            base,
            "processors.tokenizerDirectory",
            manifest.processors.tokenizer_directory,
        ),
    )
    output = _resolved_relative(
        base, manifest.output.run_directory, "output.runDirectory"
    )
    if _protected_demo(output):
        _fail("output.runDirectory", "must not target immutable demo-v1 or demo-v2.")
    if output.is_symlink():
        _fail("output.runDirectory", "must not be a symbolic link.")
    validate_output_input_disjointness(
        output_path=output,
        manifest_path=manifest.manifest_path,
        inventory_entries=entries,
    )
    if output.exists() and not output.is_dir():
        _fail("output.runDirectory", "existing destination must be a directory.")
    ancestor = _existing_ancestor(output, base)
    return ResolvedRunnerInputs(
        manifest.manifest_path,
        manifest.manifest_sha256,
        base,
        entries,
        output,
        ancestor,
    )
