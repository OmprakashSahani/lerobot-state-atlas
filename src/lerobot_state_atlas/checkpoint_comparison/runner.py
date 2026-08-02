"""Strict local-only orchestration for a complete checkpoint-comparison run."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Protocol
from uuid import uuid4

import torch

from lerobot_state_atlas.checkpoint_comparison.artifact import (
    build_checkpoint_comparison_documents,
    build_checkpoint_comparison_v1_1,
)
from lerobot_state_atlas.checkpoint_comparison.camera_inputs import (
    prepare_bound_policy_observation_input,
)
from lerobot_state_atlas.checkpoint_comparison.checkpoint_staging import (
    resolve_runner_staging_root,
)
from lerobot_state_atlas.checkpoint_comparison.compatibility import (
    adapt_pi05_finetuned_config,
)
from lerobot_state_atlas.checkpoint_comparison.deterministic_runtime import (
    DeterministicTorchSettings,
    deterministic_torch_execution,
)
from lerobot_state_atlas.checkpoint_comparison.inference import (
    run_sequential_policy_comparison,
    validate_processed_observation_device,
)
from lerobot_state_atlas.checkpoint_comparison.local_files import (
    read_stable_file_snapshot,
)
from lerobot_state_atlas.checkpoint_comparison.models import (
    ActionInterpretationProvenance,
    AvailablePolicyComparisonProjection,
    PolicyIdentity,
    RobotProjectionProvenance,
)
from lerobot_state_atlas.checkpoint_comparison.observation import (
    COMPONENT_NAMES,
    load_checkpoint_observation,
)
from lerobot_state_atlas.checkpoint_comparison.pi05_factory import (
    create_pi05_policy_factory,
)
from lerobot_state_atlas.checkpoint_comparison.preflight import (
    preflight_checkpoint_comparison_run,
)
from lerobot_state_atlas.checkpoint_comparison.processor_compatibility import (
    verify_pi05_processor_assets,
)
from lerobot_state_atlas.checkpoint_comparison.processors import (
    build_verified_pi05_processor_pair,
)
from lerobot_state_atlas.checkpoint_comparison.projection import (
    ABSOLUTE_INTERPRETATION_ID,
    ABSOLUTE_INTERPRETATION_VERSION,
    ABSOLUTE_TRANSFORMATIONS,
    PI05AbsolutePositionTargetsAdapter,
    project_policy_comparison_trajectories,
    unavailable_policy_comparison_trajectory_result,
)
from lerobot_state_atlas.checkpoint_comparison.receipt import (
    RECEIPT_SCHEMA_NAME,
    validate_checkpoint_comparison_run_receipt,
)
from lerobot_state_atlas.checkpoint_comparison.run_installation import (
    install_checkpoint_comparison_run,
)
from lerobot_state_atlas.checkpoint_comparison.runner_manifest import (
    load_checkpoint_comparison_runner_manifest,
    resolve_checkpoint_comparison_runner_inputs,
    validate_output_input_disjointness,
)
from lerobot_state_atlas.checkpoint_comparison.runner_models import (
    CheckpointComparisonRunnerManifest,
    CheckpointComparisonRunnerResult,
    HardwarePreflightReport,
    ResolvedRunnerInputs,
)
from lerobot_state_atlas.checkpoint_comparison.serialize import sha256_bytes
from lerobot_state_atlas.checkpoint_comparison.tokenizer_assets import (
    load_verified_local_pi05_tokenizer,
    verify_local_pi05_tokenizer_directory,
)
from lerobot_state_atlas.trajectory import build_trlc_dk1_joint_component_map
from lerobot_state_atlas.transforms import RigidTransform
from lerobot_state_atlas.urdf import load_robot_model


BASE_REPOSITORY = "lerobot/pi05_base"
BASE_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
FINE_REPOSITORY = "DreamMachines/actuator_unboxing_4h_diverse_fullft_bs256"
FINE_REVISION = "6c50dbbccd576e4e384ed51a8244272aab5f3c62"


@dataclass(frozen=True)
class RunnerCleanupFailure:
    """One secondary cleanup failure attached to a runner failure."""

    resource: str
    exception_type: str
    message: str
    recoverable_path: Path | None
    manual_retry_possible: bool
    installed_output_remains_valid: bool


class CheckpointComparisonRunnerError(RuntimeError):
    """Phase-tagged fatal comparison-run error."""

    def __init__(
        self,
        phase: str,
        message: str,
        *,
        recoverable_paths: tuple[Path, ...] = (),
        cleanup_failures: tuple[RunnerCleanupFailure, ...] = (),
    ) -> None:
        super().__init__(f"{phase}: {message}")
        self.phase = phase
        self.message = message
        self.recoverable_paths = recoverable_paths
        self.cleanup_failures = cleanup_failures

    def attach_cleanup_failures(
        self, failures: tuple[RunnerCleanupFailure, ...]
    ) -> None:
        """Attach secondary cleanup diagnostics without replacing this error."""
        if not failures:
            return
        self.cleanup_failures = (*self.cleanup_failures, *failures)
        paths = tuple(
            failure.recoverable_path
            for failure in failures
            if failure.recoverable_path is not None
        )
        self.recoverable_paths = tuple(dict.fromkeys((*self.recoverable_paths, *paths)))
        for failure in failures:
            path = (
                f" Recoverable path: {failure.recoverable_path}."
                if failure.recoverable_path is not None
                else " No recoverable temporary path remains."
            )
            self.add_note(
                f"Secondary cleanup failure for {failure.resource}: "
                f"{failure.exception_type}: {failure.message}.{path}"
            )


def _attempt_cleanups(
    operations: tuple[tuple[str, Path, Any], ...],
    *,
    installed_output_remains_valid: bool,
) -> tuple[RunnerCleanupFailure, ...]:
    """Attempt every cleanup operation and return deterministic diagnostics."""
    failures: list[RunnerCleanupFailure] = []
    for resource, path, operation in operations:
        try:
            operation()
        except BaseException as error:
            recoverable_path = path if os.path.lexists(path) else None
            failures.append(
                RunnerCleanupFailure(
                    resource=resource,
                    exception_type=type(error).__name__,
                    message=str(error),
                    recoverable_path=recoverable_path,
                    manual_retry_possible=recoverable_path is not None,
                    installed_output_remains_valid=installed_output_remains_valid,
                )
            )
    return tuple(failures)


def _resolve_cleanup_outcome(
    primary_error: BaseException | None,
    primary_traceback: Any,
    cleanup_failures: tuple[RunnerCleanupFailure, ...],
) -> None:
    """Preserve a primary failure, or raise cleanup after otherwise-successful work."""
    if primary_error is not None:
        if isinstance(primary_error, CheckpointComparisonRunnerError):
            primary_error.attach_cleanup_failures(cleanup_failures)
        elif cleanup_failures:
            for cleanup_failure in cleanup_failures:
                primary_error.add_note(
                    "Secondary cleanup failure for "
                    f"{cleanup_failure.resource}: "
                    f"{cleanup_failure.exception_type}: {cleanup_failure.message}; "
                    f"recoverable path: {cleanup_failure.recoverable_path}."
                )
        raise primary_error.with_traceback(primary_traceback)
    if not cleanup_failures:
        return
    details = "; ".join(
        f"{failure.resource}: {failure.exception_type}: {failure.message}"
        for failure in cleanup_failures
    )
    recoverable = tuple(
        failure.recoverable_path
        for failure in cleanup_failures
        if failure.recoverable_path is not None
    )
    installed_remains_valid = any(
        failure.installed_output_remains_valid for failure in cleanup_failures
    )
    installed_message = (
        " The installed run remains valid;" if installed_remains_valid else ""
    )
    raise CheckpointComparisonRunnerError(
        "cleanup",
        "required temporary-resource cleanup failed after the run was otherwise "
        f"complete: {details}.{installed_message} retry cleanup manually for any "
        "reported recoverable path.",
        recoverable_paths=recoverable,
        cleanup_failures=cleanup_failures,
    )


class RunnerDependencies(Protocol):
    """Injectable runner boundary used by CPU-only tests and production."""

    def load_manifest(
        self, value: str | Path
    ) -> CheckpointComparisonRunnerManifest: ...
    def resolve(
        self, manifest: CheckpointComparisonRunnerManifest
    ) -> ResolvedRunnerInputs: ...
    def preflight(
        self,
        manifest: CheckpointComparisonRunnerManifest,
        resolved: ResolvedRunnerInputs,
    ) -> HardwarePreflightReport: ...
    def execution_device(self, manifest: CheckpointComparisonRunnerManifest) -> str: ...


def _inventory_path(resolved: ResolvedRunnerInputs, logical_id: str) -> Path:
    for entry in resolved.inventory:
        if entry.logical_input_id == logical_id:
            return entry.canonical_path
    raise ValueError(f"resolved input inventory is missing {logical_id!r}.")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _source_identity() -> str:
    override = os.environ.get("LEROBOT_STATE_ATLAS_SOURCE_IDENTITY")
    if override:
        return override
    repository = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        return "package-without-git-identity"
    return f"{revision}-{'dirty' if dirty else 'clean'}"


@dataclass(frozen=True)
class DefaultCheckpointComparisonRunnerDependencies:
    """Production local-only dependency implementation."""

    def load_manifest(self, value: str | Path) -> CheckpointComparisonRunnerManifest:
        return load_checkpoint_comparison_runner_manifest(value)

    def resolve(
        self, manifest: CheckpointComparisonRunnerManifest
    ) -> ResolvedRunnerInputs:
        return resolve_checkpoint_comparison_runner_inputs(manifest)

    def preflight(
        self,
        manifest: CheckpointComparisonRunnerManifest,
        resolved: ResolvedRunnerInputs,
    ) -> HardwarePreflightReport:
        return preflight_checkpoint_comparison_run(manifest, resolved)

    def load_observation(self, resolved: ResolvedRunnerInputs):
        return load_checkpoint_observation(
            _inventory_path(resolved, "observationManifest")
        )

    def execution_device(self, manifest: CheckpointComparisonRunnerManifest) -> str:
        return manifest.runtime.device

    def prepare_cameras(self, observation):
        return prepare_bound_policy_observation_input(observation)

    def adapt_configuration(self, manifest, resolved):
        path = _inventory_path(resolved, "configuration")
        raw = read_stable_file_snapshot(path)
        expected = manifest.configuration
        if len(raw) != expected.byte_count:
            raise ValueError(
                "configuration.byteCount changed before configuration adaptation."
            )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected.sha256:
            raise ValueError(
                "configuration.sha256 does not match the stable configuration snapshot."
            )
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"configuration is not valid UTF-8 JSON: {error}"
            ) from error
        return adapt_pi05_finetuned_config(
            decoded,
            _package_version("lerobot"),
            {
                "device": "cpu",
                "dtype": manifest.runtime.model_dtype,
                "compile_model": False,
                "gradient_checkpointing": False,
            },
        )

    def verify_processors(self, manifest, resolved):
        return verify_pi05_processor_assets(manifest, resolved)

    def verify_tokenizer(self, manifest, resolved, processor_verification):
        return verify_local_pi05_tokenizer_directory(
            manifest, resolved, processor_verification
        )

    def load_tokenizer(self, tokenizer_verification):
        return load_verified_local_pi05_tokenizer(tokenizer_verification)

    def build_processors(
        self, processor_verification, tokenizer_verification, tokenizer, *, device
    ):
        return build_verified_pi05_processor_pair(
            processor_verification,
            tokenizer_verification,
            tokenizer,
            device=device,
        )

    def deterministic_context(
        self, manifest
    ) -> AbstractContextManager[DeterministicTorchSettings]:
        return deterministic_torch_execution(
            device=manifest.runtime.device,
            model_dtype=manifest.runtime.model_dtype,
        )

    def policy_factory(
        self, policy_id, manifest, resolved, compatibility, staging_parent
    ):
        if policy_id == "base-pi05":
            checkpoint_input = manifest.base_checkpoint
            path = _inventory_path(resolved, "checkpoints.base")
        else:
            checkpoint_input = manifest.fine_tuned_checkpoint
            path = _inventory_path(resolved, "checkpoints.fineTuned")
        return create_pi05_policy_factory(
            policy_id=policy_id,
            effective_config=compatibility,
            checkpoint_input=checkpoint_input,
            checkpoint_path=path,
            runtime=manifest.runtime,
            staging_parent=staging_parent,
        )

    def project(self, manifest, resolved, observation, inference_result):
        if manifest.projection.mode == "unavailable":
            return unavailable_policy_comparison_trajectory_result(
                observation,
                inference_result,
                reason=manifest.projection.unavailable_reason or "",
            )
        robot = load_robot_model(_inventory_path(resolved, "robot.urdf"))
        left = RigidTransform(
            manifest.robot.left_arm_transform.translation_xyz,
            manifest.robot.left_arm_transform.rotation_rpy,
        )
        right = RigidTransform(
            manifest.robot.right_arm_transform.translation_xyz,
            manifest.robot.right_arm_transform.rotation_rpy,
        )
        left_mapping = tuple(build_trlc_dk1_joint_component_map("left").items())
        right_mapping = tuple(build_trlc_dk1_joint_component_map("right").items())
        robot_provenance = RobotProjectionProvenance(
            robot_model_name=robot.name,
            root_link=robot.root_link,
            target_link="tool0",
            urdf_sha256=manifest.robot.urdf.sha256,
            upstream_revision=manifest.robot.upstream_revision_identity,
            fk_implementation_id="lerobot-state-atlas.compute-tool-trajectory",
            fk_implementation_version="1.0",
            left_joint_mapping=left_mapping,
            right_joint_mapping=right_mapping,
            left_transform_translation_xyz=tuple(left.translation_xyz),
            left_transform_rotation_rpy=tuple(left.rotation_rpy),
            right_transform_translation_xyz=tuple(right.translation_xyz),
            right_transform_rotation_rpy=tuple(right.rotation_rpy),
            length_unit="metre",
            angle_unit="radian",
            handedness="right-handed",
            output_coordinate_frame="comparison-shared-world",
            rotation_representation="quaternion",
            rotation_component_order=("X", "Y", "Z", "W"),
            gripper_semantic_disclaimer=(
                "Raw generated gripper targets are device-specific unproven scalars, "
                "not physical jaw widths or calibrated gripper geometry."
            ),
            calibrated_gripper_geometry=False,
            calibrated_arm_transforms=manifest.robot.calibrated_arm_transforms,
        )
        interpretation = ActionInterpretationProvenance(
            interpretation_id=ABSOLUTE_INTERPRETATION_ID,
            interpretation_version=ABSOLUTE_INTERPRETATION_VERSION,
            use_relative_actions=False,
            delta_actions_preprocessor_enabled=False,
            absolute_actions_postprocessor_enabled=False,
            component_names=tuple(COMPONENT_NAMES),
            initial_state_participates=False,
            transformations=ABSOLUTE_TRANSFORMATIONS,
        )
        return project_policy_comparison_trajectories(
            observation,
            inference_result,
            rollout_adapter=PI05AbsolutePositionTargetsAdapter(interpretation),
            robot_model=robot,
            robot_provenance=robot_provenance,
            left_arm_transform=left,
            right_arm_transform=right,
            joint_limit_policy=manifest.projection.joint_limit_policy,
        )

    def install(
        self,
        destination,
        manifest_document,
        plans_document,
        receipt_document,
        *,
        replace_existing,
    ):
        return install_checkpoint_comparison_run(
            destination,
            manifest_document,
            plans_document,
            receipt_document,
            replace_existing=replace_existing,
        )


def _phase(name: str, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except CheckpointComparisonRunnerError:
        raise
    except BaseException as error:
        raise CheckpointComparisonRunnerError(name, str(error)) from error


@dataclass
class _RunnerStagingDirectory:
    path: Path

    @property
    def name(self) -> str:
        return str(self.path)

    def cleanup(self) -> None:
        shutil.rmtree(self.path)


def _create_runner_staging_directory(
    resolved: ResolvedRunnerInputs,
    additional_inputs: tuple[tuple[str, Path], ...],
    *,
    staging_root: str | Path | None = None,
    temp_directory: str | Path | None = None,
) -> _RunnerStagingDirectory:
    parent = resolve_runner_staging_root(
        temp_directory=staging_root if staging_root is not None else temp_directory
    )
    for _ in range(8):
        candidate = parent / f"lerobot-state-atlas-comparison-{uuid4().hex}"
        validate_output_input_disjointness(
            output_path=candidate,
            manifest_path=resolved.manifest_path,
            inventory_entries=resolved.inventory,
            additional_inputs=(
                ("output.runDirectory", resolved.output_run_directory),
                *additional_inputs,
            ),
        )
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return _RunnerStagingDirectory(candidate)
    raise ValueError("could not allocate a unique private runner staging directory.")


def _policy_identities(manifest) -> tuple[PolicyIdentity, PolicyIdentity]:
    return (
        PolicyIdentity("base-pi05", "Base π0.5", BASE_REPOSITORY, BASE_REVISION),
        PolicyIdentity(
            "fine-tuned-pi05", "Fine-tuned π0.5", FINE_REPOSITORY, FINE_REVISION
        ),
    )


def _policy_failure_phase(message: str) -> str:
    prefix = "fine-tuned" if "fine-tuned-pi05" in message else "base"
    lowered = message.lower()
    if "processed_observation" in lowered and "device" in lowered:
        return "processor-construction"
    if "noise.device" in lowered or "noise_device" in lowered:
        return "deterministic-runtime"
    if "cleanup" in lowered:
        return f"{prefix}-cleanup"
    if (
        "stage" in lowered
        or "checkpoint sha" in lowered
        or "checkpoint byte" in lowered
    ):
        return f"{prefix}-staging"
    if (
        "verification" in lowered
        or "loading failed" in lowered
        or "missing=" in lowered
    ):
        return f"{prefix}-loading"
    if "construct" in lowered or "could not move" in lowered:
        return f"{prefix}-construction"
    return f"{prefix}-inference"


def _receipt_document(
    manifest,
    resolved,
    preflight,
    observation,
    compatibility,
    processor_verification,
    tokenizer_verification,
    settings,
    inference_result,
    trajectory_result,
    manifest_bytes,
    plans_bytes,
):
    def relative(path: str | Path) -> str:
        return Path(path).relative_to(resolved.manifest_directory).as_posix()

    cameras = [
        {
            "featureName": camera.feature_name,
            "relativePath": relative(camera.path),
            "byteCount": camera.byte_count,
            "sha256": camera.sha256,
        }
        for camera in observation.cameras
    ]
    projection = trajectory_result.projection
    violations = (
        sum(len(plan.joint_limit_violations) for plan in projection.policies)
        if isinstance(projection, AvailablePolicyComparisonProjection)
        else 0
    )
    document = {
        "schema": {"name": RECEIPT_SCHEMA_NAME, "major": 1, "minor": 0},
        "runnerManifestSha256": manifest.manifest_sha256,
        "software": {
            "projectVersion": _package_version("lerobot-state-atlas"),
            "sourceIdentity": _source_identity(),
            "pythonVersion": platform.python_version(),
            "lerobotVersion": _package_version("lerobot"),
            "torchVersion": _package_version("torch"),
            "safetensorsVersion": _package_version("safetensors"),
            "pillowVersion": _package_version("pillow"),
            "transformersVersion": _package_version("transformers"),
        },
        "runtime": {
            "device": manifest.runtime.device,
            "modelDtype": manifest.runtime.model_dtype,
            "noiseDtype": manifest.runtime.noise_dtype,
            "cudaRuntime": str(torch.version.cuda or "unavailable"),
            "cudaDriver": "not-probed",
            "gpuName": preflight.gpu_name,
            "computeCapability": list(preflight.compute_capability),
            "deterministicSettings": dict(settings.as_items()),
        },
        "dataset": {
            "repositoryId": manifest.dataset.repository_id,
            "revision": manifest.dataset.revision,
        },
        "observation": {
            # These values identify the immutable snapshot parsed by the
            # observation loader, not the pathname's contents at receipt time.
            "manifestSha256": observation.manifest_sha256,
            "manifestByteCount": observation.manifest_byte_count,
            "observationId": observation.observation_id,
        },
        "cameras": cameras,
        "checkpoints": [
            {
                "policyId": "base-pi05",
                "repositoryId": manifest.base_checkpoint.repository_id,
                "revision": manifest.base_checkpoint.revision,
                "byteCount": manifest.base_checkpoint.byte_count,
                "sha256": manifest.base_checkpoint.sha256,
            },
            {
                "policyId": "fine-tuned-pi05",
                "repositoryId": manifest.fine_tuned_checkpoint.repository_id,
                "revision": manifest.fine_tuned_checkpoint.revision,
                "byteCount": manifest.fine_tuned_checkpoint.byte_count,
                "sha256": manifest.fine_tuned_checkpoint.sha256,
            },
        ],
        "modelConfiguration": {
            "sourceSha256": compatibility.source_sha256,
            "effectiveSha256": compatibility.effective_sha256,
            "transformations": [
                item.operation for item in compatibility.transformations
            ],
        },
        "processors": {
            "sourcePreprocessorSha256": processor_verification.compatibility.source_preprocessor_sha256,
            "effectivePreprocessorSha256": processor_verification.compatibility.effective_preprocessor_sha256,
            "preprocessorStateSha256": processor_verification.preprocessor_state.sha256,
            "sourcePostprocessorSha256": processor_verification.compatibility.source_postprocessor_sha256,
            "effectivePostprocessorSha256": processor_verification.compatibility.effective_postprocessor_sha256,
            "postprocessorStateSha256": processor_verification.postprocessor_state.sha256,
            "sharedForPolicyIds": list(processor_verification.shared_for_policy_ids),
        },
        "tokenizer": {
            "repositoryId": tokenizer_verification.repository_id,
            "directoryIdentitySha256": tokenizer_verification.directory_identity_sha256,
        },
        "comparison": {
            "policyOrder": ["base-pi05", "fine-tuned-pi05"],
            "numInferenceSteps": manifest.runtime.num_inference_steps,
            "noise": {
                "seed": inference_result.noise.seed,
                "shape": list(inference_result.noise.shape),
                "dtype": inference_result.noise.dtype,
                "sha256": inference_result.noise.sha256,
                "generator": inference_result.noise.generator,
            },
        },
        "projection": {
            "available": bool(projection.available),
            "actionInterpretationId": ABSOLUTE_INTERPRETATION_ID,
            "actionInterpretationVersion": ABSOLUTE_INTERPRETATION_VERSION,
            "urdfSha256": manifest.robot.urdf.sha256,
            "fkImplementationId": "lerobot-state-atlas.compute-tool-trajectory",
            "calibratedArmTransforms": manifest.robot.calibrated_arm_transforms,
            "calibratedGripperGeometry": False,
            "jointLimitPolicy": manifest.projection.joint_limit_policy,
            "jointLimitViolationCount": violations,
        },
        "artifact": {
            "schemaVersion": "1.1",
            "manifest": {
                "byteCount": len(manifest_bytes),
                "sha256": sha256_bytes(manifest_bytes),
            },
            "plans": {
                "byteCount": len(plans_bytes),
                "sha256": sha256_bytes(plans_bytes),
            },
        },
    }
    return validate_checkpoint_comparison_run_receipt(document)


def preflight_checkpoint_comparison_runner(
    runner_manifest_or_path: CheckpointComparisonRunnerManifest | str | Path,
    *,
    dependencies: Any | None = None,
) -> HardwarePreflightReport:
    """Load, resolve, and preflight without decoding cameras or constructing models."""
    deps = dependencies or DefaultCheckpointComparisonRunnerDependencies()
    manifest = (
        runner_manifest_or_path
        if isinstance(runner_manifest_or_path, CheckpointComparisonRunnerManifest)
        else _phase("manifest", deps.load_manifest, runner_manifest_or_path)
    )
    resolved = _phase("resolution", deps.resolve, manifest)
    return _phase("preflight", deps.preflight, manifest, resolved)


def execute_checkpoint_comparison_run(
    runner_manifest_or_path: CheckpointComparisonRunnerManifest | str | Path,
    *,
    dependencies: Any | None = None,
) -> CheckpointComparisonRunnerResult:
    """Execute one strict, local-only comparison and install no partial result."""
    deps = dependencies or DefaultCheckpointComparisonRunnerDependencies()
    manifest = (
        runner_manifest_or_path
        if isinstance(runner_manifest_or_path, CheckpointComparisonRunnerManifest)
        else _phase("manifest", deps.load_manifest, runner_manifest_or_path)
    )
    resolved = _phase("resolution", deps.resolve, manifest)
    preflight = _phase("preflight", deps.preflight, manifest, resolved)
    observation = _phase("observation", deps.load_observation, resolved)
    expected_observation = manifest.observation_manifest
    expected_observation_path = _inventory_path(resolved, "observationManifest")
    if observation.manifest_path != expected_observation_path:
        raise CheckpointComparisonRunnerError(
            "observation",
            "validated manifest path does not match the resolved runner input: "
            f"expected {expected_observation_path}, received "
            f"{observation.manifest_path}.",
        )
    if observation.manifest_byte_count != expected_observation.byte_count:
        raise CheckpointComparisonRunnerError(
            "observation",
            "validated manifest snapshot byte count does not match "
            "observationManifest.byteCount: expected "
            f"{expected_observation.byte_count}, received "
            f"{observation.manifest_byte_count}.",
        )
    if observation.manifest_sha256 != expected_observation.sha256:
        raise CheckpointComparisonRunnerError(
            "observation",
            "validated manifest snapshot SHA-256 does not match "
            "observationManifest.sha256: expected "
            f"{expected_observation.sha256}, received "
            f"{observation.manifest_sha256}.",
        )
    if (
        observation.dataset.repository_id != manifest.dataset.repository_id
        or observation.dataset.revision != manifest.dataset.revision
    ):
        raise CheckpointComparisonRunnerError(
            "observation",
            "dataset repository/revision does not match the runner manifest.",
        )
    camera_inputs = tuple(
        (f"observation.cameras[{index}]", camera.path)
        for index, camera in enumerate(observation.cameras)
    )
    camera_directories = tuple(
        (f"observation.cameras[{index}].sourceDirectory", camera.path.parent)
        for index, camera in enumerate(observation.cameras)
    )
    _phase(
        "resolution",
        validate_output_input_disjointness,
        output_path=resolved.output_run_directory,
        manifest_path=resolved.manifest_path,
        inventory_entries=resolved.inventory,
        additional_inputs=(
            ("observation.manifest", observation.manifest_path),
            *camera_inputs,
            *camera_directories,
        ),
    )
    bound_input = _phase("cameras", deps.prepare_cameras, observation)
    compatibility = _phase(
        "configuration", deps.adapt_configuration, manifest, resolved
    )
    processor_verification = _phase(
        "processor-verification", deps.verify_processors, manifest, resolved
    )
    tokenizer_verification = _phase(
        "tokenizer-verification",
        deps.verify_tokenizer,
        manifest,
        resolved,
        processor_verification,
    )
    tokenizer = _phase("tokenizer-loading", deps.load_tokenizer, tokenizer_verification)
    execution_device = _phase("processor-construction", deps.execution_device, manifest)
    processors = _phase(
        "processor-construction",
        deps.build_processors,
        processor_verification,
        tokenizer_verification,
        tokenizer,
        device=execution_device,
    )

    staging_additional_inputs = (
        ("observation.manifest", observation.manifest_path),
        *camera_inputs,
        *camera_directories,
    )
    temporary = _phase(
        "base-staging",
        _create_runner_staging_directory,
        resolved,
        staging_additional_inputs,
        staging_root=getattr(preflight, "checkpoint_staging_root", None),
    )
    primary_error: BaseException | None = None
    primary_traceback = None
    installation = None
    try:
        staging_parent = Path(temporary.name)

        def policy_factory(policy_id: str, phase: str):
            def create():
                return _phase(
                    phase,
                    deps.policy_factory,
                    policy_id,
                    manifest,
                    resolved,
                    compatibility,
                    staging_parent,
                )()

            return create

        base_factory = policy_factory("base-pi05", "base-construction")
        fine_factory = policy_factory("fine-tuned-pi05", "fine-tuned-construction")
        try:
            runtime_context = deps.deterministic_context(manifest)
            with runtime_context as settings, torch.inference_mode():
                inference_result = run_sequential_policy_comparison(
                    observation,
                    bound_input=bound_input,
                    preprocessor=processors.preprocessor,
                    postprocessor=processors.postprocessor,
                    base_policy_factory=base_factory,
                    fine_tuned_policy_factory=fine_factory,
                    noise_seed=manifest.runtime.noise_seed,
                    noise_device=execution_device,
                    noise_dtype=torch.float32,
                    num_inference_steps=manifest.runtime.num_inference_steps,
                    processed_observation_validator=lambda value: (
                        validate_processed_observation_device(value, execution_device)
                    ),
                )
        except BaseException as error:
            message = str(error)
            phase = _policy_failure_phase(message)
            raise CheckpointComparisonRunnerError(phase, message) from error
        trajectory_result = _phase(
            "projection",
            deps.project,
            manifest,
            resolved,
            observation,
            inference_result,
        )
        manifest_document, plans_document = _phase(
            "artifact",
            build_checkpoint_comparison_v1_1,
            observation,
            inference_result,
            trajectory_result,
            bundle_id=manifest.output.bundle_id,
            policies=_policy_identities(manifest),
            joint_limit_policy=manifest.projection.joint_limit_policy,
        )
        manifest_bytes, plans_bytes = build_checkpoint_comparison_documents(
            manifest_document, plans_document
        )
        receipt_document = _phase(
            "receipt",
            _receipt_document,
            manifest,
            resolved,
            preflight,
            observation,
            compatibility,
            processor_verification,
            tokenizer_verification,
            settings,
            inference_result,
            trajectory_result,
            manifest_bytes,
            plans_bytes,
        )
        installation = _phase(
            "installation",
            deps.install,
            resolved.output_run_directory,
            manifest_document,
            plans_document,
            receipt_document,
            replace_existing=manifest.output.replace_existing,
        )
    except BaseException as error:
        primary_error = error
        primary_traceback = error.__traceback__

    cleanup_failures = _attempt_cleanups(
        (("temporary-run-directory", Path(temporary.name), temporary.cleanup),),
        installed_output_remains_valid=installation is not None,
    )
    _resolve_cleanup_outcome(primary_error, primary_traceback, cleanup_failures)
    projection = trajectory_result.projection
    violation_count = (
        sum(len(plan.joint_limit_violations) for plan in projection.policies)
        if isinstance(projection, AvailablePolicyComparisonProjection)
        else 0
    )
    return CheckpointComparisonRunnerResult(
        run_directory=installation.run_directory,
        comparison_directory=installation.comparison_directory,
        receipt_path=installation.receipt_path,
        observation_id=observation.observation_id,
        schema_version="1.1",
        policy_order=("base-pi05", "fine-tuned-pi05"),
        projection_available=bool(projection.available),
        calibrated_arm_transforms=manifest.robot.calibrated_arm_transforms,
        joint_limit_violation_count=violation_count,
        checkpoint_sha256=(
            ("base-pi05", manifest.base_checkpoint.sha256),
            ("fine-tuned-pi05", manifest.fine_tuned_checkpoint.sha256),
        ),
        preflight=preflight,
    )
