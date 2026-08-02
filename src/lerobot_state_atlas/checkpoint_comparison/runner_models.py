"""Immutable inputs and reports for checkpoint-comparison runner preflight."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunnerSchemaVersion:
    name: str
    major: int
    minor: int


@dataclass(frozen=True)
class RunnerDatasetIdentity:
    repository_id: str
    revision: str


@dataclass(frozen=True)
class RunnerFileInput:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class RunnerCheckpointInput(RunnerFileInput):
    repository_id: str
    revision: str


@dataclass(frozen=True)
class RunnerTokenizerInput:
    path: str
    identity_sha256: str


@dataclass(frozen=True)
class RunnerProcessorInputs:
    preprocessor_config: RunnerFileInput
    preprocessor_state: RunnerFileInput
    postprocessor_config: RunnerFileInput
    postprocessor_state: RunnerFileInput
    tokenizer_directory: RunnerTokenizerInput


@dataclass(frozen=True)
class RunnerRigidTransform:
    translation_xyz: tuple[float, float, float]
    rotation_rpy: tuple[float, float, float]


@dataclass(frozen=True)
class RunnerRobotConfiguration:
    urdf: RunnerFileInput
    upstream_revision_identity: str
    left_arm_transform: RunnerRigidTransform
    right_arm_transform: RunnerRigidTransform
    calibrated_arm_transforms: bool


@dataclass(frozen=True)
class RunnerRuntimeConfiguration:
    device: str
    model_dtype: str
    noise_dtype: str
    noise_seed: int
    num_inference_steps: int
    minimum_free_vram_bytes: int
    minimum_available_ram_bytes: int
    minimum_free_disk_bytes: int


@dataclass(frozen=True)
class RunnerProjectionConfiguration:
    mode: str
    joint_limit_policy: str
    unavailable_reason: str | None
    acknowledge_uncalibrated_arm_transforms: bool
    acknowledge_recorded_limit_violations: bool


@dataclass(frozen=True)
class RunnerOutputConfiguration:
    run_directory: str
    bundle_id: str
    replace_existing: bool


@dataclass(frozen=True)
class CheckpointComparisonRunnerManifest:
    manifest_path: Path
    manifest_sha256: str
    schema: RunnerSchemaVersion
    dataset: RunnerDatasetIdentity
    observation_manifest: RunnerFileInput
    base_checkpoint: RunnerCheckpointInput
    fine_tuned_checkpoint: RunnerCheckpointInput
    configuration: RunnerFileInput
    processors: RunnerProcessorInputs
    robot: RunnerRobotConfiguration
    runtime: RunnerRuntimeConfiguration
    projection: RunnerProjectionConfiguration
    output: RunnerOutputConfiguration


@dataclass(frozen=True)
class InputInventoryEntry:
    logical_input_id: str
    canonical_path: Path
    expected_byte_count: int | None
    actual_stat_byte_count: int | None
    expected_sha256: str
    kind: str
    content_hash_pending: bool
    stat_device: int
    stat_inode: int
    stat_mtime_ns: int
    stat_ctime_ns: int


@dataclass(frozen=True)
class ResolvedRunnerInputs:
    manifest_path: Path
    manifest_sha256: str
    manifest_directory: Path
    inventory: tuple[InputInventoryEntry, ...]
    output_run_directory: Path
    output_existing_ancestor: Path


@dataclass(frozen=True)
class HardwareProbeResult:
    cuda_available: bool
    device_count: int
    device_index: int
    gpu_name: str
    compute_capability: tuple[int, int]
    supported_cuda_architectures: tuple[str, ...]
    bfloat16_supported: bool
    total_vram_bytes: int
    free_vram_bytes: int
    total_ram_bytes: int
    available_ram_bytes: int


@dataclass(frozen=True)
class ResourceMeasurement:
    resource_id: str
    path: Path
    filesystem_identity: str
    observed_bytes: int
    required_bytes: int


@dataclass(frozen=True)
class HardwarePreflightReport:
    manifest_sha256: str
    requested_device: str
    device_index: int
    cuda_available: bool
    gpu_name: str
    compute_capability: tuple[int, int]
    torch_supported_cuda_architectures: tuple[str, ...]
    bfloat16_supported: bool
    total_vram_bytes: int
    free_vram_bytes: int
    total_ram_bytes: int
    available_ram_bytes: int
    resource_measurements: tuple[ResourceMeasurement, ...]
    configured_minimum_free_vram_bytes: int
    configured_minimum_available_ram_bytes: int
    configured_minimum_free_disk_bytes: int
    input_inventory: tuple[InputInventoryEntry, ...]
    base_checkpoint_stat_bytes: int
    fine_tuned_checkpoint_stat_bytes: int
    checkpoint_staging_root: Path
    checkpoint_staging_largest_checkpoint_bytes: int
    checkpoint_staging_metadata_overhead_bytes: int
    checkpoint_staging_required_free_bytes: int
    warnings: tuple[str, ...]
    required_deterministic_settings: tuple[str, ...]
    passed: bool


@dataclass(frozen=True)
class ProcessorConfigTransformation:
    processor_side: str
    source_path: str
    transformation_id: str
    source_value: str
    effective_value: str
    reason: str


@dataclass(frozen=True)
class ProcessorStepSummary:
    processor_side: str
    index: int
    source_step_name: str
    effective_step_name: str
    processor_type: str
    enabled: bool | None
    learned_state_required: bool
    learned_state_logical_input_id: str | None


@dataclass(frozen=True)
class ProcessorNormalizationContract:
    visual_mode: str
    state_mode: str
    action_mode: str
    state_component_names: tuple[str, ...]
    action_component_names: tuple[str, ...]
    expected_state_quantile_keys: tuple[str, str]
    expected_action_quantile_keys: tuple[str, str]
    state_dimension: int
    action_dimension: int


@dataclass(frozen=True)
class ProcessorStateSummary:
    logical_input_id: str
    source_path: Path
    byte_count: int
    sha256: str
    tensor_keys: tuple[str, ...]
    tensor_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    tensor_dtypes: tuple[tuple[str, str], ...]
    finite_values_verified: bool


@dataclass(frozen=True)
class PI05ProcessorCompatibilityResult:
    installed_lerobot_version: str
    source_preprocessor_sha256: str
    source_postprocessor_sha256: str
    effective_preprocessor_sha256: str
    effective_postprocessor_sha256: str
    effective_preprocessor_json: str
    effective_postprocessor_json: str
    transformations: tuple[ProcessorConfigTransformation, ...]
    preprocessor_steps: tuple[ProcessorStepSummary, ...]
    postprocessor_steps: tuple[ProcessorStepSummary, ...]
    tokenizer_repository_id: str
    tokenizer_verification_pending: bool
    use_relative_actions: bool
    delta_actions_preprocessor_enabled: bool
    absolute_actions_postprocessor_enabled: bool
    normalization: ProcessorNormalizationContract


@dataclass(frozen=True)
class PI05ProcessorVerificationResult:
    processor_identity: str
    compatibility: PI05ProcessorCompatibilityResult
    preprocessor_state: ProcessorStateSummary
    postprocessor_state: ProcessorStateSummary
    cross_state_consistent: bool
    shared_for_policy_ids: tuple[str, str]
    fairness_statement: str


@dataclass(frozen=True)
class TokenizerFileIdentity:
    relative_path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class PI05TokenizerVerificationResult:
    repository_id: str
    directory_path: Path
    directory_identity_sha256: str
    files: tuple[TokenizerFileIdentity, ...]
    required_layout_satisfied: bool
    layout_uncertainty: tuple[str, ...]
    local_only_required: bool


@dataclass(frozen=True)
class CheckpointComparisonRunnerResult:
    """Installed result of one complete, all-or-nothing comparison run."""

    run_directory: Path
    comparison_directory: Path
    receipt_path: Path
    observation_id: str
    schema_version: str
    policy_order: tuple[str, str]
    projection_available: bool
    calibrated_arm_transforms: bool
    joint_limit_violation_count: int
    checkpoint_sha256: tuple[tuple[str, str], tuple[str, str]]
    preflight: HardwarePreflightReport
