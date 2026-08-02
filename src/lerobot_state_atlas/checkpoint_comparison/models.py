"""Small immutable domain models for checkpoint-comparison artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PolicyIdentity:
    """Immutable identity and display label for one compared policy."""

    policy_id: str
    label: str
    repository_id: str
    revision: str


@dataclass(frozen=True)
class NoiseProvenance:
    """Identity of shared diffusion noise without storing its tensor values."""

    shape: tuple[int, int, int]
    dtype: str
    seed: int
    sha256: str


@dataclass(frozen=True)
class GeneratedPolicyPlan:
    """One authoritative postprocessed action chunk."""

    policy_id: str
    relative_times_seconds: tuple[float, ...]
    actions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RecordedGroundTruth:
    """Optional recorded action evidence associated with the observation."""

    available: bool
    reason: str | None
    relative_times_seconds: tuple[float, ...] | None = None
    actions: tuple[tuple[float, ...], ...] | None = None


@dataclass(frozen=True)
class CheckpointComparisonExport:
    """Result of a validated atomic comparison-bundle installation."""

    output_path: Path
    bundle_id: str
    manifest_byte_count: int
    plans_byte_count: int
    plans_sha256: str


@dataclass(frozen=True)
class ObservationDatasetIdentity:
    """Pinned dataset sample identity for one policy observation."""

    repository_id: str
    revision: str
    episode_index: int
    frame_index: int
    timestamp_seconds: float
    fps: float
    task: str


@dataclass(frozen=True)
class ObservationState:
    """Canonical 14-component robot state supplied to both policies."""

    feature_name: str
    dtype: str
    component_names: tuple[str, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class ObservationCamera:
    """Validated camera-file identity without decoded pixel storage."""

    feature_name: str
    filename: str
    path: Path
    width: int
    height: int
    channels: int
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class ObservationRecordedGroundTruth:
    """Optional authoritative recorded action chunk."""

    available: bool
    reason: str | None
    component_names: tuple[str, ...] | None = None
    actions: tuple[tuple[float, ...], ...] | None = None
    relative_times_seconds: tuple[float, ...] | None = None
    frame_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PolicyComparisonObservation:
    """Fully validated local input shared by a policy comparison."""

    manifest_path: Path
    manifest_sha256: str
    manifest_byte_count: int
    observation_id: str
    dataset: ObservationDatasetIdentity
    prompt: str
    state: ObservationState
    cameras: tuple[ObservationCamera, ...]
    recorded_ground_truth: ObservationRecordedGroundTruth


@dataclass(frozen=True)
class BoundCameraInput:
    """Prepared camera tensor with declared validated-file provenance."""

    feature_name: str
    tensor: "torch.Tensor"
    source_sha256: str
    source_byte_count: int
    source_path: Path


@dataclass(frozen=True)
class BoundPolicyObservationInput:
    """Externally prepared camera inputs bound to one validated observation."""

    observation_id: str
    cameras: tuple[BoundCameraInput, ...]


@dataclass(frozen=True)
class CompatibilityTransformation:
    """One ordered fine-tuned configuration compatibility operation."""

    field: str
    operation: str
    detail: str


@dataclass(frozen=True)
class PI05CompatibilityResult:
    """Immutable, hashed PI05 constructor compatibility result."""

    source_config: tuple[tuple[str, object], ...]
    effective_config: tuple[tuple[str, object], ...]
    transformations: tuple[CompatibilityTransformation, ...]
    source_sha256: str
    effective_sha256: str
    installed_lerobot_version: str
    drop_unused_lm_head: bool
    runtime_overrides: tuple[tuple[str, object], ...]
    unknown_fields_resolved: bool


@dataclass(frozen=True)
class CheckpointTensorMetadata:
    """One source tensor and its verified target mapping."""

    original_key: str
    reconciled_target_key: str | None
    shape: tuple[int, ...]
    source_dtype: str
    transformation: str


@dataclass(frozen=True)
class CheckpointVerificationReport:
    """Complete immutable result of checkpoint-to-module verification."""

    checkpoint_kind: str
    checkpoint_sha256: str
    source_tensor_count: int
    expected_tensor_count: int
    loaded_tensor_count: int
    source_dtype_counts: tuple[tuple[str, int], ...]
    tensors: tuple[CheckpointTensorMetadata, ...]
    transformations: tuple[str, ...]
    explicitly_dropped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    duplicate_target_collisions: tuple[str, ...]
    verified: bool


@dataclass(frozen=True)
class DeterministicNoiseProvenance:
    """Reproducible shared diffusion-noise identity without tensor contents."""

    seed: int
    generator: str
    construction_device: str
    dtype: str
    shape: tuple[int, int, int]
    sha256: str


@dataclass(frozen=True)
class InferredPolicyPlan:
    """One immutable postprocessed policy action plan."""

    policy_id: str
    label: str
    relative_times_seconds: tuple[float, ...]
    actions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PolicyComparisonInferenceResult:
    """Complete deterministic result from exactly two supplied PI05 policies."""

    observation_id: str
    policies: tuple[InferredPolicyPlan, InferredPolicyPlan]
    noise: DeterministicNoiseProvenance
    action_dimension: int
    chunk_length: int
    num_inference_steps: int | None
    shared_preprocessing: bool
    shared_postprocessing: bool


@dataclass(frozen=True)
class ActionInterpretationProvenance:
    """Verified meaning of postprocessed policy actions."""

    interpretation_id: str
    interpretation_version: str
    use_relative_actions: bool
    delta_actions_preprocessor_enabled: bool
    absolute_actions_postprocessor_enabled: bool
    component_names: tuple[str, ...]
    initial_state_participates: bool
    transformations: tuple[str, ...]


@dataclass(frozen=True)
class ActionRolloutResult:
    """Immutable target states produced by one action-interpretation adapter."""

    interpretation: ActionInterpretationProvenance
    target_states: tuple[tuple[float, ...], ...]
    component_names: tuple[str, ...]
    fps: float
    initial_state: tuple[float, ...]


@dataclass(frozen=True)
class RobotProjectionProvenance:
    """Robot, FK, transform, and coordinate identity for a projection."""

    robot_model_name: str
    root_link: str
    target_link: str
    urdf_sha256: str
    upstream_revision: str | None
    fk_implementation_id: str
    fk_implementation_version: str
    left_joint_mapping: tuple[tuple[str, str], ...]
    right_joint_mapping: tuple[tuple[str, str], ...]
    left_transform_translation_xyz: tuple[float, float, float]
    left_transform_rotation_rpy: tuple[float, float, float]
    right_transform_translation_xyz: tuple[float, float, float]
    right_transform_rotation_rpy: tuple[float, float, float]
    length_unit: str
    angle_unit: str
    handedness: str
    output_coordinate_frame: str
    rotation_representation: str
    rotation_component_order: tuple[str, str, str, str]
    gripper_semantic_disclaimer: str
    calibrated_gripper_geometry: bool
    calibrated_arm_transforms: bool


@dataclass(frozen=True)
class JointLimitViolation:
    """One unchanged arm target outside a declared URDF joint bound."""

    policy_id: str
    step_index: int
    component_name: str
    urdf_joint_name: str
    value: float
    bound: float
    violation_kind: str


@dataclass(frozen=True)
class ProjectedArmTrajectory:
    """One generated arm tool trajectory and raw gripper targets."""

    arm_id: str
    target_link: str
    positions_xyz: tuple[tuple[float, float, float], ...]
    orientations_xyzw: tuple[tuple[float, float, float, float], ...]
    generated_raw_gripper_targets: tuple[float, ...]


@dataclass(frozen=True)
class ProjectedPolicyTrajectory:
    """Visualization projection for one generated policy plan."""

    policy_id: str
    label: str
    relative_times_seconds: tuple[float, ...]
    left: ProjectedArmTrajectory
    right: ProjectedArmTrajectory
    joint_limit_violations: tuple[JointLimitViolation, ...]
    action_interpretation: ActionInterpretationProvenance


@dataclass(frozen=True)
class AvailablePolicyComparisonProjection:
    """Available projection containing exactly two policy trajectories."""

    available: bool
    policies: tuple[ProjectedPolicyTrajectory, ProjectedPolicyTrajectory]
    robot: RobotProjectionProvenance


@dataclass(frozen=True)
class UnavailablePolicyComparisonProjection:
    """Explicit projection absence without fabricated trajectory fields."""

    available: bool
    reason: str


@dataclass(frozen=True)
class PolicyComparisonTrajectoryResult:
    """Available or explicitly unavailable comparison trajectory projection."""

    observation_id: str
    action_dimension: int
    chunk_length: int
    projection: (
        AvailablePolicyComparisonProjection | UnavailablePolicyComparisonProjection
    )
    shared_projection_configuration: bool
