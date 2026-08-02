"""Injectable, non-mutating hardware and resource preflight for cloud runners."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat
from typing import Callable, Protocol

from lerobot_state_atlas.checkpoint_comparison.checkpoint_staging import (
    CheckpointStagingError,
    RUNNER_STAGING_METADATA_OVERHEAD_BYTES,
    resolve_runner_staging_root,
)

from lerobot_state_atlas.checkpoint_comparison.runner_models import (
    CheckpointComparisonRunnerManifest,
    HardwarePreflightReport,
    HardwareProbeResult,
    InputInventoryEntry,
    ResolvedRunnerInputs,
    ResourceMeasurement,
)
from lerobot_state_atlas.checkpoint_comparison.runner_manifest import (
    CheckpointComparisonRunnerManifestError,
    resolve_checkpoint_comparison_runner_inputs,
    validate_output_input_disjointness,
)


class CheckpointComparisonPreflightError(ValueError):
    """Raised when a real comparison cannot safely pass preflight."""


def _fail(field: str, message: str) -> None:
    raise CheckpointComparisonPreflightError(f"{field} {message}")


class HardwareProbe(Protocol):
    def measure(self, device_index: int) -> HardwareProbeResult: ...


class FilesystemProbe(Protocol):
    def disk_measurement(
        self, resource_id: str, path: Path, required_bytes: int
    ) -> ResourceMeasurement: ...

    def is_writable_directory(self, path: Path) -> bool: ...


@dataclass(frozen=True)
class LocalHardwareProbe:
    """Read local Torch/Linux measurements without allocating CUDA tensors."""

    def measure(self, device_index: int) -> HardwareProbeResult:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        if not cuda_available or device_index >= device_count:
            return HardwareProbeResult(
                cuda_available=cuda_available,
                device_count=device_count,
                device_index=device_index,
                gpu_name="",
                compute_capability=(0, 0),
                supported_cuda_architectures=tuple(torch.cuda.get_arch_list()),
                bfloat16_supported=False,
                total_vram_bytes=0,
                free_vram_bytes=0,
                total_ram_bytes=_total_ram_bytes(),
                available_ram_bytes=_available_ram_bytes(),
            )
        properties = torch.cuda.get_device_properties(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        free_vram, total_vram = torch.cuda.mem_get_info(device_index)
        return HardwareProbeResult(
            cuda_available=True,
            device_count=device_count,
            device_index=device_index,
            gpu_name=str(properties.name),
            compute_capability=(int(capability[0]), int(capability[1])),
            supported_cuda_architectures=tuple(torch.cuda.get_arch_list()),
            # PI05 requires native BF16 support; capability >= 8 is the
            # allocation-free CUDA criterion used by Torch itself.
            bfloat16_supported=bool(int(capability[0]) >= 8),
            total_vram_bytes=int(total_vram),
            free_vram_bytes=int(free_vram),
            total_ram_bytes=_total_ram_bytes(),
            available_ram_bytes=_available_ram_bytes(),
        )


@dataclass(frozen=True)
class LocalFilesystemProbe:
    """Read filesystem capacity and permission metadata without writing output."""

    def disk_measurement(
        self, resource_id: str, path: Path, required_bytes: int
    ) -> ResourceMeasurement:
        usage = shutil.disk_usage(path)
        filesystem_identity = str(path.stat().st_dev)
        return ResourceMeasurement(
            resource_id,
            path,
            filesystem_identity,
            int(usage.free),
            required_bytes,
        )

    def is_writable_directory(self, path: Path) -> bool:
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def _total_ram_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def _available_ram_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))


def _device_index(device: str) -> int:
    try:
        prefix, raw_index = device.split(":", 1)
        if prefix != "cuda" or not raw_index.isdigit():
            raise ValueError
        return int(raw_index)
    except ValueError as error:
        raise CheckpointComparisonPreflightError(
            "runtime.device must use syntax 'cuda:<nonnegative index>'."
        ) from error


def _default_capability_checker(
    capability: tuple[int, int], supported: tuple[str, ...]
) -> bool:
    expected = f"sm_{capability[0]}{capability[1]}"
    return expected in supported


def _recheck_entry(entry: InputInventoryEntry) -> None:
    path = entry.canonical_path
    if path.is_symlink():
        _fail(entry.logical_input_id, f"path became a symbolic link: {path}.")
    if not path.exists():
        _fail(entry.logical_input_id, f"path disappeared after resolution: {path}.")
    info = path.stat()
    expected_mode = stat.S_ISREG if entry.kind == "file" else stat.S_ISDIR
    if not expected_mode(info.st_mode):
        _fail(
            entry.logical_input_id,
            f"expected {entry.kind} at {path} but its type changed.",
        )
    if entry.kind == "file" and info.st_size != entry.actual_stat_byte_count:
        _fail(
            f"{entry.logical_input_id}.byteCount",
            f"changed from {entry.actual_stat_byte_count} to {info.st_size} bytes.",
        )
    previous_identity = (
        entry.stat_device,
        entry.stat_inode,
        entry.stat_mtime_ns,
        entry.stat_ctime_ns,
    )
    current_identity = (
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    if current_identity != previous_identity:
        _fail(entry.logical_input_id, f"metadata changed after resolution for {path}.")


def _validate_probe_result(result: HardwareProbeResult, requested_index: int) -> None:
    if not isinstance(result, HardwareProbeResult):
        _fail("hardwareProbe", "must return HardwareProbeResult.")
    for field in (
        "device_count",
        "device_index",
        "total_vram_bytes",
        "free_vram_bytes",
        "total_ram_bytes",
        "available_ram_bytes",
    ):
        value = getattr(result, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"hardwareProbe.{field}", "must be a nonnegative integer.")
    if result.device_index != requested_index:
        _fail(
            "hardwareProbe.device_index",
            f"must report requested device index {requested_index}.",
        )
    for field in ("cuda_available", "bfloat16_supported"):
        if not isinstance(getattr(result, field), bool):
            _fail(f"hardwareProbe.{field}", "must be boolean.")
    if not isinstance(result.gpu_name, str):
        _fail("hardwareProbe.gpu_name", "must be a string.")
    if (
        not isinstance(result.compute_capability, tuple)
        or len(result.compute_capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in result.compute_capability
        )
    ):
        _fail(
            "hardwareProbe.compute_capability", "must contain two nonnegative integers."
        )
    if not isinstance(result.supported_cuda_architectures, tuple) or any(
        not isinstance(value, str) or not value
        for value in result.supported_cuda_architectures
    ):
        _fail("hardwareProbe.supported_cuda_architectures", "must contain strings.")


def preflight_checkpoint_comparison_run(
    manifest: CheckpointComparisonRunnerManifest,
    resolved_inputs: ResolvedRunnerInputs,
    *,
    hardware_probe: HardwareProbe | None = None,
    filesystem_probe: FilesystemProbe | None = None,
    capability_checker: Callable[[tuple[int, int], tuple[str, ...]], bool]
    | None = None,
    temp_directory: str | Path | None = None,
) -> HardwarePreflightReport:
    """Measure a CUDA host and recheck static inputs without loading their content."""
    if manifest.manifest_sha256 != resolved_inputs.manifest_sha256:
        _fail("resolvedInputs.manifestSha256", "must match the runner manifest.")
    if manifest.manifest_path != resolved_inputs.manifest_path:
        _fail("resolvedInputs.manifestPath", "must match the runner manifest.")
    try:
        validate_output_input_disjointness(
            output_path=resolved_inputs.output_run_directory,
            manifest_path=resolved_inputs.manifest_path,
            inventory_entries=resolved_inputs.inventory,
        )
    except CheckpointComparisonRunnerManifestError as error:
        raise CheckpointComparisonPreflightError(
            f"resolvedInputs output/input overlap detected: {error}"
        ) from error
    try:
        current_resolution = resolve_checkpoint_comparison_runner_inputs(manifest)
    except CheckpointComparisonRunnerManifestError as error:
        raise CheckpointComparisonPreflightError(
            f"resolvedInputs changed after resolution: {error}"
        ) from error
    if (
        current_resolution.inventory != resolved_inputs.inventory
        or current_resolution.output_run_directory
        != resolved_inputs.output_run_directory
        or current_resolution.manifest_path != resolved_inputs.manifest_path
        or current_resolution.manifest_sha256 != resolved_inputs.manifest_sha256
    ):
        _fail("resolvedInputs", "changed after initial resolution.")
    for entry in resolved_inputs.inventory:
        _recheck_entry(entry)
    by_id = {entry.logical_input_id: entry for entry in resolved_inputs.inventory}
    expected_ids = {
        "observationManifest",
        "checkpoints.base",
        "checkpoints.fineTuned",
        "configuration",
        "processors.preprocessorConfig",
        "processors.preprocessorState",
        "processors.postprocessorConfig",
        "processors.postprocessorState",
        "robot.urdf",
        "processors.tokenizerDirectory",
    }
    if set(by_id) != expected_ids or len(by_id) != len(resolved_inputs.inventory):
        _fail(
            "resolvedInputs.inventory",
            "does not contain the exact required logical inputs.",
        )
    base_size = by_id["checkpoints.base"].actual_stat_byte_count or 0
    fine_size = by_id["checkpoints.fineTuned"].actual_stat_byte_count or 0
    if base_size <= 0:
        _fail("checkpoints.base.byteCount", "must be nonzero.")
    if fine_size <= 0:
        _fail("checkpoints.fineTuned.byteCount", "must be nonzero.")

    output = resolved_inputs.output_run_directory
    if output.is_symlink():
        _fail("output.runDirectory", "must not be a symbolic link.")
    if output.exists():
        if not output.is_dir():
            _fail("output.runDirectory", "existing destination must be a directory.")
        if not manifest.output.replace_existing:
            _fail(
                "output.replaceExisting",
                f"must be true because destination already exists: {output}.",
            )
    fs_probe = filesystem_probe or LocalFilesystemProbe()
    try:
        staging_root = resolve_runner_staging_root(
            temp_directory=temp_directory,
            usability_checker=fs_probe.is_writable_directory,
        )
    except CheckpointStagingError as error:
        _fail("resources.checkpoint-staging-filesystem", str(error))
    if not fs_probe.is_writable_directory(resolved_inputs.output_existing_ancestor):
        _fail(
            "output.runDirectory",
            f"existing ancestor is not writable: {resolved_inputs.output_existing_ancestor}.",
        )

    index = _device_index(manifest.runtime.device)
    measured = (hardware_probe or LocalHardwareProbe()).measure(index)
    _validate_probe_result(measured, index)
    if not measured.cuda_available:
        _fail(
            "runtime.device",
            f"requested {manifest.runtime.device} but CUDA is unavailable.",
        )
    if index >= measured.device_count:
        _fail(
            "runtime.device",
            f"requested CUDA index {index}, but device count is {measured.device_count}.",
        )
    if not measured.supported_cuda_architectures:
        _fail(
            "hardware.computeCapability",
            "Torch supported CUDA architecture list could not be determined.",
        )
    checker = capability_checker or _default_capability_checker
    if not checker(measured.compute_capability, measured.supported_cuda_architectures):
        _fail(
            "hardware.computeCapability",
            f"observed {measured.compute_capability[0]}.{measured.compute_capability[1]} "
            f"but Torch reports {list(measured.supported_cuda_architectures)!r}.",
        )
    if manifest.runtime.model_dtype == "bfloat16" and not measured.bfloat16_supported:
        _fail(
            "runtime.modelDtype",
            f"bfloat16 is unsupported on requested device {manifest.runtime.device}.",
        )
    if measured.free_vram_bytes < manifest.runtime.minimum_free_vram_bytes:
        _fail(
            "runtime.minimumFreeVramBytes",
            f"requires {manifest.runtime.minimum_free_vram_bytes} bytes but observed "
            f"{measured.free_vram_bytes}.",
        )
    if measured.available_ram_bytes < manifest.runtime.minimum_available_ram_bytes:
        _fail(
            "runtime.minimumAvailableRamBytes",
            f"requires {manifest.runtime.minimum_available_ram_bytes} bytes but observed "
            f"{measured.available_ram_bytes}.",
        )

    largest_checkpoint_size = max(base_size, fine_size)
    staging_required_bytes = max(
        manifest.runtime.minimum_free_disk_bytes,
        largest_checkpoint_size + RUNNER_STAGING_METADATA_OVERHEAD_BYTES,
    )
    disk_paths = (
        (
            "runner-manifest-filesystem",
            resolved_inputs.manifest_directory,
            manifest.runtime.minimum_free_disk_bytes,
        ),
        (
            "checkpoint-staging-filesystem",
            staging_root,
            staging_required_bytes,
        ),
        (
            "output-filesystem",
            resolved_inputs.output_existing_ancestor,
            manifest.runtime.minimum_free_disk_bytes,
        ),
    )
    disk = tuple(
        fs_probe.disk_measurement(resource_id, path, required_bytes)
        for resource_id, path, required_bytes in disk_paths
    )
    for (resource_id, path, required_bytes), measurement in zip(
        disk_paths, disk, strict=True
    ):
        if not isinstance(measurement, ResourceMeasurement):
            _fail(f"resources.{resource_id}", "probe must return ResourceMeasurement.")
        if measurement.resource_id != resource_id or measurement.path != path:
            _fail(
                f"resources.{resource_id}",
                "probe result must identify the requested resource and path.",
            )
        if (
            isinstance(measurement.observed_bytes, bool)
            or not isinstance(measurement.observed_bytes, int)
            or measurement.observed_bytes < 0
        ):
            _fail(f"resources.{resource_id}.observedBytes", "must be nonnegative.")
        if measurement.required_bytes != required_bytes:
            _fail(
                f"resources.{resource_id}.requiredBytes",
                f"must equal effective requirement {required_bytes}.",
            )
        if (
            not isinstance(measurement.filesystem_identity, str)
            or not measurement.filesystem_identity
        ):
            _fail(f"resources.{resource_id}.filesystemIdentity", "must be non-empty.")
        if measurement.observed_bytes < measurement.required_bytes:
            if resource_id == "checkpoint-staging-filesystem":
                shortfall = measurement.required_bytes - measurement.observed_bytes
                _fail(
                    "resources.checkpoint-staging-filesystem",
                    f"checkpoint staging at {measurement.path} observed "
                    f"{measurement.observed_bytes} free bytes; largest checkpoint is "
                    f"{largest_checkpoint_size} bytes; configured minimumFreeDiskBytes is "
                    f"{manifest.runtime.minimum_free_disk_bytes}; effective required free "
                    f"bytes are {measurement.required_bytes}; shortfall is {shortfall} "
                    f"{'byte' if shortfall == 1 else 'bytes'}.",
                )
            _fail(
                f"resources.{measurement.resource_id}",
                f"requires {measurement.required_bytes} free bytes at {measurement.path} "
                f"but observed {measurement.observed_bytes}.",
            )

    return HardwarePreflightReport(
        manifest_sha256=manifest.manifest_sha256,
        requested_device=manifest.runtime.device,
        device_index=index,
        cuda_available=measured.cuda_available,
        gpu_name=measured.gpu_name,
        compute_capability=measured.compute_capability,
        torch_supported_cuda_architectures=measured.supported_cuda_architectures,
        bfloat16_supported=measured.bfloat16_supported,
        total_vram_bytes=measured.total_vram_bytes,
        free_vram_bytes=measured.free_vram_bytes,
        total_ram_bytes=measured.total_ram_bytes,
        available_ram_bytes=measured.available_ram_bytes,
        resource_measurements=disk,
        configured_minimum_free_vram_bytes=manifest.runtime.minimum_free_vram_bytes,
        configured_minimum_available_ram_bytes=manifest.runtime.minimum_available_ram_bytes,
        configured_minimum_free_disk_bytes=manifest.runtime.minimum_free_disk_bytes,
        input_inventory=resolved_inputs.inventory,
        base_checkpoint_stat_bytes=base_size,
        fine_tuned_checkpoint_stat_bytes=fine_size,
        checkpoint_staging_root=staging_root,
        checkpoint_staging_largest_checkpoint_bytes=largest_checkpoint_size,
        checkpoint_staging_metadata_overhead_bytes=RUNNER_STAGING_METADATA_OVERHEAD_BYTES,
        checkpoint_staging_required_free_bytes=staging_required_bytes,
        warnings=(
            "Input content hashes, tokenizer completeness, and checkpoint tensors remain unverified.",
            "Model activations, decoded tensors, mmap behavior, and CUDA allocator overhead are not fully predicted.",
            "Required deterministic backend settings are reported but were not applied by preflight.",
        ),
        required_deterministic_settings=(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization",
            "torch.use_deterministic_algorithms(True)",
            "torch.backends.cudnn.deterministic=True",
            "torch.backends.cudnn.benchmark=False",
            "torch.backends.cuda.matmul.allow_tf32=False",
            "torch.backends.cudnn.allow_tf32=False",
            "torch.set_float32_matmul_precision('highest')",
            "policy.eval() under torch.inference_mode()",
        ),
        passed=True,
    )
