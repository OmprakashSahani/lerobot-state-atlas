from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import socket

import pytest

from lerobot_state_atlas.checkpoint_comparison import (
    CheckpointComparisonPreflightError,
    HardwareProbeResult,
    ResourceMeasurement,
    RUNNER_STAGING_METADATA_OVERHEAD_BYTES,
    load_checkpoint_comparison_runner_manifest,
    preflight_checkpoint_comparison_run,
    resolve_checkpoint_comparison_runner_inputs,
)
from test_checkpoint_runner_manifest import _manifest


class FakeHardware:
    def __init__(self, **changes):
        values = {
            "cuda_available": True,
            "device_count": 1,
            "device_index": 0,
            "gpu_name": "Fake GPU",
            "compute_capability": (8, 0),
            "supported_cuda_architectures": ("sm_80", "sm_90"),
            "bfloat16_supported": True,
            "total_vram_bytes": 10_000,
            "free_vram_bytes": 9_000,
            "total_ram_bytes": 20_000,
            "available_ram_bytes": 18_000,
        }
        values.update(changes)
        self.result = HardwareProbeResult(**values)
        self.calls = 0

    def measure(self, device_index: int) -> HardwareProbeResult:
        self.calls += 1
        return self.result


class FakeFilesystem:
    def __init__(self, *, free: int = 100_000, writable: bool = True):
        self.free = free
        self.writable = writable
        self.calls: list[tuple[str, Path, int]] = []

    def disk_measurement(self, resource_id, path, required_bytes):
        self.calls.append((resource_id, path, required_bytes))
        return ResourceMeasurement(
            resource_id, path, "fake-fs", self.free, required_bytes
        )

    def is_writable_directory(self, path):
        return self.writable


class PathFilesystem(FakeFilesystem):
    def __init__(self, free_by_resource: dict[str, int], *, writable=True):
        super().__init__(writable=writable)
        self.free_by_resource = free_by_resource

    def disk_measurement(self, resource_id, path, required_bytes):
        self.calls.append((resource_id, path, required_bytes))
        return ResourceMeasurement(
            resource_id,
            path,
            "fake-fs",
            self.free_by_resource[resource_id],
            required_bytes,
        )


def _loaded(tmp_path: Path, mutate=None):
    path, _ = _manifest(tmp_path, mutate)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    return manifest, resolve_checkpoint_comparison_runner_inputs(manifest)


def test_success_report_is_immutable_deterministic_and_non_mutating(
    tmp_path: Path,
) -> None:
    manifest, resolved = _loaded(tmp_path)
    report = preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=FakeFilesystem(),
    )
    repeated = preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=FakeFilesystem(),
    )
    assert report == repeated
    assert report.passed is True
    assert report.compute_capability == (8, 0)
    assert report.base_checkpoint_stat_bytes == 4
    assert len(report.resource_measurements) == 3
    assert "not fully predicted" in " ".join(report.warnings)
    assert "were not applied" in " ".join(report.warnings)
    assert not resolved.output_run_directory.exists()
    with pytest.raises(FrozenInstanceError):
        report.passed = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"cuda_available": False, "device_count": 0}, "CUDA is unavailable"),
        ({"device_count": 0}, "device count"),
        ({"bfloat16_supported": False}, "bfloat16"),
        ({"supported_cuda_architectures": ()}, "could not be determined"),
        ({"compute_capability": (7, 5)}, "observed 7.5"),
        ({"free_vram_bytes": 0}, "minimumFreeVramBytes"),
        ({"available_ram_bytes": 0}, "minimumAvailableRamBytes"),
    ],
)
def test_hardware_failures(tmp_path: Path, changes: dict, match: str) -> None:
    manifest, resolved = _loaded(tmp_path)
    with pytest.raises(CheckpointComparisonPreflightError, match=match):
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=FakeHardware(**changes),
            filesystem_probe=FakeFilesystem(),
        )


def test_invalid_requested_device_index(tmp_path: Path) -> None:
    manifest, resolved = _loaded(
        tmp_path, lambda d: d["runtime"].update(device="cuda:1")
    )
    with pytest.raises(CheckpointComparisonPreflightError, match="device count"):
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=FakeHardware(device_count=1, device_index=1),
            filesystem_probe=FakeFilesystem(),
        )


def test_injectable_capability_decision(tmp_path: Path) -> None:
    manifest, resolved = _loaded(tmp_path)
    report = preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(compute_capability=(8, 6)),
        filesystem_probe=FakeFilesystem(),
        capability_checker=lambda observed, supported: observed == (8, 6),
    )
    assert report.passed


def test_insufficient_disk_and_unwritable_output(tmp_path: Path) -> None:
    manifest, resolved = _loaded(tmp_path)
    with pytest.raises(
        CheckpointComparisonPreflightError, match="requires 1 free bytes"
    ):
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=FakeHardware(),
            filesystem_probe=FakeFilesystem(free=0),
        )
    with pytest.raises(CheckpointComparisonPreflightError, match="not writable"):
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=FakeHardware(),
            filesystem_probe=FakeFilesystem(writable=False),
        )


def test_changed_or_missing_input_is_rejected_before_hardware(tmp_path: Path) -> None:
    manifest, resolved = _loaded(tmp_path)
    checkpoint = tmp_path / manifest.base_checkpoint.path
    checkpoint.write_bytes(b"longer")
    hardware = FakeHardware()
    with pytest.raises(
        CheckpointComparisonPreflightError, match="changed after resolution"
    ):
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=hardware,
            filesystem_probe=FakeFilesystem(),
        )
    assert hardware.calls == 0


def test_preflight_rechecks_output_inventory_disjointness_before_hardware(
    tmp_path: Path,
) -> None:
    manifest, resolved = _loaded(tmp_path)
    output = resolved.output_run_directory
    overlapping_entry = replace(
        resolved.inventory[1], canonical_path=output / "base.safetensors"
    )
    changed = replace(
        resolved,
        inventory=(
            resolved.inventory[0],
            overlapping_entry,
            *resolved.inventory[2:],
        ),
    )
    hardware = FakeHardware()
    with pytest.raises(CheckpointComparisonPreflightError) as caught:
        preflight_checkpoint_comparison_run(
            manifest,
            changed,
            hardware_probe=hardware,
            filesystem_probe=FakeFilesystem(),
        )
    assert "output/input overlap" in str(caught.value)
    assert "checkpoints.base" in str(caught.value)
    assert "output-contains-input" in str(caught.value)
    assert hardware.calls == 0


def test_output_conflict_respects_replace_intent(tmp_path: Path) -> None:
    manifest, resolved = _loaded(tmp_path)
    resolved.output_run_directory.mkdir(parents=True)
    with pytest.raises(CheckpointComparisonPreflightError, match="replaceExisting"):
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=FakeHardware(),
            filesystem_probe=FakeFilesystem(),
        )
    replacement_manifest = replace(
        manifest, output=replace(manifest.output, replace_existing=True)
    )
    replacement_resolved = resolve_checkpoint_comparison_runner_inputs(
        replacement_manifest
    )
    report = preflight_checkpoint_comparison_run(
        replacement_manifest,
        replacement_resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=FakeFilesystem(),
    )
    assert report.passed


def test_no_network_model_or_checkpoint_content_calls(
    monkeypatch, tmp_path: Path
) -> None:
    manifest, resolved = _loaded(tmp_path)
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("network")
    )
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail(f"content read: {self}")
    )
    report = preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=FakeFilesystem(),
    )
    assert report.passed


def test_preflight_does_not_mutate_torch_determinism_settings(tmp_path: Path) -> None:
    import torch

    manifest, resolved = _loaded(tmp_path)
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
    )
    preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=FakeFilesystem(),
    )
    after = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
    )
    assert after == before


def test_staging_capacity_uses_injected_temp_filesystem_and_largest_checkpoint(
    tmp_path: Path,
) -> None:
    manifest, resolved = _loaded(tmp_path)
    staging_root = tmp_path / "system-temp"
    staging_root.mkdir()
    filesystem = PathFilesystem(
        {
            "runner-manifest-filesystem": 10_000,
            "checkpoint-staging-filesystem": (
                RUNNER_STAGING_METADATA_OVERHEAD_BYTES + 3
            ),
            "output-filesystem": 10_000,
        }
    )
    with pytest.raises(CheckpointComparisonPreflightError) as caught:
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=FakeHardware(),
            filesystem_probe=filesystem,
            temp_directory=staging_root,
        )
    message = str(caught.value)
    assert "checkpoint staging" in message
    assert str(staging_root) in message
    assert "largest checkpoint is 4 bytes" in message
    effective = RUNNER_STAGING_METADATA_OVERHEAD_BYTES + 4
    assert f"effective required free bytes are {effective}" in message
    assert "shortfall is 1 byte" in message
    assert filesystem.calls[1] == (
        "checkpoint-staging-filesystem",
        staging_root,
        effective,
    )


@pytest.mark.parametrize("extra_free", [0, 1])
def test_staging_capacity_boundary_passes_and_does_not_sum_sequential_copies(
    tmp_path: Path, extra_free: int
) -> None:
    manifest, resolved = _loaded(tmp_path)
    staging_root = tmp_path / "system-temp"
    staging_root.mkdir()
    filesystem = PathFilesystem(
        {
            "runner-manifest-filesystem": 10_000,
            "checkpoint-staging-filesystem": (
                RUNNER_STAGING_METADATA_OVERHEAD_BYTES + 4 + extra_free
            ),
            "output-filesystem": 10_000,
        }
    )
    report = preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=filesystem,
        temp_directory=staging_root,
    )
    assert report.checkpoint_staging_root == staging_root
    assert report.checkpoint_staging_largest_checkpoint_bytes == 4
    assert (
        report.checkpoint_staging_metadata_overhead_bytes
        == RUNNER_STAGING_METADATA_OVERHEAD_BYTES
    )
    assert (
        report.checkpoint_staging_required_free_bytes
        == RUNNER_STAGING_METADATA_OVERHEAD_BYTES + 4
    )
    assert (
        report.base_checkpoint_stat_bytes + report.fine_tuned_checkpoint_stat_bytes == 8
    )


def test_configured_disk_minimum_can_determine_staging_requirement(
    tmp_path: Path,
) -> None:
    manifest, resolved = _loaded(
        tmp_path, lambda value: value["runtime"].update(minimumFreeDiskBytes=100_000)
    )
    staging_root = tmp_path / "system-temp"
    staging_root.mkdir()
    filesystem = FakeFilesystem(free=100_000)
    report = preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=filesystem,
        temp_directory=staging_root,
    )
    assert report.checkpoint_staging_required_free_bytes == 100_000
    assert filesystem.calls[1] == (
        "checkpoint-staging-filesystem",
        staging_root,
        100_000,
    )


def test_larger_fine_tuned_checkpoint_determines_sequential_requirement(
    tmp_path: Path,
) -> None:
    path, document = _manifest(tmp_path)
    fine_path = tmp_path / document["checkpoints"]["fineTuned"]["path"]
    fine_bytes = b"fine-tuned-is-larger"
    fine_path.write_bytes(fine_bytes)
    document["checkpoints"]["fineTuned"].update(
        byteCount=len(fine_bytes), sha256=hashlib.sha256(fine_bytes).hexdigest()
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    manifest = load_checkpoint_comparison_runner_manifest(path)
    resolved = resolve_checkpoint_comparison_runner_inputs(manifest)
    staging_root = tmp_path / "system-temp"
    staging_root.mkdir()
    report = preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=FakeFilesystem(
            free=len(fine_bytes) + RUNNER_STAGING_METADATA_OVERHEAD_BYTES
        ),
        temp_directory=staging_root,
    )
    assert report.base_checkpoint_stat_bytes == 4
    assert report.fine_tuned_checkpoint_stat_bytes == len(fine_bytes)
    assert report.checkpoint_staging_largest_checkpoint_bytes == len(fine_bytes)
    assert report.checkpoint_staging_required_free_bytes == (
        len(fine_bytes) + RUNNER_STAGING_METADATA_OVERHEAD_BYTES
    )


def test_low_source_capacity_is_not_used_for_staging_measurement(
    tmp_path: Path,
) -> None:
    manifest, resolved = _loaded(tmp_path)
    staging_root = tmp_path / "system-temp"
    staging_root.mkdir()
    filesystem = PathFilesystem(
        {
            "runner-manifest-filesystem": 1,
            "checkpoint-staging-filesystem": (
                RUNNER_STAGING_METADATA_OVERHEAD_BYTES + 4
            ),
            "output-filesystem": 1,
        }
    )
    report = preflight_checkpoint_comparison_run(
        manifest,
        resolved,
        hardware_probe=FakeHardware(),
        filesystem_probe=filesystem,
        temp_directory=staging_root,
    )
    assert report.passed
    assert filesystem.calls[1][1] == staging_root


@pytest.mark.parametrize(
    "kind", ["missing", "file", "direct-symlink", "intermediate-symlink"]
)
def test_invalid_staging_roots_are_rejected(tmp_path: Path, kind: str) -> None:
    real = tmp_path / "real"
    real.mkdir()
    if kind == "missing":
        staging_root = tmp_path / "missing"
    elif kind == "file":
        staging_root = tmp_path / "file"
        staging_root.write_text("not a directory", encoding="utf-8")
    elif kind == "direct-symlink":
        staging_root = tmp_path / "link"
        staging_root.symlink_to(real, target_is_directory=True)
    else:
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        staging_root = link / "child"
        (real / "child").mkdir()
    manifest, resolved = _loaded(tmp_path / "manifest")
    with pytest.raises(CheckpointComparisonPreflightError, match="temporary directory"):
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=FakeHardware(),
            filesystem_probe=FakeFilesystem(),
            temp_directory=staging_root,
        )


def test_unusable_staging_root_fails_before_hardware(tmp_path: Path) -> None:
    manifest, resolved = _loaded(tmp_path / "manifest")
    staging_root = tmp_path / "system-temp"
    staging_root.mkdir()
    hardware = FakeHardware()
    with pytest.raises(CheckpointComparisonPreflightError, match="not writable"):
        preflight_checkpoint_comparison_run(
            manifest,
            resolved,
            hardware_probe=hardware,
            filesystem_probe=FakeFilesystem(writable=False),
            temp_directory=staging_root,
        )
    assert hardware.calls == 0
