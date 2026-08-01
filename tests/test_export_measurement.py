import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from lerobot_state_atlas.export_measurement import (
    ArmCoverageCounts,
    ExportMeasurementSession,
    PeakRssMeasurement,
    measure_bundle_artifacts,
    validate_measurement_report_path,
    write_measurement_report_atomic,
)


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        result = self.value
        self.value += 1.0
        return result


def make_artifact_bundle(path: Path) -> dict:
    path.mkdir()
    (path / "manifest.json").write_bytes(b'{"bundleId":"measured"}\n')
    (path / "coverage.json").write_bytes(b'{"arms":[]}\n')
    media = path / "media/episode-0/top.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"test-video-bytes")
    return {
        "payloads": [
            {
                "kind": "coverage",
                "filename": "coverage.json",
            }
        ]
    }


def test_measurement_report_path_must_be_outside_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"

    with pytest.raises(ValueError, match="outside the browser-data bundle"):
        validate_measurement_report_path(bundle, bundle)
    with pytest.raises(ValueError, match="outside the browser-data bundle"):
        validate_measurement_report_path(bundle / "measurement.json", bundle)
    with pytest.raises(ValueError, match="outside the browser-data bundle"):
        validate_measurement_report_path(
            bundle / "nested/reports/measurement.json",
            bundle,
        )

    validate_measurement_report_path(tmp_path / "measurement.json", bundle)
    validate_measurement_report_path(tmp_path / "reports/measurement.json", bundle)


def test_artifact_measurement_records_exact_and_gzip_sizes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    manifest = make_artifact_bundle(bundle)

    first = measure_bundle_artifacts(bundle, manifest)
    second = measure_bundle_artifacts(bundle, manifest)

    assert first == second
    assert first["files"] == [
        {
            "filename": "coverage.json",
            "payloadKind": "coverage",
            "uncompressedBytes": 12,
            "gzipLevel9Bytes": first["files"][0]["gzipLevel9Bytes"],
        },
        {
            "filename": "manifest.json",
            "payloadKind": "manifest",
            "uncompressedBytes": 24,
            "gzipLevel9Bytes": first["files"][1]["gzipLevel9Bytes"],
        },
        {
            "filename": "media/episode-0/top.mp4",
            "payloadKind": None,
            "uncompressedBytes": 16,
            "gzipLevel9Bytes": first["files"][2]["gzipLevel9Bytes"],
        },
    ]
    assert all(value["gzipLevel9Bytes"] > 0 for value in first["files"])
    assert first["totalUncompressedBytes"] == 52
    assert first["totalGzipLevel9Bytes"] == sum(
        value["gzipLevel9Bytes"] for value in first["files"]
    )
    assert not tuple(bundle.rglob("*.gz"))


def test_exact_report_structure_uses_injected_clocks_and_rss() -> None:
    clock = IncrementingClock()
    session = ExportMeasurementSession(
        monotonic_clock=clock,
        wall_clock=lambda: datetime(2026, 7, 31, 12, 30, tzinfo=timezone.utc),
        rss_reader=lambda: PeakRssMeasurement(
            True,
            123_456,
            "fake.peak_rss",
            "bytes",
        ),
    )
    session.start_export()
    with session.stage("sourcePreparation"):
        pass
    with session.stage("coverageAggregation"):
        pass
    batch_started = session.begin_coverage_batch()
    session.complete_coverage_batch(
        started_at=batch_started,
        episode_ids=(2, 3),
        frame_count=7,
        cumulative_frame_count=7,
        arms={
            "left": ArmCoverageCounts(2, 3, 7),
            "right": ArmCoverageCounts(1, 2, 7),
        },
    )
    session.finish_export()
    artifacts = {
        "files": [],
        "totalUncompressedBytes": 0,
        "totalGzipLevel9Bytes": 0,
    }

    report = session.build_report(
        repository_id="organization/dataset",
        requested_revision="v3.0",
        resolved_revision="a" * 40,
        source_episode_count=1_344,
        coverage_episode_ids=(2, 3),
        trajectory_episode_ids=(2,),
        episode_batch_size=2,
        voxel_size=0.02,
        arm_spacing=0.8,
        selected_frame_count=7,
        final_arms={
            "left": ArmCoverageCounts(2, 3, 7),
            "right": ArmCoverageCounts(1, 2, 7),
        },
        trajectory_sample_counts={2: 4},
        artifacts=artifacts,
    )

    assert report["reportFormat"] == {
        "name": "lerobot-state-atlas.export-measurement",
        "version": 1,
    }
    assert report["generatedAtUtc"] == "2026-07-31T12:30:00Z"
    assert "machine- and platform-dependent" in report["machineDependentDisclosure"]
    assert set(report["platform"]) == {"system", "release", "machine"}
    assert report["pythonVersion"]
    assert report["dataset"] == {
        "repositoryId": "organization/dataset",
        "requestedRevision": "v3.0",
        "resolvedRevision": "a" * 40,
        "sourceEpisodeCount": 1_344,
    }
    assert report["selection"] == {
        "coverageEpisodeIds": [2, 3],
        "coverageEpisodeRange": {"minimum": 2, "maximum": 3},
        "trajectoryEpisodeIds": [2],
        "episodeBatchSize": 2,
        "voxelSize": 0.02,
        "armSpacing": 0.8,
        "selectedEpisodeCount": 2,
        "selectedFrameCount": 7,
    }
    assert report["coverageBatches"] == [
        {
            "batchIndex": 1,
            "episodeIds": [2, 3],
            "episodeRange": {"minimum": 2, "maximum": 3},
            "episodeCount": 2,
            "frameCount": 7,
            "cumulativeFrameCount": 7,
            "elapsedSeconds": 1.0,
            "arms": {
                "left": {
                    "occupiedEntries": 2,
                    "csrIncidence": 3,
                    "rawVisits": 7,
                },
                "right": {
                    "occupiedEntries": 1,
                    "csrIncidence": 2,
                    "rawVisits": 7,
                },
            },
        }
    ]
    assert report["finalCoverage"]["totals"] == {
        "occupiedEntries": 3,
        "csrIncidence": 5,
        "rawVisits": 14,
    }
    assert report["trajectorySampleCounts"] == [{"episodeId": 2, "sampleCount": 4}]
    assert report["artifacts"] == artifacts
    assert report["peakProcessRss"] == {
        "available": True,
        "bytes": 123_456,
        "source": "fake.peak_rss",
        "sourceUnit": "bytes",
    }
    assert report["timingsSeconds"] == {
        "sourcePreparation": 1.0,
        "robotModelPreparation": 0.0,
        "coverageAggregation": 1.0,
        "trajectoryGeneration": None,
        "bundleSerialization": 0.0,
        "bundleWritingInstall": 0.0,
        "bundleValidation": 0.0,
        "totalExport": 7.0,
    }


def test_unavailable_rss_is_explicit() -> None:
    session = ExportMeasurementSession(
        wall_clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        rss_reader=lambda: PeakRssMeasurement(False, None, None, None),
    )
    report = session.build_report(
        repository_id="organization/dataset",
        requested_revision="v3.0",
        resolved_revision="a" * 40,
        source_episode_count=1,
        coverage_episode_ids=(0,),
        trajectory_episode_ids=(),
        episode_batch_size=1,
        voxel_size=0.02,
        arm_spacing=0.8,
        selected_frame_count=1,
        final_arms={
            "left": ArmCoverageCounts(1, 1, 1),
            "right": ArmCoverageCounts(1, 1, 1),
        },
        trajectory_sample_counts={},
        artifacts={
            "files": [],
            "totalUncompressedBytes": 0,
            "totalGzipLevel9Bytes": 0,
        },
    )

    assert report["peakProcessRss"] == {
        "available": False,
        "bytes": None,
        "source": None,
        "sourceUnit": None,
    }


def test_measurement_report_write_is_atomic_and_repeatable(tmp_path: Path) -> None:
    path = tmp_path / "reports/measurement.json"
    first = {"reportFormat": {"version": 1}, "value": 1}
    second = {"reportFormat": {"version": 1}, "value": 2}

    write_measurement_report_atomic(path, first)
    assert json.loads(path.read_text(encoding="utf-8")) == first

    write_measurement_report_atomic(path, second)
    assert json.loads(path.read_text(encoding="utf-8")) == second
    assert [value.name for value in path.parent.iterdir()] == ["measurement.json"]


def test_atomic_report_failure_preserves_existing_file_and_no_partial(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurement.json"
    original = b'{"existing":true}\n'
    path.write_bytes(original)

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(
        "lerobot_state_atlas.export_measurement.os.replace",
        fail_replace,
    )

    with pytest.raises(OSError, match="simulated atomic replacement failure"):
        write_measurement_report_atomic(path, {"replacement": True})

    assert path.read_bytes() == original
    assert [value.name for value in tmp_path.iterdir()] == ["measurement.json"]


def test_report_building_does_not_mutate_inputs() -> None:
    artifacts = {
        "files": [{"filename": "manifest.json"}],
        "totalUncompressedBytes": 1,
        "totalGzipLevel9Bytes": 1,
    }
    original = copy.deepcopy(artifacts)
    session = ExportMeasurementSession()

    session.build_report(
        repository_id="organization/dataset",
        requested_revision="v3.0",
        resolved_revision="a" * 40,
        source_episode_count=1,
        coverage_episode_ids=(0,),
        trajectory_episode_ids=(),
        episode_batch_size=1,
        voxel_size=0.02,
        arm_spacing=0.8,
        selected_frame_count=1,
        final_arms={
            "left": ArmCoverageCounts(1, 1, 1),
            "right": ArmCoverageCounts(1, 1, 1),
        },
        trajectory_sample_counts={},
        artifacts=artifacts,
    )

    assert artifacts == original
