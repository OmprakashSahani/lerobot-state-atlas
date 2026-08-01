"""Operational measurement support for browser-data exports.

Measurement reports are deliberately separate from deterministic browser-data
bundles. Their timestamps, timings, platform details, and process memory values
describe one machine-dependent export run and are not scientific payload data.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Iterator

REPORT_FORMAT_NAME = "lerobot-state-atlas.export-measurement"
REPORT_FORMAT_VERSION = 1
MACHINE_DEPENDENT_DISCLOSURE = (
    "Timings and peak process RSS are machine- and platform-dependent operational "
    "measurements. They are intentionally excluded from deterministic browser-data "
    "bundles and their checksums."
)


@dataclass(frozen=True)
class ArmCoverageCounts:
    """Cumulative coverage counts for one arm."""

    occupied_entries: int
    csr_incidence: int
    raw_visits: int

    def as_dict(self) -> dict[str, int]:
        """Return the report representation."""
        return {
            "occupiedEntries": self.occupied_entries,
            "csrIncidence": self.csr_incidence,
            "rawVisits": self.raw_visits,
        }


@dataclass(frozen=True)
class CoverageBatchMeasurement:
    """Measurements captured after one coverage batch completes."""

    batch_index: int
    episode_ids: tuple[int, ...]
    frame_count: int
    cumulative_frame_count: int
    elapsed_seconds: float
    arms: Mapping[str, ArmCoverageCounts]

    def as_dict(self) -> dict[str, Any]:
        """Return the report representation."""
        return {
            "batchIndex": self.batch_index,
            "episodeIds": list(self.episode_ids),
            "episodeRange": {
                "minimum": min(self.episode_ids),
                "maximum": max(self.episode_ids),
            },
            "episodeCount": len(self.episode_ids),
            "frameCount": self.frame_count,
            "cumulativeFrameCount": self.cumulative_frame_count,
            "elapsedSeconds": self.elapsed_seconds,
            "arms": {
                arm: counts.as_dict() for arm, counts in sorted(self.arms.items())
            },
        }


@dataclass(frozen=True)
class PeakRssMeasurement:
    """Best-effort process peak RSS normalized to bytes when supported."""

    available: bool
    bytes: int | None
    source: str | None
    source_unit: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the report representation."""
        return {
            "available": self.available,
            "bytes": self.bytes,
            "source": self.source,
            "sourceUnit": self.source_unit,
        }


MonotonicClock = Callable[[], float]
WallClock = Callable[[], datetime]
RssReader = Callable[[], PeakRssMeasurement]
ProgressCallback = Callable[[CoverageBatchMeasurement], None]


def read_peak_process_rss() -> PeakRssMeasurement:
    """Read peak process RSS using standard-library platform conventions."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return PeakRssMeasurement(False, None, None, None)

    system = platform.system()
    source = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    if system == "Darwin":
        return PeakRssMeasurement(True, value, source, "bytes")
    if system in {"Linux", "FreeBSD"}:
        return PeakRssMeasurement(True, value * 1024, source, "kibibytes")
    return PeakRssMeasurement(False, None, source, "platform-dependent")


def validate_measurement_report_path(report_path: Path, bundle_path: Path) -> None:
    """Reject reports placed at or below the immutable bundle directory."""
    resolved_report = report_path.resolve()
    resolved_bundle = bundle_path.resolve()
    if resolved_report == resolved_bundle or resolved_report.is_relative_to(
        resolved_bundle
    ):
        raise ValueError(
            "Measurement report path must be outside the browser-data bundle."
        )


def _gzip_size(path: Path) -> int:
    class ByteCounter:
        def __init__(self) -> None:
            self.count = 0

        def write(self, value: bytes) -> int:
            self.count += len(value)
            return len(value)

        def flush(self) -> None:
            return None

    output = ByteCounter()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=output,
        mtime=0,
    ) as compressed:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                compressed.write(chunk)
    return output.count


def measure_bundle_artifacts(
    bundle_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure installed bundle files without creating compressed artifacts."""
    payload_kinds = {
        str(payload["filename"]): str(payload["kind"])
        for payload in manifest["payloads"]
    }
    files: list[dict[str, Any]] = []
    for path in sorted(value for value in bundle_path.rglob("*") if value.is_file()):
        filename = path.relative_to(bundle_path).as_posix()
        kind = (
            "manifest" if filename == "manifest.json" else payload_kinds.get(filename)
        )
        files.append(
            {
                "filename": filename,
                "payloadKind": kind,
                "uncompressedBytes": path.stat().st_size,
                "gzipLevel9Bytes": _gzip_size(path),
            }
        )
    return {
        "files": files,
        "totalUncompressedBytes": sum(
            int(value["uncompressedBytes"]) for value in files
        ),
        "totalGzipLevel9Bytes": sum(int(value["gzipLevel9Bytes"]) for value in files),
    }


def write_measurement_report_atomic(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically write a measurement report without leaving partial output."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            content = (
                json.dumps(
                    report,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class ExportMeasurementSession:
    """Collect optional operational evidence for one browser-data export."""

    def __init__(
        self,
        *,
        monotonic_clock: MonotonicClock = time.monotonic,
        wall_clock: WallClock = lambda: datetime.now(timezone.utc),
        rss_reader: RssReader = read_peak_process_rss,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._rss_reader = rss_reader
        self._progress_callback = progress_callback
        self._stage_seconds: dict[str, float | None] = {
            "sourcePreparation": 0.0,
            "robotModelPreparation": 0.0,
            "coverageAggregation": 0.0,
            "trajectoryGeneration": None,
            "bundleSerialization": 0.0,
            "bundleWritingInstall": 0.0,
            "bundleValidation": 0.0,
            "totalExport": 0.0,
        }
        self._batches: list[CoverageBatchMeasurement] = []
        self._export_started_at: float | None = None
        self._report: dict[str, Any] | None = None

    def start_export(self) -> None:
        """Start the total export timer."""
        self._export_started_at = self._monotonic_clock()

    def finish_export(self) -> None:
        """Stop the total export timer after atomic bundle installation."""
        if self._export_started_at is None:
            raise RuntimeError("Measurement export timer was not started.")
        self._stage_seconds["totalExport"] = (
            self._monotonic_clock() - self._export_started_at
        )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure one non-overlapping exporter stage."""
        started_at = self._monotonic_clock()
        try:
            yield
        finally:
            self.add_elapsed(name, started_at)

    def timer_started(self) -> float:
        """Return a timestamp for a later accumulated timing."""
        return self._monotonic_clock()

    def add_elapsed(self, name: str, started_at: float) -> None:
        """Accumulate elapsed time under an existing timing field."""
        previous = self._stage_seconds[name]
        elapsed = self._monotonic_clock() - started_at
        self._stage_seconds[name] = elapsed + (0.0 if previous is None else previous)

    def begin_coverage_batch(self) -> float:
        """Start timing source loading and processing for one coverage batch."""
        return self._monotonic_clock()

    def complete_coverage_batch(
        self,
        *,
        started_at: float,
        episode_ids: Sequence[int],
        frame_count: int,
        cumulative_frame_count: int,
        arms: Mapping[str, ArmCoverageCounts],
    ) -> None:
        """Record authoritative accumulator state after a completed batch."""
        measurement = CoverageBatchMeasurement(
            batch_index=len(self._batches) + 1,
            episode_ids=tuple(int(value) for value in episode_ids),
            frame_count=frame_count,
            cumulative_frame_count=cumulative_frame_count,
            elapsed_seconds=self._monotonic_clock() - started_at,
            arms=dict(arms),
        )
        self._batches.append(measurement)
        if self._progress_callback is not None:
            self._progress_callback(measurement)

    def build_report(
        self,
        *,
        repository_id: str,
        requested_revision: str,
        resolved_revision: str,
        source_episode_count: int,
        coverage_episode_ids: Sequence[int],
        trajectory_episode_ids: Sequence[int],
        episode_batch_size: int,
        voxel_size: float,
        arm_spacing: float,
        selected_frame_count: int,
        final_arms: Mapping[str, ArmCoverageCounts],
        trajectory_sample_counts: Mapping[int, int],
        artifacts: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the standalone operational report document."""
        generated_at = self._wall_clock().astimezone(timezone.utc)
        coverage_ids = tuple(int(value) for value in coverage_episode_ids)
        rss = self._rss_reader()
        total_counts = ArmCoverageCounts(
            occupied_entries=sum(
                value.occupied_entries for value in final_arms.values()
            ),
            csr_incidence=sum(value.csr_incidence for value in final_arms.values()),
            raw_visits=sum(value.raw_visits for value in final_arms.values()),
        )
        return {
            "reportFormat": {
                "name": REPORT_FORMAT_NAME,
                "version": REPORT_FORMAT_VERSION,
            },
            "generatedAtUtc": generated_at.isoformat().replace("+00:00", "Z"),
            "machineDependentDisclosure": MACHINE_DEPENDENT_DISCLOSURE,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "pythonVersion": platform.python_version(),
            "dataset": {
                "repositoryId": repository_id,
                "requestedRevision": requested_revision,
                "resolvedRevision": resolved_revision,
                "sourceEpisodeCount": source_episode_count,
            },
            "selection": {
                "coverageEpisodeIds": list(coverage_ids),
                "coverageEpisodeRange": {
                    "minimum": min(coverage_ids),
                    "maximum": max(coverage_ids),
                },
                "trajectoryEpisodeIds": [
                    int(value) for value in trajectory_episode_ids
                ],
                "episodeBatchSize": episode_batch_size,
                "voxelSize": voxel_size,
                "armSpacing": arm_spacing,
                "selectedEpisodeCount": len(coverage_ids),
                "selectedFrameCount": selected_frame_count,
            },
            "timingsSeconds": dict(self._stage_seconds),
            "coverageBatches": [value.as_dict() for value in self._batches],
            "finalCoverage": {
                "selectedFrameCount": selected_frame_count,
                "arms": {
                    arm: counts.as_dict() for arm, counts in sorted(final_arms.items())
                },
                "totals": total_counts.as_dict(),
            },
            "trajectorySampleCounts": [
                {"episodeId": episode_id, "sampleCount": sample_count}
                for episode_id, sample_count in sorted(trajectory_sample_counts.items())
            ],
            "artifacts": dict(artifacts),
            "peakProcessRss": rss.as_dict(),
        }

    def store_report(self, report: Mapping[str, Any]) -> None:
        """Store the completed report until the CLI writes it atomically."""
        self._report = dict(report)

    @property
    def report(self) -> Mapping[str, Any]:
        """Return the completed report."""
        if self._report is None:
            raise RuntimeError("Export measurement report has not been completed.")
        return self._report

    def write_report(self, path: Path) -> None:
        """Write the completed report atomically."""
        write_measurement_report_atomic(path, self.report)
