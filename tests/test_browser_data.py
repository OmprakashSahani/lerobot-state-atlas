import copy
import json
from math import cos, pi, sin
from pathlib import Path
import subprocess
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest
import torch

from lerobot_state_atlas.browser_data.export import (
    SourceProvenance,
    _git_source_provenance,
    _trajectory_payload,
    _write_bundle,
    build_browser_data_documents,
    export_browser_data,
)
from lerobot_state_atlas.browser_data.schema import SCHEMA_MINOR
from lerobot_state_atlas.browser_data.serialize import (
    deterministic_json_bytes,
    sha256_bytes,
)
from lerobot_state_atlas.browser_data.validate import (
    BrowserDataValidationError,
    validate_browser_data,
)
from lerobot_state_atlas.coverage import WorkspaceCoverage
from lerobot_state_atlas.export_measurement import (
    ArmCoverageCounts,
    ExportMeasurementSession,
    measure_bundle_artifacts,
)
from lerobot_state_atlas.schema import DatasetSummary, FeatureSummary
from lerobot_state_atlas.trajectory import ToolTrajectory
from lerobot_state_atlas.transforms import RigidTransform, transform_tool_trajectory
from lerobot_state_atlas.urdf import RobotModel


def make_summary() -> DatasetSummary:
    return DatasetSummary(
        repo_id="organization/demo",
        requested_revision="v3.0",
        resolved_revision="c" * 40,
        lerobot_codebase_version="v3.0",
        robot_type="trlc-dk1",
        fps=50.0,
        total_episodes=2,
        total_frames=4,
        total_tasks=1,
        total_duration_seconds=0.08,
        features=(
            FeatureSummary(
                name="observation.state",
                dtype="float32",
                shape=(14,),
                component_names=tuple(f"state_{index}" for index in range(14)),
            ),
        ),
    )


def make_coverage(
    arm: str,
    indices: list[list[int]],
    visits: list[int],
    episode_ids: tuple[tuple[int, ...], ...],
) -> WorkspaceCoverage:
    episode_counts = torch.tensor(
        [len(values) for values in episode_ids],
        dtype=torch.int64,
    )
    return WorkspaceCoverage(
        arm=arm,
        link_name="tool0",
        num_points=sum(visits),
        num_episodes=2,
        voxel_size=0.02,
        minimum_xyz=(0.0, 0.0, 0.0),
        voxel_origin_xyz=(0.0, 0.0, 0.0),
        maximum_xyz=(0.04, 0.04, 0.04),
        span_xyz=(0.04, 0.04, 0.04),
        centroid_xyz=(0.02, 0.02, 0.02),
        grid_shape=(3, 3, 3),
        occupied_voxels=len(indices),
        total_voxels=27,
        occupancy_ratio=len(indices) / 27,
        bounding_box_volume=0.04**3,
        occupied_volume=len(indices) * 0.02**3,
        voxel_indices=torch.tensor(indices, dtype=torch.int64),
        visit_counts=torch.tensor(visits, dtype=torch.int64),
        episode_counts=episode_counts,
        episode_frequencies=episode_counts.to(dtype=torch.float64) / 2,
        episode_ids_by_voxel=episode_ids,
    )


def make_documents() -> tuple[dict, bytes, bytes]:
    coverages = (
        make_coverage(
            "left",
            [[0, 0, 0], [1, 0, 0]],
            [2, 1],
            ((0, 1), (1,)),
        ),
        make_coverage(
            "right",
            [[0, 0, 0], [2, 0, 0]],
            [2, 1],
            ((0, 1), (0,)),
        ),
    )
    trajectory_payload = {
        "schema": {
            "name": "lerobot-state-atlas.browser-data",
            "major": 1,
            "minor": 2,
        },
        "episodes": [
            {
                "episodeId": 0,
                "frameIndices": [0, 1],
                "timestampsSeconds": [0.0, 0.02],
                "leftPositionsXyz": [[0.0, 0.4, 0.0], [0.1, 0.4, 0.0]],
                "rightPositionsXyz": [[0.0, -0.4, 0.0], [0.1, -0.4, 0.0]],
            }
        ],
    }
    (
        manifest,
        coverage,
        trajectory,
        episode_video,
    ) = build_browser_data_documents(
        bundle_id="test-demo-v1",
        summary=make_summary(),
        selected_episodes=(0, 1),
        model=RobotModel(
            name="TRLC-DK1-Follower",
            root_link="base_link",
            links=("base_link", "tool0"),
            joints=(),
            mesh_paths=(),
        ),
        urdf_sha256="a" * 64,
        urdf_upstream_identity="upstream-commit",
        voxel_size=0.02,
        arm_spacing=0.8,
        coverages=coverages,
        source_provenance=SourceProvenance(
            repository_head_commit="b" * 40,
            working_tree_dirty=True,
        ),
        trajectory_payload=trajectory_payload,
    )
    assert trajectory is not None
    assert episode_video is None
    return manifest, coverage, trajectory


def write_documents(
    path: Path,
    manifest: dict,
    coverage: bytes,
    trajectory: bytes,
) -> None:
    path.mkdir()
    (path / "manifest.json").write_bytes(deterministic_json_bytes(manifest))
    (path / "coverage.json").write_bytes(coverage)
    (path / "trajectories.json").write_bytes(trajectory)


def add_episode_video_documents(
    path: Path,
    manifest: dict,
    *,
    episode_id: int = 0,
    media_filename: str = "media/episode-000000/top.mp4",
    write_media: bool = True,
) -> dict:
    media_bytes = b"\x00\x00\x00\x18ftypmp42state-atlas-test-video"
    payload = {
        "schema": {
            "name": "lerobot-state-atlas.browser-data",
            "major": 1,
            "minor": 2,
        },
        "defaultCameraId": "top",
        "cameras": [
            {
                "cameraId": "top",
                "datasetFeature": "observation.images.top",
                "label": "Top camera",
                "width": 224,
                "height": 224,
            }
        ],
        "episodes": [
            {
                "episodeId": episode_id,
                "videos": [
                    {
                        "cameraId": "top",
                        "filename": media_filename,
                        "mimeType": "video/mp4",
                        "fromTimestampSeconds": 0.0,
                        "toTimestampSeconds": 0.04,
                        "byteSize": len(media_bytes),
                        "sha256": sha256_bytes(media_bytes),
                    }
                ],
            }
        ],
    }
    payload_bytes = deterministic_json_bytes(payload)
    manifest["payloads"].append(
        {
            "kind": "episode-videos",
            "filename": "episode-videos.json",
            "required": False,
            "encoding": "json",
            "byteSize": len(payload_bytes),
            "sha256": sha256_bytes(payload_bytes),
        }
    )
    (path / "manifest.json").write_bytes(deterministic_json_bytes(manifest))
    (path / "episode-videos.json").write_bytes(payload_bytes)

    if write_media:
        media_path = path / media_filename
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(media_bytes)

    return payload


def test_deterministic_serialization_and_manifest_generation() -> None:
    first = make_documents()
    second = make_documents()

    assert deterministic_json_bytes(first[0]) == deterministic_json_bytes(second[0])
    assert first[1:] == second[1:]
    assert first[0]["schema"] == {
        "name": "lerobot-state-atlas.browser-data",
        "major": 1,
        "minor": 2,
    }
    assert "generationTimestamp" not in first[0]
    assert first[0]["dataset"]["requestedRevision"] == "v3.0"
    assert first[0]["dataset"]["resolvedRevision"] == "c" * 40
    assert (
        first[0]["dataset"]["requestedRevision"]
        != first[0]["dataset"]["resolvedRevision"]
    )
    assert first[0]["exporter"]["repositoryHeadCommit"] == "b" * 40
    assert first[0]["exporter"]["workingTreeDirty"] is True
    assert "uncommitted" in first[0]["exporter"]["sourceDescription"]


def test_measurement_reporting_does_not_change_bundle_bytes_or_checksums(
    tmp_path: Path,
) -> None:
    manifest, coverage, trajectory = make_documents()
    baseline = tmp_path / "baseline"
    measured = tmp_path / "measured"
    report_path = tmp_path / "reports/measurement.json"

    _write_bundle(baseline, manifest, coverage, trajectory)
    session = ExportMeasurementSession()
    session.start_export()
    _write_bundle(
        measured,
        manifest,
        coverage,
        trajectory,
        measurement=session,
    )
    session.finish_export()
    artifacts = measure_bundle_artifacts(measured, manifest)
    report = session.build_report(
        repository_id="organization/demo",
        requested_revision="v3.0",
        resolved_revision="c" * 40,
        source_episode_count=2,
        coverage_episode_ids=(0, 1),
        trajectory_episode_ids=(0,),
        episode_batch_size=2,
        voxel_size=0.02,
        arm_spacing=0.8,
        selected_frame_count=3,
        final_arms={
            "left": ArmCoverageCounts(2, 3, 3),
            "right": ArmCoverageCounts(2, 3, 3),
        },
        trajectory_sample_counts={0: 2},
        artifacts=artifacts,
    )
    session.store_report(report)
    session.write_report(report_path)

    baseline_files = {
        path.relative_to(baseline): path.read_bytes()
        for path in sorted(baseline.rglob("*"))
        if path.is_file()
    }
    measured_files = {
        path.relative_to(measured): path.read_bytes()
        for path in sorted(measured.rglob("*"))
        if path.is_file()
    }
    assert measured_files == baseline_files
    assert report_path.is_file()
    assert not (measured / "measurement.json").exists()
    installed_manifest = json.loads((measured / "manifest.json").read_text())
    assert all(
        payload["filename"] != report_path.name
        for payload in installed_manifest["payloads"]
    )


def test_manifest_checksums_csr_and_aggregate_totals(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)

    validated = validate_browser_data(bundle)
    coverage_payload = json.loads(coverage)
    left = coverage_payload["arms"][0]

    assert manifest["payloads"][0]["sha256"] == sha256_bytes(coverage)
    assert manifest["payloads"][1]["sha256"] == sha256_bytes(trajectory)
    assert left["episodeIdOffsets"] == [0, 2, 3]
    assert left["episodeIds"] == [0, 1, 1]
    assert left["episodeCounts"] == [2, 1]
    assert validated["totals"] == {
        "datasetFrameCount": 3,
        "toolPointVisitCount": 6,
        "armVoxelEntryCount": 4,
        "uniqueSharedGridCellCount": 3,
    }


def test_documents_validate_against_published_json_schema() -> None:
    manifest, coverage, trajectory = make_documents()
    schema_path = Path(__file__).parents[1] / "schemas" / "browser-data-v1.schema.json"
    validator = Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )

    validator.validate(manifest)
    validator.validate(json.loads(coverage))
    validator.validate(json.loads(trajectory))


def test_episode_video_payload_and_media_validate(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)
    payload = add_episode_video_documents(bundle, manifest)

    validated = validate_browser_data(bundle)
    schema_path = Path(__file__).parents[1] / "schemas" / "browser-data-v1.schema.json"
    validator = Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )

    validator.validate(payload)
    assert validated["schema"]["minor"] == 2
    assert validated["payloads"][-1]["kind"] == "episode-videos"


def test_missing_episode_video_media_is_rejected(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)
    add_episode_video_documents(
        bundle,
        manifest,
        write_media=False,
    )

    with pytest.raises(BrowserDataValidationError, match="Missing episode-video media"):
        validate_browser_data(bundle)


def test_episode_video_path_traversal_is_rejected(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)
    add_episode_video_documents(
        bundle,
        manifest,
        media_filename="../private.mp4",
        write_media=False,
    )

    with pytest.raises(BrowserDataValidationError, match="safe bundle-relative"):
        validate_browser_data(bundle)


def test_episode_video_episode_requires_trajectory_data(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)
    add_episode_video_documents(
        bundle,
        manifest,
        episode_id=1,
        media_filename="media/episode-000001/top.mp4",
    )

    with pytest.raises(
        BrowserDataValidationError,
        match="trajectory episode selection",
    ):
        validate_browser_data(bundle)


def test_atomic_writer_replaces_existing_valid_bundle(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")

    result = _write_bundle(destination, manifest, coverage, trajectory)

    assert result.output_path == destination.resolve()
    assert not (destination / "old.txt").exists()
    assert validate_browser_data(destination)["bundleId"] == "test-demo-v1"
    assert not list(tmp_path.glob(".bundle.previous-*"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["schema"].update({"major": 2}),
            "unsupported schema major",
        ),
        (
            lambda manifest: manifest.update({"localPath": "/private/dataset"}),
            "unsupported fields",
        ),
        (
            lambda manifest: manifest["robot"].update(
                {"urdfUpstreamIdentity": "/private/robot.urdf"}
            ),
            "absolute filesystem path",
        ),
    ],
)
def test_invalid_manifests_are_rejected(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    manifest, coverage, trajectory = make_documents()
    mutation(manifest)
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)

    with pytest.raises(BrowserDataValidationError, match=message):
        validate_browser_data(bundle)


def test_corrupted_payload_checksum_is_rejected(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage + b"corrupt", trajectory)

    with pytest.raises(BrowserDataValidationError, match="byte size"):
        validate_browser_data(bundle)


def test_inconsistent_episode_count_is_rejected(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    coverage_payload = json.loads(coverage)
    coverage_payload["arms"][0]["episodeCounts"][0] = 1
    invalid_coverage = deterministic_json_bytes(coverage_payload)
    manifest = copy.deepcopy(manifest)
    manifest["payloads"][0]["byteSize"] = len(invalid_coverage)
    manifest["payloads"][0]["sha256"] = sha256_bytes(invalid_coverage)
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, invalid_coverage, trajectory)

    with pytest.raises(BrowserDataValidationError, match="count and CSR"):
        validate_browser_data(bundle)


def test_invalid_resolved_dataset_sha_is_rejected(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    manifest["dataset"]["resolvedRevision"] = "v3.0"
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)

    with pytest.raises(BrowserDataValidationError, match="full lowercase"):
        validate_browser_data(bundle)


def _run_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def test_git_source_provenance_detects_clean_and_untracked_states(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_git(repository, "init")
    _run_git(repository, "config", "user.email", "test@example.com")
    _run_git(repository, "config", "user.name", "Test User")
    (repository / ".gitignore").write_text("build/\n", encoding="utf-8")
    (repository / "exporter.py").write_text("VERSION = 1\n", encoding="utf-8")
    _run_git(repository, "add", ".gitignore", "exporter.py")
    _run_git(repository, "commit", "-m", "Initial source")

    clean = _git_source_provenance(repository)
    assert clean.working_tree_dirty is False
    assert len(clean.repository_head_commit) == 40

    ignored_output = repository / "build" / "output.js"
    ignored_output.parent.mkdir()
    ignored_output.write_text("ignored", encoding="utf-8")
    assert _git_source_provenance(repository).working_tree_dirty is False

    (repository / "new_exporter.py").write_text("VERSION = 2\n", encoding="utf-8")
    dirty = _git_source_provenance(repository)
    assert dirty.working_tree_dirty is True
    assert dirty.repository_head_commit == clean.repository_head_commit


def _episode_video_export_fixture() -> tuple[dict, bytes, str]:
    media_bytes = b"\x00\x00\x00\x18ftypmp42exporter-media-test"
    filename = "media/episode-000000/top.mp4"
    payload = {
        "schema": {
            "name": "lerobot-state-atlas.browser-data",
            "major": 1,
            "minor": 2,
        },
        "defaultCameraId": "top",
        "cameras": [
            {
                "cameraId": "top",
                "datasetFeature": "observation.images.top",
                "label": "Top camera",
                "width": 224,
                "height": 224,
            }
        ],
        "episodes": [
            {
                "episodeId": 0,
                "videos": [
                    {
                        "cameraId": "top",
                        "filename": filename,
                        "mimeType": "video/mp4",
                        "fromTimestampSeconds": 0.0,
                        "toTimestampSeconds": 0.04,
                        "byteSize": len(media_bytes),
                        "sha256": sha256_bytes(media_bytes),
                    }
                ],
            }
        ],
    }
    return payload, media_bytes, filename


def _add_episode_video_reference(
    manifest: dict,
    payload_bytes: bytes,
) -> None:
    manifest["payloads"].append(
        {
            "kind": "episode-videos",
            "filename": "episode-videos.json",
            "required": False,
            "encoding": "json",
            "byteSize": len(payload_bytes),
            "sha256": sha256_bytes(payload_bytes),
        }
    )


def test_atomic_writer_packages_episode_video_media(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    payload, media_bytes, filename = _episode_video_export_fixture()
    payload_bytes = deterministic_json_bytes(payload)
    _add_episode_video_reference(manifest, payload_bytes)

    source = tmp_path / "top-source.mp4"
    source.write_bytes(media_bytes)
    destination = tmp_path / "bundle"

    result = _write_bundle(
        destination,
        manifest,
        coverage,
        trajectory,
        payload_bytes,
        {filename: source},
    )

    assert validate_browser_data(destination)["bundleId"] == "test-demo-v1"
    assert (destination / "episode-videos.json").read_bytes() == payload_bytes
    assert (destination / filename).read_bytes() == media_bytes
    assert result.payload_byte_count == sum(
        payload["byteSize"] for payload in manifest["payloads"]
    )


def test_media_mapping_failure_preserves_existing_bundle(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    payload, _, _ = _episode_video_export_fixture()
    payload_bytes = deterministic_json_bytes(payload)
    _add_episode_video_reference(manifest, payload_bytes)

    destination = tmp_path / "bundle"
    destination.mkdir()
    marker = destination / "existing.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly match"):
        _write_bundle(
            destination,
            manifest,
            coverage,
            trajectory,
            payload_bytes,
            {},
        )

    assert marker.read_text(encoding="utf-8") == "preserve me"
    assert not list(tmp_path.glob(".bundle.previous-*"))
    assert not list(tmp_path.glob(".bundle.*"))


def _rotation_z(angle: float) -> torch.Tensor:
    return torch.tensor(
        [
            [cos(angle), -sin(angle), 0.0],
            [sin(angle), cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _enhanced_trajectory_payload() -> dict:
    episode_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    local_rotations = torch.stack(
        [
            torch.eye(3, dtype=torch.float64),
            _rotation_z(pi / 2),
            _rotation_z(pi),
            _rotation_z(-pi / 2),
        ]
    )
    trajectories = {}
    for arm, gripper_values in (
        ("left", [-0.5, 2.0, 3.5, -4.0]),
        ("right", [10.0, -20.0, 30.0, 40.0]),
    ):
        local = ToolTrajectory(
            arm=arm,
            link_name="tool0",
            positions=torch.zeros((4, 3), dtype=torch.float64),
            rotation_matrices=local_rotations,
            episode_indices=episode_indices,
            recorded_gripper_values=torch.tensor(
                gripper_values,
                dtype=torch.float64,
            ),
        )
        trajectories[arm] = transform_tool_trajectory(
            local,
            RigidTransform(
                translation_xyz=(1.0, -2.0, 3.0),
                rotation_rpy=(0.0, 0.0, pi / 2),
            ),
        )
    return _trajectory_payload(
        trajectories=trajectories,
        episode_indices=episode_indices,
        frame_indices=torch.tensor([10, 11, 20, 21], dtype=torch.int64),
        timestamps=torch.tensor([0.0, 0.02, 0.0, 0.02], dtype=torch.float64),
        selected_episodes=(0, 1),
    )


def _documents_with_trajectory_payload(
    trajectory_payload: dict,
) -> tuple[dict, bytes, bytes]:
    coverages = (
        make_coverage("left", [[0, 0, 0]], [2], ((0, 1),)),
        make_coverage("right", [[0, 0, 0]], [2], ((0, 1),)),
    )
    manifest, coverage, trajectory, episode_video = build_browser_data_documents(
        bundle_id="enhanced-v1.2",
        summary=make_summary(),
        selected_episodes=(0, 1),
        model=RobotModel(
            name="TRLC-DK1-Follower",
            root_link="base_link",
            links=("base_link", "tool0"),
            joints=(),
            mesh_paths=(),
        ),
        urdf_sha256="a" * 64,
        urdf_upstream_identity="upstream-commit",
        voxel_size=0.02,
        arm_spacing=0.8,
        coverages=coverages,
        source_provenance=SourceProvenance("b" * 40, False),
        trajectory_payload=trajectory_payload,
    )
    assert trajectory is not None
    assert episode_video is None
    return manifest, coverage, trajectory


def _replace_trajectory(
    manifest: dict,
    payload: dict,
) -> bytes:
    trajectory = deterministic_json_bytes(payload)
    reference = next(
        value for value in manifest["payloads"] if value["kind"] == "trajectories"
    )
    reference["byteSize"] = len(trajectory)
    reference["sha256"] = sha256_bytes(trajectory)
    return trajectory


def test_schema_minor_and_enhanced_trajectory_export_contract() -> None:
    payload = _enhanced_trajectory_payload()
    manifest, _, trajectory_bytes = _documents_with_trajectory_payload(payload)
    repeated = _enhanced_trajectory_payload()

    assert SCHEMA_MINOR == 2
    assert deterministic_json_bytes(payload) == deterministic_json_bytes(repeated)
    assert trajectory_bytes == deterministic_json_bytes(payload)
    assert manifest["trajectoryState"] == {
        "orientation": {
            "available": True,
            "representation": "unit-quaternion",
            "componentOrder": ["x", "y", "z", "w"],
            "frame": "canonical-shared-world",
            "samplePolicy": "recorded-sample",
        },
        "gripper": {
            "available": True,
            "leftSourceComponent": "left_gripper.pos",
            "rightSourceComponent": "right_gripper.pos",
            "valueSemantics": "raw-device-specific-unproven",
            "physicalJawWidthCalibrated": False,
            "polarityEstablished": False,
            "visualizationGeometryCalibrated": False,
        },
    }
    first = payload["episodes"][0]
    assert first["leftOrientationsXyzw"][0] == pytest.approx(
        [0.0, 0.0, 2**-0.5, 2**-0.5]
    )
    assert first["leftRecordedGripperValues"] == [-0.5, 2.0]
    assert first["rightRecordedGripperValues"] == [10.0, -20.0]
    for episode in payload["episodes"]:
        expected = len(episode["frameIndices"])
        assert all(
            len(episode[field]) == expected
            for field in (
                "timestampsSeconds",
                "leftPositionsXyz",
                "rightPositionsXyz",
                "leftOrientationsXyzw",
                "rightOrientationsXyzw",
                "leftRecordedGripperValues",
                "rightRecordedGripperValues",
            )
        )


def test_quaternion_sign_continuity_resets_at_episode_boundaries() -> None:
    payload = _enhanced_trajectory_payload()
    first_episode = payload["episodes"][0]["leftOrientationsXyzw"]
    second_episode = payload["episodes"][1]["leftOrientationsXyzw"]

    assert (
        sum(a * b for a, b in zip(first_episode[0], first_episode[1], strict=True)) >= 0
    )
    assert second_episode[0][3] > 0
    assert (
        sum(a * b for a, b in zip(first_episode[-1], second_episode[0], strict=True))
        < 0
    )


@pytest.mark.parametrize(
    ("orientation", "gripper"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_v1_2_optional_capability_combinations_validate(
    tmp_path: Path,
    orientation: bool,
    gripper: bool,
) -> None:
    payload = _enhanced_trajectory_payload()
    for episode in payload["episodes"]:
        if not orientation:
            episode.pop("leftOrientationsXyzw")
            episode.pop("rightOrientationsXyzw")
        if not gripper:
            episode.pop("leftRecordedGripperValues")
            episode.pop("rightRecordedGripperValues")
    manifest, coverage, trajectory = _documents_with_trajectory_payload(payload)
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)

    validated = validate_browser_data(bundle)

    assert validated["trajectoryState"]["orientation"]["available"] is orientation
    assert validated["trajectoryState"]["gripper"]["available"] is gripper


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["episodes"][0].pop("rightOrientationsXyzw"),
            "both left and right orientation",
        ),
        (
            lambda payload: payload["episodes"][0].pop("rightRecordedGripperValues"),
            "both left and right gripper",
        ),
        (
            lambda payload: payload["episodes"][0]["leftOrientationsXyzw"].pop(),
            "equal, non-zero length",
        ),
        (
            lambda payload: payload["episodes"][0]["leftOrientationsXyzw"].__setitem__(
                0, [0.0, 0.0, 1.0]
            ),
            "four numbers",
        ),
        (
            lambda payload: payload["episodes"][0]["leftOrientationsXyzw"].__setitem__(
                0, [0.0, 0.0, 0.0, 2.0]
            ),
            "unit length",
        ),
        (
            lambda payload: payload["episodes"][1].pop("leftOrientationsXyzw") and None,
            "both left and right orientation",
        ),
    ],
)
def test_invalid_v1_2_trajectory_state_is_rejected(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _enhanced_trajectory_payload()
    mutation(payload)
    manifest, coverage, trajectory = _documents_with_trajectory_payload(payload)
    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)

    with pytest.raises(BrowserDataValidationError, match=message):
        validate_browser_data(bundle)


def test_nonfinite_enhanced_values_are_rejected_semantically() -> None:
    payload = _enhanced_trajectory_payload()
    payload["episodes"][0]["leftOrientationsXyzw"][0][0] = float("nan")
    with pytest.raises(BrowserDataValidationError, match="finite"):
        from lerobot_state_atlas.browser_data.validate import _validate_trajectories

        _validate_trajectories(payload, {0, 1})

    payload = _enhanced_trajectory_payload()
    payload["episodes"][0]["leftRecordedGripperValues"][0] = float("inf")
    with pytest.raises(BrowserDataValidationError, match="finite"):
        _validate_trajectories(payload, {0, 1})


def test_manifest_capability_disagreement_and_episode_sparsity_fail(
    tmp_path: Path,
) -> None:
    payload = _enhanced_trajectory_payload()
    manifest, coverage, _ = _documents_with_trajectory_payload(payload)
    manifest["trajectoryState"]["orientation"]["available"] = False
    trajectory = _replace_trajectory(manifest, payload)
    bundle = tmp_path / "disagreement"
    write_documents(bundle, manifest, coverage, trajectory)
    with pytest.raises(BrowserDataValidationError, match="does not agree"):
        validate_browser_data(bundle)

    payload = _enhanced_trajectory_payload()
    payload["episodes"][1].pop("leftRecordedGripperValues")
    payload["episodes"][1].pop("rightRecordedGripperValues")
    manifest, coverage, trajectory = _documents_with_trajectory_payload(payload)
    sparse_bundle = tmp_path / "sparse"
    write_documents(sparse_bundle, manifest, coverage, trajectory)
    with pytest.raises(BrowserDataValidationError, match="every trajectory episode"):
        validate_browser_data(sparse_bundle)


@pytest.mark.parametrize("minor", [0, 1])
def test_legacy_position_only_bundles_validate_and_reject_v1_2_fields(
    tmp_path: Path,
    minor: int,
) -> None:
    manifest, coverage, trajectory = make_documents()
    manifest.pop("trajectoryState")
    manifest["schema"]["minor"] = minor
    coverage_payload = json.loads(coverage)
    coverage_payload["schema"]["minor"] = minor
    coverage = deterministic_json_bytes(coverage_payload)
    trajectory_payload = json.loads(trajectory)
    trajectory_payload["schema"]["minor"] = minor
    trajectory = deterministic_json_bytes(trajectory_payload)
    for reference, content in zip(
        manifest["payloads"],
        (coverage, trajectory),
        strict=True,
    ):
        reference["byteSize"] = len(content)
        reference["sha256"] = sha256_bytes(content)
    bundle = tmp_path / f"legacy-{minor}"
    write_documents(bundle, manifest, coverage, trajectory)
    validate_browser_data(bundle)

    trajectory_payload["episodes"][0]["leftRecordedGripperValues"] = [0.0, 1.0]
    trajectory_payload["episodes"][0]["rightRecordedGripperValues"] = [0.0, 1.0]
    trajectory = _replace_trajectory(manifest, trajectory_payload)
    (bundle / "manifest.json").write_bytes(deterministic_json_bytes(manifest))
    (bundle / "trajectories.json").write_bytes(trajectory)
    with pytest.raises(BrowserDataValidationError, match="unsupported fields"):
        validate_browser_data(bundle)


def test_v1_1_video_capable_bundle_remains_valid(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    manifest.pop("trajectoryState")
    manifest["schema"]["minor"] = 1
    coverage_payload = json.loads(coverage)
    coverage_payload["schema"]["minor"] = 1
    coverage = deterministic_json_bytes(coverage_payload)
    trajectory_payload = json.loads(trajectory)
    trajectory_payload["schema"]["minor"] = 1
    trajectory = deterministic_json_bytes(trajectory_payload)
    for reference, content in zip(
        manifest["payloads"],
        (coverage, trajectory),
        strict=True,
    ):
        reference["byteSize"] = len(content)
        reference["sha256"] = sha256_bytes(content)
    bundle = tmp_path / "v1.1-video"
    write_documents(bundle, manifest, coverage, trajectory)
    video_payload = add_episode_video_documents(bundle, manifest)
    video_payload["schema"]["minor"] = 1
    video_bytes = deterministic_json_bytes(video_payload)
    video_reference = manifest["payloads"][-1]
    video_reference["byteSize"] = len(video_bytes)
    video_reference["sha256"] = sha256_bytes(video_bytes)
    (bundle / "manifest.json").write_bytes(deterministic_json_bytes(manifest))
    (bundle / "episode-videos.json").write_bytes(video_bytes)

    assert validate_browser_data(bundle)["schema"]["minor"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda episode: episode.update({"frameIndices": [11, 10]}),
            "sorted and distinct",
        ),
        (
            lambda episode: episode.update({"frameIndices": [10, 10]}),
            "sorted and distinct",
        ),
        (
            lambda episode: episode.update({"timestampsSeconds": [0.02, 0.0]}),
            "must be monotonic",
        ),
    ],
)
def test_v1_2_trajectory_ordering_is_semantic_only(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _enhanced_trajectory_payload()
    mutation(payload["episodes"][0])
    manifest, coverage, trajectory = _documents_with_trajectory_payload(payload)
    schema = Draft202012Validator(
        json.loads(
            (
                Path(__file__).parents[1] / "schemas" / "browser-data-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
    )
    schema.validate(payload)
    bundle = tmp_path / "v1.2-ordering"
    write_documents(bundle, manifest, coverage, trajectory)

    with pytest.raises(BrowserDataValidationError, match=message):
        validate_browser_data(bundle)


@pytest.mark.parametrize("minor", [0, 1])
@pytest.mark.parametrize(
    "mutation",
    [
        lambda episode: episode.update({"frameIndices": [1, 0]}),
        lambda episode: episode.update({"frameIndices": [0, 0]}),
        lambda episode: episode.update({"timestampsSeconds": [0.02, 0.0]}),
    ],
)
def test_legacy_trajectory_ordering_retains_previous_acceptance(
    tmp_path: Path,
    minor: int,
    mutation,
) -> None:
    manifest, coverage, trajectory = make_documents()
    manifest.pop("trajectoryState")
    manifest["schema"]["minor"] = minor
    coverage_payload = json.loads(coverage)
    coverage_payload["schema"]["minor"] = minor
    coverage = deterministic_json_bytes(coverage_payload)
    trajectory_payload = json.loads(trajectory)
    trajectory_payload["schema"]["minor"] = minor
    mutation(trajectory_payload["episodes"][0])
    trajectory = deterministic_json_bytes(trajectory_payload)
    for reference, content in zip(
        manifest["payloads"],
        (coverage, trajectory),
        strict=True,
    ):
        reference["byteSize"] = len(content)
        reference["sha256"] = sha256_bytes(content)
    bundle = tmp_path / f"legacy-ordering-{minor}"
    write_documents(bundle, manifest, coverage, trajectory)

    validate_browser_data(bundle)


def test_json_schema_and_semantic_validation_agree_for_v1_2(
    tmp_path: Path,
) -> None:
    payload = _enhanced_trajectory_payload()
    manifest, coverage, trajectory = _documents_with_trajectory_payload(payload)
    validator = Draft202012Validator(
        json.loads(
            (
                Path(__file__).parents[1] / "schemas" / "browser-data-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
    )
    validator.validate(manifest)
    validator.validate(payload)
    invalid = copy.deepcopy(payload)
    invalid["episodes"][0].pop("rightOrientationsXyzw")
    assert list(validator.iter_errors(invalid))

    bundle = tmp_path / "bundle"
    write_documents(bundle, manifest, coverage, trajectory)
    validate_browser_data(bundle)


def test_payload_schema_must_match_manifest(tmp_path: Path) -> None:
    manifest, coverage, trajectory = make_documents()
    trajectory_payload = json.loads(trajectory)
    trajectory_payload["schema"]["minor"] = 1
    trajectory = _replace_trajectory(manifest, trajectory_payload)
    bundle = tmp_path / "schema-mismatch"
    write_documents(bundle, manifest, coverage, trajectory)

    with pytest.raises(BrowserDataValidationError, match="must match the manifest"):
        validate_browser_data(bundle)


def test_v1_2_trajectory_state_presence_is_exact(
    tmp_path: Path,
) -> None:
    schema = Draft202012Validator(
        json.loads(
            (
                Path(__file__).parents[1] / "schemas" / "browser-data-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
    )
    manifest, coverage, trajectory = make_documents()
    missing_state = copy.deepcopy(manifest)
    missing_state.pop("trajectoryState")
    assert list(schema.iter_errors(missing_state))
    missing_bundle = tmp_path / "missing-state"
    write_documents(missing_bundle, missing_state, coverage, trajectory)
    with pytest.raises(
        BrowserDataValidationError,
        match="must declare trajectoryState",
    ):
        validate_browser_data(missing_bundle)

    state_without_trajectory = copy.deepcopy(manifest)
    state_without_trajectory["payloads"] = [
        reference
        for reference in state_without_trajectory["payloads"]
        if reference["kind"] != "trajectories"
    ]
    assert list(schema.iter_errors(state_without_trajectory))
    extra_state_bundle = tmp_path / "extra-state"
    extra_state_bundle.mkdir()
    (extra_state_bundle / "manifest.json").write_bytes(
        deterministic_json_bytes(state_without_trajectory)
    )
    (extra_state_bundle / "coverage.json").write_bytes(coverage)
    with pytest.raises(
        BrowserDataValidationError,
        match="must declare trajectoryState",
    ):
        validate_browser_data(extra_state_bundle)


@pytest.mark.parametrize("minor", [0, 1])
def test_legacy_manifests_reject_trajectory_state(
    tmp_path: Path,
    minor: int,
) -> None:
    manifest, coverage, trajectory = make_documents()
    manifest["schema"]["minor"] = minor
    coverage_payload = json.loads(coverage)
    coverage_payload["schema"]["minor"] = minor
    coverage = deterministic_json_bytes(coverage_payload)
    trajectory_payload = json.loads(trajectory)
    trajectory_payload["schema"]["minor"] = minor
    trajectory = deterministic_json_bytes(trajectory_payload)
    for reference, content in zip(
        manifest["payloads"],
        (coverage, trajectory),
        strict=True,
    ):
        reference["byteSize"] = len(content)
        reference["sha256"] = sha256_bytes(content)
    bundle = tmp_path / f"legacy-manifest-{minor}"
    write_documents(bundle, manifest, coverage, trajectory)

    with pytest.raises(BrowserDataValidationError, match="unsupported fields"):
        validate_browser_data(bundle)


def _documents_with_episode_video_payload(
    episode_video_payload: dict,
) -> tuple[dict, bytes, bytes, bytes]:
    trajectory_payload = _enhanced_trajectory_payload()
    coverages = (
        make_coverage("left", [[0, 0, 0]], [2], ((0, 1),)),
        make_coverage("right", [[0, 0, 0]], [2], ((0, 1),)),
    )
    manifest, coverage, trajectory, episode_video = build_browser_data_documents(
        bundle_id="episode-video-v1.2",
        summary=make_summary(),
        selected_episodes=(0, 1),
        model=RobotModel(
            name="TRLC-DK1-Follower",
            root_link="base_link",
            links=("base_link", "tool0"),
            joints=(),
            mesh_paths=(),
        ),
        urdf_sha256="a" * 64,
        urdf_upstream_identity="upstream-commit",
        voxel_size=0.02,
        arm_spacing=0.8,
        coverages=coverages,
        source_provenance=SourceProvenance("b" * 40, False),
        trajectory_payload=trajectory_payload,
        episode_video_payload=episode_video_payload,
    )
    assert trajectory is not None
    assert episode_video is not None
    return manifest, coverage, trajectory, episode_video


def _write_episode_video_bundle(
    path: Path,
    manifest: dict,
    coverage: bytes,
    trajectory: bytes,
    episode_video: bytes,
    media_bytes: bytes,
    media_filename: str,
) -> None:
    write_documents(path, manifest, coverage, trajectory)
    (path / "episode-videos.json").write_bytes(episode_video)
    media_path = path / media_filename
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(media_bytes)


def test_episode_video_schema_is_preserved_without_mutating_input() -> None:
    payload, _, _ = _episode_video_export_fixture()
    original = copy.deepcopy(payload)

    _, _, _, episode_video = _documents_with_episode_video_payload(payload)

    assert json.loads(episode_video) == original
    assert episode_video == deterministic_json_bytes(original)
    assert payload == original


@pytest.mark.parametrize(
    ("schema_mutation", "message"),
    [
        (lambda payload: payload.pop("schema"), "schema must be an object"),
        (
            lambda payload: payload["schema"].update({"minor": 1}),
            "must match the manifest",
        ),
        (
            lambda payload: payload["schema"].update({"major": 2}),
            "unsupported schema major",
        ),
        (
            lambda payload: payload["schema"].update({"name": "wrong.schema"}),
            "unsupported schema name",
        ),
    ],
)
def test_invalid_episode_video_schema_is_rejected_without_rewriting(
    tmp_path: Path,
    schema_mutation,
    message: str,
) -> None:
    payload, media_bytes, media_filename = _episode_video_export_fixture()
    schema_mutation(payload)
    original = copy.deepcopy(payload)
    manifest, coverage, trajectory, episode_video = (
        _documents_with_episode_video_payload(payload)
    )
    bundle = tmp_path / "invalid-video-schema"
    _write_episode_video_bundle(
        bundle,
        manifest,
        coverage,
        trajectory,
        episode_video,
        media_bytes,
        media_filename,
    )

    assert json.loads(episode_video) == original
    assert payload == original
    with pytest.raises(BrowserDataValidationError, match=message):
        validate_browser_data(bundle)


def test_public_demo_v1_files_remain_unchanged() -> None:
    demo = Path(__file__).parents[1] / "apps/web/public/atlas-data/demo-v1"
    assert {
        path.name: sha256_bytes(path.read_bytes()) for path in sorted(demo.iterdir())
    } == {
        "coverage.json": "27d039408348afe3b1b87c602c62bbb822e71dc98e310120b9d00ca81faa7828",
        "manifest.json": "7f0470aff874bccb74a043850c3cbe6804ec8f323e0736bee54fe84a6511fa8c",
        "trajectories.json": "cd274f52dfea7f0643e7729ed1bfdf8a929bb202bc2f1e51f9d7d08cb531f654",
    }


def test_export_requests_raw_grippers_only_for_playback_trajectories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='test'/>", encoding="utf-8")
    model = RobotModel(
        name="TRLC-DK1-Follower",
        root_link="base_link",
        links=("base_link", "tool0"),
        joints=(),
        mesh_paths=(),
    )
    coverages = (
        make_coverage("left", [[0, 0, 0]], [2], ((0, 1),)),
        make_coverage("right", [[0, 0, 0]], [2], ((0, 1),)),
    )
    batch = SimpleNamespace(
        states=torch.zeros((2, 14), dtype=torch.float64),
        episode_indices=torch.tensor([0, 0], dtype=torch.int64),
        frame_indices=torch.tensor([0, 1], dtype=torch.int64),
        timestamps=torch.tensor([0.0, 0.02], dtype=torch.float64),
    )
    aggregation_calls: list[dict] = []
    gripper_components: list[str] = []
    written: dict[str, object] = {}

    monkeypatch.setattr(
        "lerobot_state_atlas.browser_data.export.resolve_dataset_revision",
        lambda *_: SimpleNamespace(requested="v3.0", resolved="c" * 40),
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.browser_data.export.load_dataset_summary",
        lambda *_, **__: make_summary(),
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.browser_data.export.load_robot_model",
        lambda *_: model,
    )

    def fake_aggregate(*args, **kwargs):
        aggregation_calls.append(kwargs)
        return SimpleNamespace(coverages=coverages)

    monkeypatch.setattr(
        "lerobot_state_atlas.browser_data.export.aggregate_workspace_coverages",
        fake_aggregate,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.browser_data.export.load_state_batch",
        lambda *_, **__: batch,
    )

    def fake_compute(*args, arm: str, gripper_component_name: str, **kwargs):
        gripper_components.append(gripper_component_name)
        values = [-0.5, 2.5] if arm == "left" else [10.0, -20.0]
        return ToolTrajectory(
            arm=arm,
            link_name="tool0",
            positions=torch.zeros((2, 3), dtype=torch.float64),
            rotation_matrices=torch.eye(3, dtype=torch.float64).repeat(2, 1, 1),
            episode_indices=batch.episode_indices,
            recorded_gripper_values=torch.tensor(values, dtype=torch.float64),
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.browser_data.export.compute_tool_trajectory",
        fake_compute,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.browser_data.export._git_source_provenance",
        lambda *_: SourceProvenance("b" * 40, False),
    )

    def fake_write(*args):
        written["args"] = args
        return SimpleNamespace(output_path=tmp_path / "bundle")

    monkeypatch.setattr(
        "lerobot_state_atlas.browser_data.export._write_bundle",
        fake_write,
    )

    export_browser_data(
        "organization/demo",
        urdf_path=urdf,
        episodes=(0, 1),
        trajectory_episodes=(0,),
        episode_batch_size=2,
        voxel_size=0.02,
        arm_spacing=0.8,
        output_path=tmp_path / "bundle",
        bundle_id="enhanced-v1.2",
        urdf_upstream_identity="upstream",
        repository_path=tmp_path,
    )

    assert gripper_components == ["left_gripper.pos", "right_gripper.pos"]
    assert len(aggregation_calls) == 1
    assert all("gripper" not in key for key in aggregation_calls[0])
    manifest = written["args"][1]
    trajectory_bytes = written["args"][3]
    trajectory = json.loads(trajectory_bytes)
    assert manifest["trajectoryState"]["orientation"]["available"] is True
    assert manifest["trajectoryState"]["gripper"]["available"] is True
    assert trajectory["episodes"][0]["leftRecordedGripperValues"] == [-0.5, 2.5]
    assert trajectory["episodes"][0]["rightRecordedGripperValues"] == [10.0, -20.0]
