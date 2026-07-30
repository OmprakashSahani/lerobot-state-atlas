import copy
import json
from pathlib import Path
import subprocess

from jsonschema import Draft202012Validator
import pytest
import torch

from lerobot_state_atlas.browser_data.export import (
    SourceProvenance,
    _git_source_provenance,
    _write_bundle,
    build_browser_data_documents,
)
from lerobot_state_atlas.browser_data.serialize import (
    deterministic_json_bytes,
    sha256_bytes,
)
from lerobot_state_atlas.browser_data.validate import (
    BrowserDataValidationError,
    validate_browser_data,
)
from lerobot_state_atlas.coverage import WorkspaceCoverage
from lerobot_state_atlas.schema import DatasetSummary, FeatureSummary
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
            "minor": 0,
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
            "minor": 1,
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
        "minor": 1,
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
    assert validated["schema"]["minor"] == 1
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
            "minor": 1,
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
