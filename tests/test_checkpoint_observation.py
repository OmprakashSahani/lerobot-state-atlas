import copy
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest
from jsonschema import Draft202012Validator

from lerobot_state_atlas.checkpoint_comparison.observation import (
    CAMERA_FEATURE_NAMES,
    COMPONENT_NAMES,
    CheckpointObservationValidationError,
    load_checkpoint_observation,
)
import lerobot_state_atlas.checkpoint_comparison.observation as observation_module
from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
)


def _image_identity(path: Path) -> tuple[int, str]:
    content = path.read_bytes()
    return len(content), hashlib.sha256(content).hexdigest()


def make_manifest(tmp_path: Path, *, ground_truth: bool = False) -> tuple[Path, dict]:
    camera_entries = []
    for index, feature_name in enumerate(CAMERA_FEATURE_NAMES):
        filename = f"images/camera-{index}.png"
        image_path = tmp_path / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (224, 224), (index * 30, 10, 20)).save(image_path)
        byte_count, sha256 = _image_identity(image_path)
        camera_entries.append(
            {
                "featureName": feature_name,
                "filename": filename,
                "width": 224,
                "height": 224,
                "channels": 3,
                "byteCount": byte_count,
                "sha256": sha256,
            }
        )
    recorded = {
        "available": False,
        "reason": "Synthetic fixture has no recorded action chunk.",
    }
    if ground_truth:
        recorded = {
            "available": True,
            "componentNames": list(COMPONENT_NAMES),
            "actions": [
                [float(step + component) / 100 for component in range(14)]
                for step in range(50)
            ],
            "relativeTimesSeconds": [step / 50 for step in range(50)],
            "frameIndices": list(range(123, 173)),
        }
    document = {
        "schema": {
            "name": "lerobot-state-atlas.checkpoint-observation",
            "major": 1,
            "minor": 0,
        },
        "observationId": "synthetic-observation-001",
        "dataset": {
            "repoId": "DreamMachines/actuator_unboxing_4h_diverse",
            "revision": "e973df866c80f52884cc68355579043cab828e78",
            "episodeIndex": 7,
            "frameIndex": 123,
            "timestampSeconds": 2.46,
            "fps": 50.0,
            "task": "Actuator Unboxing",
        },
        "prompt": "Unbox the actuator.",
        "state": {
            "featureName": "observation.state",
            "dtype": "float32",
            "componentNames": list(COMPONENT_NAMES),
            "values": [float(index) / 10 for index in range(14)],
        },
        "cameras": camera_entries,
        "recordedGroundTruth": recorded,
    }
    manifest_path = tmp_path / "observation.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    return manifest_path, document


def write_manifest(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def test_loads_valid_unavailable_observation_without_modifying_sources(
    tmp_path: Path,
) -> None:
    manifest_path, document = make_manifest(tmp_path)
    source_bytes = {
        path: path.read_bytes()
        for path in [manifest_path, *sorted((tmp_path / "images").iterdir())]
    }
    observation = load_checkpoint_observation(manifest_path)
    assert observation.observation_id == document["observationId"]
    assert observation.manifest_path == manifest_path.resolve()
    assert observation.manifest_byte_count == len(source_bytes[manifest_path])
    assert (
        observation.manifest_sha256
        == hashlib.sha256(source_bytes[manifest_path]).hexdigest()
    )
    assert observation.state.component_names == COMPONENT_NAMES
    assert (
        tuple(camera.feature_name for camera in observation.cameras)
        == CAMERA_FEATURE_NAMES
    )
    assert all(camera.path.is_absolute() for camera in observation.cameras)
    assert observation.recorded_ground_truth.available is False
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes


def test_loads_valid_available_ground_truth(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path, ground_truth=True)
    observation = load_checkpoint_observation(manifest_path)
    recorded = observation.recorded_ground_truth
    assert recorded.available is True
    assert recorded.reason is None
    assert recorded.component_names == COMPONENT_NAMES
    assert len(recorded.actions or ()) == 50
    assert recorded.frame_indices == tuple(range(123, 173))


def test_observation_manifest_is_acquired_exactly_once(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    real_snapshot = observation_module.read_stable_file_snapshot
    acquisitions = []

    def acquire(path: Path) -> bytes:
        if path == manifest_path.resolve():
            acquisitions.append(path)
        return real_snapshot(path)

    monkeypatch.setattr(observation_module, "read_stable_file_snapshot", acquire)
    load_checkpoint_observation(manifest_path)
    assert acquisitions == [manifest_path.resolve()]


@pytest.mark.parametrize("dangling", [False, True])
def test_observation_manifest_symlink_is_rejected_without_following_target(
    tmp_path: Path, dangling: bool
) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    target = tmp_path / "stored-observation.json"
    original = manifest_path.read_bytes()
    if not dangling:
        target.write_bytes(original)
    manifest_path.unlink()
    manifest_path.symlink_to(target)

    with pytest.raises(
        CheckpointObservationValidationError, match="manifest.*symbolic link"
    ):
        load_checkpoint_observation(manifest_path)

    assert manifest_path.is_symlink()
    if not dangling:
        assert target.read_bytes() == original


def test_observation_manifest_replacement_after_snapshot_is_not_reparsed(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, document = make_manifest(tmp_path)
    original = manifest_path.read_bytes()
    replacement_document = json.loads(original)
    replacement_document["observationId"] = "x" * len(document["observationId"])
    replacement = json.dumps(replacement_document).encode()
    assert len(replacement) == len(original)
    real_snapshot = observation_module.read_stable_file_snapshot

    def acquire_then_replace(path: Path) -> bytes:
        content = real_snapshot(path)
        if path == manifest_path.resolve():
            path.write_bytes(replacement)
        return content

    monkeypatch.setattr(
        observation_module, "read_stable_file_snapshot", acquire_then_replace
    )

    loaded = load_checkpoint_observation(manifest_path)

    assert loaded.observation_id == document["observationId"]
    assert loaded.manifest_path == manifest_path.resolve()
    assert manifest_path.read_bytes() == replacement


def test_observation_manifest_invalid_utf8_is_precise(tmp_path: Path) -> None:
    path = tmp_path / "observation.json"
    path.write_bytes(b"\xff")
    with pytest.raises(CheckpointObservationValidationError, match="not valid UTF-8"):
        load_checkpoint_observation(path)


def test_observation_manifest_mutation_during_acquisition_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _ = make_manifest(tmp_path)

    def changed(path: Path) -> bytes:
        raise StableFileSnapshotError(f"{path.name} changed while it was being read")

    monkeypatch.setattr(observation_module, "read_stable_file_snapshot", changed)
    with pytest.raises(
        CheckpointObservationValidationError,
        match=r"manifest.*observation\.json.*changed while it was being read",
    ):
        load_checkpoint_observation(manifest_path)


def test_observation_manifest_missing_nonregular_and_unreadable_errors_are_precise(
    tmp_path: Path, monkeypatch
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(CheckpointObservationValidationError, match="does not exist"):
        load_checkpoint_observation(missing)

    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(CheckpointObservationValidationError, match="regular file"):
        load_checkpoint_observation(directory)

    manifest_path, _ = make_manifest(tmp_path)

    def unreadable(path: Path) -> bytes:
        raise StableFileSnapshotError(f"could not read {path.name}")

    monkeypatch.setattr(observation_module, "read_stable_file_snapshot", unreadable)
    with pytest.raises(
        CheckpointObservationValidationError,
        match=r"manifest.*observation\.json.*could not read",
    ):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize("ground_truth", [False, True])
def test_json_schema_accepts_valid_manifest_forms(
    tmp_path: Path, ground_truth: bool
) -> None:
    _, document = make_manifest(tmp_path, ground_truth=ground_truth)
    schema_path = (
        Path(__file__).parents[1] / "schemas/checkpoint-observation-v1.schema.json"
    )
    validator = Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    validator.validate(document)


def test_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "observation.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(CheckpointObservationValidationError, match="not valid JSON"):
        load_checkpoint_observation(path)


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        ("name", "wrong", "schema.name"),
        ("major", 2, "schema.major"),
        ("minor", 1, "schema.minor"),
        ("major", True, "schema.major"),
        ("minor", False, "schema.minor"),
    ],
)
def test_rejects_invalid_schema(
    tmp_path: Path, field: str, value: object, path: str
) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document["schema"][field] = value
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=path):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize("revision", ["ABC", "A" * 40, "a" * 39, "a" * 41])
def test_rejects_invalid_revision(tmp_path: Path, revision: str) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document["dataset"]["revision"] = revision
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match="dataset.revision"):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episodeIndex", True),
        ("frameIndex", False),
        ("episodeIndex", -1),
        ("frameIndex", -1),
    ],
)
def test_rejects_invalid_dataset_indices(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document["dataset"][field] = value
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=f"dataset.{field}"):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize("field", ["observationId", "prompt"])
def test_rejects_empty_top_level_strings(tmp_path: Path, field: str) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document[field] = ""
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=field):
        load_checkpoint_observation(manifest_path)


def test_rejects_empty_task(tmp_path: Path) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document["dataset"]["task"] = ""
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match="dataset.task"):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (lambda state: state.update(featureName="state"), "state.featureName"),
        (lambda state: state.update(dtype="float64"), "state.dtype"),
        (lambda state: state["componentNames"].reverse(), "state.componentNames"),
        (lambda state: state["componentNames"].pop(), "state.componentNames"),
        (lambda state: state["values"].pop(), "state.values"),
        (
            lambda state: state["values"].__setitem__(4, float("nan")),
            r"state.values\[4\]",
        ),
        (lambda state: state["values"].__setitem__(5, True), r"state.values\[5\]"),
    ],
)
def test_rejects_invalid_state(tmp_path: Path, mutation, path: str) -> None:
    manifest_path, document = make_manifest(tmp_path)
    mutation(document["state"])
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=path):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cameras: cameras.pop(),
        lambda cameras: cameras.append(copy.deepcopy(cameras[0])),
    ],
)
def test_rejects_missing_or_extra_camera(tmp_path: Path, mutation) -> None:
    manifest_path, document = make_manifest(tmp_path)
    mutation(document["cameras"])
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match="cameras"):
        load_checkpoint_observation(manifest_path)


def test_rejects_duplicate_or_reordered_cameras(tmp_path: Path) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document["cameras"][1] = copy.deepcopy(document["cameras"][0])
    write_manifest(manifest_path, document)
    with pytest.raises(
        CheckpointObservationValidationError, match=r"cameras\[1\].featureName"
    ):
        load_checkpoint_observation(manifest_path)

    manifest_path, document = make_manifest(tmp_path)
    document["cameras"][0], document["cameras"][1] = (
        document["cameras"][1],
        document["cameras"][0],
    )
    write_manifest(manifest_path, document)
    with pytest.raises(
        CheckpointObservationValidationError, match=r"cameras\[0\].featureName"
    ):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize(
    "filename",
    [
        "/tmp/image.png",
        "../image.png",
        "images/../image.png",
        "./image.png",
        "images//image.png",
        "images\\image.png",
        "https://example/image.png",
        "//server/image.png",
    ],
)
def test_rejects_unsafe_camera_paths(tmp_path: Path, filename: str) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document["cameras"][0]["filename"] = filename
    write_manifest(manifest_path, document)
    with pytest.raises(
        CheckpointObservationValidationError, match=r"cameras\[0\].filename"
    ):
        load_checkpoint_observation(manifest_path)


def test_rejects_camera_symlink_escaping_manifest_directory(tmp_path: Path) -> None:
    manifest_path, document = make_manifest(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    Image.new("RGB", (224, 224)).save(outside)
    link = tmp_path / "images/escape.png"
    link.symlink_to(outside)
    document["cameras"][0]["filename"] = "images/escape.png"
    write_manifest(manifest_path, document)
    try:
        with pytest.raises(
            CheckpointObservationValidationError, match="resolve inside"
        ):
            load_checkpoint_observation(manifest_path)
    finally:
        outside.unlink()


def test_rejects_internal_camera_symlink(tmp_path: Path) -> None:
    manifest_path, document = make_manifest(tmp_path)
    target = tmp_path / document["cameras"][0]["filename"]
    stored = target.with_name("stored.png")
    target.replace(stored)
    target.symlink_to(stored.name)

    with pytest.raises(CheckpointObservationValidationError, match="symbolic link"):
        load_checkpoint_observation(manifest_path)


def test_rejects_missing_directory_and_unreadable_camera(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, document = make_manifest(tmp_path)
    missing = tmp_path / document["cameras"][0]["filename"]
    missing.unlink()
    with pytest.raises(CheckpointObservationValidationError, match="does not exist"):
        load_checkpoint_observation(manifest_path)

    manifest_path, document = make_manifest(tmp_path)
    directory = tmp_path / "images/directory.png"
    directory.mkdir()
    document["cameras"][0]["filename"] = "images/directory.png"
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match="regular file"):
        load_checkpoint_observation(manifest_path)

    manifest_path, document = make_manifest(tmp_path)
    camera_path = (tmp_path / document["cameras"][0]["filename"]).resolve()
    import lerobot_state_atlas.checkpoint_comparison.observation as observation_module

    real_access = observation_module.os.access
    monkeypatch.setattr(
        observation_module.os,
        "access",
        lambda path, mode: (
            False if Path(path) == camera_path else real_access(path, mode)
        ),
    )
    with pytest.raises(CheckpointObservationValidationError, match="readable file"):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize("field", ["byteCount", "sha256"])
def test_rejects_camera_integrity_mismatch(tmp_path: Path, field: str) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document["cameras"][0][field] = 1 if field == "byteCount" else "f" * 64
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=field):
        load_checkpoint_observation(manifest_path)


def test_each_camera_is_acquired_once_and_same_bytes_are_hashed_and_decoded(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    real_snapshot = observation_module.read_stable_file_snapshot
    real_hash = observation_module.sha256_bytes
    real_open = observation_module.Image.open
    acquired: list[bytes] = []
    hashed: list[bytes] = []
    decoded: list[bytes] = []

    def acquire(path: Path) -> bytes:
        content = real_snapshot(path)
        if path.suffix == ".png":
            acquired.append(content)
        return content

    def hash_snapshot(content: bytes) -> str:
        hashed.append(content)
        return real_hash(content)

    def open_snapshot(stream, *args, **kwargs):
        decoded.append(stream.getvalue())
        return real_open(stream, *args, **kwargs)

    monkeypatch.setattr(observation_module, "read_stable_file_snapshot", acquire)
    monkeypatch.setattr(observation_module, "sha256_bytes", hash_snapshot)
    monkeypatch.setattr(observation_module.Image, "open", open_snapshot)

    load_checkpoint_observation(manifest_path)

    assert len(acquired) == 3
    assert hashed[:-1] == acquired
    assert hashed[-1] == manifest_path.read_bytes()
    assert decoded == acquired


def test_camera_path_replacement_after_snapshot_cannot_mix_image_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, document = make_manifest(tmp_path)
    camera = document["cameras"][0]
    camera_path = tmp_path / "images/camera-0.bmp"
    Image.new("RGB", (224, 224), "red").save(camera_path)
    original = camera_path.read_bytes()
    replacement_path = tmp_path / "replacement.bmp"
    Image.new("RGB", (223, 224), "blue").save(replacement_path)
    replacement = replacement_path.read_bytes()
    assert len(replacement) == len(original)
    camera.update(
        filename="images/camera-0.bmp",
        byteCount=len(original),
        sha256=hashlib.sha256(original).hexdigest(),
    )
    write_manifest(manifest_path, document)
    real_snapshot = observation_module.read_stable_file_snapshot

    def acquire_then_replace(path: Path) -> bytes:
        content = real_snapshot(path)
        if path == camera_path.resolve():
            path.write_bytes(replacement)
        return content

    monkeypatch.setattr(
        observation_module, "read_stable_file_snapshot", acquire_then_replace
    )

    observation = load_checkpoint_observation(manifest_path)

    assert observation.cameras[0].width == 224
    assert camera_path.read_bytes() == replacement


def test_malformed_camera_snapshot_fails_clearly(tmp_path: Path) -> None:
    manifest_path, document = make_manifest(tmp_path)
    camera = document["cameras"][0]
    camera_path = tmp_path / camera["filename"]
    malformed = b"not an image"
    camera_path.write_bytes(malformed)
    camera.update(
        byteCount=len(malformed), sha256=hashlib.sha256(malformed).hexdigest()
    )
    write_manifest(manifest_path, document)

    with pytest.raises(
        CheckpointObservationValidationError,
        match=r"cameras\[0\]\.filename must reference a valid decodable image",
    ):
        load_checkpoint_observation(manifest_path)


def test_camera_mutation_during_snapshot_acquisition_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _ = make_manifest(tmp_path)

    real_snapshot = observation_module.read_stable_file_snapshot

    def changed(path: Path) -> bytes:
        if path.suffix == ".png":
            raise StableFileSnapshotError("camera changed while it was being read")
        return real_snapshot(path)

    monkeypatch.setattr(observation_module, "read_stable_file_snapshot", changed)
    with pytest.raises(
        CheckpointObservationValidationError,
        match=r"cameras\[0\]\.filename.*changed while it was being read",
    ):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize(
    ("size", "mode", "path"),
    [
        ((223, 224), "RGB", r"cameras\[0\].width"),
        ((224, 223), "RGB", r"cameras\[0\].height"),
        ((224, 224), "L", r"cameras\[0\].channels"),
    ],
)
def test_rejects_wrong_decoded_image_contract(
    tmp_path: Path, size: tuple[int, int], mode: str, path: str
) -> None:
    manifest_path, document = make_manifest(tmp_path)
    camera_path = tmp_path / document["cameras"][0]["filename"]
    Image.new(mode, size).save(camera_path)
    byte_count, sha256 = _image_identity(camera_path)
    document["cameras"][0].update(byteCount=byte_count, sha256=sha256)
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=path):
        load_checkpoint_observation(manifest_path)


def test_rejects_multiframe_image(tmp_path: Path) -> None:
    manifest_path, document = make_manifest(tmp_path)
    camera_path = tmp_path / "images/animated.gif"
    frames = [Image.new("RGB", (224, 224), color) for color in ("red", "blue")]
    frames[0].save(camera_path, save_all=True, append_images=frames[1:])
    byte_count, sha256 = _image_identity(camera_path)
    document["cameras"][0].update(
        filename="images/animated.gif", byteCount=byte_count, sha256=sha256
    )
    write_manifest(manifest_path, document)
    with pytest.raises(
        CheckpointObservationValidationError, match="exactly one image frame"
    ):
        load_checkpoint_observation(manifest_path)


def test_unavailable_ground_truth_requires_reason_and_forbids_arrays(
    tmp_path: Path,
) -> None:
    manifest_path, document = make_manifest(tmp_path)
    document["recordedGroundTruth"] = {"available": False}
    write_manifest(manifest_path, document)
    with pytest.raises(
        CheckpointObservationValidationError, match="recordedGroundTruth.reason"
    ):
        load_checkpoint_observation(manifest_path)

    manifest_path, document = make_manifest(tmp_path)
    document["recordedGroundTruth"]["actions"] = []
    write_manifest(manifest_path, document)
    with pytest.raises(
        CheckpointObservationValidationError, match="must not include fields"
    ):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (
            lambda value: value["componentNames"].reverse(),
            "recordedGroundTruth.componentNames",
        ),
        (lambda value: value["actions"].pop(), "recordedGroundTruth.actions"),
        (lambda value: value["actions"][2].pop(), r"recordedGroundTruth.actions\[2\]"),
        (
            lambda value: value["actions"][3].__setitem__(4, float("inf")),
            r"recordedGroundTruth.actions\[3\]\[4\]",
        ),
        (
            lambda value: value["actions"][3].__setitem__(4, False),
            r"recordedGroundTruth.actions\[3\]\[4\]",
        ),
    ],
)
def test_rejects_invalid_available_actions(tmp_path: Path, mutation, path: str) -> None:
    manifest_path, document = make_manifest(tmp_path, ground_truth=True)
    mutation(document["recordedGroundTruth"])
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=path):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (
            lambda value: value["relativeTimesSeconds"].pop(),
            "recordedGroundTruth.relativeTimesSeconds",
        ),
        (
            lambda value: value["relativeTimesSeconds"].__setitem__(0, 0.01),
            r"relativeTimesSeconds\[0\]",
        ),
        (
            lambda value: value["relativeTimesSeconds"].__setitem__(
                2, value["relativeTimesSeconds"][1]
            ),
            "strictly increasing",
        ),
        (
            lambda value: value["relativeTimesSeconds"].__setitem__(4, 0.081),
            r"relativeTimesSeconds\[4\]",
        ),
        (
            lambda value: value["relativeTimesSeconds"].__setitem__(5, True),
            r"relativeTimesSeconds\[5\]",
        ),
        (
            lambda value: value["relativeTimesSeconds"].__setitem__(5, float("nan")),
            r"relativeTimesSeconds\[5\]",
        ),
    ],
)
def test_rejects_invalid_recorded_times(tmp_path: Path, mutation, path: str) -> None:
    manifest_path, document = make_manifest(tmp_path, ground_truth=True)
    mutation(document["recordedGroundTruth"])
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=path):
        load_checkpoint_observation(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (lambda value: value["frameIndices"].pop(), "recordedGroundTruth.frameIndices"),
        (lambda value: value["frameIndices"].__setitem__(0, 122), r"frameIndices\[0\]"),
        (
            lambda value: value["frameIndices"].__setitem__(
                3, value["frameIndices"][2]
            ),
            r"recordedGroundTruth\.frameIndices\[3\] must equal the previous frame index plus one\.",
        ),
        (
            lambda value: value["frameIndices"].__setitem__(3, 124),
            r"recordedGroundTruth\.frameIndices\[3\] must equal the previous frame index plus one\.",
        ),
        (
            lambda value: value["frameIndices"].__setitem__(1, 125),
            r"recordedGroundTruth\.frameIndices\[1\] must equal the previous frame index plus one\.",
        ),
        (
            lambda value: value["frameIndices"].__setitem__(27, 151),
            r"recordedGroundTruth\.frameIndices\[27\] must equal the previous frame index plus one\.",
        ),
        (
            lambda value: value["frameIndices"].__setitem__(3, True),
            r"frameIndices\[3\]",
        ),
        (lambda value: value["frameIndices"].__setitem__(3, -1), r"frameIndices\[3\]"),
    ],
)
def test_rejects_invalid_recorded_frame_indices(
    tmp_path: Path, mutation, path: str
) -> None:
    manifest_path, document = make_manifest(tmp_path, ground_truth=True)
    mutation(document["recordedGroundTruth"])
    write_manifest(manifest_path, document)
    with pytest.raises(CheckpointObservationValidationError, match=path):
        load_checkpoint_observation(manifest_path)
