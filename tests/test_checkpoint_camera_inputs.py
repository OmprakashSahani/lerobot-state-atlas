from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest
import torch

from lerobot_state_atlas.checkpoint_comparison import (
    CheckpointComparisonCameraInputError,
    prepare_bound_policy_observation_input,
    run_policy_comparison,
)
from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
)
from lerobot_state_atlas.checkpoint_comparison.observation import (
    CAMERA_FEATURE_NAMES,
    load_checkpoint_observation,
)
from test_checkpoint_observation import make_manifest, write_manifest


def _set_camera_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (224, 224), color)
    image.putpixel((1, 0), (0, 255, 128))
    image.save(path)


def _observation(tmp_path: Path):
    manifest_path, document = make_manifest(tmp_path)
    colors = ((0, 64, 255), (10, 20, 30), (40, 50, 60))
    for entry, color in zip(document["cameras"], colors, strict=True):
        path = tmp_path / entry["filename"]
        _set_camera_image(path, color)
        content = path.read_bytes()
        entry["byteCount"] = len(content)
        entry["sha256"] = hashlib.sha256(content).hexdigest()
    write_manifest(manifest_path, document)
    return load_checkpoint_observation(manifest_path), manifest_path


def _camera_bytes(mode="RGB", size=(224, 224), *, format="PNG", frames=1) -> bytes:
    stream = BytesIO()
    images = [Image.new(mode, size, index) for index in range(frames)]
    if frames == 1:
        images[0].save(stream, format=format)
    else:
        images[0].save(stream, format=format, save_all=True, append_images=images[1:])
    return stream.getvalue()


def _replace_camera_source(observation, tmp_path: Path, content: bytes, index: int = 0):
    path = tmp_path / f"replacement-{index}.image"
    path.write_bytes(content)
    camera = replace(
        observation.cameras[index],
        filename=path.name,
        path=path.resolve(),
        byte_count=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    cameras = list(observation.cameras)
    cameras[index] = camera
    return replace(observation, cameras=tuple(cameras)), path


def test_successful_conversion_is_canonical_independent_and_repeatable(
    tmp_path: Path,
) -> None:
    observation, manifest_path = _observation(tmp_path)
    sources = {manifest_path: manifest_path.read_bytes()}
    sources.update(
        {camera.path: camera.path.read_bytes() for camera in observation.cameras}
    )
    result = prepare_bound_policy_observation_input(observation)
    repeated = prepare_bound_policy_observation_input(observation)
    assert result.observation_id == observation.observation_id
    assert (
        tuple(camera.feature_name for camera in result.cameras) == CAMERA_FEATURE_NAMES
    )
    assert tuple(camera.source_path for camera in result.cameras) == tuple(
        camera.path for camera in observation.cameras
    )
    for camera, again in zip(result.cameras, repeated.cameras, strict=True):
        tensor = camera.tensor
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert tensor.requires_grad is False
        assert torch.isfinite(tensor).all()
        assert 0.0 <= tensor.min() and tensor.max() <= 1.0
        assert torch.equal(tensor, again.tensor)
        assert (
            tensor.untyped_storage().data_ptr()
            != again.tensor.untyped_storage().data_ptr()
        )
    first = result.cameras[0].tensor
    assert first[0, 0, 0, 0].item() == 0.0
    assert first[0, 2, 0, 0].item() == 1.0
    assert (
        first[0, 1, 0, 0].item() == torch.tensor(64 / 255, dtype=torch.float32).item()
    )
    assert (
        first[0, 2, 0, 1].item() == torch.tensor(128 / 255, dtype=torch.float32).item()
    )
    assert first.min().item() >= 0.0  # no PI05 [-1, 1] transform
    assert (
        len({item.tensor.untyped_storage().data_ptr() for item in result.cameras}) == 3
    )
    assert {path: path.read_bytes() for path in sources} == sources
    with pytest.raises(FrozenInstanceError):
        result.observation_id = "changed"  # type: ignore[misc]


def test_each_camera_is_acquired_exactly_once(tmp_path: Path) -> None:
    observation, _ = _observation(tmp_path)
    calls = []

    def reader(path: Path) -> bytes:
        calls.append(path)
        return path.read_bytes()

    prepare_bound_policy_observation_input(observation, snapshot_reader=reader)
    assert calls == [camera.path for camera in observation.cameras]


def test_path_replacement_after_snapshot_does_not_change_decoded_tensor(
    tmp_path: Path,
) -> None:
    observation, _ = _observation(tmp_path)
    original = observation.cameras[0].path.read_bytes()
    calls = 0

    def reader(path: Path) -> bytes:
        nonlocal calls
        calls += 1
        content = path.read_bytes()
        if calls == 1:
            path.write_bytes(b"x" * len(content))
        return content

    result = prepare_bound_policy_observation_input(observation, snapshot_reader=reader)
    assert result.cameras[0].source_sha256 == hashlib.sha256(original).hexdigest()
    assert result.cameras[0].tensor[0, 2, 0, 0].item() == 1.0


def test_mutation_during_snapshot_is_rejected_without_partial_result(
    tmp_path: Path,
) -> None:
    observation, _ = _observation(tmp_path)
    calls = []

    def reader(path: Path) -> bytes:
        calls.append(path)
        if len(calls) == 2:
            raise StableFileSnapshotError("changed while it was being read")
        return path.read_bytes()

    with pytest.raises(
        CheckpointComparisonCameraInputError, match=r"cameras\[1\].*changed"
    ):
        prepare_bound_policy_observation_input(observation, snapshot_reader=reader)
    assert len(calls) == 2


@pytest.mark.parametrize("dangling", [False, True])
def test_direct_and_dangling_camera_symlink_rejected(
    tmp_path: Path, dangling: bool
) -> None:
    observation, _ = _observation(tmp_path)
    camera_path = observation.cameras[0].path
    content = camera_path.read_bytes()
    target = tmp_path / "target.png"
    if not dangling:
        target.write_bytes(content)
    camera_path.unlink()
    camera_path.symlink_to(target)
    with pytest.raises(CheckpointComparisonCameraInputError, match="symbolic-link"):
        prepare_bound_policy_observation_input(observation)


def test_intermediate_symlink_missing_and_directory_rejected(tmp_path: Path) -> None:
    observation, _ = _observation(tmp_path)
    images = tmp_path / "images"
    stored = tmp_path / "stored-images"
    images.rename(stored)
    images.symlink_to(stored, target_is_directory=True)
    with pytest.raises(CheckpointComparisonCameraInputError, match="symbolic-link"):
        prepare_bound_policy_observation_input(observation)
    images.unlink()
    stored.rename(images)
    observation.cameras[0].path.unlink()
    with pytest.raises(CheckpointComparisonCameraInputError, match="does not exist"):
        prepare_bound_policy_observation_input(observation)
    observation.cameras[0].path.mkdir()
    with pytest.raises(CheckpointComparisonCameraInputError, match="regular file"):
        prepare_bound_policy_observation_input(observation)


def test_digest_byte_count_and_canonical_path_mismatches(tmp_path: Path) -> None:
    observation, _ = _observation(tmp_path)
    wrong_size = replace(
        observation,
        cameras=(
            replace(observation.cameras[0], byte_count=1),
            *observation.cameras[1:],
        ),
    )
    with pytest.raises(CheckpointComparisonCameraInputError, match="byteCount"):
        prepare_bound_policy_observation_input(wrong_size)
    wrong_hash = replace(
        observation,
        cameras=(
            replace(observation.cameras[0], sha256="0" * 64),
            *observation.cameras[1:],
        ),
    )
    with pytest.raises(CheckpointComparisonCameraInputError, match="sha256"):
        prepare_bound_policy_observation_input(wrong_hash)
    wrong_path = replace(
        observation,
        cameras=(
            replace(observation.cameras[0], path=tmp_path / "other.png"),
            *observation.cameras[1:],
        ),
    )
    with pytest.raises(CheckpointComparisonCameraInputError, match="canonical path"):
        prepare_bound_policy_observation_input(wrong_path)


@pytest.mark.parametrize(
    ("mode", "format"),
    [("L", "PNG"), ("RGBA", "PNG"), ("P", "PNG"), ("CMYK", "TIFF")],
)
def test_non_rgb_modes_rejected(tmp_path: Path, mode: str, format: str) -> None:
    observation, _ = _observation(tmp_path)
    malformed, _ = _replace_camera_source(
        observation, tmp_path, _camera_bytes(mode, format=format)
    )
    with pytest.raises(CheckpointComparisonCameraInputError, match="mode"):
        prepare_bound_policy_observation_input(malformed)


@pytest.mark.parametrize("size", [(223, 224), (224, 223)])
def test_wrong_dimensions_rejected(tmp_path: Path, size: tuple[int, int]) -> None:
    observation, _ = _observation(tmp_path)
    malformed, _ = _replace_camera_source(
        observation, tmp_path, _camera_bytes(size=size)
    )
    with pytest.raises(CheckpointComparisonCameraInputError, match="decodedSize"):
        prepare_bound_policy_observation_input(malformed)


def test_multiframe_truncated_and_malformed_images_rejected(tmp_path: Path) -> None:
    observation, _ = _observation(tmp_path)
    multiframe, _ = _replace_camera_source(
        observation, tmp_path, _camera_bytes(format="GIF", frames=2)
    )
    with pytest.raises(CheckpointComparisonCameraInputError, match="frameCount"):
        prepare_bound_policy_observation_input(multiframe)
    valid = _camera_bytes()
    truncated, _ = _replace_camera_source(
        observation, tmp_path, valid[: len(valid) // 2]
    )
    with pytest.raises(CheckpointComparisonCameraInputError, match="decodable"):
        prepare_bound_policy_observation_input(truncated)
    malformed, _ = _replace_camera_source(observation, tmp_path, b"not an image")
    with pytest.raises(CheckpointComparisonCameraInputError, match="decodable"):
        prepare_bound_policy_observation_input(malformed)


def test_malformed_camera_order_count_and_feature_rejected(tmp_path: Path) -> None:
    observation, _ = _observation(tmp_path)
    for cameras in (
        observation.cameras[:2],
        (*observation.cameras, observation.cameras[0]),
        (observation.cameras[1], observation.cameras[0], observation.cameras[2]),
        (observation.cameras[0], observation.cameras[0], observation.cameras[2]),
    ):
        with pytest.raises(CheckpointComparisonCameraInputError):
            prepare_bound_policy_observation_input(
                replace(observation, cameras=cameras)
            )
    malformed = replace(
        observation,
        cameras=(
            replace(observation.cameras[0], feature_name="wrong"),
            *observation.cameras[1:],
        ),
    )
    with pytest.raises(CheckpointComparisonCameraInputError, match="featureName"):
        prepare_bound_policy_observation_input(malformed)


def test_exif_orientation_is_not_applied(tmp_path: Path) -> None:
    observation, _ = _observation(tmp_path)
    image = Image.new("RGB", (224, 224), (0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = 3
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=100, subsampling=0, exif=exif)
    bound_observation, _ = _replace_camera_source(
        observation, tmp_path, stream.getvalue()
    )
    result = prepare_bound_policy_observation_input(bound_observation)
    tensor = result.cameras[0].tensor
    assert tensor[0, 0, 0, 0] > tensor[0, 0, -1, -1]


def test_result_satisfies_existing_inference_binding_contract(tmp_path: Path) -> None:
    observation, _ = _observation(tmp_path)
    bound = prepare_bound_policy_observation_input(observation)

    class Policy:
        def predict_action_chunk(self, processed, *, noise, num_inference_steps=None):
            return torch.zeros((1, 50, 14), dtype=torch.float32)

    result = run_policy_comparison(
        observation,
        bound_input=bound,
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
        base_policy=Policy(),
        fine_tuned_policy=Policy(),
        noise_seed=1,
    )
    assert result.observation_id == observation.observation_id
    assert tuple(plan.policy_id for plan in result.policies) == (
        "base-pi05",
        "fine-tuned-pi05",
    )
