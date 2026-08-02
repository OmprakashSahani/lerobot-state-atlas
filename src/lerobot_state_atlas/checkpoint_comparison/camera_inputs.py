"""Verified local camera snapshots converted to canonical PI05 input tensors."""

from collections.abc import Callable
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
import stat

from PIL import Image, UnidentifiedImageError
import torch

from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
    read_stable_file_snapshot,
)
from lerobot_state_atlas.checkpoint_comparison.models import (
    BoundCameraInput,
    BoundPolicyObservationInput,
    ObservationCamera,
    PolicyComparisonObservation,
)


CAMERA_FEATURE_NAMES = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top",
)
IMAGE_SIZE = (224, 224)
TENSOR_SHAPE = (1, 3, 224, 224)
CONVERSION_ID = "rgb224-uint8-to-contiguous-nchw-float32-unit-range-v1"


class CheckpointComparisonCameraInputError(ValueError):
    """Raised when verified camera inputs cannot be prepared atomically."""


def _fail(field: str, message: str) -> None:
    raise CheckpointComparisonCameraInputError(f"{field} {message}")


def _camera_field(index: int, camera: ObservationCamera | None = None) -> str:
    feature = camera.feature_name if camera is not None else CAMERA_FEATURE_NAMES[index]
    path = camera.path if camera is not None else "<unknown>"
    return f"observation.cameras[{index}] ({feature!r}, path={path})"


def _lexical_camera_path(
    observation: PolicyComparisonObservation,
    camera: ObservationCamera,
    index: int,
) -> Path:
    field = _camera_field(index, camera)
    if not isinstance(camera.filename, str) or not camera.filename:
        _fail(f"{field}.filename", "must be a non-empty relative POSIX path.")
    relative = PurePosixPath(camera.filename)
    if (
        "\\" in camera.filename
        or "://" in camera.filename
        or camera.filename.startswith("//")
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != camera.filename
    ):
        _fail(f"{field}.filename", "must be a safe manifest-relative POSIX path.")
    manifest_directory = observation.manifest_path.parent.resolve()
    current = manifest_directory
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail(
                f"{field}.path", f"must not contain symbolic-link component {current}."
            )
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(manifest_directory)
    except ValueError:
        _fail(f"{field}.path", "must remain inside the observation manifest directory.")
    if not isinstance(camera.path, Path):
        _fail(f"{field}.path", "must be a pathlib.Path.")
    if resolved != camera.path:
        _fail(
            f"{field}.path",
            f"must resolve to validated canonical path {camera.path}; received {resolved}.",
        )
    return current


def _snapshot(
    path: Path,
    camera: ObservationCamera,
    index: int,
    reader: Callable[[Path], bytes],
) -> bytes:
    field = _camera_field(index, camera)
    if path.is_symlink():
        _fail(f"{field}.path", "must not be a symbolic link.")
    if not path.exists():
        _fail(f"{field}.path", "does not exist.")
    try:
        info = path.stat()
    except OSError as error:
        raise CheckpointComparisonCameraInputError(
            f"{field}.path could not be inspected: {error}."
        ) from error
    if not stat.S_ISREG(info.st_mode):
        _fail(f"{field}.path", "must be an ordinary regular file.")
    try:
        content = reader(path)
    except StableFileSnapshotError as error:
        raise CheckpointComparisonCameraInputError(
            f"{field}.path could not be acquired as a stable snapshot: {error}."
        ) from error
    except OSError as error:
        raise CheckpointComparisonCameraInputError(
            f"{field}.path could not be read as a stable snapshot: {error}."
        ) from error
    if not isinstance(content, bytes):
        _fail(f"{field}.snapshot", "reader must return immutable bytes.")
    return content


def _decode_tensor(
    snapshot: bytes, camera: ObservationCamera, index: int
) -> torch.Tensor:
    field = _camera_field(index, camera)
    try:
        with Image.open(BytesIO(snapshot)) as image:
            frame_count = int(getattr(image, "n_frames", 1))
            if frame_count != 1:
                _fail(
                    f"{field}.frameCount",
                    f"must be 1; decoded {frame_count} frames.",
                )
            if image.size != IMAGE_SIZE:
                _fail(
                    f"{field}.decodedSize",
                    f"must be [224, 224]; decoded [{image.width}, {image.height}].",
                )
            if image.mode != "RGB":
                _fail(f"{field}.mode", f"must be 'RGB'; decoded {image.mode!r}.")
            bands = image.getbands()
            if len(bands) != 3 or bands != ("R", "G", "B"):
                _fail(
                    f"{field}.channels",
                    f"must be ordered RGB with 3 channels; decoded {bands!r}.",
                )
            image.load()
            pixels = image.tobytes()
    except CheckpointComparisonCameraInputError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise CheckpointComparisonCameraInputError(
            f"{field}.image must be a fully decodable single-frame RGB image: {error}."
        ) from error
    expected_pixels = 224 * 224 * 3
    if len(pixels) != expected_pixels:
        _fail(
            f"{field}.pixels",
            f"must contain {expected_pixels} uint8 channel values; decoded {len(pixels)}.",
        )
    # bytearray gives Torch writable temporary storage; clone severs that storage
    # before the CHW float conversion and the temporary is then discarded.
    hwc = (
        torch.frombuffer(bytearray(pixels), dtype=torch.uint8)
        .reshape(224, 224, 3)
        .clone()
    )
    tensor = (
        hwc.permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div(255.0)
        .unsqueeze(0)
        .contiguous()
    )
    if tuple(tensor.shape) != TENSOR_SHAPE:
        _fail(f"{field}.tensor.shape", f"must be {list(TENSOR_SHAPE)}.")
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        _fail(f"{field}.tensor", "must be a CPU float32 tensor.")
    if not tensor.is_contiguous():
        _fail(f"{field}.tensor", "must use contiguous storage.")
    if tensor.requires_grad:
        _fail(f"{field}.tensor.requiresGrad", "must be false.")
    if not torch.isfinite(tensor).all().item():
        _fail(f"{field}.tensor", "must contain only finite values.")
    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())
    if minimum < 0.0 or maximum > 1.0:
        _fail(
            f"{field}.tensor",
            f"must lie in [0.0, 1.0]; received [{minimum}, {maximum}].",
        )
    return tensor


def prepare_bound_policy_observation_input(
    observation: PolicyComparisonObservation,
    *,
    snapshot_reader: Callable[[Path], bytes] | None = None,
) -> BoundPolicyObservationInput:
    """Decode canonical tensors directly from reverified immutable snapshots.

    This proves that each returned tensor was decoded from the exact local bytes
    whose size and SHA-256 were rechecked here. Later transformations must retain
    their own provenance; this function does not perform PI05 preprocessing.
    """
    if not isinstance(observation, PolicyComparisonObservation):
        _fail("observation", "must be a validated PolicyComparisonObservation.")
    if not isinstance(observation.cameras, tuple) or len(observation.cameras) != 3:
        _fail("observation.cameras", "must contain exactly three ordered cameras.")
    reader = snapshot_reader or read_stable_file_snapshot
    if not callable(reader):
        _fail("snapshot_reader", "must be callable.")
    prepared: list[BoundCameraInput] = []
    for index, expected_feature in enumerate(CAMERA_FEATURE_NAMES):
        camera = observation.cameras[index]
        if not isinstance(camera, ObservationCamera):
            _fail(f"observation.cameras[{index}]", "must be an ObservationCamera.")
        field = _camera_field(index, camera)
        if camera.feature_name != expected_feature:
            _fail(f"{field}.featureName", f"must be {expected_feature!r}.")
        if (camera.width, camera.height, camera.channels) != (224, 224, 3):
            _fail(
                f"{field}.declaredShape",
                "must be [224, 224, 3].",
            )
        lexical_path = _lexical_camera_path(observation, camera, index)
        snapshot = _snapshot(lexical_path, camera, index, reader)
        # Recheck lexical containment after acquisition so an intermediate
        # component introduced during the read cannot silently remain accepted.
        _lexical_camera_path(observation, camera, index)
        if len(snapshot) != camera.byte_count:
            _fail(
                f"{field}.byteCount",
                f"expected {camera.byte_count}; acquired {len(snapshot)}.",
            )
        digest = hashlib.sha256(snapshot).hexdigest()
        if digest != camera.sha256:
            _fail(
                f"{field}.sha256",
                f"expected {camera.sha256}; acquired {digest}.",
            )
        prepared.append(
            BoundCameraInput(
                feature_name=camera.feature_name,
                tensor=_decode_tensor(snapshot, camera, index),
                source_sha256=camera.sha256,
                source_byte_count=camera.byte_count,
                source_path=camera.path,
            )
        )
    return BoundPolicyObservationInput(observation.observation_id, tuple(prepared))
