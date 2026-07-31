"""Rotation-matrix validation and deterministic XYZW quaternion conversion."""

import torch
from torch import Tensor


ROTATION_ATOL = 1e-9
ROTATION_RTOL = 1e-9
QUATERNION_NORM_EPSILON = 1e-12
QUATERNION_SIGN_TOLERANCE = 1e-12


def validate_rotation_matrices(rotation_matrices: Tensor) -> Tensor:
    """Return validated rotation matrices as a detached CPU float64 tensor.

    Float64 forward kinematics should remain well within the ``1e-9`` absolute
    and relative orthonormality tolerances. Determinants must be within
    ``1e-9`` of positive one. Invalid matrices are rejected rather than
    projected onto SO(3).
    """
    if rotation_matrices.ndim != 3 or rotation_matrices.shape[1:] != (3, 3):
        raise ValueError("Rotation matrices must have shape (num_matrices, 3, 3).")
    if rotation_matrices.shape[0] == 0:
        raise ValueError("Rotation matrices must contain at least one matrix.")

    matrices = rotation_matrices.detach().to(device="cpu", dtype=torch.float64).clone()
    if not torch.isfinite(matrices).all().item():
        raise ValueError("Rotation matrices must contain only finite values.")

    identity = torch.eye(3, dtype=torch.float64).expand(matrices.shape[0], -1, -1)
    gram_matrices = matrices.transpose(1, 2) @ matrices
    if not torch.allclose(
        gram_matrices,
        identity,
        atol=ROTATION_ATOL,
        rtol=ROTATION_RTOL,
    ):
        raise ValueError("Rotation matrices must be orthonormal.")

    determinants = torch.linalg.det(matrices)
    if not torch.allclose(
        determinants,
        torch.ones_like(determinants),
        atol=ROTATION_ATOL,
        rtol=0.0,
    ):
        raise ValueError("Rotation matrices must have determinant +1.")

    return matrices


def _validated_episode_indices(
    episode_indices: Tensor | None,
    *,
    count: int,
) -> Tensor | None:
    if episode_indices is None:
        return None
    if episode_indices.ndim != 1:
        raise ValueError("Episode indices must be one-dimensional.")
    if episode_indices.shape[0] != count:
        raise ValueError("Episode index count must match the quaternion count.")
    if (
        episode_indices.dtype == torch.bool
        or episode_indices.is_floating_point()
        or episode_indices.is_complex()
    ):
        raise ValueError("Episode indices must use an integer dtype.")

    indices = episode_indices.detach().to(device="cpu", dtype=torch.int64).clone()
    completed: set[int] = set()
    previous = int(indices[0].item())
    for value in indices[1:]:
        current = int(value.item())
        if current == previous:
            continue
        completed.add(previous)
        if current in completed:
            raise ValueError("Episode IDs must form contiguous blocks.")
        previous = current
    return indices


def _canonicalize_first_sign(quaternion: Tensor) -> None:
    w = float(quaternion[3].item())
    negate = w < -QUATERNION_SIGN_TOLERANCE
    if abs(w) <= QUATERNION_SIGN_TOLERANCE:
        for component in quaternion[:3]:
            value = float(component.item())
            if abs(value) > QUATERNION_SIGN_TOLERANCE:
                negate = value < 0.0
                break
    if negate:
        quaternion.neg_()


def canonicalize_quaternions_xyzw(
    quaternions: Tensor,
    episode_indices: Tensor | None = None,
) -> Tensor:
    """Normalize XYZW quaternions and apply deterministic episode-local signs.

    At each episode boundary, positive ``w`` is preferred. When ``w`` is
    within ``1e-12`` of zero, the first component among ``x, y, z`` whose
    magnitude exceeds ``1e-12`` is made positive. Later samples are negated
    only when their dot product with the preceding canonical sample is
    negative.
    """
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("Quaternions must have shape (num_quaternions, 4).")
    if quaternions.shape[0] == 0:
        raise ValueError("Quaternions must contain at least one quaternion.")

    values = quaternions.detach().to(device="cpu", dtype=torch.float64).clone()
    if not torch.isfinite(values).all().item():
        raise ValueError("Quaternions must contain only finite values.")

    norms = torch.linalg.vector_norm(values, dim=1)
    if torch.any(norms <= QUATERNION_NORM_EPSILON).item():
        raise ValueError("Quaternion norm must be greater than 1e-12.")
    values /= norms.unsqueeze(1)

    indices = _validated_episode_indices(
        episode_indices,
        count=values.shape[0],
    )
    for index in range(values.shape[0]):
        starts_episode = index == 0 or (
            indices is not None and indices[index] != indices[index - 1]
        )
        if starts_episode:
            _canonicalize_first_sign(values[index])
        elif torch.dot(values[index - 1], values[index]).item() < 0.0:
            values[index].neg_()

    return values


def rotation_matrices_to_quaternions_xyzw(
    rotation_matrices: Tensor,
    episode_indices: Tensor | None = None,
) -> Tensor:
    """Convert valid rotation matrices to normalized XYZW quaternions."""
    matrices = validate_rotation_matrices(rotation_matrices)
    count = matrices.shape[0]
    quaternions = torch.empty((count, 4), dtype=torch.float64)
    trace = matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]

    trace_mask = trace > 0.0
    if trace_mask.any():
        scale = 2.0 * torch.sqrt(1.0 + trace[trace_mask])
        selected = matrices[trace_mask]
        quaternions[trace_mask, 0] = (selected[:, 2, 1] - selected[:, 1, 2]) / scale
        quaternions[trace_mask, 1] = (selected[:, 0, 2] - selected[:, 2, 0]) / scale
        quaternions[trace_mask, 2] = (selected[:, 1, 0] - selected[:, 0, 1]) / scale
        quaternions[trace_mask, 3] = 0.25 * scale

    diagonal = torch.diagonal(matrices, dim1=1, dim2=2)
    largest_diagonal = torch.argmax(diagonal, dim=1)
    for axis in range(3):
        mask = ~trace_mask & (largest_diagonal == axis)
        if not mask.any():
            continue
        selected = matrices[mask]
        first = axis
        second = (axis + 1) % 3
        third = (axis + 2) % 3
        scale = 2.0 * torch.sqrt(
            1.0
            + selected[:, first, first]
            - selected[:, second, second]
            - selected[:, third, third]
        )
        quaternions[mask, first] = 0.25 * scale
        quaternions[mask, second] = (
            selected[:, second, first] + selected[:, first, second]
        ) / scale
        quaternions[mask, third] = (
            selected[:, third, first] + selected[:, first, third]
        ) / scale
        quaternions[mask, 3] = (
            selected[:, third, second] - selected[:, second, third]
        ) / scale

    return canonicalize_quaternions_xyzw(quaternions, episode_indices)
