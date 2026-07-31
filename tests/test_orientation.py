import math

import pytest
import torch

from lerobot_state_atlas.orientation import (
    canonicalize_quaternions_xyzw,
    rotation_matrices_to_quaternions_xyzw,
    validate_rotation_matrices,
)


def rotation_x(angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=torch.float64,
    )


def rotation_y(angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float64,
    )


def rotation_z(angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion
    return torch.tensor(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=torch.float64,
    )


@pytest.mark.parametrize(
    ("matrix", "expected"),
    (
        (torch.eye(3, dtype=torch.float64), (0.0, 0.0, 0.0, 1.0)),
        (rotation_x(math.pi / 2.0), (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))),
        (rotation_x(-math.pi / 2.0), (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))),
        (rotation_y(math.pi / 2.0), (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))),
        (rotation_y(-math.pi / 2.0), (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))),
        (rotation_z(math.pi / 2.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))),
        (rotation_z(-math.pi / 2.0), (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))),
    ),
)
def test_axis_rotation_quaternions(
    matrix: torch.Tensor,
    expected: tuple[float, float, float, float],
) -> None:
    quaternion = rotation_matrices_to_quaternions_xyzw(matrix.unsqueeze(0))

    torch.testing.assert_close(
        quaternion,
        torch.tensor([expected], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_fixed_tool0_rotation() -> None:
    matrix = rotation_z(-math.pi / 2.0) @ rotation_y(0.0) @ rotation_x(-math.pi / 2.0)

    quaternion = rotation_matrices_to_quaternions_xyzw(matrix.unsqueeze(0))

    torch.testing.assert_close(
        quaternion,
        torch.tensor([[-0.5, 0.5, -0.5, 0.5]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_composed_rotations_are_unit_and_round_trip() -> None:
    matrices = torch.stack(
        (
            rotation_z(0.3) @ rotation_y(-0.7) @ rotation_x(1.1),
            rotation_z(-2.4) @ rotation_y(0.2) @ rotation_x(-0.5),
            rotation_z(math.pi) @ rotation_y(math.pi / 3.0),
        )
    )

    quaternions = rotation_matrices_to_quaternions_xyzw(matrices)
    reconstructed = torch.stack(
        tuple(quaternion_xyzw_to_matrix(value) for value in quaternions)
    )

    torch.testing.assert_close(
        torch.linalg.vector_norm(quaternions, dim=1),
        torch.ones(3, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(reconstructed, matrices, atol=1e-12, rtol=1e-12)


def test_first_sample_prefers_positive_w() -> None:
    canonical = canonicalize_quaternions_xyzw(
        torch.tensor([[0.0, 0.0, 0.5, -0.5]], dtype=torch.float64)
    )

    assert canonical[0, 3] > 0.0


def test_pi_rotation_uses_xyz_tie_break() -> None:
    canonical = canonicalize_quaternions_xyzw(
        torch.tensor([[-1.0, 0.0, 0.0, 1e-14]], dtype=torch.float64)
    )

    assert canonical[0, 0] > 0.0


def test_adjacent_equivalent_signs_are_continuity_corrected() -> None:
    canonical = canonicalize_quaternions_xyzw(
        torch.tensor(
            [
                [0.5, -0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5, -0.5],
            ],
            dtype=torch.float64,
        )
    )

    torch.testing.assert_close(canonical[1], canonical[0])


def test_continuity_resets_at_episode_boundaries() -> None:
    canonical = canonicalize_quaternions_xyzw(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0, -0.6],
            ],
            dtype=torch.float64,
        ),
        torch.tensor([4, 4, 9], dtype=torch.int64),
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-0.8, 0.0, 0.0, 0.6],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(canonical, expected)


@pytest.mark.parametrize(
    "matrices",
    (
        torch.zeros((3, 3), dtype=torch.float64),
        torch.zeros((1, 4, 4), dtype=torch.float64),
        torch.zeros((1, 3, 3, 1), dtype=torch.float64),
    ),
)
def test_rejects_invalid_matrix_shape(matrices: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        validate_rotation_matrices(matrices)


def test_rejects_empty_matrix_batch() -> None:
    with pytest.raises(ValueError, match="at least one matrix"):
        validate_rotation_matrices(torch.empty((0, 3, 3), dtype=torch.float64))


@pytest.mark.parametrize("invalid_value", (float("nan"), float("inf"), float("-inf")))
def test_rejects_nonfinite_matrix(invalid_value: float) -> None:
    matrix = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    matrix[0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite values"):
        validate_rotation_matrices(matrix)


@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (
            torch.tensor(
                [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            "orthonormal",
        ),
        (2.0 * torch.eye(3, dtype=torch.float64), "orthonormal"),
        (
            torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
            "determinant \\+1",
        ),
        (torch.zeros((3, 3), dtype=torch.float64), "orthonormal"),
    ),
)
def test_rejects_invalid_rotation_matrix(
    matrix: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_rotation_matrices(matrix.unsqueeze(0))


def test_rejects_near_zero_quaternion_norm() -> None:
    with pytest.raises(ValueError, match="norm must be greater"):
        canonicalize_quaternions_xyzw(torch.full((1, 4), 1e-15, dtype=torch.float64))


@pytest.mark.parametrize(
    ("episode_indices", "message"),
    (
        (torch.tensor([[1], [1]]), "one-dimensional"),
        (torch.tensor([1]), "must match"),
        (torch.tensor([1.0, 1.0]), "integer dtype"),
        (torch.tensor([True, False]), "integer dtype"),
    ),
)
def test_rejects_invalid_episode_indices(
    episode_indices: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rotation_matrices_to_quaternions_xyzw(
            torch.eye(3, dtype=torch.float64).expand(2, -1, -1),
            episode_indices,
        )


def test_rejects_noncontiguous_episode_recurrence() -> None:
    with pytest.raises(ValueError, match="contiguous blocks"):
        rotation_matrices_to_quaternions_xyzw(
            torch.eye(3, dtype=torch.float64).expand(3, -1, -1),
            torch.tensor([1, 2, 1], dtype=torch.int64),
        )


def test_inputs_are_not_mutated_and_outputs_are_cpu_float64() -> None:
    matrices = rotation_z(math.pi / 2.0).to(dtype=torch.float32).unsqueeze(0)
    original_matrices = matrices.clone()
    episode_indices = torch.tensor([3], dtype=torch.int32)
    original_indices = episode_indices.clone()

    validated = validate_rotation_matrices(matrices)
    quaternions = rotation_matrices_to_quaternions_xyzw(
        matrices,
        episode_indices,
    )

    torch.testing.assert_close(matrices, original_matrices)
    assert torch.equal(episode_indices, original_indices)
    assert validated.device.type == "cpu"
    assert validated.dtype == torch.float64
    assert quaternions.device.type == "cpu"
    assert quaternions.dtype == torch.float64
    assert quaternions.shape == (1, 4)
