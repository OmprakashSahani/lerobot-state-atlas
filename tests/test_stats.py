import pytest
import torch

from lerobot_state_atlas.stats import compute_state_statistics


def test_compute_state_statistics() -> None:
    states = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )

    statistics = compute_state_statistics(
        states,
        component_names=["joint_1", "joint_2"],
    )

    assert statistics.num_frames == 3
    assert statistics.state_dimension == 2

    joint_1 = statistics.components[0]

    assert joint_1.name == "joint_1"
    assert joint_1.minimum == pytest.approx(1.0)
    assert joint_1.maximum == pytest.approx(5.0)
    assert joint_1.mean == pytest.approx(3.0)
    assert joint_1.standard_deviation == pytest.approx(1.6329931619)
    assert joint_1.movement_range == pytest.approx(4.0)


def test_compute_state_statistics_uses_default_names() -> None:
    states = torch.tensor(
        [[1.0, 2.0]],
        dtype=torch.float32,
    )

    statistics = compute_state_statistics(states)

    assert statistics.components[0].name == "state_0"
    assert statistics.components[1].name == "state_1"


def test_compute_state_statistics_rejects_non_matrix() -> None:
    with pytest.raises(
        ValueError,
        match="States must have shape",
    ):
        compute_state_statistics(torch.tensor([1.0, 2.0]))


def test_compute_state_statistics_rejects_empty_states() -> None:
    with pytest.raises(
        ValueError,
        match="at least one frame",
    ):
        compute_state_statistics(torch.empty((0, 2)))


def test_compute_state_statistics_rejects_wrong_name_count() -> None:
    states = torch.tensor(
        [[1.0, 2.0]],
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="component names",
    ):
        compute_state_statistics(
            states,
            component_names=["joint_1"],
        )
