import math

import pytest
import torch

from lerobot_state_atlas.kinematics import (
    compute_link_transforms,
    get_link_position,
)
from lerobot_state_atlas.urdf import JointDefinition, RobotModel


def make_joint(
    *,
    name: str,
    joint_type: str,
    parent_link: str,
    child_link: str,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: tuple[float, float, float] | None = None,
) -> JointDefinition:
    return JointDefinition(
        name=name,
        joint_type=joint_type,
        parent_link=parent_link,
        child_link=child_link,
        origin_xyz=origin_xyz,
        origin_rpy=origin_rpy,
        axis=axis,
        lower_limit=None,
        upper_limit=None,
    )


def test_compute_link_transforms_for_fixed_chain() -> None:
    model = RobotModel(
        name="fixed_robot",
        root_link="base_link",
        links=("base_link", "arm_link", "tool0"),
        joints=(
            make_joint(
                name="base_to_arm",
                joint_type="fixed",
                parent_link="base_link",
                child_link="arm_link",
                origin_xyz=(1.0, 2.0, 3.0),
            ),
            make_joint(
                name="arm_to_tool",
                joint_type="fixed",
                parent_link="arm_link",
                child_link="tool0",
                origin_xyz=(0.5, 0.0, 0.0),
            ),
        ),
        mesh_paths=(),
    )

    transforms = compute_link_transforms(model)

    assert get_link_position(
        transforms,
        "base_link",
    ).tolist() == pytest.approx([0.0, 0.0, 0.0])

    assert get_link_position(
        transforms,
        "arm_link",
    ).tolist() == pytest.approx([1.0, 2.0, 3.0])

    assert get_link_position(
        transforms,
        "tool0",
    ).tolist() == pytest.approx([1.5, 2.0, 3.0])


def test_compute_revolute_joint_transform() -> None:
    model = RobotModel(
        name="revolute_robot",
        root_link="base_link",
        links=("base_link", "arm_link", "tool0"),
        joints=(
            make_joint(
                name="joint1",
                joint_type="revolute",
                parent_link="base_link",
                child_link="arm_link",
                axis=(0.0, 0.0, 1.0),
            ),
            make_joint(
                name="tool_joint",
                joint_type="fixed",
                parent_link="arm_link",
                child_link="tool0",
                origin_xyz=(1.0, 0.0, 0.0),
            ),
        ),
        mesh_paths=(),
    )

    transforms = compute_link_transforms(
        model,
        {"joint1": math.pi / 2.0},
    )

    position = get_link_position(transforms, "tool0")

    assert position.tolist() == pytest.approx(
        [0.0, 1.0, 0.0],
        abs=1e-9,
    )


def test_compute_prismatic_joint_transform() -> None:
    model = RobotModel(
        name="prismatic_robot",
        root_link="base_link",
        links=("base_link", "slider"),
        joints=(
            make_joint(
                name="slider_joint",
                joint_type="prismatic",
                parent_link="base_link",
                child_link="slider",
                origin_xyz=(1.0, 0.0, 0.0),
                axis=(0.0, 2.0, 0.0),
            ),
        ),
        mesh_paths=(),
    )

    transforms = compute_link_transforms(
        model,
        {"slider_joint": 0.5},
    )

    position = get_link_position(transforms, "slider")

    assert position.tolist() == pytest.approx(
        [1.0, 0.5, 0.0],
    )


def test_compute_origin_rotation() -> None:
    model = RobotModel(
        name="origin_rotation_robot",
        root_link="base_link",
        links=("base_link", "rotated_link", "tool0"),
        joints=(
            make_joint(
                name="rotated_origin",
                joint_type="fixed",
                parent_link="base_link",
                child_link="rotated_link",
                origin_rpy=(0.0, 0.0, math.pi / 2.0),
            ),
            make_joint(
                name="tool_joint",
                joint_type="fixed",
                parent_link="rotated_link",
                child_link="tool0",
                origin_xyz=(1.0, 0.0, 0.0),
            ),
        ),
        mesh_paths=(),
    )

    transforms = compute_link_transforms(model)

    assert get_link_position(
        transforms,
        "tool0",
    ).tolist() == pytest.approx(
        [0.0, 1.0, 0.0],
        abs=1e-9,
    )


def test_compute_link_transforms_uses_zero_positions() -> None:
    model = RobotModel(
        name="zero_robot",
        root_link="base_link",
        links=("base_link", "arm_link"),
        joints=(
            make_joint(
                name="joint1",
                joint_type="revolute",
                parent_link="base_link",
                child_link="arm_link",
                origin_xyz=(0.0, 0.0, 1.0),
                axis=(0.0, 0.0, 1.0),
            ),
        ),
        mesh_paths=(),
    )

    transforms = compute_link_transforms(model)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        transforms["arm_link"],
        expected,
    )


def test_compute_link_transforms_rejects_unknown_joint() -> None:
    model = RobotModel(
        name="test_robot",
        root_link="base_link",
        links=("base_link",),
        joints=(),
        mesh_paths=(),
    )

    with pytest.raises(
        KeyError,
        match="Unknown joint positions",
    ):
        compute_link_transforms(
            model,
            {"missing_joint": 1.0},
        )


def test_compute_link_transforms_rejects_zero_axis() -> None:
    model = RobotModel(
        name="zero_axis_robot",
        root_link="base_link",
        links=("base_link", "arm_link"),
        joints=(
            make_joint(
                name="joint1",
                joint_type="revolute",
                parent_link="base_link",
                child_link="arm_link",
                axis=(0.0, 0.0, 0.0),
            ),
        ),
        mesh_paths=(),
    )

    with pytest.raises(
        ValueError,
        match="zero-length axis",
    ):
        compute_link_transforms(model)


def test_compute_link_transforms_rejects_unreachable_links() -> None:
    model = RobotModel(
        name="disconnected_robot",
        root_link="base_link",
        links=("base_link", "disconnected_link"),
        joints=(),
        mesh_paths=(),
    )

    with pytest.raises(
        ValueError,
        match="unreachable links",
    ):
        compute_link_transforms(model)


def test_get_link_position_rejects_unknown_link() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown link",
    ):
        get_link_position(
            {"base_link": torch.eye(4)},
            "missing_link",
        )


def test_get_link_position_rejects_invalid_transform() -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \(4, 4\)",
    ):
        get_link_position(
            {"base_link": torch.eye(3)},
            "base_link",
        )
