from pathlib import Path

import pytest

from lerobot_state_atlas.urdf import load_robot_model


VALID_URDF = """\
<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="meshes/base.glb" />
      </geometry>
    </visual>
  </link>

  <link name="arm_link" />
  <link name="tool0" />

  <joint name="joint1" type="revolute">
    <origin xyz="0.1 0.2 0.3" rpy="0.0 0.0 1.57" />
    <parent link="base_link" />
    <child link="arm_link" />
    <axis xyz="0 0 1" />
    <limit lower="-1.0" upper="2.0" effort="10" velocity="3" />
  </joint>

  <joint name="tool_joint" type="fixed">
    <parent link="arm_link" />
    <child link="tool0" />
  </joint>
</robot>
"""


def write_urdf(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "robot.urdf"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_robot_model(tmp_path: Path) -> None:
    model = load_robot_model(write_urdf(tmp_path, VALID_URDF))

    assert model.name == "test_robot"
    assert model.root_link == "base_link"
    assert model.links == ("base_link", "arm_link", "tool0")
    assert model.mesh_paths == ("meshes/base.glb",)

    joint = model.joint("joint1")

    assert joint.joint_type == "revolute"
    assert joint.parent_link == "base_link"
    assert joint.child_link == "arm_link"
    assert joint.origin_xyz == pytest.approx((0.1, 0.2, 0.3))
    assert joint.origin_rpy == pytest.approx((0.0, 0.0, 1.57))
    assert joint.axis == pytest.approx((0.0, 0.0, 1.0))
    assert joint.lower_limit == pytest.approx(-1.0)
    assert joint.upper_limit == pytest.approx(2.0)


def test_load_robot_model_uses_joint_defaults(tmp_path: Path) -> None:
    model = load_robot_model(write_urdf(tmp_path, VALID_URDF))

    joint = model.joint("tool_joint")

    assert joint.joint_type == "fixed"
    assert joint.origin_xyz == (0.0, 0.0, 0.0)
    assert joint.origin_rpy == (0.0, 0.0, 0.0)
    assert joint.axis is None
    assert joint.lower_limit is None
    assert joint.upper_limit is None


def test_robot_model_rejects_unknown_joint(tmp_path: Path) -> None:
    model = load_robot_model(write_urdf(tmp_path, VALID_URDF))

    with pytest.raises(KeyError, match="Unknown joint"):
        model.joint("missing_joint")


def test_load_robot_model_rejects_invalid_root(tmp_path: Path) -> None:
    path = write_urdf(
        tmp_path,
        "<not_robot name='test'><link name='base_link' /></not_robot>",
    )

    with pytest.raises(
        ValueError,
        match="root element must be <robot>",
    ):
        load_robot_model(path)


def test_load_robot_model_rejects_duplicate_links(tmp_path: Path) -> None:
    path = write_urdf(
        tmp_path,
        """\
<robot name="test">
  <link name="base_link" />
  <link name="base_link" />
</robot>
""",
    )

    with pytest.raises(
        ValueError,
        match="link names must be unique",
    ):
        load_robot_model(path)


def test_load_robot_model_rejects_unknown_child_link(
    tmp_path: Path,
) -> None:
    path = write_urdf(
        tmp_path,
        """\
<robot name="test">
  <link name="base_link" />
  <joint name="joint1" type="fixed">
    <parent link="base_link" />
    <child link="missing_link" />
  </joint>
</robot>
""",
    )

    with pytest.raises(
        ValueError,
        match="unknown child link",
    ):
        load_robot_model(path)


def test_load_robot_model_requires_one_root_link(
    tmp_path: Path,
) -> None:
    path = write_urdf(
        tmp_path,
        """\
<robot name="test">
  <link name="base_link" />
  <link name="unconnected_link" />
</robot>
""",
    )

    with pytest.raises(
        ValueError,
        match="exactly one root link",
    ):
        load_robot_model(path)


def test_load_robot_model_rejects_invalid_vector(
    tmp_path: Path,
) -> None:
    path = write_urdf(
        tmp_path,
        """\
<robot name="test">
  <link name="base_link" />
  <link name="arm_link" />
  <joint name="joint1" type="fixed">
    <origin xyz="0 1" />
    <parent link="base_link" />
    <child link="arm_link" />
  </joint>
</robot>
""",
    )

    with pytest.raises(
        ValueError,
        match="three-component vector",
    ):
        load_robot_model(path)
