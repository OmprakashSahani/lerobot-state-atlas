from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class JointDefinition:
    """Kinematic definition of one URDF joint."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: Vector3
    origin_rpy: Vector3
    axis: Vector3 | None
    lower_limit: float | None
    upper_limit: float | None


@dataclass(frozen=True)
class RobotModel:
    """Parsed kinematic structure of a URDF robot."""

    name: str
    root_link: str
    links: tuple[str, ...]
    joints: tuple[JointDefinition, ...]
    mesh_paths: tuple[str, ...]

    def joint(self, name: str) -> JointDefinition:
        """Return a joint by name."""
        for joint in self.joints:
            if joint.name == name:
                return joint

        raise KeyError(f"Unknown joint: {name}")


def _parse_vector(
    value: str | None,
    *,
    default: Vector3 = (0.0, 0.0, 0.0),
) -> Vector3:
    if value is None:
        return default

    components = tuple(float(component) for component in value.split())

    if len(components) != 3:
        raise ValueError(f"Expected a three-component vector, received: {value}")

    return components


def load_robot_model(path: str | Path) -> RobotModel:
    """Load and validate a robot model from a URDF file."""
    urdf_path = Path(path)

    try:
        root = ElementTree.parse(urdf_path).getroot()
    except ElementTree.ParseError as error:
        raise ValueError(f"Invalid URDF XML: {urdf_path}") from error

    if root.tag != "robot":
        raise ValueError("URDF root element must be <robot>.")

    robot_name = root.attrib.get("name")

    if not robot_name:
        raise ValueError("URDF robot must define a name.")

    links = tuple(link.attrib["name"] for link in root.findall("link"))

    if not links:
        raise ValueError("URDF must define at least one link.")

    if len(set(links)) != len(links):
        raise ValueError("URDF link names must be unique.")

    joints: list[JointDefinition] = []

    for element in root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")

        if parent is None or child is None:
            raise ValueError(
                f"Joint {element.attrib.get('name', '<unnamed>')} "
                "must define parent and child links."
            )

        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")

        joints.append(
            JointDefinition(
                name=element.attrib["name"],
                joint_type=element.attrib["type"],
                parent_link=parent.attrib["link"],
                child_link=child.attrib["link"],
                origin_xyz=_parse_vector(
                    origin.attrib.get("xyz") if origin is not None else None
                ),
                origin_rpy=_parse_vector(
                    origin.attrib.get("rpy") if origin is not None else None
                ),
                axis=(
                    _parse_vector(axis.attrib.get("xyz")) if axis is not None else None
                ),
                lower_limit=(
                    float(limit.attrib["lower"])
                    if limit is not None and "lower" in limit.attrib
                    else None
                ),
                upper_limit=(
                    float(limit.attrib["upper"])
                    if limit is not None and "upper" in limit.attrib
                    else None
                ),
            )
        )

    joint_names = tuple(joint.name for joint in joints)

    if len(set(joint_names)) != len(joint_names):
        raise ValueError("URDF joint names must be unique.")

    link_set = set(links)

    for joint in joints:
        if joint.parent_link not in link_set:
            raise ValueError(
                f"Joint {joint.name} references unknown parent "
                f"link: {joint.parent_link}"
            )

        if joint.child_link not in link_set:
            raise ValueError(
                f"Joint {joint.name} references unknown child link: {joint.child_link}"
            )

    child_links = {joint.child_link for joint in joints}
    root_links = tuple(link for link in links if link not in child_links)

    if len(root_links) != 1:
        raise ValueError("URDF must contain exactly one root link.")

    mesh_paths = tuple(mesh.attrib["filename"] for mesh in root.findall(".//mesh"))

    return RobotModel(
        name=robot_name,
        root_link=root_links[0],
        links=links,
        joints=tuple(joints),
        mesh_paths=mesh_paths,
    )
