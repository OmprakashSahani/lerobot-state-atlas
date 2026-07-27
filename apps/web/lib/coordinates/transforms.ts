import type { RigidTransform, Vector3 } from "@/lib/atlas-schema/types";

export function applyRigidTransform(
  point: Vector3,
  transform: RigidTransform,
): Vector3 {
  const [roll, pitch, yaw] = transform.rotationRpy;
  const cx = Math.cos(roll);
  const sx = Math.sin(roll);
  const cy = Math.cos(pitch);
  const sy = Math.sin(pitch);
  const cz = Math.cos(yaw);
  const sz = Math.sin(yaw);

  const rotatedX =
    (cz * cy) * point[0] +
    (cz * sy * sx - sz * cx) * point[1] +
    (cz * sy * cx + sz * sx) * point[2];
  const rotatedY =
    (sz * cy) * point[0] +
    (sz * sy * sx + cz * cx) * point[1] +
    (sz * sy * cx - cz * sx) * point[2];
  const rotatedZ =
    -sy * point[0] + cy * sx * point[1] + cy * cx * point[2];

  return [
    rotatedX + transform.translationXyz[0],
    rotatedY + transform.translationXyz[1],
    rotatedZ + transform.translationXyz[2],
  ];
}
