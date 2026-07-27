import type { Vector3 } from "@/lib/atlas-schema/types";

export type Arm = "left" | "right";

export function spacingDeltaY(
  arm: Arm,
  spacing: number,
  manifestSpacing: number,
): number {
  return (arm === "left" ? 1 : -1) * (spacing - manifestSpacing) / 2;
}

export function applyRuntimeSpacing(
  point: Vector3,
  arm: Arm,
  spacing: number,
  manifestSpacing: number,
): Vector3 {
  return [
    point[0],
    point[1] + spacingDeltaY(arm, spacing, manifestSpacing),
    point[2],
  ];
}
