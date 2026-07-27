"use client";

import type { AtlasManifest } from "@/lib/atlas-schema/types";
import { applyRuntimeSpacing } from "@/lib/coordinates/runtimeSpacing";

function BaseReference({
  arm,
  position,
}: {
  arm: "left" | "right";
  position: [number, number, number];
}) {
  const color = arm === "left" ? "#67e8f9" : "#fb7185";
  return (
    <group name={`${arm}-base-reference`} position={position}>
      <mesh position={[0, 0, 0.018]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.055, 0.07, 0.036, 24]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.2} />
      </mesh>
      <axesHelper args={[0.12]} />
    </group>
  );
}

export function BaseReferenceLayer({
  manifest,
  spacing,
}: {
  manifest: AtlasManifest;
  spacing: number;
}) {
  return (
    <group name="robot-base-references">
      <BaseReference
        arm="left"
        position={applyRuntimeSpacing(
          manifest.coverage.canonicalTransforms.left.translationXyz,
          "left",
          spacing,
          manifest.coverage.armSpacing,
        )}
      />
      <BaseReference
        arm="right"
        position={applyRuntimeSpacing(
          manifest.coverage.canonicalTransforms.right.translationXyz,
          "right",
          spacing,
          manifest.coverage.armSpacing,
        )}
      />
    </group>
  );
}
