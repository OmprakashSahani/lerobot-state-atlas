"use client";

export interface EnvironmentSource {
  id: string;
  kind: "grid" | "gaussian-splat";
  coordinateFrame: "canonical-shared-world";
}

export const gridEnvironment: EnvironmentSource = {
  id: "clean-grid",
  kind: "grid",
  coordinateFrame: "canonical-shared-world",
};

export function EnvironmentLayer({ source }: { source: EnvironmentSource }) {
  if (source.kind !== "grid") return null;
  return (
    <group name={`environment:${source.id}`}>
      <gridHelper
        args={[2.4, 48, "#33516c", "#172c3d"]}
        position={[0.25, 0, 0]}
        rotation={[Math.PI / 2, 0, 0]}
      />
      <mesh position={[0.25, 0, -0.012]}>
        <planeGeometry args={[2.4, 2.4]} />
        <meshStandardMaterial color="#07111b" roughness={0.92} />
      </mesh>
    </group>
  );
}
