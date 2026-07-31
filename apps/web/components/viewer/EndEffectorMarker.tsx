import type {
  QuaternionXyzw,
  Vector3,
} from "@/lib/atlas-schema/types";

export const END_EFFECTOR_MARKER_COLORS = {
  left: "#5ee4ff",
  right: "#ff6f91",
} as const;

export interface EndEffectorMarkerDescription {
  arm: "left" | "right";
  color: string;
  orientationGlyph?: {
    quaternionXyzw: QuaternionXyzw;
    forwardAxis: "local +Z";
    crossAxes: readonly ["local +X", "local +Y"];
  };
}

export function describeEndEffectorMarker(
  arm: "left" | "right",
  orientationXyzw?: QuaternionXyzw,
): EndEffectorMarkerDescription {
  return {
    arm,
    color: END_EFFECTOR_MARKER_COLORS[arm],
    ...(orientationXyzw === undefined
      ? {}
      : {
          orientationGlyph: {
            quaternionXyzw: orientationXyzw,
            forwardAxis: "local +Z" as const,
            crossAxes: ["local +X", "local +Y"] as const,
          },
        }),
  };
}

function MarkerMaterial({ color }: { color: string }) {
  return (
    <meshStandardMaterial
      color={color}
      depthTest={false}
      depthWrite={false}
      emissive={color}
      emissiveIntensity={0.8}
    />
  );
}

export function EndEffectorMarker({
  arm,
  position,
  orientationXyzw,
}: {
  arm: "left" | "right";
  position: Vector3;
  orientationXyzw?: QuaternionXyzw;
}) {
  const description = describeEndEffectorMarker(arm, orientationXyzw);
  return (
    <group name={`${arm}-end-effector-marker`} position={position}>
      <mesh name={`${arm}-tool-center`} renderOrder={30}>
        <sphereGeometry args={[0.014, 18, 12]} />
        <MarkerMaterial color={description.color} />
      </mesh>
      {description.orientationGlyph ? (
        <group
          name={`${arm}-tool-orientation-glyph`}
          quaternion={description.orientationGlyph.quaternionXyzw}
        >
          <mesh
            name={`${arm}-tool-forward-local-z`}
            position={[0, 0, 0.026]}
            rotation={[Math.PI / 2, 0, 0]}
            renderOrder={31}
          >
            <cylinderGeometry args={[0.0026, 0.0026, 0.04, 10]} />
            <MarkerMaterial color={description.color} />
          </mesh>
          <mesh
            name={`${arm}-tool-forward-tip-local-z`}
            position={[0, 0, 0.05]}
            rotation={[Math.PI / 2, 0, 0]}
            renderOrder={31}
          >
            <coneGeometry args={[0.006, 0.012, 10]} />
            <MarkerMaterial color={description.color} />
          </mesh>
          <mesh name={`${arm}-tool-local-x`} renderOrder={31}>
            <boxGeometry args={[0.034, 0.004, 0.004]} />
            <MarkerMaterial color="#ffbf69" />
          </mesh>
          <mesh name={`${arm}-tool-local-y`} renderOrder={31}>
            <boxGeometry args={[0.004, 0.028, 0.004]} />
            <MarkerMaterial color="#8cff98" />
          </mesh>
        </group>
      ) : null}
    </group>
  );
}
