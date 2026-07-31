import type {
  QuaternionXyzw,
  Vector3,
} from "@/lib/atlas-schema/types";

export const END_EFFECTOR_MARKER_COLORS = {
  left: "#5ee4ff",
  right: "#ff6f91",
} as const;

export const MIN_SYMBOLIC_FINGER_SEPARATION = 0.012;
export const MAX_SYMBOLIC_FINGER_SEPARATION = 0.034;

export function rawGripperValueToSymbolicSeparation(value: number): number {
  const midpoint =
    (MIN_SYMBOLIC_FINGER_SEPARATION + MAX_SYMBOLIC_FINGER_SEPARATION) / 2;
  const halfRange =
    (MAX_SYMBOLIC_FINGER_SEPARATION - MIN_SYMBOLIC_FINGER_SEPARATION) / 2;
  return Math.min(
    MAX_SYMBOLIC_FINGER_SEPARATION,
    Math.max(
      MIN_SYMBOLIC_FINGER_SEPARATION,
      midpoint + halfRange * Math.tanh(value),
    ),
  );
}

export interface EndEffectorMarkerDescription {
  arm: "left" | "right";
  color: string;
  orientationGlyph?: {
    quaternionXyzw: QuaternionXyzw;
    forwardAxis: "local +Z";
    crossAxes: readonly ["local +X", "local +Y"];
    symbolicGripper?: {
      rawRecordedValue: number;
      separation: number;
      fingerPositionsX: readonly [number, number];
      fingerAxis: "local +Z";
    };
  };
}

export function describeEndEffectorMarker(
  arm: "left" | "right",
  orientationXyzw?: QuaternionXyzw,
  recordedGripperValue?: number,
): EndEffectorMarkerDescription {
  const separation =
    recordedGripperValue === undefined
      ? undefined
      : rawGripperValueToSymbolicSeparation(recordedGripperValue);
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
            ...(recordedGripperValue === undefined || separation === undefined
              ? {}
              : {
                  symbolicGripper: {
                    rawRecordedValue: recordedGripperValue,
                    separation,
                    fingerPositionsX: [-separation / 2, separation / 2] as const,
                    fingerAxis: "local +Z" as const,
                  },
                }),
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
  recordedGripperValue,
}: {
  arm: "left" | "right";
  position: Vector3;
  orientationXyzw?: QuaternionXyzw;
  recordedGripperValue?: number;
}) {
  const description = describeEndEffectorMarker(
    arm,
    orientationXyzw,
    recordedGripperValue,
  );
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
          {description.orientationGlyph.symbolicGripper ? (
            <group name={`${arm}-tool-symbolic-gripper`}>
              <mesh
                name={`${arm}-tool-symbolic-finger-negative-x`}
                position={[
                  description.orientationGlyph.symbolicGripper
                    .fingerPositionsX[0],
                  0,
                  0.026,
                ]}
                renderOrder={31}
              >
                <boxGeometry args={[0.004, 0.006, 0.04]} />
                <MarkerMaterial color={description.color} />
              </mesh>
              <mesh
                name={`${arm}-tool-symbolic-finger-positive-x`}
                position={[
                  description.orientationGlyph.symbolicGripper
                    .fingerPositionsX[1],
                  0,
                  0.026,
                ]}
                renderOrder={31}
              >
                <boxGeometry args={[0.004, 0.006, 0.04]} />
                <MarkerMaterial color={description.color} />
              </mesh>
            </group>
          ) : null}
        </group>
      ) : null}
    </group>
  );
}
