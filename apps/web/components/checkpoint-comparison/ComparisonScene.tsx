"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  BufferGeometry,
  Line,
  LineBasicMaterial,
  LineDashedMaterial,
  PerspectiveCamera,
  Vector3,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { TrajectorySceneBounds } from "@/lib/checkpoint-comparison/sceneBounds";
import type { AvailableProjection, ProjectedArm } from "@/lib/checkpoint-comparison/types";

/* eslint-disable react-hooks/immutability -- Three.js cameras and OrbitControls are imperative scene objects. */
function CameraControls({
  bounds,
  fitRequest,
}: {
  bounds: TrajectorySceneBounds;
  fitRequest: number;
}) {
  const { camera, gl, size } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  useEffect(() => {
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, gl.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;
    const update = () => {
      controls.update();
      frame = requestAnimationFrame(update);
    };
    let frame = requestAnimationFrame(update);
    return () => {
      cancelAnimationFrame(frame);
      controlsRef.current = null;
      controls.dispose();
    };
  }, [camera, gl.domElement]);

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls || !(camera instanceof PerspectiveCamera)) return;
    const center = new Vector3(...bounds.center);
    const direction = camera.position.clone().sub(controls.target);
    if (!direction.lengthSq()) direction.set(1.35, 1.15, 0.9);
    direction.normalize();
    const radius = Math.max(bounds.extent * 0.64, 0.2);
    const verticalFov = (camera.fov * Math.PI) / 180;
    const aspect = Math.max(size.width / size.height, 0.1);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * aspect);
    const distance =
      Math.max(
        radius / Math.tan(verticalFov / 2),
        radius / Math.tan(horizontalFov / 2),
      ) * 1.18;
    camera.position.copy(center).addScaledVector(direction, distance);
    camera.near = Math.max(distance / 100, 0.001);
    camera.far = Math.max(distance * 25, 10);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
  }, [bounds, camera, fitRequest, size.height, size.width]);
  return null;
}
/* eslint-enable react-hooks/immutability */

function Path({
  arm,
  color,
  visible,
  dashed,
  bounds,
}: {
  arm: ProjectedArm;
  color: string;
  visible: boolean;
  dashed: boolean;
  bounds: TrajectorySceneBounds;
}) {
  const line = useMemo(() => {
    const geometry = new BufferGeometry().setFromPoints(
      arm.positionsXyz.map((point) => new Vector3(...point)),
    );
    const material = dashed
      ? new LineDashedMaterial({
          color,
          dashSize: bounds.dashSize,
          gapSize: bounds.gapSize,
          transparent: true,
          opacity: 0.9,
        })
      : new LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
    const result = new Line(geometry, material);
    if (dashed) result.computeLineDistances();
    return result;
  }, [arm.positionsXyz, bounds.dashSize, bounds.gapSize, color, dashed]);
  useEffect(
    () => () => {
      line.geometry.dispose();
      (line.material as LineBasicMaterial).dispose();
    },
    [line],
  );
  return visible ? <primitive object={line} /> : null;
}

function Marker({
  arm,
  step,
  color,
  visible,
  showOrientation,
  showGripper,
  bounds,
}: {
  arm: ProjectedArm;
  step: number;
  color: string;
  visible: boolean;
  showOrientation: boolean;
  showGripper: boolean;
  bounds: TrajectorySceneBounds;
}) {
  const rawTarget = Math.abs(arm.generatedRawGripperTargets[step]);
  const symbolicScale = bounds.gripperSize * Math.min(1.35, 1 + rawTarget * 0.05);
  return visible ? (
    <group position={arm.positionsXyz[step]} quaternion={arm.orientationsXyzw[step]}>
      <mesh>
        <sphereGeometry args={[bounds.markerRadius, 20, 20]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.35}
        />
      </mesh>
      {showOrientation && <axesHelper args={[bounds.orientationSize]} />}
      {showGripper && (
        <mesh scale={[symbolicScale, symbolicScale, symbolicScale]}>
          <torusGeometry args={[1, 0.22, 12, 24]} />
          <meshStandardMaterial color={color} wireframe />
        </mesh>
      )}
    </group>
  ) : null;
}

export function ComparisonScene({
  projection,
  step,
  visibility,
  showPaths,
  showMarkers,
  showOrientations,
  showGrippers,
  bounds,
  fitRequest,
}: {
  projection: AvailableProjection;
  step: number;
  visibility: readonly [boolean, boolean];
  showPaths: boolean;
  showMarkers: boolean;
  showOrientations: boolean;
  showGrippers: boolean;
  bounds: TrajectorySceneBounds;
  fitRequest: number;
}) {
  const colors = ["#67e8f9", "#fb7185"] as const;
  const gridPosition: [number, number, number] = [
    bounds.center[0],
    bounds.center[1],
    bounds.minimum[2] - bounds.extent * 0.08,
  ];
  return (
    <Canvas
      camera={{ fov: 44, position: [1.35, 1.15, 0.9] }}
      aria-label="Shared-world derived trajectory comparison"
    >
      <color attach="background" args={["#06101a"]} />
      <ambientLight intensity={0.9} />
      <directionalLight position={[2, 3, 2]} intensity={2} />
      <gridHelper
        args={[bounds.gridSize, 18, "#27465a", "#142c3d"]}
        position={gridPosition}
        rotation={[Math.PI / 2, 0, 0]}
      />
      {projection.plans.map((plan, index) => (
        <group key={plan.policyId}>
          {showPaths && (
            <Path
              arm={plan.left}
              color={colors[index]}
              visible={visibility[index]}
              dashed={index === 1}
              bounds={bounds}
            />
          )}
          {showPaths && (
            <Path
              arm={plan.right}
              color={colors[index]}
              visible={visibility[index]}
              dashed={index === 1}
              bounds={bounds}
            />
          )}
          {showMarkers && (
            <Marker
              arm={plan.left}
              step={step}
              color={colors[index]}
              visible={visibility[index]}
              showOrientation={showOrientations}
              showGripper={showGrippers}
              bounds={bounds}
            />
          )}
          {showMarkers && (
            <Marker
              arm={plan.right}
              step={step}
              color={colors[index]}
              visible={visibility[index]}
              showOrientation={showOrientations}
              showGripper={showGrippers}
              bounds={bounds}
            />
          )}
        </group>
      ))}
      <CameraControls bounds={bounds} fitRequest={fitRequest} />
    </Canvas>
  );
}
