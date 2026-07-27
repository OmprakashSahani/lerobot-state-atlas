"use client";

/* Three.js cameras and controls are intentionally updated through their
 * imperative API inside effects. */
/* eslint-disable react-hooks/immutability */

import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { Box3, PerspectiveCamera, Sphere, Vector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { AtlasData } from "@/lib/atlas-schema/types";

import { EnvironmentLayer, gridEnvironment } from "./EnvironmentLayer";
import { InteractionLayer } from "./InteractionLayer";
import { RobotDataLayer } from "./RobotDataLayer";
import { useViewerStore } from "./ViewerStore";

function CameraController({ data }: { data: AtlasData }) {
  const { camera, gl } = useThree();
  const controls = useRef<OrbitControls | null>(null);
  const { cameraResetToken } = useViewerStore();
  const framing = useMemo(() => {
    const bounds = new Box3(
      new Vector3(...data.manifest.sceneBounds.minimumXyz),
      new Vector3(...data.manifest.sceneBounds.maximumXyz),
    );
    const sphere = bounds.getBoundingSphere(new Sphere());
    return { center: sphere.center, radius: Math.max(sphere.radius, 0.4) };
  }, [data.manifest.sceneBounds]);

  useEffect(() => {
    camera.up.set(0, 0, 1);
    const orbit = new OrbitControls(camera, gl.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.065;
    orbit.screenSpacePanning = true;
    orbit.minDistance = framing.radius * 0.45;
    orbit.maxDistance = framing.radius * 8;
    controls.current = orbit;
    let frame = 0;
    const update = () => {
      orbit.update();
      frame = requestAnimationFrame(update);
    };
    update();
    return () => {
      cancelAnimationFrame(frame);
      orbit.dispose();
      controls.current = null;
    };
  }, [camera, framing.radius, gl.domElement]);

  useEffect(() => {
    const perspective = camera as PerspectiveCamera;
    const distance =
      framing.radius /
      Math.sin((perspective.fov * Math.PI) / 360) *
      1.15;
    camera.position.set(
      framing.center.x + distance * 0.68,
      framing.center.y + distance * 0.5,
      framing.center.z + distance * 0.56,
    );
    camera.near = Math.max(0.001, distance / 1000);
    camera.far = distance * 20;
    camera.updateProjectionMatrix();
    controls.current?.target.copy(framing.center);
    controls.current?.update();
  }, [camera, cameraResetToken, framing]);
  return null;
}

export function ViewerCanvas({ data }: { data: AtlasData }) {
  return (
    <Canvas
      camera={{ fov: 42 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.setClearColor("#06101a");
      }}
    >
      <ambientLight intensity={0.8} />
      <directionalLight intensity={2.1} position={[2, 3, 2]} />
      <directionalLight
        color="#5ee4ff"
        intensity={0.7}
        position={[-2, 1, -2]}
      />
      <EnvironmentLayer source={gridEnvironment} />
      <RobotDataLayer data={data} />
      <InteractionLayer />
      <CameraController data={data} />
    </Canvas>
  );
}
