"use client";

/* Three.js cameras and controls are intentionally updated through their
 * imperative API inside effects. */
/* eslint-disable react-hooks/immutability */

import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { Box3, MathUtils, PerspectiveCamera, Sphere, Vector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type {
  AtlasData,
  TrajectoryEpisode,
  TrajectoryEpisodeOrientations,
  TrajectoryEpisodeRecordedGripperValues,
} from "@/lib/atlas-schema/types";
import type { ValidatedEnvironmentRenderRequest } from "@/lib/environment/types";
import type { EnvironmentAlignment } from "@/lib/environment/types";
import {
  assessCameraSupport,
  reconstructionCameraStateFromScene,
  sceneFromReconstructionMatrix,
  type SupportedCameraManifold,
} from "@/lib/environment/camera-viewing-constraint";
import {
  applyRegisteredCamera,
  type ConvertedRegisteredCamera,
} from "@/lib/environment/camera-manifold";
import type {
  IntegratedEnvironmentPerformanceMetrics,
  IntegratedEnvironmentSource,
  SparkResourceLifecycleEvent,
} from "@/lib/environment/integrated-performance-diagnostic";

import { EnvironmentLayer, gridEnvironment } from "./EnvironmentLayer";
import { SparkEnvironmentAdapter, type SparkAdapterPhase } from "./SparkEnvironmentAdapter";
import { InteractionLayer } from "./InteractionLayer";
import { RobotDataLayer } from "./RobotDataLayer";
import { useViewerStore } from "./ViewerStore";

export interface GaussianViewingConstraint {
  active: boolean;
  camera: ConvertedRegisteredCamera;
  manifold: SupportedCameraManifold | null;
  alignment: EnvironmentAlignment;
}

export interface GaussianViewingTelemetry {
  cameraPosition: [number, number, number];
  nearestColmapImageId: number;
  nearestSourceImage: string;
  distance: number;
  orientationDifferenceDegrees: number;
  inside: boolean;
}

export function CameraController({
  data,
  gaussianViewingConstraint,
  onGaussianViewingTelemetry,
}: {
  data: AtlasData;
  gaussianViewingConstraint?: GaussianViewingConstraint;
  onGaussianViewingTelemetry?: (telemetry: GaussianViewingTelemetry) => void;
}) {
  const { camera, gl, size } = useThree();
  const controls = useRef<OrbitControls | null>(null);
  const { cameraResetToken, autoRotate } = useViewerStore();
  const framing = useMemo(() => {
    const bounds = new Box3(
      new Vector3(...data.manifest.sceneBounds.minimumXyz),
      new Vector3(...data.manifest.sceneBounds.maximumXyz),
    );
    const sphere = bounds.getBoundingSphere(new Sphere());
    return { center: sphere.center, radius: Math.max(sphere.radius, 0.4) };
  }, [data.manifest.sceneBounds]);

  useEffect(() => {
    if (gaussianViewingConstraint?.active) {
      controls.current = null;
      return;
    }
    camera.up.set(0, 0, 1);
    const orbit = new OrbitControls(camera, gl.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.065;
    orbit.screenSpacePanning = true;
    orbit.autoRotateSpeed = 1.2;
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
  }, [
    camera,
    framing.radius,
    gaussianViewingConstraint?.active,
    gl.domElement,
  ]);

  useEffect(() => {
    if (controls.current) controls.current.autoRotate = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    if (gaussianViewingConstraint?.active) return;
    const perspective = camera as PerspectiveCamera;
    perspective.clearViewOffset();
    perspective.fov = 42;
    perspective.aspect = size.width / Math.max(1, size.height);
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
  }, [
    camera,
    cameraResetToken,
    framing,
    gaussianViewingConstraint?.active,
    size.height,
    size.width,
  ]);

  useEffect(() => {
    if (
      !gaussianViewingConstraint?.active ||
      !(camera instanceof PerspectiveCamera)
    ) {
      return;
    }
    const { alignment, camera: capturedCamera, manifold } =
      gaussianViewingConstraint;
    applyRegisteredCamera(
      camera,
      capturedCamera,
      sceneFromReconstructionMatrix(alignment),
    );
    controls.current?.target
      .copy(camera.position)
      .add(new Vector3(0, 0, -1).applyQuaternion(camera.quaternion));
    const reconstructionState = reconstructionCameraStateFromScene(
      camera.position,
      new Vector3(0, 0, -1).applyQuaternion(camera.quaternion),
      alignment,
    );
    const assessment = manifold
      ? assessCameraSupport(
          reconstructionState.position,
          reconstructionState.viewingDirection,
          manifold,
        )
      : {
          inside: true,
          nearestCamera: capturedCamera,
          distance: reconstructionState.position.distanceTo(
            capturedCamera.center,
          ),
          orientationDifferenceRadians:
            reconstructionState.viewingDirection.angleTo(
              capturedCamera.viewingDirection,
            ),
        };
    onGaussianViewingTelemetry?.({
      cameraPosition: reconstructionState.position.toArray(),
      nearestColmapImageId: assessment.nearestCamera.colmapImageId,
      nearestSourceImage: assessment.nearestCamera.sourceImage,
      distance: assessment.distance,
      orientationDifferenceDegrees: MathUtils.radToDeg(
        assessment.orientationDifferenceRadians,
      ),
      inside: assessment.inside,
    });
  }, [
    camera,
    cameraResetToken,
    gaussianViewingConstraint,
    onGaussianViewingTelemetry,
  ]);
  return null;
}

export function ViewerCanvas({
  data,
  episode,
  orientationEpisode,
  recordedGripperEpisode,
  playbackFrame,
  environmentRequest,
  onEnvironmentPhase,
  onEnvironmentError,
  onWebGl2Support,
  integratedEnvironmentSource,
  environmentReadyRequestedAt,
  onEnvironmentPerformanceMetrics,
  environmentSourceSwitchToken,
  onEnvironmentResourceLifecycleEvent,
  gaussianViewingConstraint,
  onGaussianViewingTelemetry,
}: {
  data: AtlasData;
  episode: TrajectoryEpisode | null;
  orientationEpisode: TrajectoryEpisodeOrientations | null;
  recordedGripperEpisode: TrajectoryEpisodeRecordedGripperValues | null;
  playbackFrame: number;
  environmentRequest: ValidatedEnvironmentRenderRequest | null;
  onEnvironmentPhase: (generation: number, phase: SparkAdapterPhase) => void;
  onEnvironmentError: (generation: number, message: string) => void;
  onWebGl2Support: (supported: boolean) => void;
  integratedEnvironmentSource?: IntegratedEnvironmentSource;
  environmentReadyRequestedAt?: number | null;
  onEnvironmentPerformanceMetrics?: (
    metrics: IntegratedEnvironmentPerformanceMetrics | null,
  ) => void;
  environmentSourceSwitchToken?: number;
  onEnvironmentResourceLifecycleEvent?: (
    event: SparkResourceLifecycleEvent,
  ) => void;
  gaussianViewingConstraint?: GaussianViewingConstraint;
  onGaussianViewingTelemetry?: (telemetry: GaussianViewingTelemetry) => void;
}) {
  return (
    <Canvas
      camera={{ fov: 42 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.setClearColor("#06101a");
        onWebGl2Support(gl.capabilities.isWebGL2);
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
      <SparkEnvironmentAdapter
        request={environmentRequest}
        onPhase={onEnvironmentPhase}
        onError={onEnvironmentError}
        integratedSource={integratedEnvironmentSource}
        readyRequestedAt={environmentReadyRequestedAt}
        onPerformanceMetrics={onEnvironmentPerformanceMetrics}
        sourceSwitchToken={environmentSourceSwitchToken}
        onResourceLifecycleEvent={onEnvironmentResourceLifecycleEvent}
      />
      <RobotDataLayer data={data} />
      <InteractionLayer
        data={data}
        episode={episode}
        orientationEpisode={orientationEpisode}
        recordedGripperEpisode={recordedGripperEpisode}
        playbackFrame={playbackFrame}
      />
      <CameraController
        data={data}
        gaussianViewingConstraint={gaussianViewingConstraint}
        onGaussianViewingTelemetry={onGaussianViewingTelemetry}
      />
    </Canvas>
  );
}
