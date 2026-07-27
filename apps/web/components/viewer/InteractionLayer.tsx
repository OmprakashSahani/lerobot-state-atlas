"use client";

import { useEffect, useMemo } from "react";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

import type {
  AtlasData,
  TrajectoryEpisode,
  Vector3,
} from "@/lib/atlas-schema/types";
import { applyRuntimeSpacing } from "@/lib/coordinates/runtimeSpacing";
import { playbackPositions } from "@/lib/playback/controller";
import { useViewerStore } from "./ViewerStore";

function WidePath({
  points,
  color,
  lineWidth,
  opacity,
  renderOrder,
}: {
  points: Vector3[];
  color: string;
  lineWidth: number;
  opacity: number;
  renderOrder: number;
}) {
  const geometry = useMemo(() => new LineGeometry(), []);
  const material = useMemo(
    () =>
      new LineMaterial({
        color,
        depthTest: false,
        depthWrite: false,
        linewidth: lineWidth,
        opacity,
        transparent: true,
        worldUnits: true,
      }),
    [color, lineWidth, opacity],
  );
  const line = useMemo(() => {
    const result = new Line2(geometry, material);
    result.frustumCulled = false;
    result.renderOrder = renderOrder;
    return result;
  }, [geometry, material, renderOrder]);
  useEffect(() => {
    const drawablePoints =
      points.length === 1 ? [points[0], points[0]] : points;
    geometry.setPositions(drawablePoints.flat());
    line.computeLineDistances();
  }, [geometry, line, points]);
  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );
  return <primitive object={line} />;
}

function ToolMarker({
  position,
  color,
}: {
  position: Vector3;
  color: string;
}) {
  return (
    <mesh position={position} renderOrder={30}>
      <sphereGeometry args={[0.014, 18, 12]} />
      <meshStandardMaterial
        color={color}
        depthTest={false}
        depthWrite={false}
        emissive={color}
        emissiveIntensity={0.8}
      />
    </mesh>
  );
}

export function InteractionLayer({
  data,
  episode,
  playbackFrame,
}: {
  data: AtlasData;
  episode: TrajectoryEpisode | null;
  playbackFrame: number;
}) {
  const viewer = useViewerStore();
  const baseline = data.manifest.coverage.armSpacing;
  const selectedCenter = viewer.selection
    ? applyRuntimeSpacing(
        viewer.selection.exportedCenter,
        viewer.selection.arm,
        viewer.spacing,
        baseline,
      )
    : null;
  const positions = episode ? playbackPositions(episode, playbackFrame) : null;
  const leftPath = useMemo(
    () =>
      episode?.leftPositionsXyz.map((point) =>
        applyRuntimeSpacing(point, "left", viewer.spacing, baseline),
      ) ?? [],
    [baseline, episode, viewer.spacing],
  );
  const rightPath = useMemo(
    () =>
      episode?.rightPositionsXyz.map((point) =>
        applyRuntimeSpacing(point, "right", viewer.spacing, baseline),
      ) ?? [],
    [baseline, episode, viewer.spacing],
  );
  const travelledLeftPath = useMemo(
    () => leftPath.slice(0, (positions?.index ?? 0) + 1),
    [leftPath, positions?.index],
  );
  const travelledRightPath = useMemo(
    () => rightPath.slice(0, (positions?.index ?? 0) + 1),
    [positions?.index, rightPath],
  );
  return (
    <group name="interaction-layer">
      {selectedCenter ? (
        <group position={selectedCenter} name="radius-query-marker">
          <mesh>
            <sphereGeometry args={[Math.max(viewer.radius, 0.002), 24, 16]} />
            <meshBasicMaterial
              color="#ffffff"
              transparent
              opacity={0.12}
              wireframe
              depthWrite={false}
            />
          </mesh>
          <mesh>
            <sphereGeometry args={[0.008, 16, 12]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
        </group>
      ) : null}
      {episode && positions ? (
        <group name="trajectory-playback">
          <WidePath
            points={leftPath}
            color="#5ee4ff"
            lineWidth={0.003}
            opacity={0.32}
            renderOrder={20}
          />
          <WidePath
            points={rightPath}
            color="#ff6f91"
            lineWidth={0.003}
            opacity={0.32}
            renderOrder={20}
          />
          <WidePath
            points={travelledLeftPath}
            color="#5ee4ff"
            lineWidth={0.006}
            opacity={0.96}
            renderOrder={21}
          />
          <WidePath
            points={travelledRightPath}
            color="#ff6f91"
            lineWidth={0.006}
            opacity={0.96}
            renderOrder={21}
          />
          <ToolMarker
            position={applyRuntimeSpacing(
              positions.left,
              "left",
              viewer.spacing,
              baseline,
            )}
            color="#5ee4ff"
          />
          <ToolMarker
            position={applyRuntimeSpacing(
              positions.right,
              "right",
              viewer.spacing,
              baseline,
            )}
            color="#ff6f91"
          />
        </group>
      ) : null}
    </group>
  );
}
