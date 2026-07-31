"use client";

import { useEffect, useMemo } from "react";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

import type {
  AtlasData,
  TrajectoryEpisode,
  TrajectoryEpisodeOrientations,
  TrajectoryEpisodeRecordedGripperValues,
  Vector3,
} from "@/lib/atlas-schema/types";
import { applyRuntimeSpacing } from "@/lib/coordinates/runtimeSpacing";
import { selectRecordedPlaybackSample } from "@/lib/playback/controller";
import { EndEffectorMarker } from "./EndEffectorMarker";
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

export function InteractionLayer({
  data,
  episode,
  orientationEpisode,
  recordedGripperEpisode,
  playbackFrame,
}: {
  data: AtlasData;
  episode: TrajectoryEpisode | null;
  orientationEpisode: TrajectoryEpisodeOrientations | null;
  recordedGripperEpisode: TrajectoryEpisodeRecordedGripperValues | null;
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
  const sample = episode
    ? selectRecordedPlaybackSample(
        episode,
        playbackFrame,
        orientationEpisode ?? undefined,
        recordedGripperEpisode ?? undefined,
      )
    : null;
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
    () => leftPath.slice(0, (sample?.index ?? 0) + 1),
    [leftPath, sample?.index],
  );
  const travelledRightPath = useMemo(
    () => rightPath.slice(0, (sample?.index ?? 0) + 1),
    [rightPath, sample?.index],
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
      {episode && sample ? (
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
          <EndEffectorMarker
            arm="left"
            position={applyRuntimeSpacing(
              sample.left.position,
              "left",
              viewer.spacing,
              baseline,
            )}
            orientationXyzw={sample.left.orientationXyzw}
          />
          <EndEffectorMarker
            arm="right"
            position={applyRuntimeSpacing(
              sample.right.position,
              "right",
              viewer.spacing,
              baseline,
            )}
            orientationXyzw={sample.right.orientationXyzw}
          />
        </group>
      ) : null}
    </group>
  );
}
