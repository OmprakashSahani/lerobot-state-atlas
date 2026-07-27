"use client";

import { useEffect, useMemo, useRef } from "react";
import {
  Color,
  DynamicDrawUsage,
  InstancedMesh,
  Matrix4,
  Quaternion,
  Vector3,
} from "three";

import type { PreparedVoxelArm } from "@/lib/atlas-schema/types";
import { spacingDeltaY } from "@/lib/coordinates/runtimeSpacing";
import { metricValue } from "@/lib/data/metrics";
import { visitColorScale } from "@/lib/data/prepareCoverage";
import { useViewerStore } from "./ViewerStore";

export function VoxelLayer({
  data,
  voxelSize,
  visible,
  range,
  spacing,
  manifestSpacing,
}: {
  data: PreparedVoxelArm;
  voxelSize: number;
  visible: boolean;
  range: [number, number];
  spacing: number;
  manifestSpacing: number;
}) {
  const viewer = useViewerStore();
  const mesh = useRef<InstancedMesh>(null);
  const matrix = useMemo(() => new Matrix4(), []);
  const quaternion = useMemo(() => new Quaternion(), []);
  const scale = useMemo(
    () => new Vector3(voxelSize * 0.86, voxelSize * 0.86, voxelSize * 0.86),
    [voxelSize],
  );

  useEffect(() => {
    if (!mesh.current) return;
    const position = new Vector3();
    const color = new Color();
    for (let index = 0; index < data.visits.length; index += 1) {
      position.fromArray(data.centers, index * 3);
      matrix.compose(position, quaternion, scale);
      mesh.current.setMatrixAt(index, matrix);
      const [red, green, blue] = visitColorScale(
        metricValue(data, index, viewer.metric),
        range[0],
        range[1],
      );
      color.setRGB(red, green, blue);
      mesh.current.setColorAt(index, color);
    }
    mesh.current.instanceMatrix.setUsage(DynamicDrawUsage);
    mesh.current.instanceMatrix.needsUpdate = true;
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true;
    mesh.current.computeBoundingSphere();
  }, [data, matrix, quaternion, range, scale, viewer.metric]);

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, data.visits.length]}
      name={`${data.arm}-workspace-voxels`}
      visible={visible}
      frustumCulled
      position={[0, spacingDeltaY(data.arm, spacing, manifestSpacing), 0]}
      onClick={(event) => {
        event.stopPropagation();
        if (event.instanceId === undefined) return;
        const offset = event.instanceId * 3;
        viewer.selectVoxel(data.arm, event.instanceId, [
          data.centers[offset],
          data.centers[offset + 1],
          data.centers[offset + 2],
        ]);
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        roughness={0.48}
        metalness={0.05}
        transparent
        opacity={0.86}
      />
    </instancedMesh>
  );
}
