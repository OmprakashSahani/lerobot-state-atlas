"use client";

import type { AtlasData } from "@/lib/atlas-schema/types";

import { BaseReferenceLayer } from "./BaseReferenceLayer";
import { useViewerStore } from "./ViewerStore";
import { VoxelLayer } from "./VoxelLayer";

export function RobotDataLayer({ data }: { data: AtlasData }) {
  const viewer = useViewerStore();
  const range: [number, number] = [
    Math.min(...data.preparedArms.map((arm) => arm.minimumVisitCount)),
    Math.max(...data.preparedArms.map((arm) => arm.maximumVisitCount)),
  ];
  return (
    <group name="robot-data-layer">
      <BaseReferenceLayer manifest={data.manifest} />
      <VoxelLayer
        data={data.preparedArms[0]}
        range={range}
        visible={viewer.leftVisible}
        voxelSize={data.manifest.coverage.voxelSize}
      />
      <VoxelLayer
        data={data.preparedArms[1]}
        range={range}
        visible={viewer.rightVisible}
        voxelSize={data.manifest.coverage.voxelSize}
      />
    </group>
  );
}
