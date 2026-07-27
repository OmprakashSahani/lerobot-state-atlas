"use client";

import { useMemo } from "react";

import type { AtlasData } from "@/lib/atlas-schema/types";
import { metricDomain } from "@/lib/data/metrics";

import { BaseReferenceLayer } from "./BaseReferenceLayer";
import { useViewerStore } from "./ViewerStore";
import { VoxelLayer } from "./VoxelLayer";

export function RobotDataLayer({ data }: { data: AtlasData }) {
  const viewer = useViewerStore();
  const range = useMemo(
    () => metricDomain(data.preparedArms, viewer.metric),
    [data.preparedArms, viewer.metric],
  );
  return (
    <group name="robot-data-layer">
      <BaseReferenceLayer
        manifest={data.manifest}
        spacing={viewer.spacing}
      />
      <VoxelLayer
        data={data.preparedArms[0]}
        range={range}
        visible={viewer.leftVisible}
        voxelSize={data.manifest.coverage.voxelSize}
        spacing={viewer.spacing}
        manifestSpacing={data.manifest.coverage.armSpacing}
      />
      <VoxelLayer
        data={data.preparedArms[1]}
        range={range}
        visible={viewer.rightVisible}
        voxelSize={data.manifest.coverage.voxelSize}
        spacing={viewer.spacing}
        manifestSpacing={data.manifest.coverage.armSpacing}
      />
    </group>
  );
}
