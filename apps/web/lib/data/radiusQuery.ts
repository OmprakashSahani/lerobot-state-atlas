import type {
  CoveragePayload,
  PreparedVoxelArm,
  Vector3,
} from "@/lib/atlas-schema/types";
import { applyRuntimeSpacing } from "@/lib/coordinates/runtimeSpacing";

export interface VoxelSelection {
  arm: "left" | "right";
  voxelEntryIndex: number;
  exportedCenter: Vector3;
}

export interface RadiusQueryResult {
  radius: number;
  selectedArm: "left" | "right";
  center: Vector3;
  entryCount: number;
  toolPointVisits: number;
  leftVisits: number;
  rightVisits: number;
  distinctEpisodeCount: number;
  matches: { arm: "left" | "right"; voxelEntryIndex: number }[];
}

export function queryRadius(
  preparedArms: PreparedVoxelArm[],
  coverage: CoveragePayload,
  selection: VoxelSelection,
  radius: number,
  spacing: number,
  manifestSpacing: number,
): RadiusQueryResult {
  const center = applyRuntimeSpacing(
    selection.exportedCenter,
    selection.arm,
    spacing,
    manifestSpacing,
  );
  const matches: RadiusQueryResult["matches"] = [];
  const episodes = new Set<number>();
  let leftVisits = 0;
  let rightVisits = 0;
  const epsilon = radius === 0 ? 0 : 1e-7;
  preparedArms.forEach((prepared, armIndex) => {
    const source = coverage.arms[armIndex];
    for (let index = 0; index < prepared.visits.length; index += 1) {
      const offset = index * 3;
      const candidate = applyRuntimeSpacing(
        [
          prepared.centers[offset],
          prepared.centers[offset + 1],
          prepared.centers[offset + 2],
        ],
        prepared.arm,
        spacing,
        manifestSpacing,
      );
      const distance = Math.hypot(
        candidate[0] - center[0],
        candidate[1] - center[1],
        candidate[2] - center[2],
      );
      if (distance <= radius + epsilon) {
        matches.push({ arm: prepared.arm, voxelEntryIndex: index });
        if (prepared.arm === "left") leftVisits += prepared.visits[index];
        else rightVisits += prepared.visits[index];
        for (
          let cursor = source.episodeIdOffsets[index];
          cursor < source.episodeIdOffsets[index + 1];
          cursor += 1
        ) {
          episodes.add(source.episodeIds[cursor]);
        }
      }
    }
  });
  return {
    radius,
    selectedArm: selection.arm,
    center,
    entryCount: matches.length,
    toolPointVisits: leftVisits + rightVisits,
    leftVisits,
    rightVisits,
    distinctEpisodeCount: episodes.size,
    matches,
  };
}
