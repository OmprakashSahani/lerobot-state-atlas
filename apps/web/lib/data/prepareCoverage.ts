import type {
  AtlasManifest,
  CoveragePayload,
  PreparedVoxelArm,
  Vector3,
} from "@/lib/atlas-schema/types";

export function voxelCenter(
  index: Vector3,
  origin: Vector3,
  voxelSize: number,
): Vector3 {
  return [
    origin[0] + (index[0] + 0.5) * voxelSize,
    origin[1] + (index[1] + 0.5) * voxelSize,
    origin[2] + (index[2] + 0.5) * voxelSize,
  ];
}

export function numericExtent(values: ArrayLike<number>): [number, number] {
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    minimum = Math.min(minimum, values[index]);
    maximum = Math.max(maximum, values[index]);
  }
  return [minimum, maximum];
}

export function prepareCoverage(
  manifest: AtlasManifest,
  coverage: CoveragePayload,
): PreparedVoxelArm[] {
  return coverage.arms.map((arm) => {
    const [minimumVisitCount, maximumVisitCount] = numericExtent(arm.visitCounts);
    const centers = new Float32Array(arm.voxelIndices.length * 3);
    const visits = Uint32Array.from(arm.visitCounts);
    const episodeCounts = Uint32Array.from(
      arm.voxelIndices.map(
        (_, index) => arm.episodeIdOffsets[index + 1] - arm.episodeIdOffsets[index],
      ),
    );
    const instanceLookup = arm.voxelIndices.map((index, voxelEntryIndex) => {
      const center = voxelCenter(
        index,
        manifest.coverage.voxelOrigin,
        manifest.coverage.voxelSize,
      );
      centers.set(center, voxelEntryIndex * 3);
      return { arm: arm.arm, voxelEntryIndex, voxelIndex: index };
    });
    return {
      arm: arm.arm,
      centers,
      visits,
      episodeCounts,
      instanceLookup,
      minimumVisitCount,
      maximumVisitCount,
    };
  });
}

export function visitColorScale(value: number, minimum: number, maximum: number) {
  const amount = maximum === minimum ? 1 : (value - minimum) / (maximum - minimum);
  const stops = [
    [29, 64, 108],
    [28, 147, 140],
    [122, 209, 81],
    [247, 209, 61],
  ];
  const scaled = Math.max(0, Math.min(1, amount)) * (stops.length - 1);
  const low = Math.floor(scaled);
  const high = Math.min(stops.length - 1, low + 1);
  const mix = scaled - low;
  return stops[low].map(
    (channel, index) =>
      (channel + (stops[high][index] - channel) * mix) / 255,
  ) as [number, number, number];
}
