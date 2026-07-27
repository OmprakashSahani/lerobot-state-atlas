import { describe, expect, it } from "vitest";

import type { CoveragePayload, PreparedVoxelArm } from "@/lib/atlas-schema/types";
import { queryRadius } from "@/lib/data/radiusQuery";

const prepared: PreparedVoxelArm[] = [
  {
    arm: "left",
    centers: Float32Array.from([0, 0.4, 0, 0.1, 0.4, 0]),
    visits: Uint32Array.from([3, 5]),
    episodeCounts: Uint32Array.from([2, 1]),
    instanceLookup: [],
    minimumVisitCount: 3,
    maximumVisitCount: 5,
  },
  {
    arm: "right",
    centers: Float32Array.from([0, -0.4, 0, 0, 0.4, 0]),
    visits: Uint32Array.from([7, 11]),
    episodeCounts: Uint32Array.from([1, 2]),
    instanceLookup: [],
    minimumVisitCount: 7,
    maximumVisitCount: 11,
  },
];
const coverage = {
  schema: { name: "lerobot-state-atlas.browser-data", major: 1, minor: 0 },
  arms: [
    { arm: "left", episodeIdOffsets: [0, 2, 3], episodeIds: [1, 2, 2] },
    { arm: "right", episodeIdOffsets: [0, 1, 3], episodeIds: [3, 2, 4] },
  ],
} as CoveragePayload;

describe("shared-world radius queries", () => {
  it("includes exact-center cross-arm entries at zero radius", () => {
    const result = queryRadius(
      prepared,
      coverage,
      {
        arm: "left",
        voxelEntryIndex: 0,
        exportedCenter: [0, prepared[0].centers[1], 0],
      },
      0,
      0.8,
      0.8,
    );
    expect(result.entryCount).toBe(2);
    expect(result.matches.map((match) => match.arm)).toEqual(["left", "right"]);
    expect(result.leftVisits).toBe(3);
    expect(result.rightVisits).toBe(11);
    expect(result.toolPointVisits).toBe(14);
    expect(result.distinctEpisodeCount).toBe(3);
  });

  it("uses Euclidean distance and adjusted query centers", () => {
    const result = queryRadius(
      prepared,
      coverage,
      {
        arm: "left",
        voxelEntryIndex: 0,
        exportedCenter: [0, prepared[0].centers[1], 0],
      },
      0.1,
      1,
      0.8,
    );
    expect(result.center[1]).toBeCloseTo(0.5);
    expect(result.matches).toContainEqual({ arm: "left", voxelEntryIndex: 1 });
  });
});
