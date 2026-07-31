import { describe, expect, it } from "vitest";

import type { CoveragePayload, PreparedVoxelArm } from "@/lib/atlas-schema/types";
import { queryRadius } from "@/lib/data/radiusQuery";
import { scoreUncommonEpisodes } from "@/lib/data/uncommonEpisodes";

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
const coverage: CoveragePayload = {
  schema: { name: "lerobot-state-atlas.browser-data", major: 1, minor: 0 },
  arms: [
    {
      arm: "left",
      toolLink: "tool0",
      voxelIndices: [
        [0, 0, 0],
        [1, 0, 0],
      ],
      visitCounts: [3, 5],
      episodeCounts: [2, 1],
      episodeIdOffsets: [0, 2, 3],
      episodeIds: [1, 2, 2],
      statistics: {
        voxelEntryCount: 2,
        minimumVisitCount: 3,
        maximumVisitCount: 5,
        minimumEpisodeCount: 1,
        maximumEpisodeCount: 2,
      },
    },
    {
      arm: "right",
      toolLink: "tool0",
      voxelIndices: [
        [0, 0, 0],
        [0, 1, 0],
      ],
      visitCounts: [7, 11],
      episodeCounts: [1, 2],
      episodeIdOffsets: [0, 1, 3],
      episodeIds: [3, 2, 4],
      statistics: {
        voxelEntryCount: 2,
        minimumVisitCount: 7,
        maximumVisitCount: 11,
        minimumEpisodeCount: 1,
        maximumEpisodeCount: 2,
      },
    },
  ],
};

const selection = {
  arm: "left" as const,
  voxelEntryIndex: 0,
  exportedCenter: [0, prepared[0].centers[1], 0] as [number, number, number],
};

function uncommonScores(
  source: CoveragePayload = coverage,
  scope?: ReturnType<typeof queryRadius>["matches"],
) {
  return scoreUncommonEpisodes({
    coverage: source,
    episodeCount: 5,
    allowedEpisodeIds: [0, 1, 2, 3, 4],
    scope,
  });
}

describe("shared-world radius queries", () => {
  it("includes exact-center cross-arm entries at zero radius", () => {
    const result = queryRadius(
      prepared,
      coverage,
      selection,
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
      selection,
      0.1,
      1,
      0.8,
    );
    expect(result.center[1]).toBeCloseTo(0.5);
    expect(result.matches).toContainEqual({ arm: "left", voxelEntryIndex: 1 });
  });

  it("passes exact radius matches directly to local uncommon scoring", () => {
    const result = queryRadius(prepared, coverage, selection, 0, 0.8, 0.8);
    const scores = uncommonScores(coverage, result.matches);

    expect(result.matches).toEqual([
      { arm: "left", voxelEntryIndex: 0 },
      { arm: "right", voxelEntryIndex: 1 },
    ]);
    expect(scores.map((item) => item.episodeId)).toEqual([2, 1, 4]);
    expect(scores[0].touchedEntryCount).toBe(2);
    expect(scores[0].scopeEntryShare).toBe(1);
    expect(scores[1].touchedEntryCount).toBe(1);
    expect(scores.every((item) => item.episodeId !== 3)).toBe(true);
  });

  it("keeps global scores invariant under runtime spacing changes", () => {
    const before = uncommonScores();
    queryRadius(prepared, coverage, selection, 0.3, 0.8, 0.8);
    const afterBaselineQuery = uncommonScores();
    queryRadius(prepared, coverage, selection, 0.3, 1.2, 0.8);
    const afterAdjustedQuery = uncommonScores();

    expect(afterBaselineQuery).toEqual(before);
    expect(afterAdjustedQuery).toEqual(before);
  });

  it("keeps local scores equal when spacing produces the same entry identities", () => {
    const first = queryRadius(prepared, coverage, selection, 0, 0.9, 0.8);
    const second = queryRadius(prepared, coverage, selection, 0, 1, 0.8);

    expect(first.matches).toEqual([{ arm: "left", voxelEntryIndex: 0 }]);
    expect(second.matches).toEqual(first.matches);
    expect(uncommonScores(coverage, second.matches)).toEqual(
      uncommonScores(coverage, first.matches),
    );
  });

  it("allows local membership to change only through spacing geometry", () => {
    const baseline = queryRadius(prepared, coverage, selection, 0, 0.8, 0.8);
    const adjusted = queryRadius(prepared, coverage, selection, 0, 1, 0.8);

    expect(baseline.matches).toEqual([
      { arm: "left", voxelEntryIndex: 0 },
      { arm: "right", voxelEntryIndex: 1 },
    ]);
    expect(adjusted.matches).toEqual([
      { arm: "left", voxelEntryIndex: 0 },
    ]);
    expect(uncommonScores(coverage, baseline.matches).map(({ episodeId }) => episodeId)).toEqual([
      2, 1, 4,
    ]);
    expect(uncommonScores(coverage, adjusted.matches).map(({ episodeId }) => episodeId)).toEqual([
      1, 2,
    ]);
  });

  it("does not let raw visits affect local uncommonness", () => {
    const result = queryRadius(prepared, coverage, selection, 0, 0.8, 0.8);
    const changedVisits = structuredClone(coverage);
    changedVisits.arms[0].visitCounts = [3_000_000, 1];
    changedVisits.arms[1].visitCounts = [1, 9_000_000];

    expect(uncommonScores(changedVisits, result.matches)).toEqual(
      uncommonScores(coverage, result.matches),
    );
  });

  it("does not mutate query or scoring inputs", () => {
    const preparedValues = () =>
      prepared.map((item) => ({
        ...item,
        centers: Array.from(item.centers),
        visits: Array.from(item.visits),
        episodeCounts: Array.from(item.episodeCounts),
      }));
    const preparedBefore = preparedValues();
    const coverageBefore = structuredClone(coverage);
    const selectionBefore = structuredClone(selection);
    const result = queryRadius(prepared, coverage, selection, 0, 0.8, 0.8);
    const matchesBefore = structuredClone(result.matches);

    uncommonScores(coverage, result.matches);

    expect(preparedValues()).toEqual(preparedBefore);
    expect(coverage).toEqual(coverageBefore);
    expect(selection).toEqual(selectionBefore);
    expect(result.matches).toEqual(matchesBefore);
  });
});
