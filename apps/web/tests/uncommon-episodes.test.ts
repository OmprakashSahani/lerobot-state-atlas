import { describe, expect, it } from "vitest";

import type { CoverageArm, CoveragePayload } from "@/lib/atlas-schema/types";
import {
  scoreUncommonEpisodes,
  type CoverageEntryReference,
} from "@/lib/data/uncommonEpisodes";

function arm(
  name: "left" | "right",
  episodeIdsByEntry: number[][],
  visitCounts = episodeIdsByEntry.map(() => 1),
  coordinates = episodeIdsByEntry.map((_, index) => [index, 0, 0]),
): CoverageArm {
  const episodeIds = episodeIdsByEntry.flat();
  const episodeIdOffsets = [0];
  for (const ids of episodeIdsByEntry) {
    episodeIdOffsets.push(episodeIdOffsets.at(-1)! + ids.length);
  }
  return {
    arm: name,
    toolLink: "tool0",
    voxelIndices: coordinates as [number, number, number][],
    visitCounts,
    episodeCounts: episodeIdsByEntry.map((ids) => ids.length),
    episodeIdOffsets,
    episodeIds,
    statistics: {
      voxelEntryCount: episodeIdsByEntry.length,
      minimumVisitCount: Math.min(...visitCounts),
      maximumVisitCount: Math.max(...visitCounts),
      minimumEpisodeCount: Math.min(...episodeIdsByEntry.map((ids) => ids.length)),
      maximumEpisodeCount: Math.max(...episodeIdsByEntry.map((ids) => ids.length)),
    },
  };
}

function coverage(
  left: number[][],
  right: number[][] = [],
  leftVisits?: number[],
  rightVisits?: number[],
  leftCoordinates?: number[][],
  rightCoordinates?: number[][],
): CoveragePayload {
  return {
    schema: { name: "lerobot-state-atlas.browser-data", major: 1, minor: 2 },
    arms: [
      arm("left", left, leftVisits, leftCoordinates),
      arm("right", right, rightVisits, rightCoordinates),
    ],
  };
}

function score(
  source: CoveragePayload,
  episodeCount: number,
  allowedEpisodeIds = Array.from({ length: episodeCount }, (_, index) => index),
  scope?: readonly CoverageEntryReference[],
) {
  return scoreUncommonEpisodes({
    coverage: source,
    episodeCount,
    allowedEpisodeIds,
    scope,
  });
}

describe("uncommon-space episode scoring", () => {
  it("returns finite zero scores when coverage contains one episode", () => {
    const result = score(coverage([[0]], [[0]]), 1);
    expect(result).toEqual([
      {
        episodeId: 0,
        score: 0,
        touchedEntryCount: 2,
        scopeEntryShare: 1,
        minimumDistinctEpisodeCount: 1,
        maximumDistinctEpisodeCount: 1,
      },
    ]);
    expect(Number.isFinite(result[0].score)).toBe(true);
  });

  it("maps one episode to rarity one and all episodes to rarity zero", () => {
    const result = score(coverage([[0], [0, 1, 2, 3]]), 4);
    expect(result.find((item) => item.episodeId === 1)?.score).toBe(0);
    expect(result.find((item) => item.episodeId === 0)?.score).toBe(0.5);
  });

  it("uses the exact intermediate logarithmic rarity", () => {
    const result = score(coverage([[0, 1]]), 8);
    expect(result[0].score).toBeCloseTo(Math.log(8 / 2) / Math.log(8));
  });

  it("does not use raw visits or repeated visits in scoring", () => {
    const highVisits = score(coverage([[0]], [], [1_000_000]), 4);
    const repeatedVisits = score(coverage([[0]], [], [7]), 4);
    const lowVisitsCommon = score(coverage([[0, 1, 2, 3]], [], [1]), 4);
    expect(highVisits[0].score).toBe(1);
    expect(repeatedVisits).toEqual(highVisits);
    expect(lowVisitsCommon.every((item) => item.score === 0)).toBe(true);
  });

  it("counts the same coordinate on opposite arms as two entries", () => {
    const source = coverage(
      [[0]],
      [[0]],
      [1],
      [1],
      [[4, 5, 6]],
      [[4, 5, 6]],
    );
    expect(score(source, 2)[0].touchedEntryCount).toBe(2);
  });

  it("uses a mean rather than a sum and reports exact evidence fields", () => {
    const result = score(coverage([[0], [0, 1], [1, 2, 3, 4]]), 5);
    const episodeZero = result.find((item) => item.episodeId === 0)!;
    expect(episodeZero.score).toBeCloseTo(
      (1 + Math.log(5 / 2) / Math.log(5)) / 2,
    );
    expect(episodeZero.touchedEntryCount).toBe(2);
    expect(episodeZero.scopeEntryShare).toBe(2 / 3);
    expect(episodeZero.minimumDistinctEpisodeCount).toBe(1);
    expect(episodeZero.maximumDistinctEpisodeCount).toBe(2);
  });

  it("omits absent episodes and supports an empty scope", () => {
    const source = coverage([[0]]);
    expect(score(source, 3).map((item) => item.episodeId)).toEqual([0]);
    expect(score(source, 3, [0, 1, 2], [])).toEqual([]);
  });

  it("makes global scoring equal an explicit all-entry scope", () => {
    const source = coverage([[0], [1]], [[0, 1]]);
    const allEntries: CoverageEntryReference[] = [
      { arm: "left", voxelEntryIndex: 0 },
      { arm: "left", voxelEntryIndex: 1 },
      { arm: "right", voxelEntryIndex: 0 },
    ];
    expect(score(source, 2, [0, 1], allEntries)).toEqual(score(source, 2));
  });

  it("orders ties by evidence count and then episode ID", () => {
    const result = score(coverage([[2], [1], [1], [0]]), 3);
    expect(result.map((item) => item.episodeId)).toEqual([1, 0, 2]);
  });

  it("does not mutate coverage, scope, or allowed episode IDs", () => {
    const source = coverage([[0], [0, 1]], [[1]]);
    const allowed = [0, 1];
    const scope: CoverageEntryReference[] = [
      { arm: "left", voxelEntryIndex: 1 },
      { arm: "right", voxelEntryIndex: 0 },
    ];
    const sourceBefore = structuredClone(source);
    const allowedBefore = structuredClone(allowed);
    const scopeBefore = structuredClone(scope);
    scoreUncommonEpisodes({
      coverage: source,
      episodeCount: 2,
      allowedEpisodeIds: allowed,
      scope,
    });
    expect(source).toEqual(sourceBefore);
    expect(allowed).toEqual(allowedBefore);
    expect(scope).toEqual(scopeBefore);
  });

  it.each([0, -1, 1.5])("rejects invalid episode count %s", (episodeCount) => {
    expect(() => score(coverage([[0]]), episodeCount, [0])).toThrow(
      "Coverage episode count must be a positive integer.",
    );
  });

  it("rejects duplicate scoped references", () => {
    const reference = { arm: "left", voxelEntryIndex: 0 } as const;
    expect(() => score(coverage([[0]]), 1, [0], [reference, reference])).toThrow(
      "Duplicate scoped coverage entry: left:0.",
    );
  });

  it("rejects unknown arms and out-of-range entry indices", () => {
    expect(() =>
      score(coverage([[0]]), 1, [0], [
        { arm: "center", voxelEntryIndex: 0 } as unknown as CoverageEntryReference,
      ]),
    ).toThrow("Unknown coverage arm: center.");
    expect(() =>
      score(coverage([[0]]), 1, [0], [
        { arm: "left", voxelEntryIndex: 1 },
      ]),
    ).toThrow("Scoped coverage entry is out of range: left:1.");
  });

  it.each([
    { entries: [[0]], offsets: [1, 1], message: "must start at zero" },
    {
      entries: [[0]],
      offsets: [0],
      message: "one offset per entry plus one",
    },
    {
      entries: [[0], [0]],
      offsets: [0, 1, 0],
      message: "must be monotonic",
    },
    {
      entries: [[0]],
      offsets: [0, 1.5],
      message: "must be a non-negative integer",
    },
    {
      entries: [[0]],
      offsets: [0, 0],
      message: "must end at the episode-ID length",
    },
  ])("rejects malformed CSR offsets", ({ entries, offsets, message }) => {
    const source = coverage(entries);
    source.arms[0].episodeIdOffsets = offsets;
    expect(() => score(source, 1, [0])).toThrow(message);
  });

  it("rejects episode IDs outside the supplied universe", () => {
    expect(() => score(coverage([[2]]), 3, [0, 1])).toThrow(
      "contains episode ID 2 outside the allowed episode universe.",
    );
  });

  it("rejects a distinct episode count greater than E", () => {
    expect(() => score(coverage([[0, 1, 2]]), 2, [0, 1, 2])).toThrow(
      "distinct episode count 3 exceeds coverage episode count 2.",
    );
  });
});
