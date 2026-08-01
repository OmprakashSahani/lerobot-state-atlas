import { describe, expect, it } from "vitest";

import coverageJson from "@/public/atlas-data/demo-v2/coverage.json";
import manifestJson from "@/public/atlas-data/demo-v2/manifest.json";
import { decodeCoverage, decodeManifest } from "@/lib/atlas-schema/validate";
import { formatEpisodeSelection } from "@/lib/data/episodeSelection";
import { numericExtent, prepareCoverage } from "@/lib/data/prepareCoverage";

describe("large coverage preparation", () => {
  it.each([
    { values: [4, 1, 9, 2], expected: [1, 9] },
    { values: [-4, -12, 3, 0], expected: [-12, 3] },
    { values: [7], expected: [7, 7] },
  ])("reduces ordinary numeric arrays: $values", ({ values, expected }) => {
    expect(numericExtent(values)).toEqual(expected);
  });

  it("reduces typed arrays without copying them", () => {
    const values = new Float64Array([8.5, -2.25, 17, 4]);

    expect(numericExtent(values)).toEqual([-2.25, 17]);
  });

  it("preserves the previous empty-input extent contract", () => {
    expect(numericExtent([])).toEqual([
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ]);
  });

  it("handles an array beyond typical function-argument limits", () => {
    const values = new Int32Array(200_000);
    values.fill(12);
    values[91_337] = -25;
    values[199_999] = 48;

    expect(numericExtent(values)).toEqual([-25, 48]);
  });

  it("preserves prepared output from the previous valid-array reduction", () => {
    const manifest = decodeManifest(manifestJson);
    const coverage = decodeCoverage(coverageJson);
    const prepared = prepareCoverage(manifest, coverage);

    for (const [index, arm] of coverage.arms.entries()) {
      expect(prepared[index].minimumVisitCount).toBe(
        Math.min(...arm.visitCounts),
      );
      expect(prepared[index].maximumVisitCount).toBe(
        Math.max(...arm.visitCounts),
      );
      expect(Array.from(prepared[index].visits)).toEqual(arm.visitCounts);
      expect(prepared[index].instanceLookup.map((entry) => entry.voxelIndex)).toEqual(
        arm.voxelIndices,
      );
    }
  });
});

describe("manifest episode selection presentation", () => {
  it("formats contiguous episode IDs", () => {
    expect(formatEpisodeSelection([0, 1, 2, 3], 4)).toBe("episodes 0–3");
    expect(formatEpisodeSelection([20, 21, 22], 3)).toBe("episodes 20–22");
  });

  it("formats one selected episode with singular wording", () => {
    expect(formatEpisodeSelection([42], 1)).toBe("episode 42");
  });

  it("does not imply a range for non-contiguous IDs", () => {
    expect(formatEpisodeSelection([2, 5, 9], 3)).toBe("3 selected episodes");
  });
});
