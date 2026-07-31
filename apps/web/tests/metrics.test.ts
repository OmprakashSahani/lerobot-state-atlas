import { describe, expect, it } from "vitest";

import coverageJson from "@/public/atlas-data/demo-v2/coverage.json";
import manifestJson from "@/public/atlas-data/demo-v2/manifest.json";
import { decodeCoverage, decodeManifest } from "@/lib/atlas-schema/validate";
import { metricDomain, metricValue } from "@/lib/data/metrics";
import { prepareCoverage } from "@/lib/data/prepareCoverage";

const arms = prepareCoverage(
  decodeManifest(manifestJson),
  decodeCoverage(coverageJson),
);

describe("coverage metrics", () => {
  it("uses raw visits and log1p without rewriting raw data", () => {
    const raw = arms[0].visits[3];
    expect(metricValue(arms[0], 3, "visits")).toBe(raw);
    expect(metricValue(arms[0], 3, "log-visits")).toBeCloseTo(Math.log1p(raw));
    expect(arms[0].visits[3]).toBe(raw);
  });

  it("uses the exact CSR distinct-episode count", () => {
    const source = decodeCoverage(coverageJson).arms[0];
    const index = source.episodeCounts.findIndex((count) => count > 1);
    expect(metricValue(arms[0], index, "episodes")).toBe(
      source.episodeIdOffsets[index + 1] - source.episodeIdOffsets[index],
    );
  });

  it("keeps one stable cross-arm domain", () => {
    expect(metricDomain(arms, "visits")).toEqual([
      Math.min(...coverageJson.arms.flatMap((arm) => arm.visitCounts)),
      Math.max(...coverageJson.arms.flatMap((arm) => arm.visitCounts)),
    ]);
  });
});
