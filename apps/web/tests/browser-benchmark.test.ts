import { describe, expect, it, vi } from "vitest";

import coverageJson from "@/public/atlas-data/demo-v2/coverage.json";
import manifestJson from "@/public/atlas-data/demo-v2/manifest.json";
import { decodeCoverage, decodeManifest } from "@/lib/atlas-schema/validate";
import {
  BENCHMARK_RADII,
  benchmarksEnabled,
  publishBrowserBenchmarkIfEnabled,
  publishBrowserBenchmarkOnce,
  runAtlasBrowserBenchmark,
  selectDeterministicOccupiedEntry,
  type AtlasBrowserBenchmarkReport,
  type BenchmarkWindow,
} from "@/lib/data/browserBenchmark";
import { loadDemoBundleForBenchmark } from "@/lib/data/loadBundle";
import { prepareCoverage } from "@/lib/data/prepareCoverage";
import { queryRadius } from "@/lib/data/radiusQuery";
import { scoreUncommonEpisodes } from "@/lib/data/uncommonEpisodes";

const manifest = decodeManifest(manifestJson);
const coverage = decodeCoverage(coverageJson);
const data = {
  manifest,
  coverage,
  preparedArms: prepareCoverage(manifest, coverage),
};

function benchmarkReport(): AtlasBrowserBenchmarkReport {
  let time = 0;
  const dates = [
    new Date("2026-07-31T00:00:00.000Z"),
    new Date("2026-07-31T00:00:01.000Z"),
  ];
  return runAtlasBrowserBenchmark(data, {
    bundleBase: "/atlas-data/__local-benchmark__",
    loadDurations: {
      manifestLoadMilliseconds: 2,
      coverageLoadMilliseconds: 3,
      coveragePreparationMilliseconds: 4,
    },
    now: () => time++,
    wallNow: () => dates.shift()!,
    userAgent: "benchmark-test-agent",
  });
}

describe("browser scaling benchmark", () => {
  it("is strictly opt-in", () => {
    const target: BenchmarkWindow = {};
    const operation = vi.fn(async () => ({ data, report: benchmarkReport() }));

    expect(benchmarksEnabled(undefined)).toBe(false);
    expect(benchmarksEnabled("0")).toBe(false);
    expect(
      publishBrowserBenchmarkIfEnabled(
        false,
        "disabled",
        target,
        operation,
      ),
    ).toBeNull();
    expect(operation).not.toHaveBeenCalled();
    expect(target.__LEROBOT_STATE_ATLAS_BENCHMARK__).toBeUndefined();
    expect(benchmarksEnabled("1")).toBe(true);
  });

  it("selects the first real occupied entry deterministically", () => {
    const selected = selectDeterministicOccupiedEntry(data);

    expect(selected).toEqual({
      arm: data.preparedArms[0].arm,
      voxelEntryIndex: 0,
      exportedCenter: Array.from(data.preparedArms[0].centers.slice(0, 3)),
    });
    expect(data.preparedArms[0].visits[0]).toBeGreaterThan(0);
  });

  it("records metadata, loader stages, and real global and local results", () => {
    const report = benchmarkReport();
    const globalScores = scoreUncommonEpisodes({
      coverage,
      episodeCount: manifest.dataset.episodeCount,
      allowedEpisodeIds: manifest.dataset.episodeIds,
    });

    expect(report).toMatchObject({
      reportFormat: 1,
      startedAtUtc: "2026-07-31T00:00:00.000Z",
      completedAtUtc: "2026-07-31T00:00:01.000Z",
      bundleBase: "/atlas-data/__local-benchmark__",
      bundleId: "demo-v2",
      episodeCount: 10,
      armSpecificEntryCount: 1224,
      csrIncidence: 1717,
      userAgent: "benchmark-test-agent",
      durations: {
        manifestLoadMilliseconds: 2,
        coverageLoadMilliseconds: 3,
        coveragePreparationMilliseconds: 4,
        globalUncommonScoringMilliseconds: 1,
      },
      globalUncommonEpisodeCount: globalScores.length,
    });
    expect(report.radiusMeasurements.map((item) => item.radiusMetres)).toEqual(
      BENCHMARK_RADII,
    );
    for (const measurement of report.radiusMeasurements) {
      const result = queryRadius(
        data.preparedArms,
        coverage,
        report.selectedEntry,
        measurement.radiusMetres,
        manifest.coverage.armSpacing,
        manifest.coverage.armSpacing,
      );
      const localScores = scoreUncommonEpisodes({
        coverage,
        episodeCount: manifest.dataset.episodeCount,
        allowedEpisodeIds: manifest.dataset.episodeIds,
        scope: result.matches,
      });
      expect(measurement).toMatchObject({
        matchedEntryCount: result.entryCount,
        rankedEpisodeCount: localScores.length,
        radiusQueryMilliseconds: 1,
        localUncommonScoringMilliseconds: 1,
      });
    }
  });

  it("publishes one completed report for concurrent activations", async () => {
    const target: BenchmarkWindow = {};
    const report = benchmarkReport();
    const operation = vi.fn(async () => ({ data, report }));
    const consoleInfo = vi.spyOn(console, "info").mockImplementation(() => {});
    const key = `exactly-once-${crypto.randomUUID()}`;

    const [first, second] = await Promise.all([
      publishBrowserBenchmarkOnce(key, target, operation),
      publishBrowserBenchmarkOnce(key, target, operation),
    ]);

    expect(first).toBe(data);
    expect(second).toBe(data);
    expect(operation).toHaveBeenCalledTimes(1);
    expect(target.__LEROBOT_STATE_ATLAS_BENCHMARK__).toBe(report);
    expect(consoleInfo).toHaveBeenCalledTimes(1);
    consoleInfo.mockRestore();
  });

  it("does not publish a partial report when loading or decoding fails", async () => {
    const target: BenchmarkWindow = {};
    const key = `failed-${crypto.randomUUID()}`;
    const invalidFetcher = vi.fn(async () =>
      new Response("not json", { status: 200 }),
    ) as unknown as typeof fetch;

    await expect(
      publishBrowserBenchmarkOnce(key, target, async () => {
        const result = await loadDemoBundleForBenchmark(
          invalidFetcher,
          "development",
          () => 0,
        );
        return {
          data: result.data,
          report: benchmarkReport(),
        };
      }),
    ).rejects.toThrow(/atlas manifest is not valid JSON/);
    expect(target.__LEROBOT_STATE_ATLAS_BENCHMARK__).toBeUndefined();
  });
});
