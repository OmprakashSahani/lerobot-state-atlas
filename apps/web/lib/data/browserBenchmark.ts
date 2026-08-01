import type { AtlasData, Vector3 } from "@/lib/atlas-schema/types";
import type { BenchmarkLoadDurations } from "@/lib/data/loadBundle";
import { queryRadius, type VoxelSelection } from "@/lib/data/radiusQuery";
import { scoreUncommonEpisodes } from "@/lib/data/uncommonEpisodes";

export const BENCHMARK_RADII = [0, 0.05, 0.3] as const;

export interface RadiusBenchmarkMeasurement {
  radiusMetres: number;
  matchedEntryCount: number;
  rankedEpisodeCount: number;
  radiusQueryMilliseconds: number;
  localUncommonScoringMilliseconds: number;
}

export interface AtlasBrowserBenchmarkReport {
  reportFormat: 1;
  startedAtUtc: string;
  completedAtUtc: string;
  bundleBase: string;
  bundleId: string;
  episodeCount: number;
  armSpecificEntryCount: number;
  csrIncidence: number;
  userAgent: string;
  selectedEntry: {
    arm: "left" | "right";
    voxelEntryIndex: number;
    exportedCenter: Vector3;
  };
  durations: BenchmarkLoadDurations & {
    globalUncommonScoringMilliseconds: number;
  };
  globalUncommonEpisodeCount: number;
  radiusMeasurements: RadiusBenchmarkMeasurement[];
}

export interface BenchmarkRunnerOptions {
  bundleBase: string;
  loadDurations: BenchmarkLoadDurations;
  startedAtUtc?: string;
  now?: () => number;
  wallNow?: () => Date;
  userAgent?: string;
}

export function benchmarksEnabled(
  value = process.env.NEXT_PUBLIC_ATLAS_ENABLE_BENCHMARKS,
): boolean {
  return value === "1";
}

export function selectDeterministicOccupiedEntry(data: AtlasData): VoxelSelection {
  for (const arm of data.preparedArms) {
    if (arm.visits.length === 0) continue;
    return {
      arm: arm.arm,
      voxelEntryIndex: 0,
      exportedCenter: Array.from(arm.centers.slice(0, 3)) as Vector3,
    };
  }
  throw new Error("Cannot benchmark radius queries without an occupied entry.");
}

export function runAtlasBrowserBenchmark(
  data: AtlasData,
  options: BenchmarkRunnerOptions,
): AtlasBrowserBenchmarkReport {
  const now = options.now ?? performance.now.bind(performance);
  const wallNow = options.wallNow ?? (() => new Date());
  const startedAtUtc = options.startedAtUtc ?? wallNow().toISOString();
  const { manifest, coverage, preparedArms } = data;
  const selectedEntry = selectDeterministicOccupiedEntry(data);

  const globalScoringStarted = now();
  const globalScores = scoreUncommonEpisodes({
    coverage,
    episodeCount: manifest.dataset.episodeCount,
    allowedEpisodeIds: manifest.dataset.episodeIds,
  });
  const globalUncommonScoringMilliseconds = now() - globalScoringStarted;

  const radiusMeasurements = BENCHMARK_RADII.map((radiusMetres) => {
    const queryStarted = now();
    const result = queryRadius(
      preparedArms,
      coverage,
      selectedEntry,
      radiusMetres,
      manifest.coverage.armSpacing,
      manifest.coverage.armSpacing,
    );
    const radiusQueryMilliseconds = now() - queryStarted;
    const localScoringStarted = now();
    const localScores = scoreUncommonEpisodes({
      coverage,
      episodeCount: manifest.dataset.episodeCount,
      allowedEpisodeIds: manifest.dataset.episodeIds,
      scope: result.matches,
    });
    const localUncommonScoringMilliseconds = now() - localScoringStarted;
    return {
      radiusMetres,
      matchedEntryCount: result.entryCount,
      rankedEpisodeCount: localScores.length,
      radiusQueryMilliseconds,
      localUncommonScoringMilliseconds,
    };
  });

  return {
    reportFormat: 1,
    startedAtUtc,
    completedAtUtc: wallNow().toISOString(),
    bundleBase: options.bundleBase,
    bundleId: manifest.bundleId,
    episodeCount: manifest.dataset.episodeCount,
    armSpecificEntryCount: preparedArms.reduce(
      (total, arm) => total + arm.visits.length,
      0,
    ),
    csrIncidence: coverage.arms.reduce(
      (total, arm) => total + arm.episodeIds.length,
      0,
    ),
    userAgent: options.userAgent ?? navigator.userAgent,
    selectedEntry,
    durations: {
      ...options.loadDurations,
      globalUncommonScoringMilliseconds,
    },
    globalUncommonEpisodeCount: globalScores.length,
    radiusMeasurements,
  };
}

export interface BenchmarkWindow {
  __LEROBOT_STATE_ATLAS_BENCHMARK__?: AtlasBrowserBenchmarkReport;
}

declare global {
  interface Window {
    __LEROBOT_STATE_ATLAS_BENCHMARK__?: AtlasBrowserBenchmarkReport;
  }
}

const benchmarkRuns = new Map<string, Promise<AtlasData>>();

export function publishBrowserBenchmarkOnce(
  key: string,
  target: BenchmarkWindow,
  operation: () => Promise<{
    data: AtlasData;
    report: AtlasBrowserBenchmarkReport;
  }>,
): Promise<AtlasData> {
  const existing = benchmarkRuns.get(key);
  if (existing) return existing;
  const run = operation().then(
    ({ data, report }) => {
      target.__LEROBOT_STATE_ATLAS_BENCHMARK__ = report;
      console.info(
        `LeRobot State Atlas benchmark complete for ${report.bundleId}.`,
      );
      return data;
    },
    (error: unknown) => {
      benchmarkRuns.delete(key);
      throw error;
    },
  );
  benchmarkRuns.set(key, run);
  return run;
}

export function publishBrowserBenchmarkIfEnabled(
  enabled: boolean,
  key: string,
  target: BenchmarkWindow,
  operation: () => Promise<{
    data: AtlasData;
    report: AtlasBrowserBenchmarkReport;
  }>,
): Promise<AtlasData> | null {
  return enabled
    ? publishBrowserBenchmarkOnce(key, target, operation)
    : null;
}
