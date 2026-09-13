import { DIAGNOSTIC_SPLAT_SOURCES } from "./spark-ply-diagnostic";

export const INTEGRATED_PERFORMANCE_DIAGNOSTIC_QUERY =
  "environmentPerformanceDiagnostic";
export const INTEGRATED_PERFORMANCE_DIAGNOSTIC_VALUE = "250k";

export type IntegratedEnvironmentSource =
  | "current"
  | "uniform250k"
  | "uniform300k";

export const INTEGRATED_SOURCE_SWITCH_CYCLE: readonly IntegratedEnvironmentSource[] = [
  "current",
  "uniform250k",
  "uniform300k",
  "current",
  "uniform300k",
  "uniform250k",
  "current",
];
export const SOURCE_SWITCH_STABILIZED_FRAME_TARGET = 120;

export const INTEGRATED_ENVIRONMENT_SOURCES = {
  current: {
    source: "current",
    label: "Current voxel SPZ (86,954)",
  },
  uniform250k: {
    source: "uniform250k",
    label: "Uniform 250k SPZ (250,000)",
  },
  uniform300k: {
    source: "uniform300k",
    label: "Uniform 300k SPZ (300,000)",
  },
} as const satisfies Record<
  IntegratedEnvironmentSource,
  { source: IntegratedEnvironmentSource; label: string }
>;

export function isIntegratedPerformanceDiagnosticEnabled(
  nodeEnv: string | undefined,
  search: string,
): boolean {
  if (nodeEnv !== "development") return false;
  return (
    new URLSearchParams(search).get(INTEGRATED_PERFORMANCE_DIAGNOSTIC_QUERY) ===
    INTEGRATED_PERFORMANCE_DIAGNOSTIC_VALUE
  );
}

export function resolveIntegratedEnvironmentSource(
  nodeEnv: string | undefined,
  search: string,
  requestedSource: IntegratedEnvironmentSource | undefined,
): IntegratedEnvironmentSource {
  if (!isIntegratedPerformanceDiagnosticEnabled(nodeEnv, search)) {
    return "current";
  }
  return requestedSource === "uniform250k" || requestedSource === "uniform300k"
    ? requestedSource
    : "current";
}

export const UNIFORM_250K_INTEGRATED_ASSET =
  DIAGNOSTIC_SPLAT_SOURCES.uniform250k;
export const UNIFORM_300K_INTEGRATED_ASSET =
  DIAGNOSTIC_SPLAT_SOURCES.uniform300k;

export interface SparkFramePerformance {
  sampleCount: number;
  windowSize: number;
  averageMilliseconds: number;
  framesPerSecond: number;
  p50Milliseconds: number;
  p90Milliseconds: number;
  p95Milliseconds: number;
  p99Milliseconds: number;
  maxMilliseconds: number;
  over20Milliseconds: number;
  over25Milliseconds: number;
  over33Point3Milliseconds: number;
  over50Milliseconds: number;
}

export interface ThreeRendererInfoSnapshot {
  geometries: number;
  textures: number;
  programs: number | null;
  renderCalls: number;
  triangles: number;
  points: number;
  lines: number;
}

export interface IntegratedEnvironmentPerformanceMetrics {
  source: IntegratedEnvironmentSource;
  decodedSplatCount: number;
  assetBytes: number;
  meshInitializationMilliseconds: number;
  firstReadyMilliseconds: number;
  initializationFrameCount: number;
  stabilizationFrameCount: number;
  stabilizationFrameTarget: number;
  stabilizedFrameCount: number;
  frame: SparkFramePerformance | null;
  rendererInfo: ThreeRendererInfoSnapshot | null;
}

export type SparkResourceLifecycleStage =
  | "before-switch"
  | "after-old-disposal"
  | "after-old-disposal-frame"
  | "after-new-initialization"
  | "after-new-attachment"
  | "after-new-attachment-frame"
  | "steady-state";

export interface SparkResourceLifecycleEvent {
  switchId: number;
  fromSource: IntegratedEnvironmentSource;
  toSource: IntegratedEnvironmentSource;
  stage: SparkResourceLifecycleStage;
  elapsedMilliseconds: number;
  rendererInfo: ThreeRendererInfoSnapshot | null;
}

export type SparkResourceLifecycleResult =
  | { status: "pending" }
  | {
      status: "stable" | "growth" | "lower";
      geometryDelta: number;
      textureDelta: number;
      programDelta: number | null;
    };

export function analyzeResourceLifecycle(
  events: readonly SparkResourceLifecycleEvent[],
): SparkResourceLifecycleResult {
  const steadyStates = events.filter(
    (event) => event.stage === "steady-state" && event.rendererInfo,
  );
  if (steadyStates.length < 2) return { status: "pending" };
  const baseline = steadyStates[0]?.rendererInfo;
  const final = steadyStates.at(-1)?.rendererInfo;
  if (!baseline || !final) return { status: "pending" };
  const geometryDelta = final.geometries - baseline.geometries;
  const textureDelta = final.textures - baseline.textures;
  const programDelta =
    baseline.programs === null || final.programs === null
      ? null
      : final.programs - baseline.programs;
  const deltas = [geometryDelta, textureDelta];
  if (programDelta !== null) deltas.push(programDelta);
  const status = deltas.some((delta) => delta > 0)
    ? "growth"
    : deltas.some((delta) => delta < 0)
      ? "lower"
      : "stable";
  return {
    status,
    geometryDelta,
    textureDelta,
    programDelta,
  };
}

export function formatSignedResourceDelta(value: number): string {
  return value >= 0 ? `+${value}` : String(value);
}

export function summarizeFrameTimes(
  samples: readonly number[],
): SparkFramePerformance | null {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((left, right) => left - right);
  const averageMilliseconds =
    samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const percentile = (fraction: number) =>
    sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
  return {
    sampleCount: samples.length,
    windowSize: samples.length,
    averageMilliseconds,
    framesPerSecond: 1_000 / averageMilliseconds,
    p50Milliseconds: percentile(0.5),
    p90Milliseconds: percentile(0.9),
    p95Milliseconds: percentile(0.95),
    p99Milliseconds: percentile(0.99),
    maxMilliseconds: sorted[sorted.length - 1],
    over20Milliseconds: samples.filter((value) => value > 20).length,
    over25Milliseconds: samples.filter((value) => value > 25).length,
    over33Point3Milliseconds: samples.filter((value) => value > 33.3).length,
    over50Milliseconds: samples.filter((value) => value > 50).length,
  };
}
