import { describe, expect, it } from "vitest";

import {
  analyzeResourceLifecycle,
  formatSignedResourceDelta,
  INTEGRATED_SOURCE_SWITCH_CYCLE,
  isIntegratedPerformanceDiagnosticEnabled,
  resolveIntegratedEnvironmentSource,
  summarizeFrameTimes,
  UNIFORM_250K_INTEGRATED_ASSET,
  UNIFORM_300K_INTEGRATED_ASSET,
} from "@/lib/environment/integrated-performance-diagnostic";
import {
  MAX_ENVIRONMENT_ASSET_BYTES,
  MAX_ENVIRONMENT_SPLATS,
} from "@/lib/environment/limits";

describe("integrated environment performance diagnostic", () => {
  const query = "?environmentPerformanceDiagnostic=250k";

  it("requires the exact development-only gate", () => {
    expect(isIntegratedPerformanceDiagnosticEnabled("development", query)).toBe(
      true,
    );
    expect(isIntegratedPerformanceDiagnosticEnabled("production", query)).toBe(
      false,
    );
    expect(
      isIntegratedPerformanceDiagnosticEnabled(
        "development",
        "?environmentPerformanceDiagnostic=true",
      ),
    ).toBe(false);
  });

  it("defines the controlled cycle without consecutive duplicate sources", () => {
    expect(INTEGRATED_SOURCE_SWITCH_CYCLE).toEqual([
      "current",
      "uniform250k",
      "uniform300k",
      "current",
      "uniform300k",
      "uniform250k",
      "current",
    ]);
  });

  it("leaves the production source and conservative production caps unchanged", () => {
    expect(
      resolveIntegratedEnvironmentSource(
        "production",
        query,
        "uniform300k",
      ),
    ).toBe("current");
    expect(
      resolveIntegratedEnvironmentSource("development", query, "uniform300k"),
    ).toBe("uniform300k");
    expect(MAX_ENVIRONMENT_SPLATS).toBe(250_000);
    expect(MAX_ENVIRONMENT_ASSET_BYTES).toBe(8 * 1024 * 1024);
    expect(UNIFORM_250K_INTEGRATED_ASSET.expectedSplatCount).toBe(250_000);
    expect(UNIFORM_300K_INTEGRATED_ASSET).toMatchObject({
      expectedSplatCount: 300_000,
      byteSize: 4_785_949,
      fileName: "diagnostic-uniform-300k.spz",
    });
  });

  it("summarizes measured frame deltas deterministically", () => {
    expect(summarizeFrameTimes([])).toBeNull();
    expect(summarizeFrameTimes([10, 20, 30, 40])).toEqual({
      sampleCount: 4,
      windowSize: 4,
      averageMilliseconds: 25,
      framesPerSecond: 40,
      p50Milliseconds: 20,
      p90Milliseconds: 40,
      p95Milliseconds: 40,
      p99Milliseconds: 40,
      maxMilliseconds: 40,
      over20Milliseconds: 2,
      over25Milliseconds: 2,
      over33Point3Milliseconds: 1,
      over50Milliseconds: 0,
    });
  });

  it("compares final steady-state resource counts with the cycle baseline", () => {
    const rendererInfo = (geometries: number) => ({
      geometries,
      textures: 7,
      programs: 6,
      renderCalls: 1,
      triangles: 1,
      points: 0,
      lines: 0,
    });
    const base = {
      switchId: 1,
      fromSource: "current" as const,
      toSource: "uniform250k" as const,
      elapsedMilliseconds: 0,
    };
    expect(
      analyzeResourceLifecycle([
        { ...base, stage: "steady-state", rendererInfo: rendererInfo(10) },
        {
          ...base,
          switchId: 2,
          stage: "steady-state",
          rendererInfo: rendererInfo(10),
        },
        {
          ...base,
          switchId: 3,
          stage: "steady-state",
          rendererInfo: rendererInfo(10),
        },
        {
          ...base,
          switchId: 4,
          stage: "steady-state",
          rendererInfo: rendererInfo(10),
        },
        {
          ...base,
          switchId: 5,
          stage: "steady-state",
          rendererInfo: rendererInfo(10),
        },
      ]),
    ).toEqual({
      status: "stable",
      geometryDelta: 0,
      textureDelta: 0,
      programDelta: 0,
    });
  });

  it("never labels negative steady-state deltas as growth", () => {
    const info = (geometries: number, textures: number, programs: number) => ({
      geometries,
      textures,
      programs,
      renderCalls: 0,
      triangles: 0,
      points: 0,
      lines: 0,
    });
    const event = {
      switchId: 0,
      fromSource: "current" as const,
      toSource: "current" as const,
      stage: "steady-state" as const,
      elapsedMilliseconds: 0,
    };
    expect(
      analyzeResourceLifecycle([
        { ...event, rendererInfo: info(10, 7, 6) },
        { ...event, switchId: 4, rendererInfo: info(9, 3, 5) },
      ]),
    ).toEqual({
      status: "lower",
      geometryDelta: -1,
      textureDelta: -4,
      programDelta: -1,
    });
    expect(
      analyzeResourceLifecycle([
        { ...event, rendererInfo: info(10, 7, 6) },
        { ...event, switchId: 4, rendererInfo: info(11, 3, 5) },
      ]).status,
    ).toBe("growth");
    expect(formatSignedResourceDelta(0)).toBe("+0");
    expect(formatSignedResourceDelta(2)).toBe("+2");
    expect(formatSignedResourceDelta(-2)).toBe("-2");
  });
});
