// @vitest-environment node
import { describe, expect, it } from "vitest";

import {
  NERFSTUDIO_EVAL_CAMERA,
  nerfstudioEvalCameraToWorldMatrix,
} from "@/lib/environment/spark-ply-diagnostic";
// @ts-expect-error repository-owned no-build diagnostic script
import * as diagnosticReduction from "@/scripts/generate-diagnostic-spz-reductions.mjs";

const {
  cameraAwareImportance,
  convertDiagnosticCameras,
  deterministicHash,
  distributionQuantiles,
  selectScaleStratifiedCameraAwareOffsets,
  selectUniformCandidateOffsets,
} = diagnosticReduction;

// @ts-expect-error repository-owned no-build diagnostic script
import * as diagnosticScaling from "@/scripts/generate-diagnostic-spz-scaling.mjs";

const { assertDiagnosticScalingOutputName, DIAGNOSTIC_SCALING_VARIANTS } =
  diagnosticScaling;

describe("development-only diagnostic SPZ reduction", () => {
  it("defines separate scaling outputs and protects current assets", () => {
    expect(
      DIAGNOSTIC_SCALING_VARIANTS.map(
        (variant: { count: number | null; fileName: string }) => [
          variant.count,
          variant.fileName,
        ],
      ),
    ).toEqual([
      [150_000, "diagnostic-uniform-150k.spz"],
      [200_000, "diagnostic-uniform-200k.spz"],
      [250_000, "diagnostic-uniform-250k.spz"],
      [300_000, "diagnostic-uniform-300k.spz"],
      [null, "diagnostic-candidate-full.spz"],
      [null, "diagnostic-full-original.spz"],
    ]);
    expect(() =>
      assertDiagnosticScalingOutputName("omprakash-workcell.spz"),
    ).toThrow(/protected/);
    expect(() =>
      assertDiagnosticScalingOutputName("diagnostic-uniform-100k.spz"),
    ).toThrow(/protected/);
    expect(
      assertDiagnosticScalingOutputName("diagnostic-uniform-300k.spz"),
    ).toBe("diagnostic-uniform-300k.spz");
  });

  it("uses deterministic uniform hash selection without duplicate candidates", () => {
    const sourceIndices = Uint32Array.from({ length: 20 }, (_, index) => index);
    const first = selectUniformCandidateOffsets(sourceIndices, 7);
    const second = selectUniformCandidateOffsets(sourceIndices, 7);

    expect(first).toEqual(second);
    expect(new Set(first)).toHaveLength(7);
    expect(first).toHaveLength(7);
    expect(deterministicHash(12)).toBe(deterministicHash(12));
    expect(deterministicHash(12)).not.toBe(deterministicHash(13));
  });

  it("rewards multi-view support and capped projected contribution", () => {
    const sparse = cameraAwareImportance({
      visibleCameraCount: 10,
      cameraCount: 210,
      meanFootprintContribution: 0.5,
    });
    const supported = cameraAwareImportance({
      visibleCameraCount: 100,
      cameraCount: 210,
      meanFootprintContribution: 0.5,
    });
    const strongerFootprint = cameraAwareImportance({
      visibleCameraCount: 100,
      cameraCount: 210,
      meanFootprintContribution: 0.9,
    });

    expect(supported).toBeGreaterThan(sparse);
    expect(strongerFootprint).toBeGreaterThan(supported);
    expect(
      cameraAwareImportance({
        visibleCameraCount: 210,
        cameraCount: 210,
        meanFootprintContribution: 100,
      }),
    ).toBe(1);
    expect(
      cameraAwareImportance({
        visibleCameraCount: 0,
        cameraCount: 210,
        meanFootprintContribution: 1,
      }),
    ).toBe(0);
  });

  it("allocates camera-aware selections across scale strata", () => {
    const sourceIndices = Uint32Array.from({ length: 20 }, (_, index) => index);
    const maxScales = Float32Array.from(
      { length: 20 },
      (_, index) => index / 1000,
    );
    const scores = Float32Array.from({ length: 20 }, (_, index) => index);
    const selected = selectScaleStratifiedCameraAwareOffsets(
      sourceIndices,
      maxScales,
      scores,
      10,
      2,
    );

    expect(selected.filter((index: number) => index < 10)).toHaveLength(5);
    expect(selected.filter((index: number) => index >= 10)).toHaveLength(5);
    expect(selected.toSorted((left: number, right: number) => left - right)).toEqual([
      5, 6, 7, 8, 9, 15, 16, 17, 18, 19,
    ]);
  });

  it("reports interpolation-based p10/p50/p90/p99 distribution summaries", () => {
    expect(distributionQuantiles([1, 2, 3, 4, 5])).toEqual({
      p10: 1.4,
      p50: 3,
      p90: 4.6,
      p99: 4.96,
    });
  });

  it("matches the established COLMAP-to-Nerfstudio camera conversion", () => {
    const [converted] = convertDiagnosticCameras({
      schemaVersion: 1,
      source: {
        transform: NERFSTUDIO_EVAL_CAMERA.dataparserTransform,
        scale: NERFSTUDIO_EVAL_CAMERA.dataparserScale,
      },
      cameras: Array.from({ length: 210 }, (_, registeredIndex) => ({
        registeredIndex,
        colmapImageId: registeredIndex + 1,
        sourceImage: `frame_${registeredIndex}.jpg`,
        width: NERFSTUDIO_EVAL_CAMERA.width,
        height: NERFSTUDIO_EVAL_CAMERA.height,
        fx: NERFSTUDIO_EVAL_CAMERA.fx,
        fy: NERFSTUDIO_EVAL_CAMERA.fy,
        cx: NERFSTUDIO_EVAL_CAMERA.cx,
        cy: NERFSTUDIO_EVAL_CAMERA.cy,
        quaternionWxyz:
          NERFSTUDIO_EVAL_CAMERA.colmapWorldToCamera.quaternionWxyz,
        translation: NERFSTUDIO_EVAL_CAMERA.colmapWorldToCamera.translation,
      })),
    });
    const expected = nerfstudioEvalCameraToWorldMatrix().invert().elements;

    converted.worldToCamera.forEach((value: number, index: number) => {
      expect(value).toBeCloseTo(expected[index], 12);
    });
  });
});
