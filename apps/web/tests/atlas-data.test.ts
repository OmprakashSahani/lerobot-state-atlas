import { describe, expect, it } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v1/manifest.json";
import coverageJson from "@/public/atlas-data/demo-v1/coverage.json";
import {
  AtlasDataError,
  decodeCoverage,
  decodeManifest,
} from "@/lib/atlas-schema/validate";
import { prepareCoverage, voxelCenter } from "@/lib/data/prepareCoverage";

describe("browser-data compatibility", () => {
  it("decodes the committed v1 manifest and coverage", () => {
    const manifest = decodeManifest(manifestJson);
    const coverage = decodeCoverage(coverageJson);

    expect(manifest.schema.major).toBe(1);
    expect(manifest.bundleId).toBe("demo-v1");
    expect(manifest.dataset.requestedRevision).toBe("v3.0");
    expect(manifest.dataset.resolvedRevision).toMatch(/^[0-9a-f]{40}$/);
    expect(manifest.dataset.requestedRevision).not.toBe(
      manifest.dataset.resolvedRevision,
    );
    expect(manifest.exporter.workingTreeDirty).toBe(true);
    expect(coverage.arms.map((arm) => arm.arm)).toEqual(["left", "right"]);
    expect(
      coverage.arms.reduce((total, arm) => total + arm.visitCounts.length, 0),
    ).toBe(manifest.totals.armVoxelEntryCount);
  });

  it("rejects an unsupported schema major version", () => {
    const incompatible = structuredClone(manifestJson);
    incompatible.schema.major = 2;

    expect(() => decodeManifest(incompatible)).toThrow(
      new AtlasDataError("Unsupported browser-data major version: 2."),
    );
  });

  it("rejects a non-immutable resolved dataset revision", () => {
    const invalid = structuredClone(manifestJson);
    invalid.dataset.resolvedRevision = "v3.0";

    expect(() => decodeManifest(invalid)).toThrow(
      /full lowercase commit SHA/,
    );
  });

  it("derives voxel centers and stable instance lookup from raw indices", () => {
    const manifest = decodeManifest(manifestJson);
    const coverage = decodeCoverage(coverageJson);
    const prepared = prepareCoverage(manifest, coverage);
    const firstIndex = coverage.arms[0].voxelIndices[0];
    const expected = voxelCenter(
      firstIndex,
      manifest.coverage.voxelOrigin,
      manifest.coverage.voxelSize,
    );

    expect(Array.from(prepared[0].centers.slice(0, 3))).toEqual(
      expected.map((value) => Math.fround(value)),
    );
    expect(prepared[0].instanceLookup[0]).toEqual({
      arm: "left",
      voxelEntryIndex: 0,
      voxelIndex: firstIndex,
    });
    expect(prepared[0].visits[0]).toBe(coverage.arms[0].visitCounts[0]);
  });

  it("detects inconsistent CSR episode counts", () => {
    const invalid = structuredClone(coverageJson);
    invalid.arms[0].episodeCounts[0] += 1;

    expect(() => decodeCoverage(invalid)).toThrow(
      /episode counts disagree with CSR data/,
    );
  });
});
