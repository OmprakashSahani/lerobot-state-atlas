// @vitest-environment node
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";

import { inspectSpzHeader } from "@/lib/environment/spz-header";
// The generator is intentionally JavaScript so it can run without a build step.
// @ts-expect-error no declaration file is needed for this repository-owned script
import { EXPECTED_SHA256, EXPECTED_SPLAT_COUNT, generateSyntheticSpz } from "@/scripts/generate-synthetic-spz.mjs";

function fixtureHeader(mutate?: (view: DataView) => void, payloadBytes = 0) {
  const bytes = new Uint8Array(16 + payloadBytes);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x5053474e, true); view.setUint32(4, 3, true); view.setUint32(8, 4, true);
  view.setUint8(12, 0); view.setUint8(13, 12); view.setUint8(14, 1); view.setUint8(15, 0);
  mutate?.(view);
  return new Uint8Array(gzipSync(bytes));
}

describe("SPZ v3 spike preflight", () => {
  it("generates byte-identical application-owned bytes with fixed identity", async () => {
    const first = generateSyntheticSpz(); const second = generateSyntheticSpz();
    expect(first).toEqual(second);
    expect(first.byteLength).toBe(541);
    expect(createHash("sha256").update(first).digest("hex")).toBe(EXPECTED_SHA256);
    await expect(inspectSpzHeader(first, EXPECTED_SPLAT_COUNT)).resolves.toMatchObject({ version: 3, shDegree: 0, splatCount: EXPECTED_SPLAT_COUNT, flags: 1, fractionalBits: 12 });
  });

  it.each([1, 2, 4])("rejects SPZ version %i", async (version) => {
    await expect(inspectSpzHeader(fixtureHeader((view) => view.setUint32(4, version, true)), 4)).rejects.toThrow(/version 3/);
  });

  it.each([
    ["magic", (view: DataView) => view.setUint32(0, 0, true), /magic/],
    ["zero count", (view: DataView) => view.setUint32(8, 0, true), /count/],
    ["excessive count", (view: DataView) => view.setUint32(8, 100001, true), /count/],
    ["SH degree", (view: DataView) => view.setUint8(12, 1), /degree 0/],
    ["fractional bits", (view: DataView) => view.setUint8(13, 17), /fractionalBits/],
    ["unknown flags", (view: DataView) => view.setUint8(14, 0x80), /flags/],
    ["reserved", (view: DataView) => view.setUint8(15, 1), /reserved/],
  ] as const)("rejects invalid %s", async (_label, mutate, message) => {
    await expect(inspectSpzHeader(fixtureHeader(mutate), 4)).rejects.toThrow(message);
  });

  it("rejects count mismatch, malformed gzip, and truncated header", async () => {
    await expect(inspectSpzHeader(fixtureHeader(), 5)).rejects.toThrow(/does not match/);
    await expect(inspectSpzHeader(new Uint8Array([1, 2, 3]), 4)).rejects.toThrow(/malformed/);
    await expect(inspectSpzHeader(new Uint8Array(gzipSync(new Uint8Array(8))), 4)).rejects.toThrow(/truncated/);
  });
});
