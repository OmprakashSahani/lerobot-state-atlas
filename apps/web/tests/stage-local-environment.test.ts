// @vitest-environment node
import { createHash } from "node:crypto";
import { readFileSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const webRoot = resolve(import.meta.dirname, "..");
const fixtureRoot = resolve(webRoot, "public/environment-data/__local-synthetic__");
const asset = resolve(fixtureRoot, "synthetic-environment.spz");

beforeEach(() => {
  rmSync(fixtureRoot, { recursive: true, force: true });
});

afterEach(() => {
  rmSync(fixtureRoot, { recursive: true, force: true });
});
// @ts-expect-error repository-owned no-build staging script
import { generateSyntheticSpz, stageSyntheticFixture } from "@/scripts/generate-synthetic-spz.mjs";

describe("local synthetic environment staging", () => {
  it("writes deterministic bytes only to the approved ignored root", async () => {
    await stageSyntheticFixture();
    const first = readFileSync(asset);
    await stageSyntheticFixture();
    const second = readFileSync(asset);
    expect(first).toEqual(second);
    expect(new Uint8Array(first)).toEqual(generateSyntheticSpz());
    expect(createHash("sha256").update(first).digest("hex")).toBe("4fb67ec298debc9ca0d5923f283427a4af442ad9a1dc8a6d487e898654f17b98");
  });

  it("refuses an output path outside the exact local root", async () => {
    await expect(stageSyntheticFixture("/tmp/not-the-approved-root")).rejects.toThrow(/Refusing to write outside/);
  });
});
