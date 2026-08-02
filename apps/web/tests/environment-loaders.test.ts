import { describe, expect, it, vi } from "vitest";

import syntheticManifest from "@/tests/fixtures/environment/synthetic-v1/manifest.json";
import { loadVerifiedEnvironmentAsset, sha256Hex } from "@/lib/environment/load-asset";
import { loadLocalEnvironmentManifest, readResponseBytes } from "@/lib/environment/load-manifest";
import { decodeEnvironmentManifest } from "@/lib/environment/validate";

const origin = "https://atlas.test";
const manifestPath = "/environment-data/__local-synthetic__/manifest.json";

function response(body: BodyInit, options: ResponseInit & { url?: string; redirected?: boolean } = {}) {
  const result = new Response(body, options);
  Object.defineProperties(result, {
    url: { value: options.url ?? `${origin}${manifestPath}` },
    redirected: { value: options.redirected ?? false },
  });
  return result;
}

describe("local environment manifest loading", () => {
  it("loads a bounded same-origin synthetic manifest", async () => {
    const fetcher = vi.fn(async () => response(JSON.stringify(syntheticManifest))) as unknown as typeof fetch;
    const loaded = await loadLocalEnvironmentManifest(manifestPath, origin, new AbortController().signal, fetcher);
    expect(loaded.manifest.provenance.sourceKind).toBe("synthetic-test");
    expect(loaded.assetPath).toMatch(/synthetic-contract-test\.spz$/);
    expect(fetcher).toHaveBeenCalledWith(new URL(`${origin}${manifestPath}`), expect.objectContaining({ redirect: "error" }));
  });

  it("rejects redirects and final URLs outside the fixed root", async () => {
    const redirected = vi.fn(async () => response("{}", { redirected: true })) as unknown as typeof fetch;
    await expect(loadLocalEnvironmentManifest(manifestPath, origin, new AbortController().signal, redirected)).rejects.toThrow(/redirect/i);
    const external = vi.fn(async () => response("{}", { url: "https://evil.test/manifest.json" })) as unknown as typeof fetch;
    await expect(loadLocalEnvironmentManifest(manifestPath, origin, new AbortController().signal, external)).rejects.toThrow(/outside/i);
  });

  it("rejects invalid JSON and non-synthetic or unavailable manifests", async () => {
    const invalid = vi.fn(async () => response("{")) as unknown as typeof fetch;
    await expect(loadLocalEnvironmentManifest(manifestPath, origin, new AbortController().signal, invalid)).rejects.toThrow(/JSON/);
    const unavailable = structuredClone(syntheticManifest) as Record<string, unknown>;
    unavailable.status = "unavailable";
    delete unavailable.asset; delete unavailable.alignment; delete unavailable.bounds;
    unavailable.unavailableReason = "Intentional test.";
    const unavailableFetch = vi.fn(async () => response(JSON.stringify(unavailable))) as unknown as typeof fetch;
    await expect(loadLocalEnvironmentManifest(manifestPath, origin, new AbortController().signal, unavailableFetch)).rejects.toThrow(/not available/);
  });
});

describe("bounded byte loading and integrity", () => {
  it("accepts absent Content-Length and validates exact actual bytes", async () => {
    await expect(readResponseBytes(response(new Uint8Array([1, 2, 3])), 3)).resolves.toEqual(new Uint8Array([1, 2, 3]));
  });

  it("rejects Content-Length over cap, mismatch, and streaming overrun", async () => {
    await expect(readResponseBytes(response("x", { headers: { "content-length": "9" } }), 8)).rejects.toThrow(/Content-Length/);
    await expect(readResponseBytes(response("x", { headers: { "content-length": "2" } }), 8)).rejects.toThrow(/does not match/);
    await expect(readResponseBytes(response("12345"), 4)).rejects.toThrow(/streaming byte limit/);
  });

  it("verifies SHA-256 only after all bytes and rejects mismatch", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const manifest = decodeEnvironmentManifest({ ...syntheticManifest, asset: { ...syntheticManifest.asset, byteSize: 3, sha256: await sha256Hex(bytes) } });
    if (manifest.status !== "available") throw new Error("fixture");
    const fetcher = vi.fn(async () => response(bytes, { url: `${origin}/environment-data/__local-synthetic__/synthetic-contract-test.spz` })) as unknown as typeof fetch;
    await expect(loadVerifiedEnvironmentAsset("/environment-data/__local-synthetic__/synthetic-contract-test.spz", manifest, origin, new AbortController().signal, fetcher)).resolves.toEqual(bytes);
    const bad = { ...manifest, asset: { ...manifest.asset, sha256: "0".repeat(64) } };
    await expect(loadVerifiedEnvironmentAsset("/environment-data/__local-synthetic__/synthetic-contract-test.spz", bad, origin, new AbortController().signal, fetcher)).rejects.toThrow(/checksum/);
  });

  it("propagates abort without replacing it with an integrity error", async () => {
    const controller = new AbortController(); controller.abort();
    const fetcher = vi.fn(async (_input, init) => { if (init?.signal instanceof AbortSignal) init.signal.throwIfAborted(); return response(""); }) as unknown as typeof fetch;
    await expect(loadLocalEnvironmentManifest(manifestPath, origin, controller.signal, fetcher)).rejects.toHaveProperty("name", "AbortError");
  });
});
