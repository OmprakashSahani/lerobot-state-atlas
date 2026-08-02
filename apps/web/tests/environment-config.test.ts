import { describe, expect, it } from "vitest";

import { isConservativeSpikeMobile, resolveLocalEnvironmentConfiguration } from "@/lib/environment/config";
import { resolveLocalAssetPath, validateLocalManifestPath } from "@/lib/environment/path-safety";

const approved = "/environment-data/__local-synthetic__/manifest.json";

describe("local environment capability configuration", () => {
  it("is always unavailable in production", () => {
    expect(resolveLocalEnvironmentConfiguration("production", approved)).toEqual({ status: "unavailable" });
  });

  it("requires an explicitly approved development path", () => {
    expect(resolveLocalEnvironmentConfiguration("development", undefined).status).toBe("unavailable");
    expect(resolveLocalEnvironmentConfiguration("development", approved)).toEqual({ status: "available", manifestPath: approved });
  });

  it.each([
    "https://example.com/manifest.json",
    "//example.com/manifest.json",
    "/environment-data/__local-synthetic__/../manifest.json",
    "/environment-data/__local-synthetic__/%2e%2e/manifest.json",
    "/environment-data/__local-synthetic__/%2Fmanifest.json",
    "/environment-data/__local-synthetic__/a//manifest.json",
    "/environment-data/__local-synthetic__/manifest.json?q=1",
    "/environment-data/__local-synthetic__/manifest.json#x",
    "/environment-data/__local-synthetic__/folder\\manifest.json",
  ])("rejects unsafe configured path %s", (path) => {
    expect(() => validateLocalManifestPath(path)).toThrow();
    expect(resolveLocalEnvironmentConfiguration("development", path).status).toBe("unavailable");
  });

  it("resolves only a safe single-segment SPZ beside the manifest", () => {
    expect(resolveLocalAssetPath(approved, "synthetic-environment.spz")).toBe("/environment-data/__local-synthetic__/synthetic-environment.spz");
    expect(() => resolveLocalAssetPath(approved, "../escape.spz")).toThrow();
    expect(() => resolveLocalAssetPath(approved, "nested/scene.spz")).toThrow();
  });

  it("uses conservative browser mobile signals rather than viewport width", () => {
    expect(isConservativeSpikeMobile({ userAgent: "Mozilla Android Mobile" } as Navigator)).toBe(true);
    expect(isConservativeSpikeMobile({ userAgent: "Desktop", userAgentData: { mobile: true } } as unknown as Navigator)).toBe(true);
    expect(isConservativeSpikeMobile({ userAgent: "Macintosh", platform: "MacIntel", maxTouchPoints: 5 } as Navigator)).toBe(true);
    expect(isConservativeSpikeMobile({ userAgent: "Desktop" } as Navigator)).toBe(false);
  });
});
