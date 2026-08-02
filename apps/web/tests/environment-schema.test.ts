import { describe, expect, it } from "vitest";

import syntheticManifest from "@/tests/fixtures/environment/synthetic-v1/manifest.json";
import {
  decodeEnvironmentManifest,
  EnvironmentManifestError,
} from "@/lib/environment/validate";

function cloneFixture(): Record<string, unknown> {
  return structuredClone(syntheticManifest) as Record<string, unknown>;
}

describe("environment-layer v1 contract", () => {
  it("accepts the explicitly synthetic test-only manifest", () => {
    const manifest = decodeEnvironmentManifest(syntheticManifest);

    expect(manifest.status).toBe("available");
    expect(manifest.provenance.sourceKind).toBe("synthetic-test");
    expect(manifest.provenance.reconstructionClaim).toBe(false);
    expect(manifest.provenance.description).toMatch(/synthetic/i);
  });

  it("accepts an intentional unavailable capability manifest", () => {
    expect(
      decodeEnvironmentManifest({
        schema: {
          name: "lerobot-state-atlas.environment-layer",
          major: 1,
          minor: 0,
        },
        environmentId: "future-demo-environment",
        label: "Future demo environment",
        status: "unavailable",
        provenance: {
          sourceKind: "real-scan",
          description: "No validated scan is currently bundled.",
          reconstructionClaim: false,
        },
        coordinateFrame: "canonical-shared-world",
        unavailableReason: "No validated scan or calibration is available.",
      }).status,
    ).toBe("unavailable");
  });

  it.each([
    ["unsupported schema", (value: Record<string, unknown>) => {
      (value.schema as Record<string, unknown>).minor = 1;
    }],
    ["unknown fields", (value: Record<string, unknown>) => {
      value.downloadUrl = "https://example.com/scene.spz";
    }],
    ["unsafe filename", (value: Record<string, unknown>) => {
      (value.asset as Record<string, unknown>).filename = "../scene.spz";
    }],
    ["non-finite transform", (value: Record<string, unknown>) => {
      (value.alignment as Record<string, unknown>).translationXyz = [0, Number.NaN, 0];
    }],
    ["non-unit quaternion", (value: Record<string, unknown>) => {
      (value.alignment as Record<string, unknown>).rotationXyzw = [0, 0, 0, 2];
    }],
    ["invalid bounds", (value: Record<string, unknown>) => {
      (value.bounds as Record<string, unknown>).maximumXyz = [-2, 1, 1];
    }],
    ["invalid checksum", (value: Record<string, unknown>) => {
      (value.asset as Record<string, unknown>).sha256 = "A".repeat(64);
    }],
  ])("rejects %s", (_label, mutate) => {
    const value = cloneFixture();
    mutate(value);
    expect(() => decodeEnvironmentManifest(value)).toThrow(EnvironmentManifestError);
  });

  it("requires synthetic provenance to be explicit and truthful", () => {
    const value = cloneFixture();
    const provenance = value.provenance as Record<string, unknown>;
    provenance.description = "Test environment.";
    expect(() => decodeEnvironmentManifest(value)).toThrow(/explicitly say synthetic/);

    provenance.description = "Synthetic test environment.";
    provenance.reconstructionClaim = "documented-real-scan";
    expect(() => decodeEnvironmentManifest(value)).toThrow(
      /must not claim a real reconstruction/,
    );
  });
});
