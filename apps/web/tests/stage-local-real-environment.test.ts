// @vitest-environment node
import { describe, expect, it } from "vitest";

// The staging implementation is intentionally JavaScript so it can run directly.
// @ts-expect-error no declaration file is needed for this repository-owned script
import { PROMOTED_ASSET, PROMOTED_CAMERA_DATA, validateCameraData, validateMetadata } from "@/scripts/stage-local-real-environment.mjs";

describe("real-environment staging policy", () => {
  it("pins the reviewed Uniform 250k asset identity", () => {
    expect(PROMOTED_ASSET).toEqual({
      variant: "uniform250k",
      selectedSplats: 250_000,
      compressedBytes: 3_990_074,
      sha256:
        "f90b5ed3d1c672f62e25fca1e6acf9534901f9e8c8c80ff1a6c6b874c1d7386d",
      selectionMethod:
        "Deterministic 32-bit hash sample of the viable cropped/filtered candidate distribution.",
    });
  });

  it("rejects a source above the reviewed 250k splat cap", () => {
    expect(() =>
      validateMetadata(
        {
          selectedSplats: 250_001,
          compressedBytes: 1,
          sha256: "0".repeat(64),
          spz: { version: 3, shDegree: 0, fractionalBits: 12, flags: 0 },
        },
        new Uint8Array([0]),
      ),
    ).toThrow(/250000 permitted splats/);
  });

  it("pins the reviewed registered-camera support data", () => {
    expect(PROMOTED_CAMERA_DATA).toEqual({
      filename: "registered-cameras.json",
      byteSize: 130_912,
      sha256:
        "49077a800713ef38070fd4ec547f73840373b914f548a0bc0e0029362a74f60a",
      registeredCameraCount: 210,
      dominantCameraCount: 205,
      detachedColmapImageIds: [197, 198, 199, 200, 201],
    });
    expect(() => validateCameraData(Buffer.from("{}"))).toThrow(
      /byte size does not match/,
    );
  });
});
