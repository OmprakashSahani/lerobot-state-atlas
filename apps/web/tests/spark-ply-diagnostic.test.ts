import { Box3, Vector3 } from "three";
import { describe, expect, it } from "vitest";

import {
  boxFromDiagnosticBounds,
  calculatePerspectiveCameraFit,
  colmapCameraToNerfstudioWorldMatrix,
  DIAGNOSTIC_SPLAT_SOURCES,
  diagnosticSplatMeshOptions,
  diagnosticBoundsFromBox,
  KNOWN_PLY_CORE_BOUNDS,
  NERFSTUDIO_EVAL_CAMERA,
  nerfstudioEvalCameraAspect,
  nerfstudioEvalCameraToWorldMatrix,
  nerfstudioEvalCameraVerticalFovDegrees,
  SPARK_CAMERA_AWARE_SPZ_DIAGNOSTIC_URL,
  SPARK_HYBRID_SPZ_DIAGNOSTIC_URL,
  SPARK_PLY_DIAGNOSTIC_URL,
  SPARK_SPZ_DIAGNOSTIC_URL,
  SPARK_UNIFORM_SPZ_DIAGNOSTIC_URL,
  validateDiagnosticSplatCount,
} from "@/lib/environment/spark-ply-diagnostic";

describe("diagnostic splat sources", () => {
  it("constructs PLY and SPZ meshes with identical non-source settings", () => {
    const fileTypes = { PLY: "ply-enum", SPZ: "spz-enum" };
    const ply = diagnosticSplatMeshOptions("ply", fileTypes);
    const spz = diagnosticSplatMeshOptions("spz", fileTypes);

    expect(ply).toEqual({
      url: SPARK_PLY_DIAGNOSTIC_URL,
      fileType: "ply-enum",
      fileName: "diagnostic-original.ply",
      editable: false,
      raycastable: false,
      lod: false,
      enableLod: false,
      nonLod: true,
      paged: false,
    });
    expect(spz).toEqual({
      ...ply,
      url: SPARK_SPZ_DIAGNOSTIC_URL,
      fileType: "spz-enum",
      fileName: "omprakash-workcell.spz",
    });
    expect(ply).not.toHaveProperty("maxSplats");
    expect(spz).not.toHaveProperty("maxSplats");
  });

  it("constructs every experimental SPZ with the shared mesh settings", () => {
    const fileTypes = { PLY: "ply-enum", SPZ: "spz-enum" };
    const expected = [
      ["uniform", SPARK_UNIFORM_SPZ_DIAGNOSTIC_URL, "diagnostic-uniform-100k.spz"],
      [
        "cameraAware",
        SPARK_CAMERA_AWARE_SPZ_DIAGNOSTIC_URL,
        "diagnostic-camera-aware-100k.spz",
      ],
      ["hybrid", SPARK_HYBRID_SPZ_DIAGNOSTIC_URL, "diagnostic-hybrid-100k.spz"],
    ] as const;

    for (const [source, url, fileName] of expected) {
      expect(diagnosticSplatMeshOptions(source, fileTypes)).toEqual({
        ...diagnosticSplatMeshOptions("spz", fileTypes),
        url,
        fileName,
      });
    }
  });

  it("defines scaling sources with generated counts and byte sizes", () => {
    const expected = {
      uniform150k: [150_000, 2_397_011, "diagnostic-uniform-150k.spz"],
      uniform200k: [200_000, 3_193_032, "diagnostic-uniform-200k.spz"],
      uniform250k: [250_000, 3_990_074, "diagnostic-uniform-250k.spz"],
      uniform300k: [300_000, 4_785_949, "diagnostic-uniform-300k.spz"],
      candidateFull: [398_598, 6_357_739, "diagnostic-candidate-full.spz"],
      fullOriginal: [466_363, 7_534_845, "diagnostic-full-original.spz"],
    } as const;
    const fileTypes = { PLY: "ply-enum", SPZ: "spz-enum" };

    for (const [source, [count, byteSize, fileName]] of Object.entries(
      expected,
    )) {
      const typedSource = source as keyof typeof expected;
      expect(DIAGNOSTIC_SPLAT_SOURCES[typedSource]).toMatchObject({
        expectedSplatCount: count,
        byteSize,
        fileName,
      });
      expect(diagnosticSplatMeshOptions(typedSource, fileTypes)).toEqual({
        ...diagnosticSplatMeshOptions("spz", fileTypes),
        url: `/environment-data/__local-real__/${fileName}`,
        fileName,
      });
      expect(validateDiagnosticSplatCount(typedSource, count)).toBe(count);
    }
  });

  it("accepts only the observed decoded count for each source", () => {
    expect(DIAGNOSTIC_SPLAT_SOURCES.ply.expectedSplatCount).toBe(466_363);
    expect(DIAGNOSTIC_SPLAT_SOURCES.spz.expectedSplatCount).toBe(86_954);
    expect(DIAGNOSTIC_SPLAT_SOURCES.uniform.expectedSplatCount).toBe(100_000);
    expect(DIAGNOSTIC_SPLAT_SOURCES.cameraAware.expectedSplatCount).toBe(
      100_000,
    );
    expect(DIAGNOSTIC_SPLAT_SOURCES.hybrid.expectedSplatCount).toBe(100_000);
    expect(validateDiagnosticSplatCount("ply", 466_363)).toBe(466_363);
    expect(validateDiagnosticSplatCount("spz", 86_954)).toBe(86_954);
    expect(validateDiagnosticSplatCount("uniform", 100_000)).toBe(100_000);
    expect(validateDiagnosticSplatCount("cameraAware", 100_000)).toBe(
      100_000,
    );
    expect(validateDiagnosticSplatCount("hybrid", 100_000)).toBe(100_000);
    expect(() => validateDiagnosticSplatCount("ply", 86_954)).toThrow(
      /does not match expected/,
    );
    expect(() => validateDiagnosticSplatCount("spz", 466_363)).toThrow(
      /does not match expected/,
    );
  });
});

describe("known Nerfstudio eval camera", () => {
  it("identifies the first held-out eval frame and its exact PINHOLE intrinsics", () => {
    expect(NERFSTUDIO_EVAL_CAMERA).toMatchObject({
      nerfstudioVersion: "1.1.5",
      evalIndex: 0,
      evalInterval: 8,
      evalRender: "eval_img_0000.png",
      sourceImage: "frame_0003.jpg",
      colmapImageId: 1,
      cameraModel: "PINHOLE",
      width: 1903,
      height: 1070,
      fx: 1688.3917720798788,
      fy: 1688.3917720798788,
      cx: 951.5,
      cy: 535,
    });
    expect(nerfstudioEvalCameraAspect()).toBeCloseTo(1903 / 1070, 12);
    expect(nerfstudioEvalCameraVerticalFovDegrees()).toBeCloseTo(
      35.163649820894065,
      12,
    );
  });

  it("converts COLMAP world-to-camera into the exported PLY coordinate system", () => {
    const cameraToWorld = nerfstudioEvalCameraToWorldMatrix();
    const expectedRows = [
      [
        0.939988086239782, -0.103751313066746, 0.32505090642096,
        0.658028055806075,
      ],
      [
        0.331116174563752, 0.047398841869632, -0.942398747876814,
        -0.555819822369986,
      ],
      [
        0.082368070314583, 0.993473208251362, 0.078908084397564,
        -0.067713549314084,
      ],
      [0, 0, 0, 1],
    ];

    for (let row = 0; row < 4; row += 1) {
      for (let column = 0; column < 4; column += 1) {
        expect(cameraToWorld.elements[column * 4 + row]).toBeCloseTo(
          expectedRows[row][column],
          12,
        );
      }
    }

    // Three.js looks down local -Z, which is Nerfstudio's OpenGL convention.
    const lookDirection = new Vector3(0, 0, -1)
      .transformDirection(cameraToWorld)
      .toArray();
    const expectedLookDirection = new Vector3(
      -0.32505090642096,
      0.942398747876814,
      -0.078908084397564,
    ).normalize();
    expect(lookDirection[0]).toBeCloseTo(expectedLookDirection.x, 12);
    expect(lookDirection[1]).toBeCloseTo(expectedLookDirection.y, 12);
    expect(lookDirection[2]).toBeCloseTo(expectedLookDirection.z, 12);
  });

  it("uses the generic registered-camera conversion for the known eval view", () => {
    const generic = colmapCameraToNerfstudioWorldMatrix(
      NERFSTUDIO_EVAL_CAMERA.colmapWorldToCamera,
      NERFSTUDIO_EVAL_CAMERA.dataparserTransform,
      NERFSTUDIO_EVAL_CAMERA.dataparserScale,
    );
    expect(generic.elements).toEqual(
      nerfstudioEvalCameraToWorldMatrix().elements,
    );
  });
});

describe("isolated Spark PLY camera fitting", () => {
  it("round-trips the known reconstruction-coordinate core bounds", () => {
    expect(
      diagnosticBoundsFromBox(boxFromDiagnosticBounds(KNOWN_PLY_CORE_BOUNDS)),
    ).toEqual(KNOWN_PLY_CORE_BOUNDS);
  });

  it("places the perspective camera outside the fitted reconstruction sphere", () => {
    const bounds = new Box3(
      new Vector3(-1, -2, -3),
      new Vector3(1, 2, 3),
    );
    const fit = calculatePerspectiveCameraFit({
      bounds,
      verticalFovDegrees: 50,
      aspect: 16 / 9,
      distanceMultiplier: 1,
      nearMultiplier: 1,
      farMultiplier: 1,
    });

    expect(fit.target.toArray()).toEqual([0, 0, 0]);
    expect(fit.position.distanceTo(fit.target)).toBeCloseTo(fit.distance);
    expect(fit.distance).toBeGreaterThan(fit.radius * 2);
    expect(fit.near).toBeGreaterThan(0);
    expect(fit.far).toBeGreaterThan(fit.distance + fit.radius);
  });

  it("moves farther away for a narrower viewport and honors clip multipliers", () => {
    const bounds = new Box3(
      new Vector3(-1, -1, -1),
      new Vector3(1, 1, 1),
    );
    const wide = calculatePerspectiveCameraFit({
      bounds,
      verticalFovDegrees: 50,
      aspect: 2,
      distanceMultiplier: 1,
      nearMultiplier: 1,
      farMultiplier: 1,
    });
    const narrow = calculatePerspectiveCameraFit({
      bounds,
      verticalFovDegrees: 50,
      aspect: 0.5,
      distanceMultiplier: 1,
      nearMultiplier: 2,
      farMultiplier: 2,
    });

    expect(narrow.distance).toBeGreaterThan(wide.distance);
    expect(narrow.near).toBeCloseTo(wide.near * 2);
    expect(narrow.far).toBeGreaterThan(wide.far * 2);
  });

  it("rejects empty bounds and invalid viewport geometry", () => {
    expect(() =>
      calculatePerspectiveCameraFit({
        bounds: new Box3(),
        verticalFovDegrees: 50,
        aspect: 1,
        distanceMultiplier: 1,
        nearMultiplier: 1,
        farMultiplier: 1,
      }),
    ).toThrow(/empty/);
    expect(() =>
      calculatePerspectiveCameraFit({
        bounds: new Box3(
          new Vector3(-1, -1, -1),
          new Vector3(1, 1, 1),
        ),
        verticalFovDegrees: 50,
        aspect: 0,
        distanceMultiplier: 1,
        nearMultiplier: 1,
        farMultiplier: 1,
      }),
    ).toThrow(/invalid/);
  });
});
