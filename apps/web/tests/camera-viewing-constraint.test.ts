import { Matrix4, Vector3 } from "three";
import { describe, expect, it } from "vitest";

import {
  assertReviewedCameraManifold,
  assessCameraSupport,
  buildSupportedCameraManifold,
  clampSupportedCameraIndex,
  isViewingConstraintDebugEnabled,
  loadReviewedCameraManifold,
  projectCameraToSupportedPose,
  reconstructionCameraStateFromScene,
  resetSupportedCameraIndex,
  sceneFromReconstructionMatrix,
  shouldConstrainGaussianViewing,
  type SupportedCameraManifold,
} from "@/lib/environment/camera-viewing-constraint";
import type { ConvertedRegisteredCamera } from "@/lib/environment/camera-manifold";
import type {
  EnvironmentAlignment,
  ValidatedEnvironmentRenderRequest,
} from "@/lib/environment/types";

function camera(
  registeredIndex: number,
  colmapImageId: number,
  center: [number, number, number],
  sourceFrame = registeredIndex + 1,
): ConvertedRegisteredCamera {
  const cameraToWorld = new Matrix4().setPosition(new Vector3(...center));
  return {
    registeredIndex,
    colmapImageId,
    sourceImage: `frame_${String(sourceFrame).padStart(4, "0")}.jpg`,
    cameraId: 1,
    cameraModel: "PINHOLE",
    width: 1903,
    height: 1070,
    fx: 1688,
    fy: 1688,
    cx: 951.5,
    cy: 535,
    quaternionWxyz: [1, 0, 0, 0],
    translation: [0, 0, 0],
    evalIndex: null,
    evalPsnr: null,
    cameraToWorld,
    center: new Vector3(...center),
    viewingDirection: new Vector3(0, 0, -1),
  };
}

function reviewedCameraSet(): ConvertedRegisteredCamera[] {
  const dominantIds = [
    ...Array.from({ length: 196 }, (_, index) => index + 1),
    ...Array.from({ length: 9 }, (_, index) => index + 202),
  ];
  const dominant = dominantIds.map((id, index) =>
    camera(index, id, [index * 0.01, 0, 0], index + 1),
  );
  const detached = [197, 198, 199, 200, 201].map((id, offset) =>
    camera(205 + offset, id, [10 + offset * 0.01, 0, 0], 197 + offset),
  );
  return [...dominant, ...detached];
}

describe("Gaussian captured-view constraint", () => {
  const cameras = reviewedCameraSet();
  const manifold = buildSupportedCameraManifold(cameras);

  it("uses the dominant 205-camera component and excludes the detached five", () => {
    expect(manifold.cameras).toHaveLength(205);
    expect(manifold.detachedCameras.map((item) => item.colmapImageId)).toEqual([
      197, 198, 199, 200, 201,
    ]);
    expect(() => assertReviewedCameraManifold(cameras, manifold)).not.toThrow();
  });

  it("accepts captured poses and rejects unsupported positions or orientations", () => {
    const supported = manifold.cameras[20];
    expect(
      assessCameraSupport(
        supported.center,
        supported.viewingDirection,
        manifold,
      ),
    ).toMatchObject({
      inside: true,
      distance: 0,
      nearestCamera: supported,
    });
    expect(
      assessCameraSupport(
        new Vector3(20, 20, 20),
        supported.viewingDirection,
        manifold,
      ).inside,
    ).toBe(false);
    expect(
      assessCameraSupport(
        supported.center,
        supported.viewingDirection.clone().negate(),
        manifold,
      ).inside,
    ).toBe(false);
  });

  it("projects deterministically to a dominant captured pose", () => {
    const detached = manifold.detachedCameras[0];
    const projected = projectCameraToSupportedPose(
      detached.center,
      detached.viewingDirection,
      manifold,
    );
    expect(manifold.cameras).toContain(projected);
    expect(manifold.detachedCameras).not.toContain(projected);

    const left = camera(0, 10, [-1, 0, 0]);
    const right = camera(1, 11, [1, 0, 0]);
    const tied: SupportedCameraManifold = {
      cameras: [left, right],
      detachedCameras: [],
      positionTolerance: 2,
      orientationToleranceRadians: Math.PI,
      defaultCameraIndex: 0,
    };
    expect(
      projectCameraToSupportedPose(
        new Vector3(0, 0, 0),
        new Vector3(0, 0, -1),
        tied,
      ).registeredIndex,
    ).toBe(0);
  });

  it("clamps navigation and resets to the known-good COLMAP 1 pose", () => {
    expect(clampSupportedCameraIndex(-10, manifold)).toBe(0);
    expect(clampSupportedCameraIndex(10_000, manifold)).toBe(204);
    expect(clampSupportedCameraIndex(12.9, manifold)).toBe(12);
    const resetIndex = resetSupportedCameraIndex(manifold);
    expect(manifold.cameras[resetIndex].colmapImageId).toBe(1);
  });

  it("transforms only camera state and does not mutate analytical or alignment inputs", () => {
    const alignment: EnvironmentAlignment = {
      translationXyz: [1, 2, 3],
      rotationXyzw: [0, 0, 0, 1],
      uniformScale: 2,
      calibrated: false,
      disclosure: "Uncalibrated test alignment.",
    };
    const alignmentBefore = structuredClone(alignment);
    const analyticalPoint = [0.2, -0.1, 0.3] as const;
    const scenePosition = new Vector3(3, 4, 5);
    const sceneDirection = new Vector3(0, 0, -1);
    const state = reconstructionCameraStateFromScene(
      scenePosition,
      sceneDirection,
      alignment,
    );
    expect(state.position.toArray()).toEqual([1, 1, 1]);
    expect(sceneFromReconstructionMatrix(alignment).elements).toHaveLength(16);
    expect(alignment).toEqual(alignmentBefore);
    expect(analyticalPoint).toEqual([0.2, -0.1, 0.3]);
    expect(scenePosition.toArray()).toEqual([3, 4, 5]);
    expect(sceneDirection.toArray()).toEqual([0, 0, -1]);
  });

  it("enforces real-scan viewing in normal behavior while debug data stays development-only", () => {
    const request = {
      visible: true,
      manifest: { provenance: { sourceKind: "real-scan" } },
    } as ValidatedEnvironmentRenderRequest;
    expect(shouldConstrainGaussianViewing(request)).toBe(true);
    expect(
      shouldConstrainGaussianViewing({
        ...request,
        manifest: {
          ...request.manifest,
          provenance: {
            ...request.manifest.provenance,
            sourceKind: "synthetic-test",
          },
        },
      }),
    ).toBe(false);
    expect(isViewingConstraintDebugEnabled("production", "?environmentViewingDebug=1")).toBe(
      false,
    );
    expect(isViewingConstraintDebugEnabled("development", "?environmentViewingDebug=1")).toBe(
      true,
    );
  });

  it("rejects camera support that does not match the reviewed staged identity", async () => {
    const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe(
        "https://atlas.example/environment-data/__local-real__/registered-cameras.json",
      );
      expect(init).toMatchObject({ cache: "no-store", redirect: "error" });
      return new Response("{}", {
        headers: { "content-length": "2" },
        status: 200,
      });
    };
    await expect(
      loadReviewedCameraManifold(
        new AbortController().signal,
        "https://atlas.example",
        fetcher as typeof fetch,
      ),
    ).rejects.toThrow(/identity is not reviewed/);
  });
});
