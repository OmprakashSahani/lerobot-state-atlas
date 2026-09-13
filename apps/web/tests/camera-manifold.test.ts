import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { describe, expect, it } from "vitest";

import {
  analyzeCameraManifold,
  applyRegisteredCamera,
  convertRegisteredCameras,
  decodeRegisteredCameraDataset,
  evalRenderUrl,
  type ConvertedRegisteredCamera,
  type RegisteredCamera,
} from "@/lib/environment/camera-manifold";
import {
  NERFSTUDIO_EVAL_CAMERA,
  nerfstudioEvalCameraToWorldMatrix,
} from "@/lib/environment/spark-ply-diagnostic";

const knownCameraDataset = {
  schemaVersion: 1,
  source: {
    colmapModel: "gs-input-model0/sparse",
    dataparserTransform: "full-30000/dataparser_transforms.json",
    evalInterval: 8,
    transform: NERFSTUDIO_EVAL_CAMERA.dataparserTransform,
    scale: NERFSTUDIO_EVAL_CAMERA.dataparserScale,
  },
  cameras: [
    {
      registeredIndex: 0,
      colmapImageId: NERFSTUDIO_EVAL_CAMERA.colmapImageId,
      sourceImage: NERFSTUDIO_EVAL_CAMERA.sourceImage,
      cameraId: 1,
      cameraModel: "PINHOLE",
      width: NERFSTUDIO_EVAL_CAMERA.width,
      height: NERFSTUDIO_EVAL_CAMERA.height,
      fx: NERFSTUDIO_EVAL_CAMERA.fx,
      fy: NERFSTUDIO_EVAL_CAMERA.fy,
      cx: NERFSTUDIO_EVAL_CAMERA.cx,
      cy: NERFSTUDIO_EVAL_CAMERA.cy,
      quaternionWxyz:
        NERFSTUDIO_EVAL_CAMERA.colmapWorldToCamera.quaternionWxyz,
      translation: NERFSTUDIO_EVAL_CAMERA.colmapWorldToCamera.translation,
      evalIndex: 0,
      evalPsnr: 28.059858,
    },
  ],
};

describe("registered COLMAP camera dataset", () => {
  it("decodes and converts the known eval camera through the shared transform", () => {
    const dataset = decodeRegisteredCameraDataset(knownCameraDataset);
    const [camera] = convertRegisteredCameras(dataset);
    const knownMatrix = nerfstudioEvalCameraToWorldMatrix();

    expect(camera.sourceImage).toBe("frame_0003.jpg");
    expect(evalRenderUrl(camera)).toMatch(/eval_img_0000\.png$/);
    camera.cameraToWorld.elements.forEach((value, index) => {
      expect(value).toBeCloseTo(knownMatrix.elements[index], 12);
    });

    const perspective = new PerspectiveCamera();
    applyRegisteredCamera(perspective, camera);
    perspective.matrixWorld.elements.forEach((value, index) => {
      expect(value).toBeCloseTo(knownMatrix.elements[index], 7);
    });
    expect(perspective.aspect).toBeCloseTo(1903 / 1070, 12);
    expect(perspective.fov).toBeCloseTo(35.16365, 5);
    expect(perspective.near).toBe(NERFSTUDIO_EVAL_CAMERA.near);
    expect(perspective.far).toBe(NERFSTUDIO_EVAL_CAMERA.far);
  });

  it("rejects unordered cameras and incomplete eval evidence", () => {
    expect(() =>
      decodeRegisteredCameraDataset({
        ...knownCameraDataset,
        cameras: [
          {
            ...knownCameraDataset.cameras[0],
            registeredIndex: 1,
          },
        ],
      }),
    ).toThrow(/registered-index order/);
    expect(() =>
      decodeRegisteredCameraDataset({
        ...knownCameraDataset,
        cameras: [
          {
            ...knownCameraDataset.cameras[0],
            evalPsnr: null,
          },
        ],
      }),
    ).toThrow(/incomplete eval metadata/);
  });
});

function convertedCamera(
  registeredIndex: number,
  sourceFrame: number,
  center: [number, number, number],
): ConvertedRegisteredCamera {
  const raw: RegisteredCamera = {
    registeredIndex,
    colmapImageId: registeredIndex + 1,
    sourceImage: `frame_${String(sourceFrame).padStart(4, "0")}.jpg`,
    cameraId: 1,
    cameraModel: "PINHOLE",
    width: 100,
    height: 50,
    fx: 80,
    fy: 80,
    cx: 50,
    cy: 25,
    quaternionWxyz: [1, 0, 0, 0],
    translation: [0, 0, 0],
    evalIndex: null,
    evalPsnr: null,
  };
  return {
    ...raw,
    cameraToWorld: new Matrix4().setPosition(new Vector3(...center)),
    center: new Vector3(...center),
    viewingDirection: new Vector3(0, 0, -1),
  };
}

describe("captured camera manifold analysis", () => {
  it("reports bounds, ordered path length, and the bridge needed to connect centers", () => {
    const stats = analyzeCameraManifold([
      convertedCamera(0, 1, [0, 0, 0]),
      convertedCamera(1, 2, [1, 0, 0]),
      convertedCamera(2, 3, [1, 2, 0]),
    ]);

    expect(stats.centerBounds).toEqual([
      [0, 0, 0],
      [1, 2, 0],
    ]);
    expect(stats.extent).toEqual([1, 2, 0]);
    expect(stats.diagonal).toBeCloseTo(Math.sqrt(5));
    expect(stats.orderedPathLength).toBe(3);
    expect(stats.maximumStep.distance).toBe(2);
    expect(stats.maximumMstBridge.distance).toBe(2);
    expect(stats.medianNearestNeighbor).toBe(1);
    expect(stats.neighborhoodRadius).toBe(4);
    expect(stats.neighborhoodComponents.map((component) => component.length)).toEqual([
      3,
    ]);
  });
});
