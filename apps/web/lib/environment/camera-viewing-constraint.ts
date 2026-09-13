import { MathUtils, Matrix4, Quaternion, Vector3 } from "three";

import {
  analyzeCameraManifold,
  convertRegisteredCameras,
  decodeRegisteredCameraDataset,
  REGISTERED_CAMERAS_URL,
  sourceFrameNumber,
  type ConvertedRegisteredCamera,
} from "./camera-manifold";
import { sha256Hex } from "./load-asset";
import { readResponseBytes } from "./load-manifest";
import { assertSafeFinalResponseUrl } from "./path-safety";
import {
  colmapCameraToNerfstudioWorldMatrix,
  NERFSTUDIO_EVAL_CAMERA,
} from "./spark-ply-diagnostic";
import type {
  EnvironmentAlignment,
  ValidatedEnvironmentRenderRequest,
} from "./types";

export const REVIEWED_REGISTERED_CAMERA_COUNT = 210;
export const REVIEWED_DOMINANT_CAMERA_COUNT = 205;
export const REVIEWED_REGISTERED_CAMERA_BYTES = 130_912;
export const REVIEWED_REGISTERED_CAMERA_SHA256 =
  "49077a800713ef38070fd4ec547f73840373b914f548a0bc0e0029362a74f60a";
export const REVIEWED_DETACHED_COLMAP_IDS = [197, 198, 199, 200, 201] as const;
export const DEFAULT_SUPPORTED_COLMAP_IMAGE_ID = 1;

export interface SupportedCameraManifold {
  cameras: ConvertedRegisteredCamera[];
  detachedCameras: ConvertedRegisteredCamera[];
  positionTolerance: number;
  orientationToleranceRadians: number;
  defaultCameraIndex: number;
}

export interface CameraSupportAssessment {
  inside: boolean;
  nearestCamera: ConvertedRegisteredCamera;
  distance: number;
  orientationDifferenceRadians: number;
}

function quantile(values: readonly number[], fraction: number): number {
  if (values.length === 0) return 0;
  const sorted = values.toSorted((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const interpolation = position - lower;
  return (
    sorted[lower] +
    (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) *
      interpolation
  );
}

export function buildSupportedCameraManifold(
  cameras: ConvertedRegisteredCamera[],
): SupportedCameraManifold {
  const stats = analyzeCameraManifold(cameras);
  const supported = [...stats.neighborhoodComponents[0]].toSorted(
    (left, right) =>
      sourceFrameNumber(left) - sourceFrameNumber(right) ||
      left.colmapImageId - right.colmapImageId,
  );
  const detached = stats.neighborhoodComponents
    .slice(1)
    .flat()
    .toSorted((left, right) => left.colmapImageId - right.colmapImageId);
  const localOrientationSteps = supported.slice(1).flatMap((camera, index) => {
    const previous = supported[index];
    return previous.center.distanceTo(camera.center) <= stats.neighborhoodRadius
      ? [previous.viewingDirection.angleTo(camera.viewingDirection)]
      : [];
  });
  const orientationToleranceRadians = Math.min(
    MathUtils.degToRad(20),
    Math.max(
      MathUtils.degToRad(5),
      quantile(localOrientationSteps, 0.95) * 1.25,
    ),
  );
  const defaultCameraIndex = Math.max(
    0,
    supported.findIndex(
      (camera) => camera.colmapImageId === DEFAULT_SUPPORTED_COLMAP_IMAGE_ID,
    ),
  );
  return {
    cameras: supported,
    detachedCameras: detached,
    positionTolerance: stats.neighborhoodRadius,
    orientationToleranceRadians,
    defaultCameraIndex,
  };
}

export function assertReviewedCameraManifold(
  allCameras: readonly ConvertedRegisteredCamera[],
  manifold: SupportedCameraManifold,
): void {
  if (
    allCameras.length !== REVIEWED_REGISTERED_CAMERA_COUNT ||
    manifold.cameras.length !== REVIEWED_DOMINANT_CAMERA_COUNT ||
    manifold.detachedCameras.length !== REVIEWED_DETACHED_COLMAP_IDS.length ||
    manifold.detachedCameras.some(
      (camera, index) =>
        camera.colmapImageId !== REVIEWED_DETACHED_COLMAP_IDS[index],
    )
  ) {
    throw new Error(
      "Registered-camera data does not match the reviewed 205 + 5 manifold.",
    );
  }
}

export function assessCameraSupport(
  position: Vector3,
  viewingDirection: Vector3,
  manifold: SupportedCameraManifold,
): CameraSupportAssessment {
  if (manifold.cameras.length === 0) {
    throw new Error("Supported camera manifold contains no cameras.");
  }
  let nearestCamera = manifold.cameras[0];
  let distance = position.distanceTo(nearestCamera.center);
  for (const camera of manifold.cameras.slice(1)) {
    const candidateDistance = position.distanceTo(camera.center);
    if (
      candidateDistance < distance ||
      (candidateDistance === distance &&
        camera.registeredIndex < nearestCamera.registeredIndex)
    ) {
      nearestCamera = camera;
      distance = candidateDistance;
    }
  }
  const normalizedDirection = viewingDirection.clone().normalize();
  const orientationDifferenceRadians = normalizedDirection.angleTo(
    nearestCamera.viewingDirection,
  );
  return {
    inside:
      distance <= manifold.positionTolerance &&
      orientationDifferenceRadians <= manifold.orientationToleranceRadians,
    nearestCamera,
    distance,
    orientationDifferenceRadians,
  };
}

export function projectCameraToSupportedPose(
  position: Vector3,
  viewingDirection: Vector3,
  manifold: SupportedCameraManifold,
): ConvertedRegisteredCamera {
  return assessCameraSupport(position, viewingDirection, manifold).nearestCamera;
}

export function clampSupportedCameraIndex(
  index: number,
  manifold: SupportedCameraManifold,
): number {
  return Math.min(
    manifold.cameras.length - 1,
    Math.max(0, Math.trunc(Number.isFinite(index) ? index : 0)),
  );
}

export function resetSupportedCameraIndex(
  manifold: SupportedCameraManifold,
): number {
  return manifold.defaultCameraIndex;
}

export function sceneFromReconstructionMatrix(
  alignment: EnvironmentAlignment,
): Matrix4 {
  return new Matrix4().compose(
    new Vector3(...alignment.translationXyz),
    new Quaternion(...alignment.rotationXyzw),
    new Vector3(
      alignment.uniformScale,
      alignment.uniformScale,
      alignment.uniformScale,
    ),
  );
}

export function reconstructionCameraStateFromScene(
  scenePosition: Vector3,
  sceneViewingDirection: Vector3,
  alignment: EnvironmentAlignment,
): { position: Vector3; viewingDirection: Vector3 } {
  const sceneFromReconstruction = sceneFromReconstructionMatrix(alignment);
  const reconstructionFromScene = sceneFromReconstruction.clone().invert();
  const inverseRotation = new Quaternion(
    ...alignment.rotationXyzw,
  ).invert();
  return {
    position: scenePosition.clone().applyMatrix4(reconstructionFromScene),
    viewingDirection: sceneViewingDirection
      .clone()
      .applyQuaternion(inverseRotation)
      .normalize(),
  };
}

export function knownGoodCapturedCamera(): ConvertedRegisteredCamera {
  const cameraToWorld = colmapCameraToNerfstudioWorldMatrix(
    NERFSTUDIO_EVAL_CAMERA.colmapWorldToCamera,
    NERFSTUDIO_EVAL_CAMERA.dataparserTransform,
    NERFSTUDIO_EVAL_CAMERA.dataparserScale,
  );
  return {
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
    evalIndex: NERFSTUDIO_EVAL_CAMERA.evalIndex,
    evalPsnr: 28.059858,
    cameraToWorld,
    center: new Vector3().setFromMatrixPosition(cameraToWorld),
    viewingDirection: new Vector3(0, 0, -1).transformDirection(cameraToWorld),
  };
}

export function isViewingConstraintDebugEnabled(
  nodeEnv: string | undefined,
  search: string,
): boolean {
  return (
    nodeEnv === "development" &&
    new URLSearchParams(search).get("environmentViewingDebug") === "1"
  );
}

export function shouldConstrainGaussianViewing(
  request: ValidatedEnvironmentRenderRequest | null,
): boolean {
  return Boolean(
    request?.visible && request.manifest.provenance.sourceKind === "real-scan",
  );
}

export async function loadReviewedCameraManifold(
  signal: AbortSignal,
  origin: string,
  fetcher: typeof fetch = fetch,
): Promise<SupportedCameraManifold> {
  const response = await fetcher(new URL(REGISTERED_CAMERAS_URL, origin), {
    cache: "no-store",
    redirect: "error",
    signal,
  });
  assertSafeFinalResponseUrl(response, REGISTERED_CAMERAS_URL, origin);
  const bytes = await readResponseBytes(
    response,
    REVIEWED_REGISTERED_CAMERA_BYTES,
  );
  if (
    bytes.byteLength !== REVIEWED_REGISTERED_CAMERA_BYTES ||
    (await sha256Hex(bytes)) !== REVIEWED_REGISTERED_CAMERA_SHA256
  ) {
    throw new Error("Registered-camera support data identity is not reviewed.");
  }
  let dataset: unknown;
  try {
    dataset = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    );
  } catch {
    throw new Error("Registered-camera support data is not valid UTF-8 JSON.");
  }
  const cameras = convertRegisteredCameras(
    decodeRegisteredCameraDataset(dataset),
  );
  const manifold = buildSupportedCameraManifold(cameras);
  assertReviewedCameraManifold(cameras, manifold);
  return manifold;
}
