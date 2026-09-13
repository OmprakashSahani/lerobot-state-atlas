import { Box3, Matrix4, PerspectiveCamera, Vector3 } from "three";

import {
  colmapCameraToNerfstudioWorldMatrix,
  diagnosticBoundsFromBox,
  NERFSTUDIO_EVAL_CAMERA,
  pinholeVerticalFovDegrees,
  type DataparserTransformTuple,
  type DiagnosticBoundsTuple,
  type QuaternionWxyzTuple,
  type Vector3Tuple,
} from "./spark-ply-diagnostic";

export const REGISTERED_CAMERAS_URL =
  "/environment-data/__local-real__/registered-cameras.json";
export const REGISTERED_CAMERAS_DIAGNOSTIC_URL = REGISTERED_CAMERAS_URL;
export const EVAL_RENDERS_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/eval-renders";

export interface RegisteredCamera {
  registeredIndex: number;
  colmapImageId: number;
  sourceImage: string;
  cameraId: number;
  cameraModel: "PINHOLE";
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  quaternionWxyz: QuaternionWxyzTuple;
  translation: Vector3Tuple;
  evalIndex: number | null;
  evalPsnr: number | null;
}

export interface RegisteredCameraDataset {
  schemaVersion: 1;
  source: {
    colmapModel: string;
    dataparserTransform: string;
    evalInterval: number;
    transform: DataparserTransformTuple;
    scale: number;
  };
  cameras: RegisteredCamera[];
}

export interface ConvertedRegisteredCamera extends RegisteredCamera {
  cameraToWorld: Matrix4;
  center: Vector3;
  viewingDirection: Vector3;
}

export interface CameraTrajectoryGap {
  distance: number;
  from: ConvertedRegisteredCamera;
  to: ConvertedRegisteredCamera;
}

export interface CameraManifoldStats {
  centerBounds: DiagnosticBoundsTuple;
  extent: Vector3Tuple;
  diagonal: number;
  orderedPathLength: number;
  medianStep: number;
  p95Step: number;
  maximumStep: CameraTrajectoryGap;
  medianNearestNeighbor: number;
  maximumMstBridge: CameraTrajectoryGap;
  neighborhoodRadius: number;
  neighborhoodComponents: ConvertedRegisteredCamera[][];
  largeTrajectoryGaps: CameraTrajectoryGap[];
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return value;
}

function integer(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (!Number.isInteger(result)) throw new Error(`${label} must be an integer.`);
  return result;
}

function tuple(
  value: unknown,
  length: number,
  label: string,
): readonly number[] {
  if (!Array.isArray(value) || value.length !== length) {
    throw new Error(`${label} must contain ${length} values.`);
  }
  return value.map((item, index) => finiteNumber(item, `${label}[${index}]`));
}

export function decodeRegisteredCameraDataset(
  value: unknown,
): RegisteredCameraDataset {
  const root = record(value, "camera dataset");
  if (root.schemaVersion !== 1) {
    throw new Error("Unsupported registered-camera schema version.");
  }
  const sourceValue = record(root.source, "camera dataset source");
  if (!Array.isArray(sourceValue.transform) || sourceValue.transform.length !== 3) {
    throw new Error("Dataparser transform must contain three rows.");
  }
  const transform = sourceValue.transform.map((row, index) =>
    tuple(row, 4, `dataparser transform row ${index}`),
  ) as unknown as DataparserTransformTuple;
  if (!Array.isArray(root.cameras) || root.cameras.length === 0) {
    throw new Error("Registered-camera dataset contains no cameras.");
  }

  const cameras = root.cameras.map((item, index): RegisteredCamera => {
    const camera = record(item, `camera ${index}`);
    const registeredIndex = integer(
      camera.registeredIndex,
      `camera ${index} registeredIndex`,
    );
    if (registeredIndex !== index) {
      throw new Error(`Camera ${index} is not in registered-index order.`);
    }
    if (camera.cameraModel !== "PINHOLE") {
      throw new Error(`Camera ${index} is not an undistorted PINHOLE camera.`);
    }
    if (typeof camera.sourceImage !== "string" || !camera.sourceImage) {
      throw new Error(`Camera ${index} has no source image.`);
    }
    const evalIndex =
      camera.evalIndex === null
        ? null
        : integer(camera.evalIndex, `camera ${index} evalIndex`);
    const evalPsnr =
      camera.evalPsnr === null
        ? null
        : finiteNumber(camera.evalPsnr, `camera ${index} evalPsnr`);
    if ((evalIndex === null) !== (evalPsnr === null)) {
      throw new Error(`Camera ${index} has incomplete eval metadata.`);
    }
    return {
      registeredIndex,
      colmapImageId: integer(
        camera.colmapImageId,
        `camera ${index} colmapImageId`,
      ),
      sourceImage: camera.sourceImage,
      cameraId: integer(camera.cameraId, `camera ${index} cameraId`),
      cameraModel: "PINHOLE",
      width: integer(camera.width, `camera ${index} width`),
      height: integer(camera.height, `camera ${index} height`),
      fx: finiteNumber(camera.fx, `camera ${index} fx`),
      fy: finiteNumber(camera.fy, `camera ${index} fy`),
      cx: finiteNumber(camera.cx, `camera ${index} cx`),
      cy: finiteNumber(camera.cy, `camera ${index} cy`),
      quaternionWxyz: tuple(
        camera.quaternionWxyz,
        4,
        `camera ${index} quaternion`,
      ) as QuaternionWxyzTuple,
      translation: tuple(
        camera.translation,
        3,
        `camera ${index} translation`,
      ) as Vector3Tuple,
      evalIndex,
      evalPsnr,
    };
  });

  const ids = new Set(cameras.map((camera) => camera.colmapImageId));
  if (ids.size !== cameras.length) {
    throw new Error("Registered-camera COLMAP image IDs are not unique.");
  }

  return {
    schemaVersion: 1,
    source: {
      colmapModel: String(sourceValue.colmapModel),
      dataparserTransform: String(sourceValue.dataparserTransform),
      evalInterval: integer(sourceValue.evalInterval, "eval interval"),
      transform,
      scale: finiteNumber(sourceValue.scale, "dataparser scale"),
    },
    cameras,
  };
}

export function convertRegisteredCameras(
  dataset: RegisteredCameraDataset,
): ConvertedRegisteredCamera[] {
  return dataset.cameras.map((camera) => {
    const cameraToWorld = colmapCameraToNerfstudioWorldMatrix(
      camera,
      dataset.source.transform,
      dataset.source.scale,
    );
    return {
      ...camera,
      cameraToWorld,
      center: new Vector3().setFromMatrixPosition(cameraToWorld),
      viewingDirection: new Vector3(0, 0, -1).transformDirection(cameraToWorld),
    };
  });
}

export function applyRegisteredCamera(
  camera: PerspectiveCamera,
  registeredCamera: ConvertedRegisteredCamera,
  sceneFromReconstruction = new Matrix4(),
): void {
  const cameraToScene = registeredCamera.cameraToWorld
    .clone()
    .premultiply(sceneFromReconstruction);
  cameraToScene.decompose(camera.position, camera.quaternion, camera.scale);
  camera.scale.set(1, 1, 1);
  camera.fov = pinholeVerticalFovDegrees(
    registeredCamera.height,
    registeredCamera.fy,
  );
  camera.aspect = registeredCamera.width / registeredCamera.height;
  camera.near = NERFSTUDIO_EVAL_CAMERA.near;
  camera.far = NERFSTUDIO_EVAL_CAMERA.far;
  camera.setViewOffset(
    registeredCamera.width,
    registeredCamera.height,
    registeredCamera.width / 2 - registeredCamera.cx,
    registeredCamera.height / 2 - registeredCamera.cy,
    registeredCamera.width,
    registeredCamera.height,
  );
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
}

function quantile(values: number[], fraction: number): number {
  if (values.length === 0) throw new Error("Cannot summarize no values.");
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

export function sourceFrameNumber(camera: RegisteredCamera): number {
  const match = camera.sourceImage.match(/frame_(\d+)\.[^.]+$/);
  return match ? Number(match[1]) : camera.registeredIndex;
}

function minimumSpanningTreeMaximumBridge(
  cameras: ConvertedRegisteredCamera[],
): CameraTrajectoryGap {
  const included = new Set([0]);
  let maximum: CameraTrajectoryGap | undefined;
  while (included.size < cameras.length) {
    let shortest: CameraTrajectoryGap | undefined;
    for (const fromIndex of included) {
      for (let toIndex = 0; toIndex < cameras.length; toIndex += 1) {
        if (included.has(toIndex)) continue;
        const candidate = {
          distance: cameras[fromIndex].center.distanceTo(cameras[toIndex].center),
          from: cameras[fromIndex],
          to: cameras[toIndex],
        };
        if (!shortest || candidate.distance < shortest.distance) {
          shortest = candidate;
        }
      }
    }
    if (!shortest) throw new Error("Could not connect registered camera centers.");
    included.add(shortest.to.registeredIndex);
    if (!maximum || shortest.distance > maximum.distance) maximum = shortest;
  }
  if (!maximum) throw new Error("At least two registered cameras are required.");
  return maximum;
}

function radiusComponents(
  cameras: ConvertedRegisteredCamera[],
  radius: number,
): ConvertedRegisteredCamera[][] {
  const visited = new Set<number>();
  const components: ConvertedRegisteredCamera[][] = [];
  for (let start = 0; start < cameras.length; start += 1) {
    if (visited.has(start)) continue;
    const component: ConvertedRegisteredCamera[] = [];
    const pending = [start];
    visited.add(start);
    while (pending.length > 0) {
      const currentIndex = pending.pop();
      if (currentIndex === undefined) break;
      component.push(cameras[currentIndex]);
      for (let candidate = 0; candidate < cameras.length; candidate += 1) {
        if (
          !visited.has(candidate) &&
          cameras[currentIndex].center.distanceTo(cameras[candidate].center) <=
            radius
        ) {
          visited.add(candidate);
          pending.push(candidate);
        }
      }
    }
    components.push(component);
  }
  return components.toSorted((left, right) => right.length - left.length);
}

export function analyzeCameraManifold(
  cameras: ConvertedRegisteredCamera[],
): CameraManifoldStats {
  if (cameras.length < 2) throw new Error("At least two cameras are required.");
  const ordered = cameras.toSorted(
    (left, right) => sourceFrameNumber(left) - sourceFrameNumber(right),
  );
  const gaps = ordered.slice(1).map((camera, index) => ({
    distance: ordered[index].center.distanceTo(camera.center),
    from: ordered[index],
    to: camera,
  }));
  const distances = gaps.map((gap) => gap.distance);
  const medianStep = quantile(distances, 0.5);
  const p95Step = quantile(distances, 0.95);
  const largeGapThreshold = Math.max(medianStep * 3, p95Step);
  const nearestNeighbors = cameras.map((camera, cameraIndex) =>
    Math.min(
      ...cameras.map((other, otherIndex) =>
        cameraIndex === otherIndex
          ? Number.POSITIVE_INFINITY
          : camera.center.distanceTo(other.center),
      ),
    ),
  );
  const bounds = new Box3().setFromPoints(cameras.map((camera) => camera.center));
  const extent = bounds.getSize(new Vector3());
  const medianNearestNeighbor = quantile(nearestNeighbors, 0.5);
  const neighborhoodRadius = medianNearestNeighbor * 4;

  return {
    centerBounds: diagnosticBoundsFromBox(bounds),
    extent: extent.toArray(),
    diagonal: extent.length(),
    orderedPathLength: distances.reduce((sum, distance) => sum + distance, 0),
    medianStep,
    p95Step,
    maximumStep: gaps.reduce((maximum, gap) =>
      gap.distance > maximum.distance ? gap : maximum,
    ),
    medianNearestNeighbor,
    maximumMstBridge: minimumSpanningTreeMaximumBridge(cameras),
    neighborhoodRadius,
    neighborhoodComponents: radiusComponents(cameras, neighborhoodRadius),
    largeTrajectoryGaps: gaps
      .filter((gap) => gap.distance > largeGapThreshold)
      .toSorted((left, right) => right.distance - left.distance),
  };
}

export function evalRenderUrl(camera: RegisteredCamera): string | null {
  if (camera.evalIndex === null) return null;
  return `${EVAL_RENDERS_DIAGNOSTIC_URL}/eval_img_${String(camera.evalIndex).padStart(4, "0")}.png`;
}
