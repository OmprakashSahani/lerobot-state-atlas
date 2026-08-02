import type { AvailableProjection, Vector3 } from "./types";

export interface TrajectorySceneBounds {
  center: Vector3;
  minimum: Vector3;
  maximum: Vector3;
  extent: number;
  gridSize: number;
  markerRadius: number;
  orientationSize: number;
  gripperSize: number;
  dashSize: number;
  gapSize: number;
}

const MINIMUM_EXTENT = 0.3;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function computeVisibleTrajectoryBounds(
  projection: AvailableProjection,
  visibility: readonly [boolean, boolean],
): TrajectorySceneBounds {
  const points = projection.plans.flatMap((plan, index) =>
    visibility[index] ? [...plan.left.positionsXyz, ...plan.right.positionsXyz] : [],
  );
  const finitePoints = points.filter((point) => point.every(Number.isFinite));
  if (finitePoints.length === 0) {
    return {
      center: [0, 0, 0],
      minimum: [-MINIMUM_EXTENT / 2, -MINIMUM_EXTENT / 2, -MINIMUM_EXTENT / 2],
      maximum: [MINIMUM_EXTENT / 2, MINIMUM_EXTENT / 2, MINIMUM_EXTENT / 2],
      extent: MINIMUM_EXTENT,
      gridSize: 0.6,
      markerRadius: 0.018,
      orientationSize: 0.09,
      gripperSize: 0.022,
      dashSize: 0.018,
      gapSize: 0.012,
    };
  }

  const minimum: Vector3 = [...finitePoints[0]];
  const maximum: Vector3 = [...finitePoints[0]];
  for (const point of finitePoints.slice(1)) {
    for (let axis = 0; axis < 3; axis += 1) {
      minimum[axis] = Math.min(minimum[axis], point[axis]);
      maximum[axis] = Math.max(maximum[axis], point[axis]);
    }
  }
  const center = minimum.map(
    (value, axis) => (value + maximum[axis]) / 2,
  ) as Vector3;
  const rawExtent = Math.max(...maximum.map((value, axis) => value - minimum[axis]));
  const extent = Math.max(MINIMUM_EXTENT, rawExtent);
  return {
    center,
    minimum,
    maximum,
    extent,
    gridSize: clamp(extent * 1.8, 0.6, 24),
    markerRadius: clamp(extent * 0.045, 0.018, 0.08),
    orientationSize: clamp(extent * 0.22, 0.09, 0.4),
    gripperSize: clamp(extent * 0.055, 0.022, 0.1),
    dashSize: clamp(extent * 0.065, 0.018, 0.12),
    gapSize: clamp(extent * 0.04, 0.012, 0.08),
  };
}
