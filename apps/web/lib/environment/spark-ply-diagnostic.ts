import { Box3, MathUtils, Matrix4, Quaternion, Sphere, Vector3 } from "three";

export type DiagnosticBoundsTuple = readonly [
  readonly [number, number, number],
  readonly [number, number, number],
];

export type Vector3Tuple = readonly [number, number, number];
export type QuaternionWxyzTuple = readonly [number, number, number, number];
export type DataparserTransformTuple = readonly [
  readonly [number, number, number, number],
  readonly [number, number, number, number],
  readonly [number, number, number, number],
];

export interface ColmapWorldToCameraPose {
  quaternionWxyz: QuaternionWxyzTuple;
  translation: Vector3Tuple;
}

export const SPARK_PLY_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-original.ply";
export const SPARK_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/omprakash-workcell.spz";
export const SPARK_UNIFORM_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-uniform-100k.spz";
export const SPARK_CAMERA_AWARE_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-camera-aware-100k.spz";
export const SPARK_HYBRID_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-hybrid-100k.spz";
export const SPARK_UNIFORM_150K_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-uniform-150k.spz";
export const SPARK_UNIFORM_200K_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-uniform-200k.spz";
export const SPARK_UNIFORM_250K_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-uniform-250k.spz";
export const SPARK_UNIFORM_300K_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-uniform-300k.spz";
export const SPARK_CANDIDATE_FULL_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-candidate-full.spz";
export const SPARK_FULL_ORIGINAL_SPZ_DIAGNOSTIC_URL =
  "/environment-data/__local-real__/diagnostic-full-original.spz";

export type DiagnosticSplatSource =
  | "ply"
  | "spz"
  | "uniform"
  | "cameraAware"
  | "hybrid"
  | "uniform150k"
  | "uniform200k"
  | "uniform250k"
  | "uniform300k"
  | "candidateFull"
  | "fullOriginal";

export interface DiagnosticSplatSourceDefinition {
  source: DiagnosticSplatSource;
  label: string;
  sourceType: "PLY" | "SPZ";
  url: string;
  fileName: string;
  expectedSplatCount: number;
  byteSize: number;
  selectionMethod: string;
}

export const DIAGNOSTIC_SPLAT_SOURCES: Readonly<
  Record<DiagnosticSplatSource, DiagnosticSplatSourceDefinition>
> = {
  ply: {
    source: "ply",
    label: "Original PLY",
    sourceType: "PLY",
    url: SPARK_PLY_DIAGNOSTIC_URL,
    fileName: "diagnostic-original.ply",
    expectedSplatCount: 466_363,
    byteSize: 115_659_617,
    selectionMethod: "Full original reconstruction",
  },
  spz: {
    source: "spz",
    label: "Current voxel SPZ",
    sourceType: "SPZ",
    url: SPARK_SPZ_DIAGNOSTIC_URL,
    fileName: "omprakash-workcell.spz",
    expectedSplatCount: 86_954,
    byteSize: 1_410_503,
    selectionMethod: "One representative per occupied 11 mm voxel",
  },
  uniform: {
    source: "uniform",
    label: "Uniform sample SPZ",
    sourceType: "SPZ",
    url: SPARK_UNIFORM_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-uniform-100k.spz",
    expectedSplatCount: 100_000,
    byteSize: 1_600_226,
    selectionMethod: "Deterministic uniform candidate sample",
  },
  cameraAware: {
    source: "cameraAware",
    label: "Camera-aware SPZ",
    sourceType: "SPZ",
    url: SPARK_CAMERA_AWARE_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-camera-aware-100k.spz",
    expectedSplatCount: 100_000,
    byteSize: 1_570_440,
    selectionMethod: "Scale-stratified 210-camera importance ranking",
  },
  hybrid: {
    source: "hybrid",
    label: "Hybrid SPZ",
    sourceType: "SPZ",
    url: SPARK_HYBRID_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-hybrid-100k.spz",
    expectedSplatCount: 100_000,
    byteSize: 1_597_288,
    selectionMethod: "80k camera-aware plus 20k uncovered-cell coverage",
  },
  uniform150k: {
    source: "uniform150k",
    label: "Uniform 150k SPZ",
    sourceType: "SPZ",
    url: SPARK_UNIFORM_150K_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-uniform-150k.spz",
    expectedSplatCount: 150_000,
    byteSize: 2_397_011,
    selectionMethod: "Deterministic uniform viable-candidate sample",
  },
  uniform200k: {
    source: "uniform200k",
    label: "Uniform 200k SPZ",
    sourceType: "SPZ",
    url: SPARK_UNIFORM_200K_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-uniform-200k.spz",
    expectedSplatCount: 200_000,
    byteSize: 3_193_032,
    selectionMethod: "Deterministic uniform viable-candidate sample",
  },
  uniform250k: {
    source: "uniform250k",
    label: "Uniform 250k SPZ",
    sourceType: "SPZ",
    url: SPARK_UNIFORM_250K_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-uniform-250k.spz",
    expectedSplatCount: 250_000,
    byteSize: 3_990_074,
    selectionMethod: "Deterministic uniform viable-candidate sample",
  },
  uniform300k: {
    source: "uniform300k",
    label: "Uniform 300k SPZ",
    sourceType: "SPZ",
    url: SPARK_UNIFORM_300K_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-uniform-300k.spz",
    expectedSplatCount: 300_000,
    byteSize: 4_785_949,
    selectionMethod: "Deterministic uniform viable-candidate sample",
  },
  candidateFull: {
    source: "candidateFull",
    label: "Candidate-full SPZ",
    sourceType: "SPZ",
    url: SPARK_CANDIDATE_FULL_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-candidate-full.spz",
    expectedSplatCount: 398_598,
    byteSize: 6_357_739,
    selectionMethod: "All viable cropped/filtered candidates",
  },
  fullOriginal: {
    source: "fullOriginal",
    label: "Full-original SPZ",
    sourceType: "SPZ",
    url: SPARK_FULL_ORIGINAL_SPZ_DIAGNOSTIC_URL,
    fileName: "diagnostic-full-original.spz",
    expectedSplatCount: 466_363,
    byteSize: 7_534_845,
    selectionMethod: "All original PLY Gaussians; no viability filter",
  },
};

interface DiagnosticSplatFileTypes<T> {
  PLY: T;
  SPZ: T;
}

export function diagnosticSplatMeshOptions<T>(
  source: DiagnosticSplatSource,
  fileTypes: DiagnosticSplatFileTypes<T>,
) {
  const definition = DIAGNOSTIC_SPLAT_SOURCES[source];
  return {
    url: definition.url,
    fileType: source === "ply" ? fileTypes.PLY : fileTypes.SPZ,
    fileName: definition.fileName,
    editable: false,
    raycastable: false,
    lod: false,
    enableLod: false,
    nonLod: true,
    paged: false,
  } as const;
}

export function validateDiagnosticSplatCount(
  source: DiagnosticSplatSource,
  decodedSplatCount: number,
): number {
  const expected = DIAGNOSTIC_SPLAT_SOURCES[source].expectedSplatCount;
  if (decodedSplatCount !== expected) {
    throw new Error(
      `Decoded ${source.toUpperCase()} splat count ${decodedSplatCount.toLocaleString()} does not match expected ${expected.toLocaleString()}.`,
    );
  }
  return decodedSplatCount;
}

// These are the 2.5%-97.5% PLY center quantiles recorded by the existing
// reduction metadata. They remain in reconstruction coordinates and are used
// only to frame the diagnostic camera.
export const KNOWN_PLY_CORE_BOUNDS: DiagnosticBoundsTuple = [
  [-1.0733494877815244, 0.014625570131465792, -0.8045600205659861],
  [1.2637607514858251, 2.898145306110383, 0.7109479665756231],
];

// Nerfstudio 1.1.5 orders COLMAP frames by image ID and uses indices divisible
// by eval_interval (8) for evaluation. Image ID 1 is therefore eval image 0.
// Nerfstudio writes [ground truth | prediction] for splatfacto eval images; the
// left half of eval_img_0000.png corresponds to frame_0003.jpg (also confirmed
// directly against the local image pixels).
export const NERFSTUDIO_EVAL_CAMERA = {
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
  near: 0.01,
  far: 1e10,
  colmapWorldToCamera: {
    quaternionWxyz: [
      0.9827840484292055, 0.03912733252069382, 0.16594013294405915,
      0.07119296511190046,
    ] as const,
    translation: [
      -3.0965993107369263, -1.1701733624306159, 5.649283705290374,
    ] as const,
  },
  dataparserTransform: [
    [
      0.9987677931785583, -0.0496276393532753, 0.00014738501340616494,
      -0.09720537066459656,
    ],
    [
      0.00014738501340616494, 0.00593592319637537, 0.9999823570251465,
      0.3238338828086853,
    ],
    [
      -0.0496276393532753, -0.9987501502037048, 0.00593592319637537,
      -0.029281016439199448,
    ],
  ] as const,
  dataparserScale: 0.13791457110789326,
} as const;

export function nerfstudioEvalCameraAspect(): number {
  return NERFSTUDIO_EVAL_CAMERA.width / NERFSTUDIO_EVAL_CAMERA.height;
}

export function nerfstudioEvalCameraVerticalFovDegrees(): number {
  return pinholeVerticalFovDegrees(
    NERFSTUDIO_EVAL_CAMERA.height,
    NERFSTUDIO_EVAL_CAMERA.fy,
  );
}

export function pinholeVerticalFovDegrees(height: number, fy: number): number {
  return MathUtils.radToDeg(2 * Math.atan(height / (2 * fy)));
}

/**
 * Reproduces Nerfstudio 1.1.5's COLMAP camera conversion in the coordinate
 * system of the exported PLY. Both Nerfstudio and Three cameras use local -Z
 * as the viewing direction after the OpenCV-to-OpenGL camera-axis conversion.
 */
export function colmapCameraToNerfstudioWorldMatrix(
  { quaternionWxyz, translation }: ColmapWorldToCameraPose,
  dataparserTransformRows: DataparserTransformTuple,
  dataparserScale: number,
): Matrix4 {
  const [w, x, y, z] = quaternionWxyz;
  const worldToCamera = new Matrix4().compose(
    new Vector3(...translation),
    new Quaternion(x, y, z, w),
    new Vector3(1, 1, 1),
  );
  const cameraToColmapWorld = worldToCamera
    .invert()
    .multiply(new Matrix4().makeScale(1, -1, -1));
  const [row0, row1, row2] = dataparserTransformRows;
  const dataparserTransform = new Matrix4().set(
    ...row0,
    ...row1,
    ...row2,
    0,
    0,
    0,
    1,
  );
  const cameraToNerfstudioWorld = dataparserTransform.multiply(
    cameraToColmapWorld,
  );

  cameraToNerfstudioWorld.setPosition(
    new Vector3().setFromMatrixPosition(cameraToNerfstudioWorld).multiplyScalar(
      dataparserScale,
    ),
  );
  return cameraToNerfstudioWorld;
}

export function nerfstudioEvalCameraToWorldMatrix(): Matrix4 {
  return colmapCameraToNerfstudioWorldMatrix(
    NERFSTUDIO_EVAL_CAMERA.colmapWorldToCamera,
    NERFSTUDIO_EVAL_CAMERA.dataparserTransform,
    NERFSTUDIO_EVAL_CAMERA.dataparserScale,
  );
}

export interface PerspectiveCameraFit {
  target: Vector3;
  position: Vector3;
  radius: number;
  distance: number;
  near: number;
  far: number;
}

export function boxFromDiagnosticBounds(bounds: DiagnosticBoundsTuple): Box3 {
  return new Box3(
    new Vector3(...bounds[0]),
    new Vector3(...bounds[1]),
  );
}

export function diagnosticBoundsFromBox(box: Box3): DiagnosticBoundsTuple {
  return [box.min.toArray(), box.max.toArray()];
}

export function calculatePerspectiveCameraFit({
  bounds,
  verticalFovDegrees,
  aspect,
  distanceMultiplier,
  nearMultiplier,
  farMultiplier,
}: {
  bounds: Box3;
  verticalFovDegrees: number;
  aspect: number;
  distanceMultiplier: number;
  nearMultiplier: number;
  farMultiplier: number;
}): PerspectiveCameraFit {
  if (bounds.isEmpty()) throw new Error("Cannot frame empty PLY bounds.");

  const sphere = bounds.getBoundingSphere(new Sphere());
  if (
    !Number.isFinite(sphere.radius) ||
    sphere.radius <= 0 ||
    !Number.isFinite(aspect) ||
    aspect <= 0
  ) {
    throw new Error("Cannot frame invalid PLY bounds or viewport.");
  }

  const verticalHalfFov = MathUtils.degToRad(verticalFovDegrees / 2);
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * aspect);
  const limitingHalfFov = Math.min(verticalHalfFov, horizontalHalfFov);
  const radius = sphere.radius;
  const distance =
    (radius / Math.sin(limitingHalfFov)) * 1.15 * distanceMultiplier;
  const target = sphere.center.clone();
  const viewDirection = new Vector3(1, -1, 0.65).normalize();
  const position = target.clone().addScaledVector(viewDirection, distance);
  const near = Math.max(radius * 0.0001, radius * 0.01 * nearMultiplier);
  const far = Math.max(near * 2, (distance + radius * 2) * farMultiplier);

  return { target, position, radius, distance, near, far };
}
