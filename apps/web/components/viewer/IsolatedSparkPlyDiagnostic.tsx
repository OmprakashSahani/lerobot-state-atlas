"use client";

/* Three.js cameras and controls are intentionally updated through their
 * imperative API inside effects. */
/* eslint-disable react-hooks/immutability */
/* The local-only Nerfstudio eval montage is intentionally displayed directly. */
/* eslint-disable @next/next/no-img-element */

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import type { ChangeEvent } from "react";
import {
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  SphereGeometry,
  type Box3,
  type Object3D,
  type Scene,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SparkRenderer } from "@sparkjsdev/spark";

import {
  analyzeCameraManifold,
  applyRegisteredCamera,
  convertRegisteredCameras,
  decodeRegisteredCameraDataset,
  evalRenderUrl,
  REGISTERED_CAMERAS_DIAGNOSTIC_URL,
  type CameraManifoldStats,
  type ConvertedRegisteredCamera,
  type RegisteredCameraDataset,
} from "@/lib/environment/camera-manifold";
import {
  boxFromDiagnosticBounds,
  calculatePerspectiveCameraFit,
  DIAGNOSTIC_SPLAT_SOURCES,
  diagnosticSplatMeshOptions,
  diagnosticBoundsFromBox,
  KNOWN_PLY_CORE_BOUNDS,
  NERFSTUDIO_EVAL_CAMERA,
  nerfstudioEvalCameraVerticalFovDegrees,
  pinholeVerticalFovDegrees,
  validateDiagnosticSplatCount,
  type DiagnosticSplatSource,
  type DiagnosticBoundsTuple,
  type PerspectiveCameraFit,
} from "@/lib/environment/spark-ply-diagnostic";

import styles from "./IsolatedSparkPlyDiagnostic.module.css";

interface RendererSettings {
  focalAdjustment: number;
  maxPixelRadius: number;
  preBlurAmount: number;
  blurAmount: number;
  sortRadial: boolean;
}

interface CameraSettings {
  distanceMultiplier: number;
  nearMultiplier: number;
  farMultiplier: number;
}

type CameraMode = "registered" | "orbit";
type CameraAssessment = "coherent" | "weak";

export interface DiagnosticControlState {
  source: DiagnosticSplatSource;
  cameraMode: CameraMode;
  selectedCameraIndex: number;
  showCameraManifold: boolean;
}

export const DEFAULT_DIAGNOSTIC_CONTROL_STATE: DiagnosticControlState = {
  source: "ply",
  cameraMode: "registered",
  selectedCameraIndex: 0,
  showCameraManifold: false,
};

export type DiagnosticControlAction =
  | { type: "select-source"; source: DiagnosticSplatSource }
  | { type: "select-camera"; index: number }
  | { type: "set-camera-mode"; cameraMode: CameraMode }
  | { type: "set-manifold-visible"; visible: boolean };

export function diagnosticControlReducer(
  state: DiagnosticControlState,
  action: DiagnosticControlAction,
): DiagnosticControlState {
  switch (action.type) {
    case "select-source":
      return { ...state, source: action.source };
    case "select-camera":
      return {
        ...state,
        selectedCameraIndex: action.index,
        cameraMode: "registered",
      };
    case "set-camera-mode":
      return { ...state, cameraMode: action.cameraMode };
    case "set-manifold-visible":
      return { ...state, showCameraManifold: action.visible };
  }
}

interface DiagnosticMesh extends Object3D {
  dispose(): void;
  getBoundingBox(centersOnly?: boolean): Box3;
  initialized: Promise<DiagnosticMesh>;
  numSplats: number;
}

export function attachDiagnosticMeshWhenReady({
  mesh,
  source,
  scene,
  initializationStartedAt = performance.now(),
  now = () => performance.now(),
  onReady,
  onError,
}: {
  mesh: DiagnosticMesh;
  source: DiagnosticSplatSource;
  scene: Scene;
  initializationStartedAt?: number;
  now?: () => number;
  onReady: (
    splatCount: number,
    bounds: DiagnosticBoundsTuple,
    initializationMilliseconds: number,
  ) => void;
  onError: (message: string) => void;
}): () => void {
  let stale = false;
  let disposed = false;
  const disposeMesh = () => {
    if (disposed) return;
    disposed = true;
    mesh.parent?.remove(mesh);
    mesh.dispose();
  };

  void mesh.initialized
    .then(() => {
      if (stale) return;
      validateDiagnosticSplatCount(source, mesh.numSplats);
      const decoded = mesh.getBoundingBox(true);
      if (decoded.isEmpty()) {
        throw new Error(`Spark decoded empty ${source.toUpperCase()} bounds.`);
      }
      scene.add(mesh);
      onReady(
        mesh.numSplats,
        diagnosticBoundsFromBox(decoded),
        Math.max(0, now() - initializationStartedAt),
      );
    })
    .catch((error: unknown) => {
      disposeMesh();
      if (!stale) {
        onError(
          error instanceof Error
            ? error.message
            : `Spark ${source.toUpperCase()} diagnostic failed.`,
        );
      }
    });

  return () => {
    stale = true;
    disposeMesh();
  };
}

const ORBIT_CAMERA_FOV = 50;

const DEFAULT_RENDERER_SETTINGS: RendererSettings = {
  focalAdjustment: 1,
  maxPixelRadius: 512,
  preBlurAmount: 0,
  blurAmount: 0.3,
  sortRadial: true,
};

const DEFAULT_CAMERA_SETTINGS: CameraSettings = {
  distanceMultiplier: 1,
  nearMultiplier: 1,
  farMultiplier: 1,
};

function updateSparkRenderer(
  renderer: SparkRenderer,
  settings: RendererSettings,
): void {
  renderer.focalAdjustment = settings.focalAdjustment;
  renderer.maxPixelRadius = settings.maxPixelRadius;
  renderer.preBlurAmount = settings.preBlurAmount;
  renderer.blurAmount = settings.blurAmount;
  renderer.sortRadial = settings.sortRadial;
  renderer.setDirty();
}

function cameraColor(
  camera: ConvertedRegisteredCamera,
  selectedIndex: number,
  assessments: Readonly<Record<number, CameraAssessment>>,
): Color {
  if (camera.registeredIndex === selectedIndex) return new Color("#ffffff");
  if (assessments[camera.colmapImageId] === "coherent") {
    return new Color("#66e08a");
  }
  if (assessments[camera.colmapImageId] === "weak") {
    return new Color("#ff5b63");
  }
  if (camera.evalPsnr !== null && camera.evalPsnr < 20) {
    return new Color("#ff9f43");
  }
  return new Color(camera.evalIndex === null ? "#31d7ff" : "#ffd166");
}

function CameraManifoldOverlay({
  cameras,
  selectedIndex,
  assessments,
  visible,
}: {
  cameras: ConvertedRegisteredCamera[];
  selectedIndex: number;
  assessments: Readonly<Record<number, CameraAssessment>>;
  visible: boolean;
}) {
  const { scene } = useThree();

  useEffect(() => {
    if (!visible || cameras.length === 0) return;
    const group = new Group();
    group.name = "development-camera-manifold";
    const centers: number[] = [];
    const colors: number[] = [];
    const directionSegments: number[] = [];
    const directionColors: number[] = [];
    const trajectorySegments: number[] = [];
    const directionLength = 0.11;

    for (const camera of cameras) {
      centers.push(...camera.center.toArray());
      const color = cameraColor(camera, selectedIndex, assessments);
      colors.push(color.r, color.g, color.b);
      directionSegments.push(
        ...camera.center.toArray(),
        ...camera.center
          .clone()
          .addScaledVector(camera.viewingDirection, directionLength)
          .toArray(),
      );
      directionColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    }
    const sourceOrdered = cameras.toSorted((left, right) =>
      left.sourceImage.localeCompare(right.sourceImage, undefined, {
        numeric: true,
      }),
    );
    for (let index = 1; index < sourceOrdered.length; index += 1) {
      trajectorySegments.push(
        ...sourceOrdered[index - 1].center.toArray(),
        ...sourceOrdered[index].center.toArray(),
      );
    }

    const pointsGeometry = new BufferGeometry();
    pointsGeometry.setAttribute(
      "position",
      new Float32BufferAttribute(centers, 3),
    );
    pointsGeometry.setAttribute("color", new Float32BufferAttribute(colors, 3));
    const pointsMaterial = new PointsMaterial({
      size: 0.025,
      sizeAttenuation: true,
      vertexColors: true,
    });
    group.add(new Points(pointsGeometry, pointsMaterial));

    const directionsGeometry = new BufferGeometry();
    directionsGeometry.setAttribute(
      "position",
      new Float32BufferAttribute(directionSegments, 3),
    );
    directionsGeometry.setAttribute(
      "color",
      new Float32BufferAttribute(directionColors, 3),
    );
    const directionsMaterial = new LineBasicMaterial({
      transparent: true,
      opacity: 0.7,
      vertexColors: true,
    });
    group.add(new LineSegments(directionsGeometry, directionsMaterial));

    const trajectoryGeometry = new BufferGeometry();
    trajectoryGeometry.setAttribute(
      "position",
      new Float32BufferAttribute(trajectorySegments, 3),
    );
    const trajectoryMaterial = new LineBasicMaterial({
      color: "#52697a",
      transparent: true,
      opacity: 0.55,
    });
    group.add(new LineSegments(trajectoryGeometry, trajectoryMaterial));

    const selectedGeometry = new SphereGeometry(0.035, 12, 8);
    const selectedMaterial = new MeshBasicMaterial({ color: "#ffffff" });
    const selectedMarker = new Mesh(selectedGeometry, selectedMaterial);
    selectedMarker.position.copy(cameras[selectedIndex].center);
    group.add(selectedMarker);
    scene.add(group);

    return () => {
      scene.remove(group);
      pointsGeometry.dispose();
      pointsMaterial.dispose();
      directionsGeometry.dispose();
      directionsMaterial.dispose();
      trajectoryGeometry.dispose();
      trajectoryMaterial.dispose();
      selectedGeometry.dispose();
      selectedMaterial.dispose();
    };
  }, [assessments, cameras, scene, selectedIndex, visible]);

  return null;
}

export function IsolatedSparkScene({
  source,
  cameraMode,
  registeredCamera,
  cameras,
  selectedCameraIndex,
  assessments,
  showCameraManifold,
  boundsMode,
  decodedBounds,
  cameraSettings,
  rendererSettings,
  onReady,
  onCameraFit,
  onError,
}: {
  source: DiagnosticSplatSource;
  cameraMode: CameraMode;
  registeredCamera: ConvertedRegisteredCamera | null;
  cameras: ConvertedRegisteredCamera[];
  selectedCameraIndex: number;
  assessments: Readonly<Record<number, CameraAssessment>>;
  showCameraManifold: boolean;
  boundsMode: "core" | "decoded";
  decodedBounds: DiagnosticBoundsTuple | null;
  cameraSettings: CameraSettings;
  rendererSettings: RendererSettings;
  onReady: (
    splatCount: number,
    bounds: DiagnosticBoundsTuple,
    initializationMilliseconds: number,
  ) => void;
  onCameraFit: (fit: PerspectiveCameraFit | null) => void;
  onError: (message: string) => void;
}) {
  const { camera, gl, scene, size } = useThree();
  const controls = useRef<OrbitControls | null>(null);
  const sparkRenderer = useRef<SparkRenderer | null>(null);
  const settingsRef = useRef(rendererSettings);
  const lastSortRadial = useRef(rendererSettings.sortRadial);

  useEffect(() => {
    settingsRef.current = rendererSettings;
  }, [rendererSettings]);

  useEffect(() => {
    if (!(camera instanceof PerspectiveCamera)) {
      onError("The isolated Spark diagnostic requires a perspective camera.");
      return;
    }
    camera.up.set(0, 0, 1);
    const orbit = new OrbitControls(camera, gl.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.065;
    orbit.screenSpacePanning = true;
    controls.current = orbit;
    return () => {
      orbit.dispose();
      controls.current = null;
    };
  }, [camera, gl.domElement, onError]);

  useFrame(() => {
    if (controls.current?.enabled) controls.current.update();
  });

  useEffect(() => {
    if (controls.current) {
      controls.current.enabled = cameraMode === "orbit";
    }
  }, [cameraMode]);

  useEffect(() => {
    if (!(camera instanceof PerspectiveCamera)) return;

    if (cameraMode === "registered" && registeredCamera) {
      applyRegisteredCamera(camera, registeredCamera);
      onCameraFit(null);
      return;
    }

    camera.clearViewOffset();
    camera.up.set(0, 0, 1);
    camera.fov = ORBIT_CAMERA_FOV;
    camera.aspect = size.width / Math.max(1, size.height);
    const selectedBounds =
      boundsMode === "decoded" && decodedBounds
        ? decodedBounds
        : KNOWN_PLY_CORE_BOUNDS;
    const fit = calculatePerspectiveCameraFit({
      bounds: boxFromDiagnosticBounds(selectedBounds),
      verticalFovDegrees: camera.fov,
      aspect: size.width / Math.max(1, size.height),
      ...cameraSettings,
    });
    camera.position.copy(fit.position);
    camera.near = fit.near;
    camera.far = fit.far;
    camera.updateProjectionMatrix();
    controls.current?.target.copy(fit.target);
    controls.current?.update();
    onCameraFit(fit);
  }, [
    boundsMode,
    camera,
    cameraMode,
    cameraSettings,
    decodedBounds,
    onCameraFit,
    registeredCamera,
    size.height,
    size.width,
  ]);

  useEffect(() => {
    let stale = false;
    let renderer: SparkRenderer | null = null;
    let disposed = false;

    const disposeRenderer = () => {
      if (!renderer || disposed) return;
      disposed = true;
      renderer.parent?.remove(renderer);
      renderer.dispose();
      if (sparkRenderer.current === renderer) sparkRenderer.current = null;
    };

    void (async () => {
      try {
        const spark = await import("@sparkjsdev/spark");
        if (stale) return;
        renderer = new spark.SparkRenderer({
          renderer: gl,
          enableLod: false,
          ...settingsRef.current,
        });
        sparkRenderer.current = renderer;
        scene.add(renderer);
      } catch (error) {
        disposeRenderer();
        if (!stale) {
          onError(
            error instanceof Error
              ? error.message
              : "Spark diagnostic renderer failed.",
          );
        }
      }
    })();

    return () => {
      stale = true;
      disposeRenderer();
    };
  }, [gl, onError, onReady, scene]);

  useEffect(() => {
    let stale = false;
    let disposeMesh: () => void = () => undefined;

    void (async () => {
      try {
        const spark = await import("@sparkjsdev/spark");
        if (stale) return;
        const initializationStartedAt = performance.now();
        const mesh = new spark.SplatMesh(
          diagnosticSplatMeshOptions(source, spark.SplatFileType),
        );
        disposeMesh = attachDiagnosticMeshWhenReady({
          mesh,
          source,
          scene,
          initializationStartedAt,
          onReady,
          onError,
        });
      } catch (error) {
        if (!stale) {
          onError(
            error instanceof Error
              ? error.message
              : `Spark ${source.toUpperCase()} diagnostic failed.`,
          );
        }
      }
    })();

    return () => {
      stale = true;
      disposeMesh();
    };
  }, [onError, onReady, scene, source]);

  useEffect(() => {
    const renderer = sparkRenderer.current;
    if (!renderer) return;
    const sortChanged = lastSortRadial.current !== rendererSettings.sortRadial;
    lastSortRadial.current = rendererSettings.sortRadial;
    updateSparkRenderer(renderer, rendererSettings);
    if (sortChanged) {
      void renderer.update({ scene, camera }).catch((error: unknown) => {
        onError(
          error instanceof Error ? error.message : "Spark re-sort failed.",
        );
      });
    }
  }, [camera, onError, rendererSettings, scene]);

  return (
    <CameraManifoldOverlay
      cameras={cameras}
      selectedIndex={selectedCameraIndex}
      assessments={assessments}
      visible={showCameraManifold}
    />
  );
}

function boundsLabel(bounds: DiagnosticBoundsTuple | null): string {
  if (!bounds) return "Waiting for Spark decode";
  return bounds
    .map((point) => point.map((value) => value.toFixed(3)).join(", "))
    .join(" → ");
}

function byteSizeLabel(bytes: number): string {
  return `${bytes.toLocaleString()} bytes (${(bytes / (1024 * 1024)).toFixed(3)} MiB)`;
}

export function IsolatedSparkPlyDiagnostic() {
  const [controlsState, dispatchControls] = useReducer(
    diagnosticControlReducer,
    DEFAULT_DIAGNOSTIC_CONTROL_STATE,
  );
  const { source, cameraMode, selectedCameraIndex, showCameraManifold } =
    controlsState;
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [error, setError] = useState<string>();
  const [splatCount, setSplatCount] = useState<number>();
  const [initializationMilliseconds, setInitializationMilliseconds] =
    useState<number>();
  const [decodedBounds, setDecodedBounds] =
    useState<DiagnosticBoundsTuple | null>(null);
  const [cameraDataset, setCameraDataset] =
    useState<RegisteredCameraDataset | null>(null);
  const [cameraDatasetError, setCameraDatasetError] = useState<string>();
  const [assessments, setAssessments] = useState<
    Record<number, CameraAssessment>
  >({});
  const [boundsMode, setBoundsMode] = useState<"core" | "decoded">("core");
  const [cameraSettings, setCameraSettings] = useState(
    DEFAULT_CAMERA_SETTINGS,
  );
  const [rendererSettings, setRendererSettings] = useState(
    DEFAULT_RENDERER_SETTINGS,
  );
  const [cameraFit, setCameraFit] = useState<PerspectiveCameraFit>();
  const cameras = useMemo(
    () => (cameraDataset ? convertRegisteredCameras(cameraDataset) : []),
    [cameraDataset],
  );
  const manifoldStats = useMemo<CameraManifoldStats | null>(
    () => (cameras.length > 1 ? analyzeCameraManifold(cameras) : null),
    [cameras],
  );
  const selectedCamera = cameras[selectedCameraIndex] ?? null;
  const selectedEvalRender = selectedCamera
    ? evalRenderUrl(selectedCamera)
    : null;
  const activeSource = DIAGNOSTIC_SPLAT_SOURCES[source];

  useEffect(() => {
    const controller = new AbortController();
    void fetch(REGISTERED_CAMERAS_DIAGNOSTIC_URL, {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Camera data request failed (${response.status}).`);
        }
        return decodeRegisteredCameraDataset(await response.json());
      })
      .then((dataset) => {
        if (dataset.cameras.length !== 210) {
          throw new Error(
            `Expected 210 registered cameras, received ${dataset.cameras.length}.`,
          );
        }
        setCameraDataset(dataset);
        setCameraDatasetError(undefined);
      })
      .catch((fetchError: unknown) => {
        if (controller.signal.aborted) return;
        setCameraDatasetError(
          fetchError instanceof Error
            ? fetchError.message
            : "Registered-camera data failed to load.",
        );
      });
    return () => controller.abort();
  }, []);

  const onReady = useCallback(
    (
      count: number,
      bounds: DiagnosticBoundsTuple,
      elapsedMilliseconds: number,
    ) => {
      setSplatCount(count);
      setInitializationMilliseconds(elapsedMilliseconds);
      setDecodedBounds(bounds);
      setError(undefined);
      setStatus("ready");
    },
    [],
  );
  const onError = useCallback((message: string) => {
    setError(message);
    setStatus("error");
  }, []);
  const onCameraFit = useCallback((fit: PerspectiveCameraFit | null) => {
    setCameraFit(fit ?? undefined);
  }, []);

  const updateCameraNumber =
    (key: keyof CameraSettings) => (event: ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value);
      setCameraSettings((current) => ({ ...current, [key]: value }));
    };
  const updateRendererNumber =
    (key: Exclude<keyof RendererSettings, "sortRadial">) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value);
      setRendererSettings((current) => ({ ...current, [key]: value }));
    };
  const selectCamera = (index: number) => {
    if (cameras.length === 0) return;
    dispatchControls({
      type: "select-camera",
      index: Math.min(cameras.length - 1, Math.max(0, Math.trunc(index))),
    });
  };
  const stepCamera = (offset: number) => {
    if (cameras.length === 0) return;
    selectCamera(
      (selectedCameraIndex + offset + cameras.length) % cameras.length,
    );
  };
  const stepMatchingCamera = (
    predicate: (camera: ConvertedRegisteredCamera) => boolean,
  ) => {
    if (cameras.length === 0) return;
    for (let offset = 1; offset <= cameras.length; offset += 1) {
      const index = (selectedCameraIndex + offset) % cameras.length;
      if (predicate(cameras[index])) {
        selectCamera(index);
        return;
      }
    }
  };
  const assessSelectedCamera = (assessment?: CameraAssessment) => {
    if (!selectedCamera) return;
    setAssessments((current) => {
      const next = { ...current };
      if (assessment) next[selectedCamera.colmapImageId] = assessment;
      else delete next[selectedCamera.colmapImageId];
      return next;
    });
  };
  const selectSource = (nextSource: DiagnosticSplatSource) => {
    if (nextSource === source) return;
    setStatus("loading");
    setError(undefined);
    setSplatCount(undefined);
    setInitializationMilliseconds(undefined);
    setDecodedBounds(null);
    dispatchControls({ type: "select-source", source: nextSource });
  };

  return (
    <section className={styles.shell} aria-labelledby="diagnostic-heading">
      <aside className={styles.panel}>
        <p className={styles.eyebrow}>Development-only diagnostic</p>
        <h1 id="diagnostic-heading">Exact Spark source comparison</h1>
        <p>
          Original Nerfstudio PLY, current voxel SPZ, and development-only
          reduction variants. No analytical grid, robot geometry, or State
          Atlas camera framing is mounted.
        </p>
        <p role={status === "error" ? "alert" : "status"}>
          {status === "loading"
            ? `Loading and decoding ${activeSource.label}…`
            : status === "ready"
              ? `Ready — ${splatCount?.toLocaleString()} decoded splats`
              : error}
        </p>

        <fieldset>
          <legend>Render source</legend>
          <div className={styles.buttonRow}>
            {(Object.keys(DIAGNOSTIC_SPLAT_SOURCES) as DiagnosticSplatSource[]).map(
              (candidate) => (
                <button
                  key={candidate}
                  type="button"
                  aria-pressed={source === candidate}
                  onClick={() => selectSource(candidate)}
                >
                  {DIAGNOSTIC_SPLAT_SOURCES[candidate].label}
                </button>
              ),
            )}
          </div>
          <dl className={styles.activeSourceDetails}>
            <div>
              <dt>Source type</dt>
              <dd>{activeSource.sourceType}</dd>
            </div>
            <div>
              <dt>Decoded splats</dt>
              <dd>{splatCount?.toLocaleString() ?? "Loading…"}</dd>
            </div>
            <div>
              <dt>Approx loaded</dt>
              <dd>{byteSizeLabel(activeSource.byteSize)}</dd>
            </div>
            <div>
              <dt>Mesh init</dt>
              <dd>
                {initializationMilliseconds === undefined
                  ? "Loading…"
                  : `${initializationMilliseconds.toFixed(1)} ms`}
              </dd>
            </div>
            <div>
              <dt>Selection</dt>
              <dd>{activeSource.selectionMethod}</dd>
            </div>
            <div>
              <dt>Active camera</dt>
              <dd>
                {selectedCamera
                  ? `COLMAP ${selectedCamera.colmapImageId} (registered ${selectedCamera.registeredIndex + 1}/${cameras.length})`
                  : "Loading…"}
              </dd>
            </div>
            <div>
              <dt>Source frame</dt>
              <dd>{selectedCamera?.sourceImage ?? "Loading…"}</dd>
            </div>
            <div>
              <dt>Eval / PSNR</dt>
              <dd>
                {selectedCamera?.evalIndex === null || !selectedCamera
                  ? "Not an eval frame"
                  : `eval ${selectedCamera.evalIndex} · ${selectedCamera.evalPsnr?.toFixed(3)} dB`}
              </dd>
            </div>
          </dl>
        </fieldset>

        <fieldset>
          <legend>Viewpoint</legend>
          <div className={styles.buttonRow}>
            <button
              type="button"
              aria-pressed={cameraMode === "registered"}
              disabled={!selectedCamera}
              onClick={() =>
                dispatchControls({
                  type: "set-camera-mode",
                  cameraMode: "registered",
                })
              }
            >
              Use selected camera
            </button>
            <button
              type="button"
              aria-pressed={cameraMode === "orbit"}
              onClick={() =>
                dispatchControls({
                  type: "set-camera-mode",
                  cameraMode: "orbit",
                })
              }
            >
              Orbit camera
            </button>
          </div>
          <p className={styles.readout} role="status">
            {cameraDatasetError
              ? `${cameraDatasetError} Run npm run environment:diagnostic:ply.`
              : cameraDataset
                ? `${cameras.length} registered cameras parsed and converted`
                : "Loading registered camera poses…"}
          </p>
        </fieldset>

        <fieldset>
          <legend>Registered camera</legend>
          <div className={styles.buttonRow}>
            <button
              type="button"
              disabled={!selectedCamera}
              onClick={() => stepCamera(-1)}
            >
              Previous
            </button>
            <button
              type="button"
              disabled={!selectedCamera}
              onClick={() => stepCamera(1)}
            >
              Next
            </button>
          </div>
          <label>
            Registered index {selectedCameraIndex + 1} / {cameras.length || "—"}
            <input
              type="range"
              min="0"
              max={Math.max(0, cameras.length - 1)}
              step="1"
              value={selectedCameraIndex}
              disabled={!selectedCamera}
              onChange={(event) => selectCamera(Number(event.target.value))}
            />
          </label>
          <select
            aria-label="Registered camera pose"
            value={selectedCameraIndex}
            disabled={!selectedCamera}
            onChange={(event) => selectCamera(Number(event.target.value))}
          >
            {cameras.map((camera) => (
              <option
                key={camera.colmapImageId}
                value={camera.registeredIndex}
              >
                {camera.registeredIndex + 1}: COLMAP {camera.colmapImageId} ·{" "}
                {camera.sourceImage}
                {camera.evalIndex === null
                  ? ""
                  : ` · eval ${camera.evalIndex}${camera.evalPsnr === null ? "" : ` · ${camera.evalPsnr.toFixed(1)} dB`}`}
              </option>
            ))}
          </select>
          <div className={styles.buttonRow}>
            <button
              type="button"
              disabled={!selectedCamera}
              onClick={() =>
                stepMatchingCamera((camera) => camera.evalIndex !== null)
              }
            >
              Next eval
            </button>
            <button
              type="button"
              disabled={!selectedCamera}
              onClick={() =>
                stepMatchingCamera(
                  (camera) =>
                    camera.evalPsnr !== null && camera.evalPsnr < 20,
                )
              }
            >
              Next low-PSNR eval
            </button>
          </div>
          <div className={styles.buttonRowThree}>
            <button
              type="button"
              aria-pressed={
                selectedCamera
                  ? assessments[selectedCamera.colmapImageId] === "coherent"
                  : false
              }
              disabled={!selectedCamera}
              onClick={() => assessSelectedCamera("coherent")}
            >
              Mark coherent
            </button>
            <button
              type="button"
              aria-pressed={
                selectedCamera
                  ? assessments[selectedCamera.colmapImageId] === "weak"
                  : false
              }
              disabled={!selectedCamera}
              onClick={() => assessSelectedCamera("weak")}
            >
              Mark weak
            </button>
            <button
              type="button"
              disabled={!selectedCamera}
              onClick={() => assessSelectedCamera()}
            >
              Clear
            </button>
          </div>
          {selectedCamera ? (
            <p className={styles.readout}>
              COLMAP image {selectedCamera.colmapImageId} ·{" "}
              {selectedCamera.sourceImage}
              <br />
              {selectedCamera.width}×{selectedCamera.height} · fx/fy{" "}
              {selectedCamera.fx.toFixed(6)} · vfov{" "}
              {pinholeVerticalFovDegrees(
                selectedCamera.height,
                selectedCamera.fy,
              ).toFixed(6)}
              °
              <br />
              center: {selectedCamera.center.toArray().map((value) => value.toFixed(4)).join(", ")}
              <br />
              look: {selectedCamera.viewingDirection
                .toArray()
                .map((value) => value.toFixed(4))
                .join(", ")}
              {selectedCamera.evalIndex === null ? null : (
                <>
                  <br />
                  eval_img_{String(selectedCamera.evalIndex).padStart(4, "0")}
                  .png · {selectedCamera.evalPsnr?.toFixed(3)} dB
                </>
              )}
            </p>
          ) : null}
          {selectedEvalRender ? (
            <a
              className={styles.evalReference}
              href={selectedEvalRender}
              target="_blank"
              rel="noreferrer"
              title="Open full-size Nerfstudio ground-truth and prediction montage"
            >
              <img
                src={selectedEvalRender}
                alt={`Nerfstudio eval ${selectedCamera?.evalIndex}: ground truth on the left and prediction on the right`}
              />
              <span>Ground truth | Nerfstudio prediction</span>
            </a>
          ) : null}
          <p className={styles.readout}>
            Session marks: {Object.values(assessments).filter((value) => value === "coherent").length} coherent ·{" "}
            {Object.values(assessments).filter((value) => value === "weak").length} weak
          </p>
        </fieldset>

        <fieldset>
          <legend>Captured camera manifold</legend>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showCameraManifold}
              onChange={(event) =>
                dispatchControls({
                  type: "set-manifold-visible",
                  visible: event.target.checked,
                })
              }
            />
            Show camera manifold
          </label>
          <p className={styles.legend}>
            <span data-color="registered">registered</span>
            <span data-color="eval">eval</span>
            <span data-color="low">eval below 20 dB</span>
            <span data-color="coherent">marked coherent</span>
            <span data-color="weak">marked weak</span>
          </p>
          {manifoldStats ? (
            <>
              <p className={styles.readout}>
                Reconstruction-coordinate centers only; not robot-world
                coordinates.
                <br />
                bounds: {boundsLabel(manifoldStats.centerBounds)}
                <br />
                extent: {manifoldStats.extent
                  .map((value) => value.toFixed(4))
                  .join(" × ")}
                <br />
                diagonal: {manifoldStats.diagonal.toFixed(4)} · ordered path:{" "}
                {manifoldStats.orderedPathLength.toFixed(4)}
                <br />
                median / p95 step: {manifoldStats.medianStep.toFixed(4)} /{" "}
                {manifoldStats.p95Step.toFixed(4)}
                <br />
                4× median-neighbor groups ({manifoldStats.neighborhoodRadius.toFixed(4)} radius):{" "}
                {manifoldStats.neighborhoodComponents
                  .map((component) => component.length)
                  .join(" + ")}
                <br />
                MST maximum bridge: {manifoldStats.maximumMstBridge.distance.toFixed(4)} ({manifoldStats.maximumMstBridge.from.sourceImage} →{" "}
                {manifoldStats.maximumMstBridge.to.sourceImage})
              </p>
              {manifoldStats.neighborhoodComponents.length > 1 ? (
                <details>
                  <summary>Detached fine-scale camera groups</summary>
                  <ul className={styles.gapList}>
                    {manifoldStats.neighborhoodComponents
                      .slice(1)
                      .map((component) => (
                        <li key={component[0].colmapImageId}>
                          {component.length} cameras: {component
                            .toSorted(
                              (left, right) =>
                                left.registeredIndex - right.registeredIndex,
                            )
                            .map(
                              (camera) =>
                                `${camera.colmapImageId}/${camera.sourceImage}`,
                            )
                            .join(", ")}
                        </li>
                      ))}
                  </ul>
                </details>
              ) : null}
              <details>
                <summary>
                  {manifoldStats.largeTrajectoryGaps.length} large ordered gaps
                </summary>
                <ul className={styles.gapList}>
                  {manifoldStats.largeTrajectoryGaps.map((gap) => (
                    <li key={`${gap.from.colmapImageId}-${gap.to.colmapImageId}`}>
                      {gap.from.sourceImage} → {gap.to.sourceImage}: {gap.distance.toFixed(4)}
                    </li>
                  ))}
                </ul>
              </details>
            </>
          ) : null}
        </fieldset>

        <fieldset>
          <legend>Orbit camera framing</legend>
          <div className={styles.buttonRow}>
            <button
              type="button"
              aria-pressed={boundsMode === "core"}
              onClick={() => {
                setBoundsMode("core");
                dispatchControls({
                  type: "set-camera-mode",
                  cameraMode: "orbit",
                });
              }}
            >
              Fit known core
            </button>
            <button
              type="button"
              aria-pressed={boundsMode === "decoded"}
              disabled={!decodedBounds}
              onClick={() => {
                setBoundsMode("decoded");
                dispatchControls({
                  type: "set-camera-mode",
                  cameraMode: "orbit",
                });
              }}
            >
              Fit all centers
            </button>
          </div>
          <label>
            Distance × {cameraSettings.distanceMultiplier.toFixed(2)}
            <input
              type="range"
              min="0.25"
              max="4"
              step="0.05"
              value={cameraSettings.distanceMultiplier}
              disabled={cameraMode !== "orbit"}
              onChange={updateCameraNumber("distanceMultiplier")}
            />
          </label>
          <label>
            Near clip ×
            <input
              type="number"
              min="0.01"
              max="100"
              step="0.1"
              value={cameraSettings.nearMultiplier}
              disabled={cameraMode !== "orbit"}
              onChange={updateCameraNumber("nearMultiplier")}
            />
          </label>
          <label>
            Far clip ×
            <input
              type="number"
              min="0.25"
              max="10"
              step="0.25"
              value={cameraSettings.farMultiplier}
              disabled={cameraMode !== "orbit"}
              onChange={updateCameraNumber("farMultiplier")}
            />
          </label>
          <p className={styles.readout}>
            Effective near/far:{" "}
            {cameraMode === "registered"
              ? `${NERFSTUDIO_EVAL_CAMERA.near} / ${NERFSTUDIO_EVAL_CAMERA.far.toExponential()}`
              : `${cameraFit?.near.toPrecision(3) ?? "—"} / ${cameraFit?.far.toPrecision(4) ?? "—"}`}
          </p>
        </fieldset>

        <fieldset>
          <legend>Official Spark renderer controls</legend>
          <label>
            Focal adjustment: {rendererSettings.focalAdjustment.toFixed(2)}
            <input
              type="range"
              min="0.25"
              max="4"
              step="0.05"
              value={rendererSettings.focalAdjustment}
              onChange={updateRendererNumber("focalAdjustment")}
            />
          </label>
          <label>
            Maximum pixel radius: {rendererSettings.maxPixelRadius}
            <input
              type="range"
              min="16"
              max="512"
              step="8"
              value={rendererSettings.maxPixelRadius}
              onChange={updateRendererNumber("maxPixelRadius")}
            />
          </label>
          <label>
            Pre-blur: {rendererSettings.preBlurAmount.toFixed(2)}
            <input
              type="range"
              min="0"
              max="0.6"
              step="0.01"
              value={rendererSettings.preBlurAmount}
              onChange={updateRendererNumber("preBlurAmount")}
            />
          </label>
          <label>
            Anti-alias blur: {rendererSettings.blurAmount.toFixed(2)}
            <input
              type="range"
              min="0"
              max="0.6"
              step="0.01"
              value={rendererSettings.blurAmount}
              onChange={updateRendererNumber("blurAmount")}
            />
          </label>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={rendererSettings.sortRadial}
              onChange={(event) =>
                setRendererSettings((current) => ({
                  ...current,
                  sortRadial: event.target.checked,
                }))
              }
            />
            Radial sorting (off selects Z-depth)
          </label>
          <button
            type="button"
            onClick={() => setRendererSettings(DEFAULT_RENDERER_SETTINGS)}
          >
            Reset Spark defaults
          </button>
        </fieldset>

        <dl className={styles.details}>
          <div>
            <dt>Known core bounds</dt>
            <dd>{boundsLabel(KNOWN_PLY_CORE_BOUNDS)}</dd>
          </div>
          <div>
            <dt>Decoded center bounds</dt>
            <dd>{boundsLabel(decodedBounds)}</dd>
          </div>
        </dl>

        <a
          className={styles.compareLink}
          href="/viewer/demo?sparkEnvironmentSource=ply"
          target="_blank"
          rel="noreferrer"
        >
          Open integrated PLY viewer
        </a>
      </aside>

      <div className={styles.canvas}>
        <div className={styles.cameraViewport}>
          <Canvas
            camera={{
              fov: nerfstudioEvalCameraVerticalFovDegrees(),
              near: NERFSTUDIO_EVAL_CAMERA.near,
              far: NERFSTUDIO_EVAL_CAMERA.far,
            }}
            dpr={[1, 2]}
            gl={{ antialias: false, powerPreference: "high-performance" }}
            onCreated={({ gl }) => {
              gl.setClearColor("#02070b");
              if (!gl.capabilities.isWebGL2) {
                onError("The isolated Spark diagnostic requires WebGL2.");
              }
            }}
          >
            <IsolatedSparkScene
              source={source}
              cameraMode={cameraMode}
              registeredCamera={selectedCamera}
              cameras={cameras}
              selectedCameraIndex={selectedCameraIndex}
              assessments={assessments}
              showCameraManifold={showCameraManifold}
              boundsMode={boundsMode}
              decodedBounds={decodedBounds}
              cameraSettings={cameraSettings}
              rendererSettings={rendererSettings}
              onReady={onReady}
              onCameraFit={onCameraFit}
              onError={onError}
            />
          </Canvas>
        </div>
      </div>
    </section>
  );
}
