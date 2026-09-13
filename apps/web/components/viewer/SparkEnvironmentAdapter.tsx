"use client";

import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";

import {
  resolveIntegratedEnvironmentSource,
  summarizeFrameTimes,
  UNIFORM_250K_INTEGRATED_ASSET,
  UNIFORM_300K_INTEGRATED_ASSET,
  type IntegratedEnvironmentPerformanceMetrics,
  type IntegratedEnvironmentSource,
  type SparkResourceLifecycleEvent,
  type SparkResourceLifecycleStage,
  type ThreeRendererInfoSnapshot,
} from "@/lib/environment/integrated-performance-diagnostic";
import { SPARK_PLY_DIAGNOSTIC_URL } from "@/lib/environment/spark-ply-diagnostic";
import type { ValidatedEnvironmentRenderRequest } from "@/lib/environment/types";

export type SparkAdapterPhase = "initializing" | "ready";
export type SparkEnvironmentSource =
  | "spz"
  | "ply"
  | "uniform250k"
  | "uniform300k";

export { SPARK_PLY_DIAGNOSTIC_URL };

export function resolveSparkEnvironmentSource(
  nodeEnv: string | undefined,
  search: string,
  requestedIntegratedSource?: IntegratedEnvironmentSource,
): SparkEnvironmentSource {
  if (nodeEnv !== "development") return "spz";
  if (new URLSearchParams(search).get("sparkEnvironmentSource") === "ply") {
    return "ply";
  }
  const resolvedSource = resolveIntegratedEnvironmentSource(
    nodeEnv,
    search,
    requestedIntegratedSource,
  );
  return resolvedSource === "current" ? "spz" : resolvedSource;
}

const FRAME_WINDOW_SIZE = 120;
const FRAME_REPORT_INTERVAL = 30;
const STABILIZATION_FRAME_COUNT = 30;

function readRendererInfo(renderer: unknown): ThreeRendererInfoSnapshot | null {
  const info = (renderer as {
    info?: {
      memory?: { geometries?: number; textures?: number };
      programs?: unknown[] | null;
      render?: {
        calls?: number;
        triangles?: number;
        points?: number;
        lines?: number;
      };
    };
  }).info;
  if (!info?.memory || !info.render) return null;
  return {
    geometries: info.memory.geometries ?? 0,
    textures: info.memory.textures ?? 0,
    programs: Array.isArray(info.programs) ? info.programs.length : null,
    renderCalls: info.render.calls ?? 0,
    triangles: info.render.triangles ?? 0,
    points: info.render.points ?? 0,
    lines: info.render.lines ?? 0,
  };
}

interface OwnedSparkResources {
  source: IntegratedEnvironmentSource;
  group: THREE.Group;
  mesh: { dispose(): void; numSplats: number } & THREE.Object3D;
  renderer: {
    dispose(): void;
    geometry?: { dispose(): void };
    material?: { dispose(): void } | Array<{ dispose(): void }>;
  } & THREE.Object3D;
  disposed: boolean;
}

type SparkRendererResource = OwnedSparkResources["renderer"];

function disposeSparkRendererResource(renderer: SparkRendererResource): void {
  renderer.dispose();
  renderer.geometry?.dispose();
  const materials = Array.isArray(renderer.material)
    ? renderer.material
    : renderer.material
      ? [renderer.material]
      : [];
  for (const material of materials) material.dispose();
}

export function disposeSparkResources(resources: OwnedSparkResources | null): void {
  if (!resources || resources.disposed) return;
  resources.disposed = true;
  resources.group.remove(resources.mesh);
  resources.group.parent?.remove(resources.group);
  resources.renderer.parent?.remove(resources.renderer);
  resources.mesh.dispose();
  disposeSparkRendererResource(resources.renderer);
}

export function SparkEnvironmentAdapter({
  request,
  onPhase,
  onError,
  integratedSource,
  readyRequestedAt,
  onPerformanceMetrics,
  sourceSwitchToken = 0,
  onResourceLifecycleEvent,
}: {
  request: ValidatedEnvironmentRenderRequest | null;
  onPhase: (generation: number, phase: SparkAdapterPhase) => void;
  onError: (generation: number, message: string) => void;
  integratedSource?: IntegratedEnvironmentSource;
  readyRequestedAt?: number | null;
  onPerformanceMetrics?: (
    metrics: IntegratedEnvironmentPerformanceMetrics | null,
  ) => void;
  sourceSwitchToken?: number;
  onResourceLifecycleEvent?: (event: SparkResourceLifecycleEvent) => void;
}) {
  const { gl, scene } = useThree();
  const owned = useRef<OwnedSparkResources | null>(null);
  const requestRef = useRef(request);
  const readyRequestedAtRef = useRef(readyRequestedAt);
  const metricsCallbackRef = useRef(onPerformanceMetrics);
  const lifecycleCallbackRef = useRef(onResourceLifecycleEvent);
  const currentMetrics = useRef<IntegratedEnvironmentPerformanceMetrics | null>(
    null,
  );
  const frameSamples = useRef<number[]>([]);
  const framesSinceReport = useRef(0);
  const initializationFrameCount = useRef(0);
  const stabilizationFrameCount = useRef(0);
  const stabilizedFrameCount = useRef(0);
  const samplingPhase = useRef<"idle" | "initializing" | "stabilizing" | "stabilized">(
    "idle",
  );
  const switchSequence = useRef(0);
  const pendingFrameLifecycleEvents = useRef<
    Array<
      Omit<SparkResourceLifecycleEvent, "elapsedMilliseconds" | "rendererInfo"> & {
        startedAt: number;
        onRecorded?: () => void;
      }
    >
  >([]);
  const steadyLifecycleContext = useRef<{
    switchId: number;
    fromSource: IntegratedEnvironmentSource;
    toSource: IntegratedEnvironmentSource;
    startedAt: number;
  } | null>(null);
  const requestGeneration = request?.generation;
  const requestVisible = request?.visible;

  useEffect(() => {
    requestRef.current = request;
  }, [request]);

  useEffect(() => {
    readyRequestedAtRef.current = readyRequestedAt;
  }, [readyRequestedAt]);

  useEffect(() => {
    metricsCallbackRef.current = onPerformanceMetrics;
  }, [onPerformanceMetrics]);

  useEffect(() => {
    lifecycleCallbackRef.current = onResourceLifecycleEvent;
  }, [onResourceLifecycleEvent]);

  useFrame((_state, delta) => {
    for (const event of pendingFrameLifecycleEvents.current.splice(0)) {
      lifecycleCallbackRef.current?.({
        switchId: event.switchId,
        fromSource: event.fromSource,
        toSource: event.toSource,
        stage: event.stage,
        elapsedMilliseconds: performance.now() - event.startedAt,
        rendererInfo: readRendererInfo(gl),
      });
      event.onRecorded?.();
    }
    if (samplingPhase.current === "initializing") {
      initializationFrameCount.current += 1;
      return;
    }
    if (samplingPhase.current === "stabilizing") {
      stabilizationFrameCount.current += 1;
      if (stabilizationFrameCount.current >= STABILIZATION_FRAME_COUNT) {
        samplingPhase.current = "stabilized";
      }
      if (
        currentMetrics.current &&
        (stabilizationFrameCount.current % 10 === 0 ||
          stabilizationFrameCount.current === STABILIZATION_FRAME_COUNT)
      ) {
        currentMetrics.current = {
          ...currentMetrics.current,
          stabilizationFrameCount: stabilizationFrameCount.current,
        };
        metricsCallbackRef.current?.(currentMetrics.current);
      }
      return;
    }
    if (!owned.current?.group.visible) return;
    const milliseconds = delta * 1_000;
    if (!Number.isFinite(milliseconds) || milliseconds <= 0) return;
    frameSamples.current.push(milliseconds);
    stabilizedFrameCount.current += 1;
    if (
      stabilizedFrameCount.current === FRAME_WINDOW_SIZE &&
      steadyLifecycleContext.current
    ) {
      const checkpoint = steadyLifecycleContext.current;
      lifecycleCallbackRef.current?.({
        switchId: checkpoint.switchId,
        fromSource: checkpoint.fromSource,
        toSource: checkpoint.toSource,
        stage: "steady-state",
        elapsedMilliseconds: performance.now() - checkpoint.startedAt,
        rendererInfo: readRendererInfo(gl),
      });
      steadyLifecycleContext.current = null;
    }
    if (frameSamples.current.length > FRAME_WINDOW_SIZE) {
      frameSamples.current.shift();
    }
    if (!currentMetrics.current) return;
    framesSinceReport.current += 1;
    if (framesSinceReport.current < FRAME_REPORT_INTERVAL) return;
    framesSinceReport.current = 0;
    const frame = summarizeFrameTimes(frameSamples.current);
    currentMetrics.current = {
      ...currentMetrics.current,
      frame,
      stabilizedFrameCount: stabilizedFrameCount.current,
      rendererInfo: readRendererInfo(gl),
    };
    metricsCallbackRef.current?.(currentMetrics.current);
  });

  useEffect(
    () => () => {
      disposeSparkResources(owned.current);
      owned.current = null;
    },
    [],
  );

  useEffect(() => {
    const source = resolveSparkEnvironmentSource(
      process.env.NODE_ENV,
      window.location.search,
      integratedSource,
    );
    const integratedResolvedSource: IntegratedEnvironmentSource =
      source === "uniform250k" || source === "uniform300k"
        ? source
        : "current";
    const diagnosticAsset =
      source === "uniform250k"
        ? UNIFORM_250K_INTEGRATED_ASSET
        : source === "uniform300k"
          ? UNIFORM_300K_INTEGRATED_ASSET
          : null;
    const previous = owned.current;
    const switchId = previous ? ++switchSequence.current : null;
    const switchStartedAt = performance.now();
    const emitLifecycle = (stage: SparkResourceLifecycleStage) => {
      if (switchId === null || !previous) return;
      lifecycleCallbackRef.current?.({
        switchId,
        fromSource: previous.source,
        toSource: integratedResolvedSource,
        stage,
        elapsedMilliseconds: performance.now() - switchStartedAt,
        rendererInfo: readRendererInfo(gl),
      });
    };
    emitLifecycle("before-switch");
    disposeSparkResources(previous);
    owned.current = null;
    emitLifecycle("after-old-disposal");
    const afterOldDisposalFrame =
      switchId !== null && previous && lifecycleCallbackRef.current
        ? new Promise<void>((resolve) => {
            pendingFrameLifecycleEvents.current.push({
              switchId,
              fromSource: previous.source,
              toSource: integratedResolvedSource,
              stage: "after-old-disposal-frame",
              startedAt: switchStartedAt,
              onRecorded: resolve,
            });
          })
        : Promise.resolve();
    currentMetrics.current = null;
    frameSamples.current = [];
    framesSinceReport.current = 0;
    initializationFrameCount.current = 0;
    stabilizationFrameCount.current = 0;
    stabilizedFrameCount.current = 0;
    steadyLifecycleContext.current = null;
    samplingPhase.current = "initializing";
    metricsCallbackRef.current?.(null);
    const currentRequest = requestRef.current;
    if (!currentRequest) {
      samplingPhase.current = "idle";
      return;
    }

    let stale = false;
    const generation = currentRequest.generation;
    const firstReadyStartedAt = readyRequestedAtRef.current ?? performance.now();
    onPhase(generation, "initializing");
    void (async () => {
      let next: OwnedSparkResources | null = null;
      let unattachedRenderer: ({ dispose(): void } & THREE.Object3D) | null = null;
      let unattachedMesh: ({ dispose(): void; numSplats: number } & THREE.Object3D) | null = null;
      try {
        await afterOldDisposalFrame;
        if (stale) return;
        const { SparkRenderer, SplatFileType, SplatMesh } = await import(
          "@sparkjsdev/spark"
        );
        if (stale) return;
        const sparkRenderer = new SparkRenderer({ renderer: gl });
        unattachedRenderer = sparkRenderer;
        const sharedMeshOptions = {
          raycastable: false,
          lod: false,
          enableLod: false,
          nonLod: true,
          paged: false,
        } as const;
        const meshInitializationStartedAt = performance.now();
        const mesh = new SplatMesh(
          source === "ply"
            ? {
                ...sharedMeshOptions,
                url: SPARK_PLY_DIAGNOSTIC_URL,
                fileType: SplatFileType.PLY,
                fileName: "diagnostic-original.ply",
              }
            : diagnosticAsset
              ? {
                  ...sharedMeshOptions,
                  url: diagnosticAsset.url,
                  fileType: SplatFileType.SPZ,
                  fileName: diagnosticAsset.fileName,
                  maxSplats: diagnosticAsset.expectedSplatCount,
                }
              : {
                ...sharedMeshOptions,
                fileBytes: currentRequest.bytes,
                fileType: SplatFileType.SPZ,
                fileName: currentRequest.manifest.asset.filename,
                maxSplats: currentRequest.splatCount,
              },
        );
        unattachedMesh = mesh;
        const group = new THREE.Group();
        group.name = `environment:spark:${source}:${currentRequest.manifest.environmentId}`;
        const [x, y, z] = currentRequest.manifest.alignment.translationXyz;
        const [qx, qy, qz, qw] = currentRequest.manifest.alignment.rotationXyzw;
        group.position.set(x, y, z);
        group.quaternion.set(qx, qy, qz, qw);
        group.scale.setScalar(currentRequest.manifest.alignment.uniformScale);
        group.visible = currentRequest.visible;
        next = {
          source: integratedResolvedSource,
          group,
          mesh,
          renderer: sparkRenderer,
          disposed: false,
        };
        unattachedRenderer = null;
        unattachedMesh = null;
        await mesh.initialized;
        const readyAt = performance.now();
        if (stale) {
          disposeSparkResources(next);
          return;
        }
        emitLifecycle("after-new-initialization");
        if (
          source === "spz" &&
          (mesh.numSplats !== currentRequest.splatCount ||
            mesh.numSplats !== currentRequest.manifest.asset.splatCount)
        ) {
          throw new Error("Decoded SPZ splat count does not match validated metadata.");
        }
        if (
          diagnosticAsset &&
          mesh.numSplats !== diagnosticAsset.expectedSplatCount
        ) {
          throw new Error(
            `Decoded ${diagnosticAsset.label} splat count does not match diagnostic metadata.`,
          );
        }
        if (source === "ply") {
          console.info(
            `[Spark PLY diagnostic] Rendering ${mesh.numSplats.toLocaleString()} original PLY splats; the reduced SPZ contains ${currentRequest.splatCount.toLocaleString()}.`,
          );
        }
        group.add(mesh);
        scene.add(sparkRenderer);
        scene.add(group);
        owned.current = next;
        emitLifecycle("after-new-attachment");
        if (switchId !== null && previous) {
          pendingFrameLifecycleEvents.current.push({
            switchId,
            fromSource: previous.source,
            toSource: integratedResolvedSource,
            stage: "after-new-attachment-frame",
            startedAt: switchStartedAt,
          });
        }
        if (source !== "ply" && metricsCallbackRef.current) {
          currentMetrics.current = {
            source: integratedResolvedSource,
            decodedSplatCount: mesh.numSplats,
            assetBytes:
              diagnosticAsset?.byteSize ?? currentRequest.bytes.byteLength,
            meshInitializationMilliseconds:
              readyAt - meshInitializationStartedAt,
            firstReadyMilliseconds: readyAt - firstReadyStartedAt,
            initializationFrameCount: initializationFrameCount.current,
            stabilizationFrameCount: 0,
            stabilizationFrameTarget: STABILIZATION_FRAME_COUNT,
            stabilizedFrameCount: 0,
            frame: null,
            rendererInfo: readRendererInfo(gl),
          };
          metricsCallbackRef.current?.(currentMetrics.current);
        }
        if (source !== "ply" && lifecycleCallbackRef.current) {
          steadyLifecycleContext.current = {
            switchId: switchId ?? 0,
            fromSource: previous?.source ?? integratedResolvedSource,
            toSource: integratedResolvedSource,
            startedAt: switchStartedAt,
          };
        }
        samplingPhase.current = "stabilizing";
        onPhase(generation, "ready");
      } catch (error) {
        disposeSparkResources(next);
        unattachedMesh?.dispose();
        if (unattachedRenderer) {
          disposeSparkRendererResource(
            unattachedRenderer as SparkRendererResource,
          );
        }
        if (!stale) onError(generation, error instanceof Error ? error.message : "Spark renderer initialization failed.");
      }
    })();

    return () => {
      stale = true;
    };
  }, [
    gl,
    integratedSource,
    onError,
    onPhase,
    requestGeneration,
    scene,
    sourceSwitchToken,
  ]);

  useEffect(() => {
    if (owned.current && requestVisible !== undefined) owned.current.group.visible = requestVisible;
  }, [requestVisible]);

  return null;
}
