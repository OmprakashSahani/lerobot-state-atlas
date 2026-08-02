"use client";

import { useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";

import type { ValidatedEnvironmentRenderRequest } from "@/lib/environment/types";

export type SparkAdapterPhase = "initializing" | "ready";

interface OwnedSparkResources {
  group: THREE.Group;
  mesh: { dispose(): void; numSplats: number } & THREE.Object3D;
  renderer: { dispose(): void } & THREE.Object3D;
  disposed: boolean;
}

export function disposeSparkResources(resources: OwnedSparkResources | null): void {
  if (!resources || resources.disposed) return;
  resources.disposed = true;
  resources.group.remove(resources.mesh);
  resources.group.parent?.remove(resources.group);
  resources.renderer.parent?.remove(resources.renderer);
  resources.mesh.dispose();
  resources.renderer.dispose();
}

export function SparkEnvironmentAdapter({
  request,
  onPhase,
  onError,
}: {
  request: ValidatedEnvironmentRenderRequest | null;
  onPhase: (generation: number, phase: SparkAdapterPhase) => void;
  onError: (generation: number, message: string) => void;
}) {
  const { gl, scene } = useThree();
  const owned = useRef<OwnedSparkResources | null>(null);
  const requestRef = useRef(request);
  const requestGeneration = request?.generation;
  const requestVisible = request?.visible;

  useEffect(() => {
    requestRef.current = request;
  }, [request]);

  useEffect(() => {
    disposeSparkResources(owned.current);
    owned.current = null;
    const currentRequest = requestRef.current;
    if (!currentRequest) return;

    let stale = false;
    const generation = currentRequest.generation;
    onPhase(generation, "initializing");
    void (async () => {
      let next: OwnedSparkResources | null = null;
      let unattachedRenderer: ({ dispose(): void } & THREE.Object3D) | null = null;
      let unattachedMesh: ({ dispose(): void; numSplats: number } & THREE.Object3D) | null = null;
      try {
        const { SparkRenderer, SplatFileType, SplatMesh } = await import(
          "@sparkjsdev/spark"
        );
        if (stale) return;
        const sparkRenderer = new SparkRenderer({ renderer: gl });
        unattachedRenderer = sparkRenderer;
        const mesh = new SplatMesh({
          fileBytes: currentRequest.bytes,
          fileType: SplatFileType.SPZ,
          fileName: currentRequest.manifest.asset.filename,
          raycastable: false,
          maxSplats: currentRequest.splatCount,
          lod: false,
          enableLod: false,
          nonLod: true,
          paged: false,
        });
        unattachedMesh = mesh;
        const group = new THREE.Group();
        group.name = `environment:spark:${currentRequest.manifest.environmentId}`;
        const [x, y, z] = currentRequest.manifest.alignment.translationXyz;
        const [qx, qy, qz, qw] = currentRequest.manifest.alignment.rotationXyzw;
        group.position.set(x, y, z);
        group.quaternion.set(qx, qy, qz, qw);
        group.scale.setScalar(currentRequest.manifest.alignment.uniformScale);
        group.visible = currentRequest.visible;
        next = { group, mesh, renderer: sparkRenderer, disposed: false };
        unattachedRenderer = null;
        unattachedMesh = null;
        await mesh.initialized;
        if (mesh.numSplats !== currentRequest.splatCount || mesh.numSplats !== currentRequest.manifest.asset.splatCount) {
          throw new Error("Decoded SPZ splat count does not match validated metadata.");
        }
        if (stale) {
          disposeSparkResources(next);
          return;
        }
        group.add(mesh);
        scene.add(sparkRenderer);
        scene.add(group);
        owned.current = next;
        onPhase(generation, "ready");
      } catch (error) {
        disposeSparkResources(next);
        unattachedMesh?.dispose();
        unattachedRenderer?.dispose();
        if (!stale) onError(generation, error instanceof Error ? error.message : "Spark renderer initialization failed.");
      }
    })();

    return () => {
      stale = true;
      disposeSparkResources(owned.current);
      owned.current = null;
    };
  }, [gl, onError, onPhase, requestGeneration, scene]);

  useEffect(() => {
    if (owned.current && requestVisible !== undefined) owned.current.group.visible = requestVisible;
  }, [requestVisible]);

  return null;
}
