"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { configuredLocalEnvironment, isConservativeSpikeMobile } from "./config";
import { loadVerifiedEnvironmentAsset } from "./load-asset";
import { loadLocalEnvironmentManifest } from "./load-manifest";
import { inspectSpzHeader } from "./spz-header";
import type { ValidatedEnvironmentRenderRequest } from "./types";
import type { SparkAdapterPhase } from "@/components/viewer/SparkEnvironmentAdapter";

export type LocalEnvironmentPhase =
  | "unavailable"
  | "idle"
  | "loading-manifest"
  | "loading-asset"
  | "verifying-checksum"
  | "inspecting-spz"
  | "initializing-renderer"
  | "ready-visible"
  | "ready-hidden"
  | "unloading"
  | "error"
  | "unsupported-webgl2"
  | "mobile-refusal";

export interface LocalEnvironmentController {
  phase: LocalEnvironmentPhase;
  request: ValidatedEnvironmentRenderRequest | null;
  error?: string;
  disclosure?: string;
  load(): void;
  hide(): void;
  show(): void;
  unload(): void;
  retry(): void;
  onRendererPhase(generation: number, phase: SparkAdapterPhase): void;
  onRendererError(generation: number, message: string): void;
  setWebGl2Supported(supported: boolean): void;
}

export function useLocalEnvironmentSpike(): LocalEnvironmentController {
  const configuration = useMemo(() => configuredLocalEnvironment(), []);
  const mobile = useSyncExternalStore(
    () => () => undefined,
    () => configuration.status === "available" && isConservativeSpikeMobile(navigator),
    () => false,
  );
  const [phase, setPhase] = useState<LocalEnvironmentPhase>(
    configuration.status === "available" && mobile ? "mobile-refusal" : configuration.status === "available" ? "idle" : "unavailable",
  );
  const [request, setRequest] = useState<ValidatedEnvironmentRenderRequest | null>(null);
  const [error, setError] = useState<string>();
  const generation = useRef(0);
  const abortController = useRef<AbortController | null>(null);
  const active = useRef(false);
  const mounted = useRef(true);
  const webGl2 = useRef<boolean | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      generation.current += 1;
      abortController.current?.abort();
      active.current = false;
    };
  }, []);

  const effectivePhase = mobile && configuration.status === "available" ? "mobile-refusal" : phase;

  const unload = useCallback(() => {
    generation.current += 1;
    abortController.current?.abort();
    abortController.current = null;
    active.current = false;
    setPhase("unloading");
    setRequest(null);
    setError(undefined);
    queueMicrotask(() => {
      if (mounted.current) setPhase(configuration.status === "available" ? "idle" : "unavailable");
    });
  }, [configuration.status]);

  const load = useCallback(() => {
    if (configuration.status !== "available" || active.current || mobile) return;
    if (webGl2.current === false) {
      setPhase("unsupported-webgl2");
      return;
    }
    const current = ++generation.current;
    active.current = true;
    setRequest(null);
    setError(undefined);
    const controller = new AbortController();
    abortController.current = controller;
    void (async () => {
      try {
        setPhase("loading-manifest");
        const loaded = await loadLocalEnvironmentManifest(
          configuration.manifestPath,
          window.location.origin,
          controller.signal,
        );
        if (generation.current !== current) return;
        setPhase("loading-asset");
        const bytes = await loadVerifiedEnvironmentAsset(
          loaded.assetPath,
          loaded.manifest,
          window.location.origin,
          controller.signal,
          fetch,
          (next) => {
            if (generation.current === current && next === "verifying") setPhase("verifying-checksum");
          },
        );
        if (generation.current !== current) return;
        setPhase("inspecting-spz");
        const header = await inspectSpzHeader(bytes, loaded.manifest.asset.splatCount);
        if (generation.current !== current) return;
        setRequest({ generation: current, manifest: loaded.manifest, bytes, splatCount: header.splatCount, visible: true });
        setPhase("initializing-renderer");
      } catch (caught) {
        if (controller.signal.aborted || generation.current !== current) return;
        active.current = false;
        setError(caught instanceof Error ? caught.message : "Environment load failed.");
        setPhase("error");
      }
    })();
  }, [configuration, mobile]);

  const onRendererPhase = useCallback((value: number, next: SparkAdapterPhase) => {
    if (value !== generation.current) return;
    if (next === "initializing") setPhase("initializing-renderer");
    if (next === "ready") {
      active.current = false;
      setPhase("ready-visible");
    }
  }, []);
  const onRendererError = useCallback((value: number, message: string) => {
    if (value !== generation.current) return;
    active.current = false;
    setRequest(null);
    setError(message);
    setPhase("error");
  }, []);

  return {
    phase: effectivePhase,
    request,
    error,
    disclosure: request?.manifest.alignment.disclosure,
    load,
    retry: load,
    hide: () => {
      setRequest((current) => current ? { ...current, visible: false } : current);
      setPhase("ready-hidden");
    },
    show: () => {
      setRequest((current) => current ? { ...current, visible: true } : current);
      setPhase("ready-visible");
    },
    unload,
    onRendererPhase,
    onRendererError,
    setWebGl2Supported: (supported) => {
      webGl2.current = supported;
      if (!supported && phase === "idle") setPhase("unsupported-webgl2");
    },
  };
}
