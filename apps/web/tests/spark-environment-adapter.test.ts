import { act, render, waitFor } from "@testing-library/react";
import { createElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";

import syntheticManifest from "@/tests/fixtures/environment/synthetic-v1/manifest.json";
import { decodeEnvironmentManifest } from "@/lib/environment/validate";

const mocks = vi.hoisted(() => ({
  moduleEvaluations: 0,
  meshDispose: vi.fn(),
  rendererDispose: vi.fn(),
  rendererGeometryDispose: vi.fn(),
  rendererMaterialDispose: vi.fn(),
  meshOptions: [] as Array<Record<string, unknown>>,
  pending: [] as Array<() => void>,
  frameCallbacks: [] as Array<(state: unknown, delta: number) => void>,
  scene: null as THREE.Scene | null,
  gl: {
    info: {
      memory: { geometries: 3, textures: 2 },
      programs: [{}, {}],
      render: { calls: 5, triangles: 7, points: 11, lines: 13 },
    },
  },
}));

vi.mock("@react-three/fiber", () => ({
  useThree: () => ({ gl: mocks.gl, scene: mocks.scene }),
  useFrame: (callback: (state: unknown, delta: number) => void) => {
    mocks.frameCallbacks.push(callback);
  },
}));

vi.mock("@sparkjsdev/spark", async () => {
  mocks.moduleEvaluations += 1;
  const three = await import("three");
  class MockRenderer extends three.Object3D {
    dispose = mocks.rendererDispose;
    geometry = { dispose: mocks.rendererGeometryDispose };
    material = { dispose: mocks.rendererMaterialDispose };
  }
  class MockMesh extends three.Object3D {
    dispose = mocks.meshDispose;
    numSplats: number;
    initialized = new Promise<this>((resolve) => mocks.pending.push(() => resolve(this)));
    constructor(options: Record<string, unknown>) {
      super();
      mocks.meshOptions.push(options);
      this.numSplats =
        options.fileName === "diagnostic-uniform-250k.spz"
          ? 250_000
          : options.fileName === "diagnostic-uniform-300k.spz"
            ? 300_000
            : 4;
    }
  }
  return {
    SparkRenderer: MockRenderer,
    SplatMesh: MockMesh,
    SplatFileType: { PLY: "ply", SPZ: "spz" },
  };
});

import {
  disposeSparkResources,
  resolveSparkEnvironmentSource,
  SPARK_PLY_DIAGNOSTIC_URL,
  SparkEnvironmentAdapter,
} from "@/components/viewer/SparkEnvironmentAdapter";

const manifest = decodeEnvironmentManifest(syntheticManifest);
if (manifest.status !== "available") throw new Error("Expected available fixture.");
const request = { generation: 1, manifest, bytes: new Uint8Array([1]), splatCount: 4, visible: true };
const adapter = (value: typeof request | null, onPhase = vi.fn(), onError = vi.fn()) =>
  createElement(SparkEnvironmentAdapter, { request: value, onPhase, onError });

beforeEach(() => {
  mocks.moduleEvaluations = 0;
  mocks.meshDispose.mockClear();
  mocks.rendererDispose.mockClear();
  mocks.rendererGeometryDispose.mockClear();
  mocks.rendererMaterialDispose.mockClear();
  mocks.meshOptions.length = 0;
  mocks.pending.length = 0;
  mocks.frameCallbacks.length = 0;
  mocks.scene = new THREE.Scene();
});

afterEach(() => {
  vi.unstubAllEnvs();
  window.history.replaceState(null, "", "/");
});

describe("Spark environment resource ownership", () => {
  it("does not evaluate Spark without a validated render request", () => {
    render(adapter(null));
    expect(mocks.moduleEvaluations).toBe(0);
    expect(mocks.scene?.children).toHaveLength(0);
  });

  it("gates the PLY diagnostic behind development and an explicit query", () => {
    expect(
      resolveSparkEnvironmentSource(
        "development",
        "?sparkEnvironmentSource=ply",
      ),
    ).toBe("ply");
    expect(
      resolveSparkEnvironmentSource(
        "production",
        "?sparkEnvironmentSource=ply",
      ),
    ).toBe("spz");
    expect(resolveSparkEnvironmentSource("development", "")).toBe("spz");
  });

  it("gates the integrated scaling sources behind development and an explicit query", () => {
    const search = "?environmentPerformanceDiagnostic=250k";
    expect(
      resolveSparkEnvironmentSource("development", search, "uniform250k"),
    ).toBe("uniform250k");
    expect(
      resolveSparkEnvironmentSource("development", "", "uniform250k"),
    ).toBe("spz");
    expect(
      resolveSparkEnvironmentSource("production", search, "uniform250k"),
    ).toBe("spz");
    expect(
      resolveSparkEnvironmentSource("development", search, "uniform300k"),
    ).toBe("uniform300k");
    expect(
      resolveSparkEnvironmentSource("development", "", "uniform300k"),
    ).toBe("spz");
    expect(
      resolveSparkEnvironmentSource("production", search, "uniform300k"),
    ).toBe("spz");
  });

  it("loads Uniform 250k with the same non-LOD settings and reports measured metadata", async () => {
    vi.stubEnv("NODE_ENV", "development");
    window.history.replaceState(
      null,
      "",
      "/?environmentPerformanceDiagnostic=250k",
    );
    const onMetrics = vi.fn();
    render(
      createElement(SparkEnvironmentAdapter, {
        request,
        onPhase: vi.fn(),
        onError: vi.fn(),
        integratedSource: "uniform250k",
        readyRequestedAt: performance.now(),
        onPerformanceMetrics: onMetrics,
      }),
    );
    await waitFor(() => expect(mocks.pending).toHaveLength(1));
    expect(mocks.meshOptions[0]).toEqual(
      expect.objectContaining({
        url: "/environment-data/__local-real__/diagnostic-uniform-250k.spz",
        fileType: "spz",
        fileName: "diagnostic-uniform-250k.spz",
        maxSplats: 250_000,
        raycastable: false,
        lod: false,
        enableLod: false,
        nonLod: true,
        paged: false,
      }),
    );
    expect(mocks.meshOptions[0]).not.toHaveProperty("fileBytes");
    act(() => {
      mocks.frameCallbacks[0]({}, 0.2);
      mocks.frameCallbacks[0]({}, 0.15);
    });
    await act(async () => mocks.pending[0]());
    await waitFor(() =>
      expect(onMetrics).toHaveBeenLastCalledWith(
        expect.objectContaining({
          source: "uniform250k",
          decodedSplatCount: 250_000,
          assetBytes: 3_990_074,
          initializationFrameCount: 2,
          frame: null,
          rendererInfo: expect.objectContaining({
            geometries: 3,
            textures: 2,
            programs: 2,
          }),
        }),
      ),
    );
    act(() => {
      for (let index = 0; index < 60; index += 1) {
        mocks.frameCallbacks[0]({}, 0.02);
      }
    });
    expect(onMetrics).toHaveBeenLastCalledWith(
      expect.objectContaining({
        frame: expect.objectContaining({
          averageMilliseconds: 20,
          framesPerSecond: 50,
          p50Milliseconds: 20,
          p95Milliseconds: 20,
        }),
      }),
    );
  });

  it("loads Uniform 300k with the shared settings and full stabilized measurement window", async () => {
    vi.stubEnv("NODE_ENV", "development");
    window.history.replaceState(
      null,
      "",
      "/?environmentPerformanceDiagnostic=250k",
    );
    const onMetrics = vi.fn();
    render(
      createElement(SparkEnvironmentAdapter, {
        request,
        onPhase: vi.fn(),
        onError: vi.fn(),
        integratedSource: "uniform300k",
        readyRequestedAt: performance.now(),
        onPerformanceMetrics: onMetrics,
      }),
    );
    await waitFor(() => expect(mocks.pending).toHaveLength(1));
    expect(mocks.meshOptions[0]).toEqual(
      expect.objectContaining({
        url: "/environment-data/__local-real__/diagnostic-uniform-300k.spz",
        fileType: "spz",
        fileName: "diagnostic-uniform-300k.spz",
        maxSplats: 300_000,
        raycastable: false,
        lod: false,
        enableLod: false,
        nonLod: true,
        paged: false,
      }),
    );
    expect(mocks.meshOptions[0]).not.toHaveProperty("fileBytes");
    await act(async () => mocks.pending[0]());
    await waitFor(() =>
      expect(onMetrics).toHaveBeenLastCalledWith(
        expect.objectContaining({
          source: "uniform300k",
          decodedSplatCount: 300_000,
          assetBytes: 4_785_949,
          stabilizationFrameTarget: 30,
          frame: null,
        }),
      ),
    );
    act(() => {
      for (let index = 0; index < 150; index += 1) {
        mocks.frameCallbacks[0]({}, 0.02);
      }
    });
    expect(onMetrics).toHaveBeenLastCalledWith(
      expect.objectContaining({
        source: "uniform300k",
        stabilizationFrameCount: 30,
        stabilizedFrameCount: 120,
        frame: expect.objectContaining({
          sampleCount: 120,
          windowSize: 120,
          averageMilliseconds: 20,
          framesPerSecond: 50,
          p50Milliseconds: 20,
          p90Milliseconds: 20,
          p95Milliseconds: 20,
          p99Milliseconds: 20,
          maxMilliseconds: 20,
          over20Milliseconds: 0,
          over25Milliseconds: 0,
          over33Point3Milliseconds: 0,
          over50Milliseconds: 0,
        }),
        rendererInfo: expect.objectContaining({
          geometries: 3,
          textures: 2,
          programs: 2,
          renderCalls: 5,
          triangles: 7,
          points: 11,
          lines: 13,
        }),
      }),
    );
  });

  it("keeps the validated production SPZ constructor path unchanged", async () => {
    vi.stubEnv("NODE_ENV", "production");
    window.history.replaceState(
      null,
      "",
      "/?environmentPerformanceDiagnostic=250k",
    );
    render(
      createElement(SparkEnvironmentAdapter, {
        request,
        onPhase: vi.fn(),
        onError: vi.fn(),
        integratedSource: "uniform300k",
      }),
    );
    await waitFor(() => expect(mocks.meshOptions).toHaveLength(1));
    expect(mocks.meshOptions[0]).toEqual(
      expect.objectContaining({
        fileBytes: request.bytes,
        fileType: "spz",
        fileName: request.manifest.asset.filename,
        maxSplats: request.splatCount,
      }),
    );
    expect(mocks.meshOptions[0]).not.toHaveProperty("url");
  });

  it("disposes a stale mesh completion when the diagnostic source changes", async () => {
    vi.stubEnv("NODE_ENV", "development");
    window.history.replaceState(
      null,
      "",
      "/?environmentPerformanceDiagnostic=250k",
    );
    const shared = { onPhase: vi.fn(), onError: vi.fn() };
    const view = render(
      createElement(SparkEnvironmentAdapter, {
        request,
        ...shared,
        integratedSource: "current",
      }),
    );
    await waitFor(() => expect(mocks.pending).toHaveLength(1));
    view.rerender(
      createElement(SparkEnvironmentAdapter, {
        request,
        ...shared,
        integratedSource: "uniform300k",
      }),
    );
    await waitFor(() => expect(mocks.pending).toHaveLength(2));
    await act(async () => mocks.pending[0]());
    await waitFor(() => expect(mocks.meshDispose).toHaveBeenCalledTimes(1));
    expect(mocks.scene?.children).toHaveLength(0);
    await act(async () => mocks.pending[1]());
    await waitFor(() => expect(mocks.scene?.children).toHaveLength(2));
    expect(shared.onPhase).toHaveBeenLastCalledWith(1, "ready");
  });

  it("reports ordered resource checkpoints around a ready source switch", async () => {
    vi.stubEnv("NODE_ENV", "development");
    window.history.replaceState(
      null,
      "",
      "/?environmentPerformanceDiagnostic=250k",
    );
    const onLifecycle = vi.fn();
    const shared = {
      request,
      onPhase: vi.fn(),
      onError: vi.fn(),
      onResourceLifecycleEvent: onLifecycle,
    };
    const view = render(
      createElement(SparkEnvironmentAdapter, {
        ...shared,
        integratedSource: "uniform250k",
      }),
    );
    await waitFor(() => expect(mocks.pending).toHaveLength(1));
    await act(async () => mocks.pending[0]());
    await waitFor(() => expect(mocks.scene?.children).toHaveLength(2));

    view.rerender(
      createElement(SparkEnvironmentAdapter, {
        ...shared,
        integratedSource: "uniform300k",
      }),
    );
    act(() => mocks.frameCallbacks[0]({}, 0.016));
    await waitFor(() => expect(mocks.pending).toHaveLength(2));
    await act(async () => mocks.pending[1]());
    await waitFor(() => expect(mocks.scene?.children).toHaveLength(2));
    act(() => mocks.frameCallbacks[0]({}, 0.016));
    act(() => {
      for (let index = 0; index < 149; index += 1) {
        mocks.frameCallbacks[0]({}, 0.016);
      }
    });

    expect(
      onLifecycle.mock.calls.map(
        (call) => (call[0] as { stage: string }).stage,
      ),
    ).toEqual([
      "before-switch",
      "after-old-disposal",
      "after-old-disposal-frame",
      "after-new-initialization",
      "after-new-attachment",
      "after-new-attachment-frame",
      "steady-state",
    ]);
    expect(onLifecycle).toHaveBeenLastCalledWith(
      expect.objectContaining({
        fromSource: "uniform250k",
        toSource: "uniform300k",
        rendererInfo: expect.objectContaining({
          geometries: 3,
          textures: 2,
          programs: 2,
        }),
      }),
    );
  });

  it("loads the ignored original PLY URL directly through Spark in diagnostic mode", async () => {
    vi.stubEnv("NODE_ENV", "development");
    window.history.replaceState(
      null,
      "",
      "/?sparkEnvironmentSource=ply",
    );
    const info = vi.spyOn(console, "info").mockImplementation(() => undefined);

    render(adapter(request));

    await waitFor(() => expect(mocks.pending).toHaveLength(1));
    expect(mocks.meshOptions[0]).toEqual(
      expect.objectContaining({
        url: SPARK_PLY_DIAGNOSTIC_URL,
        fileType: "ply",
        fileName: "diagnostic-original.ply",
      }),
    );
    expect(mocks.meshOptions[0]).not.toHaveProperty("fileBytes");
    expect(mocks.meshOptions[0]).not.toHaveProperty("maxSplats");
    await act(async () => mocks.pending[0]());
    await waitFor(() => expect(info).toHaveBeenCalledOnce());
    info.mockRestore();
  });

  it("never attaches a stale completion and disposes it immediately", async () => {
    const onPhase = vi.fn();
    const onError = vi.fn();
    const view = render(adapter(request, onPhase, onError));
    await waitFor(() => expect(mocks.pending).toHaveLength(1));
    view.rerender(adapter(null, onPhase, onError));
    await act(async () => mocks.pending[0]());
    await waitFor(() => expect(mocks.meshDispose).toHaveBeenCalledTimes(1));
    expect(mocks.rendererDispose).toHaveBeenCalledTimes(1);
    expect(mocks.rendererGeometryDispose).toHaveBeenCalledTimes(1);
    expect(mocks.rendererMaterialDispose).toHaveBeenCalledTimes(1);
    expect(mocks.scene?.children).toHaveLength(0);
    expect(onPhase).not.toHaveBeenCalledWith(1, "ready");
  });

  it("attaches once, hides without disposal, then unloads idempotently", async () => {
    const onPhase = vi.fn();
    const onError = vi.fn();
    const view = render(adapter(request, onPhase, onError));
    await waitFor(() => expect(mocks.pending).toHaveLength(1));
    await act(async () => mocks.pending[0]());
    await waitFor(() => expect(onPhase).toHaveBeenCalledWith(1, "ready"));
    expect(mocks.scene?.children).toHaveLength(2);
    view.rerender(adapter({ ...request, visible: false }, onPhase, onError));
    expect(mocks.meshDispose).not.toHaveBeenCalled();
    view.rerender(adapter(null, onPhase, onError));
    await waitFor(() => expect(mocks.meshDispose).toHaveBeenCalledTimes(1));
    view.unmount();
    expect(mocks.meshDispose).toHaveBeenCalledTimes(1);
    expect(mocks.rendererDispose).toHaveBeenCalledTimes(1);
    expect(mocks.rendererGeometryDispose).toHaveBeenCalledTimes(1);
    expect(mocks.rendererMaterialDispose).toHaveBeenCalledTimes(1);
  });

  it("makes direct final cleanup idempotent", () => {
    const group = { remove: vi.fn(), parent: { remove: vi.fn() } };
    const mesh = { dispose: vi.fn() };
    const renderer = {
      dispose: vi.fn(),
      geometry: { dispose: vi.fn() },
      material: { dispose: vi.fn() },
      parent: { remove: vi.fn() },
    };
    const resources = {
      source: "current",
      group,
      mesh,
      renderer,
      disposed: false,
    };
    disposeSparkResources(resources as never);
    disposeSparkResources(resources as never);
    expect(mesh.dispose).toHaveBeenCalledTimes(1);
    expect(renderer.dispose).toHaveBeenCalledTimes(1);
    expect(renderer.geometry.dispose).toHaveBeenCalledTimes(1);
    expect(renderer.material.dispose).toHaveBeenCalledTimes(1);
  });
});
