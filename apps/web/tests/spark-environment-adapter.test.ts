import { act, render, waitFor } from "@testing-library/react";
import { createElement } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";

import syntheticManifest from "@/tests/fixtures/environment/synthetic-v1/manifest.json";
import { decodeEnvironmentManifest } from "@/lib/environment/validate";

const mocks = vi.hoisted(() => ({
  moduleEvaluations: 0,
  meshDispose: vi.fn(),
  rendererDispose: vi.fn(),
  pending: [] as Array<() => void>,
  scene: null as THREE.Scene | null,
  gl: {},
}));

vi.mock("@react-three/fiber", () => ({
  useThree: () => ({ gl: mocks.gl, scene: mocks.scene }),
}));

vi.mock("@sparkjsdev/spark", async () => {
  mocks.moduleEvaluations += 1;
  const three = await import("three");
  class MockRenderer extends three.Object3D { dispose = mocks.rendererDispose; }
  class MockMesh extends three.Object3D {
    dispose = mocks.meshDispose;
    numSplats = 4;
    initialized = new Promise<this>((resolve) => mocks.pending.push(() => resolve(this)));
  }
  return { SparkRenderer: MockRenderer, SplatMesh: MockMesh, SplatFileType: { SPZ: "spz" } };
});

import { disposeSparkResources, SparkEnvironmentAdapter } from "@/components/viewer/SparkEnvironmentAdapter";

const manifest = decodeEnvironmentManifest(syntheticManifest);
if (manifest.status !== "available") throw new Error("Expected available fixture.");
const request = { generation: 1, manifest, bytes: new Uint8Array([1]), splatCount: 4, visible: true };
const adapter = (value: typeof request | null, onPhase = vi.fn(), onError = vi.fn()) =>
  createElement(SparkEnvironmentAdapter, { request: value, onPhase, onError });

beforeEach(() => {
  mocks.moduleEvaluations = 0;
  mocks.meshDispose.mockClear();
  mocks.rendererDispose.mockClear();
  mocks.pending.length = 0;
  mocks.scene = new THREE.Scene();
});

describe("Spark environment resource ownership", () => {
  it("does not evaluate Spark without a validated render request", () => {
    render(adapter(null));
    expect(mocks.moduleEvaluations).toBe(0);
    expect(mocks.scene?.children).toHaveLength(0);
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
  });

  it("makes direct final cleanup idempotent", () => {
    const group = { remove: vi.fn(), parent: { remove: vi.fn() } };
    const mesh = { dispose: vi.fn() };
    const renderer = { dispose: vi.fn(), parent: { remove: vi.fn() } };
    const resources = { group, mesh, renderer, disposed: false };
    disposeSparkResources(resources as never);
    disposeSparkResources(resources as never);
    expect(mesh.dispose).toHaveBeenCalledTimes(1);
    expect(renderer.dispose).toHaveBeenCalledTimes(1);
  });
});
