import { act } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";

import {
  attachDiagnosticMeshWhenReady,
  DEFAULT_DIAGNOSTIC_CONTROL_STATE,
  diagnosticControlReducer,
} from "@/components/viewer/IsolatedSparkPlyDiagnostic";

describe("isolated comparison controls", () => {
  it("hides the camera manifold by default", () => {
    expect(DEFAULT_DIAGNOSTIC_CONTROL_STATE).toMatchObject({
      source: "ply",
      cameraMode: "registered",
      selectedCameraIndex: 0,
      showCameraManifold: false,
    });
  });

  it("preserves camera selection, mode, and manifold state across sources", () => {
    let state = diagnosticControlReducer(DEFAULT_DIAGNOSTIC_CONTROL_STATE, {
      type: "select-camera",
      index: 88,
    });
    state = diagnosticControlReducer(state, {
      type: "set-manifold-visible",
      visible: true,
    });
    state = diagnosticControlReducer(state, {
      type: "set-camera-mode",
      cameraMode: "orbit",
    });
    state = diagnosticControlReducer(state, {
      type: "select-source",
      source: "spz",
    });

    expect(state).toEqual({
      source: "spz",
      cameraMode: "orbit",
      selectedCameraIndex: 88,
      showCameraManifold: true,
    });
  });

  it("never changes manifold visibility when camera controls change", () => {
    const visible = {
      ...DEFAULT_DIAGNOSTIC_CONTROL_STATE,
      showCameraManifold: true,
    };
    const orbit = diagnosticControlReducer(visible, {
      type: "set-camera-mode",
      cameraMode: "orbit",
    });
    const stepped = diagnosticControlReducer(orbit, {
      type: "select-camera",
      index: 89,
    });

    expect(orbit.showCameraManifold).toBe(true);
    expect(stepped.showCameraManifold).toBe(true);
    expect(stepped.selectedCameraIndex).toBe(89);
  });
});

class DeferredDiagnosticMesh extends THREE.Object3D {
  dispose = vi.fn();
  initialized: Promise<this>;
  numSplats: number;
  private resolveInitialization!: (mesh: this) => void;

  constructor(numSplats: number) {
    super();
    this.numSplats = numSplats;
    this.initialized = new Promise<this>((resolve) => {
      this.resolveInitialization = resolve;
    });
  }

  getBoundingBox() {
    return new THREE.Box3(
      new THREE.Vector3(-1, -1, -1),
      new THREE.Vector3(1, 1, 1),
    );
  }

  resolve() {
    this.resolveInitialization(this);
  }
}

describe("isolated Spark mesh ownership", () => {
  let scene: THREE.Scene;

  beforeEach(() => {
    scene = new THREE.Scene();
  });

  it("disposes a stale async mesh without attaching or reporting ready", async () => {
    const mesh = new DeferredDiagnosticMesh(466_363);
    const onReady = vi.fn();
    const onError = vi.fn();
    const dispose = attachDiagnosticMeshWhenReady({
      mesh,
      source: "ply",
      scene,
      onReady,
      onError,
    });

    dispose();
    await act(async () => mesh.resolve());

    expect(mesh.dispose).toHaveBeenCalledTimes(1);
    expect(mesh.parent).toBeNull();
    expect(scene.children).toHaveLength(0);
    expect(onReady).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });

  it("attaches the current SPZ once and reports its decoded count", async () => {
    const mesh = new DeferredDiagnosticMesh(86_954);
    const onReady = vi.fn();
    const onError = vi.fn();
    const dispose = attachDiagnosticMeshWhenReady({
      mesh,
      source: "spz",
      scene,
      initializationStartedAt: 100,
      now: () => 145.5,
      onReady,
      onError,
    });

    await act(async () => mesh.resolve());

    expect(mesh.parent).toBe(scene);
    expect(onReady).toHaveBeenCalledWith(
      86_954,
      [
        [-1, -1, -1],
        [1, 1, 1],
      ],
      45.5,
    );
    expect(onError).not.toHaveBeenCalled();

    dispose();
    dispose();
    expect(mesh.dispose).toHaveBeenCalledTimes(1);
    expect(scene.children).toHaveLength(0);
  });
});
