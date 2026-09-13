import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Matrix4, Vector3 } from "three";

import coverageJson from "@/public/atlas-data/demo-v2/coverage.json";
import manifestJson from "@/public/atlas-data/demo-v2/manifest.json";
import type { AtlasData } from "@/lib/atlas-schema/types";
import { decodeCoverage, decodeManifest } from "@/lib/atlas-schema/validate";
import { prepareCoverage } from "@/lib/data/prepareCoverage";
import type { SupportedCameraManifold } from "@/lib/environment/camera-viewing-constraint";
import type { ConvertedRegisteredCamera } from "@/lib/environment/camera-manifold";
import type {
  AvailableEnvironmentManifest,
  ValidatedEnvironmentRenderRequest,
} from "@/lib/environment/types";
import type { LocalEnvironmentController } from "@/lib/environment/use-local-environment";
import { AtlasViewer } from "@/components/viewer/AtlasViewer";

const manifest = decodeManifest(manifestJson);
const coverage = decodeCoverage(coverageJson);
const atlasData: AtlasData = {
  manifest,
  coverage,
  preparedArms: prepareCoverage(manifest, coverage),
};
const resetCamera = vi.fn();

function capturedCamera(
  registeredIndex: number,
  colmapImageId: number,
  sourceFrame: number,
): ConvertedRegisteredCamera {
  const center = new Vector3(registeredIndex * 0.01, 0, 0);
  const cameraToWorld = new Matrix4().setPosition(center);
  return {
    registeredIndex,
    colmapImageId,
    sourceImage: `frame_${String(sourceFrame).padStart(4, "0")}.jpg`,
    cameraId: 1,
    cameraModel: "PINHOLE",
    width: 1903,
    height: 1070,
    fx: 1688.3917720798788,
    fy: 1688.3917720798788,
    cx: 951.5,
    cy: 535,
    quaternionWxyz: [1, 0, 0, 0],
    translation: [0, 0, 0],
    evalIndex: colmapImageId === 1 ? 0 : null,
    evalPsnr: colmapImageId === 1 ? 28.059858 : null,
    cameraToWorld,
    center,
    viewingDirection: new Vector3(0, 0, -1),
  };
}

const supportedCameras = [
  capturedCamera(0, 2, 1),
  capturedCamera(1, 3, 2),
  capturedCamera(2, 1, 3),
  capturedCamera(3, 4, 4),
];
const supportedManifold: SupportedCameraManifold = {
  cameras: supportedCameras,
  detachedCameras: [],
  positionTolerance: 0.1,
  orientationToleranceRadians: Math.PI / 12,
  defaultCameraIndex: 2,
};
const environmentManifest: AvailableEnvironmentManifest = {
  schema: {
    name: "lerobot-state-atlas.environment-layer",
    major: 1,
    minor: 0,
  },
  environmentId: "omprakash-workcell-real-scan",
  label: "Omprakash workcell — uncalibrated real scan",
  status: "available",
  provenance: {
    sourceKind: "real-scan",
    description: "Reviewed test environment.",
    reconstructionClaim: false,
  },
  coordinateFrame: "canonical-shared-world",
  alignment: {
    translationXyz: [0, 0, 0],
    rotationXyzw: [0, 0, 0, 1],
    uniformScale: 1,
    calibrated: false,
    disclosure: "No robot-world calibration.",
  },
  bounds: {
    minimumXyz: [-1, -1, -1],
    maximumXyz: [1, 1, 1],
  },
  asset: {
    filename: "omprakash-workcell.spz",
    format: "spz",
    mimeType: "application/octet-stream",
    byteSize: 3_990_074,
    sha256:
      "f90b5ed3d1c672f62e25fca1e6acf9534901f9e8c8c80ff1a6c6b874c1d7386d",
    splatCount: 250_000,
  },
};
const environmentRequest: ValidatedEnvironmentRenderRequest = {
  generation: 1,
  manifest: environmentManifest,
  bytes: new Uint8Array([1]),
  splatCount: 250_000,
  visible: true,
};

let localController: LocalEnvironmentController;
let viewerCanvasProps: {
  gaussianViewingConstraint?: {
    active: boolean;
    camera: ConvertedRegisteredCamera;
  };
} | null = null;
let rerenderViewer: (() => void) | null = null;

vi.mock("@/components/viewer/AtlasDataProvider", () => ({
  useAtlasData: () => ({ status: "ready", data: atlasData }),
}));

vi.mock("@/components/viewer/ViewerStore", () => ({
  useViewerStore: () => ({
    leftVisible: true,
    rightVisible: true,
    cameraResetToken: 0,
    metric: "visits",
    spacing: manifest.coverage.armSpacing,
    radius: 0.05,
    selection: null,
    autoRotate: false,
    toggleArm: vi.fn(),
    resetCamera,
    setMetric: vi.fn(),
    setSpacing: vi.fn(),
    setRadius: vi.fn(),
    selectVoxel: vi.fn(),
    clearSelection: vi.fn(),
    setAutoRotate: vi.fn(),
  }),
}));

vi.mock("@/components/viewer/ViewerCanvas", () => ({
  ViewerCanvas: (props: NonNullable<typeof viewerCanvasProps>) => {
    viewerCanvasProps = props;
    return <div data-testid="viewer-canvas" />;
  },
}));

vi.mock("@/lib/environment/use-local-environment", () => ({
  useLocalEnvironmentSpike: () => localController,
}));

vi.mock("@/lib/environment/camera-viewing-constraint", async (importOriginal) => {
  const original = await importOriginal<
    typeof import("@/lib/environment/camera-viewing-constraint")
  >();
  return {
    ...original,
    loadReviewedCameraManifold: vi.fn(async () => supportedManifold),
  };
});

function controller(
  request: ValidatedEnvironmentRenderRequest | null,
): LocalEnvironmentController {
  return {
    phase: request
      ? request.visible
        ? "ready-visible"
        : "ready-hidden"
      : "idle",
    request,
    loadRequestedAt: null,
    disclosure: request?.manifest.alignment.disclosure,
    load: vi.fn(),
    hide: vi.fn(),
    show: vi.fn(),
    unload: vi.fn(),
    retry: vi.fn(),
    onRendererPhase: vi.fn(),
    onRendererError: vi.fn(),
    setWebGl2Supported: vi.fn(),
  };
}

function interactiveController(
  request: ValidatedEnvironmentRenderRequest,
): LocalEnvironmentController {
  const result = controller(request);
  result.hide = vi.fn(() => {
    localController = interactiveController({ ...request, visible: false });
    rerenderViewer?.();
  });
  result.show = vi.fn(() => {
    localController = interactiveController({ ...request, visible: true });
    rerenderViewer?.();
  });
  return result;
}

beforeEach(() => {
  localController = interactiveController(environmentRequest);
  viewerCanvasProps = null;
  rerenderViewer = null;
  resetCamera.mockClear();
});

afterEach(() => cleanup());

describe("supported captured-view controls", () => {
  it("shows Reset camera and returns a changed view to COLMAP 1 / frame_0003.jpg", async () => {
    const sceneBoundsBefore = structuredClone(atlasData.manifest.sceneBounds);
    const analyticalCentersBefore = Array.from(atlasData.preparedArms[0].centers);
    render(<AtlasViewer />);

    const controls = await screen.findByRole("group", {
      name: "Supported captured views",
    });
    const reset = within(controls).getByRole("button", {
      name: "Reset camera",
    });
    expect(reset).toBeVisible();
    await waitFor(() =>
      expect(within(controls).getByText("Captured view 3 of 4")).toBeVisible(),
    );
    expect(viewerCanvasProps?.gaussianViewingConstraint?.camera).toBe(
      supportedCameras[2],
    );

    fireEvent.click(within(controls).getByRole("button", { name: "Next view" }));
    await waitFor(() =>
      expect(within(controls).getByText("Captured view 4 of 4")).toBeVisible(),
    );
    expect(viewerCanvasProps?.gaussianViewingConstraint?.camera).toBe(
      supportedCameras[3],
    );

    fireEvent.click(reset);
    await waitFor(() =>
      expect(within(controls).getByText("Captured view 3 of 4")).toBeVisible(),
    );
    expect(viewerCanvasProps?.gaussianViewingConstraint?.camera).toBe(
      supportedCameras[2],
    );
    expect(resetCamera).toHaveBeenCalledTimes(1);
    expect(atlasData.manifest.sceneBounds).toEqual(sceneBoundsBefore);
    expect(Array.from(atlasData.preparedArms[0].centers)).toEqual(
      analyticalCentersBefore,
    );
  });

  it("does not render supported-view controls without an active real environment", () => {
    localController = controller(null);
    render(<AtlasViewer />);
    expect(
      screen.queryByRole("group", { name: "Supported captured views" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reset camera" })).toBeVisible();
  });

  it("hides the rail for analytical orbit and restores the selected view when shown again", async () => {
    const sceneBoundsBefore = structuredClone(atlasData.manifest.sceneBounds);
    const analyticalCentersBefore = Array.from(atlasData.preparedArms[0].centers);
    const rendered = render(<AtlasViewer />);
    rerenderViewer = () => rendered.rerender(<AtlasViewer />);

    let controls = await screen.findByRole("group", {
      name: "Supported captured views",
    });
    await waitFor(() =>
      expect(within(controls).getByText("Captured view 3 of 4")).toBeVisible(),
    );
    fireEvent.click(within(controls).getByRole("button", { name: "Next view" }));
    await waitFor(() =>
      expect(within(controls).getByText("Captured view 4 of 4")).toBeVisible(),
    );

    fireEvent.click(screen.getByRole("button", { name: "Hide" }));
    expect(
      screen.queryByRole("group", { name: "Supported captured views" }),
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText("Auto rotate")).toBeEnabled();
    expect(screen.getByText(/Drag to orbit · Scroll to zoom/)).toBeVisible();
    expect(screen.getByText("Environment hidden")).toBeVisible();
    expect(viewerCanvasProps?.gaussianViewingConstraint).toMatchObject({
      active: false,
      camera: supportedCameras[3],
    });

    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    controls = await screen.findByRole("group", {
      name: "Supported captured views",
    });
    expect(within(controls).getByText("Captured view 4 of 4")).toBeVisible();
    expect(viewerCanvasProps?.gaussianViewingConstraint).toMatchObject({
      active: true,
      camera: supportedCameras[3],
    });
    expect(screen.getByLabelText("Auto rotate")).toBeDisabled();
    expect(atlasData.manifest.sceneBounds).toEqual(sceneBoundsBefore);
    expect(Array.from(atlasData.preparedArms[0].centers)).toEqual(
      analyticalCentersBefore,
    );
  });
});
