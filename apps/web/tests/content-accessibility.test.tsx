import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v1/manifest.json";
import coverageJson from "@/public/atlas-data/demo-v1/coverage.json";
import trajectoriesJson from "@/public/atlas-data/demo-v1/trajectories.json";
import HomePage from "@/app/page";
import MethodologyPage from "@/app/methodology/page";
import { AtlasViewer } from "@/components/viewer/AtlasViewer";
import {
  decodeCoverage,
  decodeManifest,
  decodeTrajectories,
} from "@/lib/atlas-schema/validate";
import { prepareCoverage } from "@/lib/data/prepareCoverage";

const manifest = decodeManifest(manifestJson);
const coverage = decodeCoverage(coverageJson);
const preparedArmsForTest = prepareCoverage(manifest, coverage);
const setSpacingMock = vi.fn();

afterEach(() => {
  cleanup();
  setSpacingMock.mockClear();
});

vi.mock("@/components/viewer/AtlasDataProvider", () => ({
  useAtlasData: () => ({
    status: "ready",
    data: {
      manifest,
      coverage,
      preparedArms: prepareCoverage(manifest, coverage),
    },
  }),
}));

vi.mock("@/components/viewer/ViewerStore", () => ({
  useViewerStore: () => ({
    leftVisible: true,
    rightVisible: true,
    cameraResetToken: 0,
    metric: "visits",
    spacing: manifest.coverage.armSpacing,
    radius: 0.05,
    selection: {
      arm: "left",
      voxelEntryIndex: 0,
      exportedCenter: Array.from(
        preparedArmsForTest[0].centers.slice(0, 3),
      ) as [number, number, number],
    },
    autoRotate: false,
    toggleArm: vi.fn(),
    resetCamera: vi.fn(),
    setMetric: vi.fn(),
    setSpacing: setSpacingMock,
    setRadius: vi.fn(),
    selectVoxel: vi.fn(),
    clearSelection: vi.fn(),
    setAutoRotate: vi.fn(),
  }),
}));

vi.mock("@/components/viewer/ViewerCanvas", () => ({
  ViewerCanvas: () => <div data-testid="viewer-canvas" />,
}));

vi.mock("@/lib/data/loadBundle", () => ({
  loadTrajectories: vi.fn(async () => decodeTrajectories(trajectoriesJson)),
}));

describe("accessible product content", () => {
  it("presents a clear landing action and calibration caveat", () => {
    render(<HomePage />);
    expect(
      screen.getByRole("link", { name: "Open shared-world viewer" }),
    ).toHaveAttribute("href", "/viewer/demo");
    expect(screen.getByText(/not calibrated physical geometry/i)).toBeVisible();
  });

  it("documents tool-point counting semantics", () => {
    render(<MethodologyPage />);
    expect(screen.getByText("Tool-point visits")).toBeVisible();
    expect(
      screen.getByText(/single dataset frame can contribute two tool points/i),
    ).toBeVisible();
  });

  it("labels viewer controls and preserves the spacing disclosure", () => {
    render(<AtlasViewer />);
    expect(
      screen.getByRole("complementary", {
        name: "Viewer controls and metadata",
      }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Reset camera" })).toBeVisible();
    expect(screen.getByLabelText("Metric")).toBeVisible();
    expect(screen.getByLabelText("Visits color range")).toBeVisible();
    expect(screen.getByLabelText("Query radius: 0.050 m")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Clear selection" }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Load playback" })).toBeVisible();
    expect(screen.getByLabelText("Auto rotate")).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Robot setup" }),
    ).toBeVisible();
    expect(screen.getByLabelText("Arm spacing (metres)")).toHaveValue(0.8);
    expect(screen.getByLabelText(/Arm spacing slider/)).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Apply spacing" }),
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Restore manifest spacing" }),
    ).toBeVisible();
    expect(screen.getByText("Manifest baseline: 0.80 m")).toBeVisible();
    expect(screen.getByText(manifest.coverage.spacingDisclosure)).toBeVisible();
    expect(screen.getByText("Requested dataset ref")).toBeVisible();
    expect(screen.getByText("Resolved HF commit")).toBeVisible();
    expect(screen.getByText("Repository HEAD")).toBeVisible();
    expect(
      screen.queryByText("Uncommitted exporter source"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(manifest.exporter.sourceDescription),
    ).not.toBeInTheDocument();
  });

  it("applies, clamps, rejects, and restores arm spacing", () => {
    render(<AtlasViewer />);

    const input = screen.getByLabelText("Arm spacing (metres)");
    const applyButton = screen.getByRole("button", {
      name: "Apply spacing",
    });

    fireEvent.change(input, { target: { value: "1.15" } });
    fireEvent.click(applyButton);
    expect(setSpacingMock).toHaveBeenLastCalledWith(1.15);

    fireEvent.change(input, { target: { value: "2.00" } });
    fireEvent.click(applyButton);
    expect(setSpacingMock).toHaveBeenLastCalledWith(1.4);

    const callCountBeforeInvalidInput = setSpacingMock.mock.calls.length;
    fireEvent.change(input, { target: { value: "" } });
    fireEvent.click(applyButton);
    expect(setSpacingMock).toHaveBeenCalledTimes(callCountBeforeInvalidInput);
    expect(input).toHaveValue(0.8);

    fireEvent.click(
      screen.getByRole("button", { name: "Restore manifest spacing" }),
    );
    expect(setSpacingMock).toHaveBeenLastCalledWith(
      manifest.coverage.armSpacing,
    );
  });

  it("exposes accessible playback controls after lazy activation", async () => {
    render(<AtlasViewer />);
    screen.getByRole("button", { name: "Load playback" }).click();
    expect(await screen.findByLabelText("Episode")).toBeVisible();
    expect(screen.getByRole("button", { name: "Play" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Restart" })).toBeVisible();
    expect(screen.getByLabelText("Timeline")).toBeVisible();
    expect(screen.getByLabelText("Playback speed")).toBeVisible();
    expect(screen.getByLabelText("Loop playback")).toBeVisible();
  });
});
