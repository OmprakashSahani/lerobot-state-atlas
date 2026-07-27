import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v1/manifest.json";
import coverageJson from "@/public/atlas-data/demo-v1/coverage.json";
import HomePage from "@/app/page";
import MethodologyPage from "@/app/methodology/page";
import { AtlasViewer } from "@/components/viewer/AtlasViewer";
import { decodeCoverage, decodeManifest } from "@/lib/atlas-schema/validate";
import { prepareCoverage } from "@/lib/data/prepareCoverage";

const manifest = decodeManifest(manifestJson);
const coverage = decodeCoverage(coverageJson);

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
    toggleArm: vi.fn(),
    resetCamera: vi.fn(),
  }),
}));

vi.mock("@/components/viewer/ViewerCanvas", () => ({
  ViewerCanvas: () => <div data-testid="viewer-canvas" />,
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
    expect(screen.getByLabelText("Visit count color range")).toBeVisible();
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
});
