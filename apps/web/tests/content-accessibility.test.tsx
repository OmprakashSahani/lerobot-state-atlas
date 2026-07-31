import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v1/manifest.json";
import coverageJson from "@/public/atlas-data/demo-v1/coverage.json";
import trajectoriesJson from "@/public/atlas-data/demo-v1/trajectories.json";
import HomePage from "@/app/page";
import MethodologyPage from "@/app/methodology/page";
import { AtlasViewer } from "@/components/viewer/AtlasViewer";
import {
  decodeCoverage,
  decodeEpisodeVideos,
  decodeManifest,
  decodeTrajectories,
} from "@/lib/atlas-schema/validate";
import {
  loadEpisodeVideos,
  loadTrajectories,
} from "@/lib/data/loadBundle";
import { prepareCoverage } from "@/lib/data/prepareCoverage";
import type {
  AtlasData,
  TrajectoryEpisode,
  TrajectoryEpisodeOrientations,
  TrajectoryEpisodeRecordedGripperValues,
  TrajectoryPayload,
} from "@/lib/atlas-schema/types";

const manifest = decodeManifest(manifestJson);
const manifestWithVideos = decodeManifest({
  ...manifestJson,
  schema: { ...manifestJson.schema, minor: 1 },
  payloads: [
    ...manifestJson.payloads,
    {
      kind: "episode-videos",
      filename: "episode-videos.json",
      required: false,
      encoding: "json",
      byteSize: 1234,
      sha256: "b".repeat(64),
    },
  ],
});
const episodeVideos = decodeEpisodeVideos({
  schema: {
    name: "lerobot-state-atlas.browser-data",
    major: 1,
    minor: 1,
  },
  defaultCameraId: "top",
  cameras: [
    {
      cameraId: "left",
      datasetFeature: "observation.images.left_wrist",
      label: "Left wrist camera",
      width: 224,
      height: 224,
    },
    {
      cameraId: "top",
      datasetFeature: "observation.images.top",
      label: "Top camera",
      width: 224,
      height: 224,
    },
  ],
  episodes: [0, 1].map((episodeId) => ({
    episodeId,
    videos: [
      {
        cameraId: "left",
        filename: `media/episode-${episodeId}/left.mp4`,
        mimeType: "video/mp4",
        fromTimestampSeconds: 0,
        toTimestampSeconds: 20,
        byteSize: 100,
        sha256: `${episodeId + 1}`.repeat(64),
      },
      {
        cameraId: "top",
        filename: `media/episode-${episodeId}/top.mp4`,
        mimeType: "video/mp4",
        fromTimestampSeconds: 0,
        toTimestampSeconds: 20,
        byteSize: 100,
        sha256: `${episodeId + 3}`.repeat(64),
      },
    ],
  })),
});
const coverage = decodeCoverage(coverageJson);
const positionOnlyTrajectories = decodeTrajectories(
  trajectoriesJson,
  manifest,
);
const preparedArmsForTest = prepareCoverage(manifest, coverage);
const setSpacingMock = vi.fn();
let activeManifest = manifest;
let viewerCanvasProps: {
  data: AtlasData;
  episode: TrajectoryEpisode | null;
  orientationEpisode: TrajectoryEpisodeOrientations | null;
  recordedGripperEpisode: TrajectoryEpisodeRecordedGripperValues | null;
  playbackFrame: number;
} | null = null;

function trajectoriesWithOptionalState(
  orientationStatus: "available" | "degraded" = "available",
  gripperStatus: "available" | "degraded" = "available",
): TrajectoryPayload {
  return {
    ...positionOnlyTrajectories,
    orientation:
      orientationStatus === "degraded"
        ? { status: "degraded", warning: "Invalid orientation fixture." }
        : {
            status: "available",
            data: {
              episodes: positionOnlyTrajectories.episodes.map((episode) => ({
                episodeId: episode.episodeId,
                leftOrientationsXyzw: episode.frameIndices.map(() => [
                  episode.episodeId === 0 ? 0 : 1,
                  0,
                  0,
                  episode.episodeId === 0 ? 1 : 0,
                ]),
                rightOrientationsXyzw: episode.frameIndices.map(() => [
                  0,
                  0,
                  1,
                  0,
                ]),
              })),
            },
          },
    gripper:
      gripperStatus === "degraded"
        ? { status: "degraded", warning: "Invalid gripper fixture." }
        : {
            status: "available",
            data: {
              episodes: positionOnlyTrajectories.episodes.map((episode) => ({
                episodeId: episode.episodeId,
                leftRecordedGripperValues: episode.frameIndices.map(
                  (_, index) =>
                    episode.episodeId === 0 ? -0.5 - index : -10 - index,
                ),
                rightRecordedGripperValues: episode.frameIndices.map(
                  (_, index) =>
                    episode.episodeId === 0 ? index + 2.25 : index + 100,
                ),
              })),
            },
          },
  };
}

function currentViewerCanvasProps() {
  if (viewerCanvasProps === null) {
    throw new Error("ViewerCanvas has not rendered.");
  }
  return viewerCanvasProps;
}

afterEach(() => {
  cleanup();
  setSpacingMock.mockClear();
  activeManifest = manifest;
  viewerCanvasProps = null;
  vi.mocked(loadTrajectories).mockResolvedValue(
    decodeTrajectories(trajectoriesJson, manifest),
  );
  vi.mocked(loadEpisodeVideos).mockResolvedValue(episodeVideos);
});

vi.mock("@/components/viewer/AtlasDataProvider", () => ({
  useAtlasData: () => ({
    status: "ready",
    data: {
      manifest: activeManifest,
      coverage,
      preparedArms: prepareCoverage(activeManifest, coverage),
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
  ViewerCanvas: (props: NonNullable<typeof viewerCanvasProps>) => {
    viewerCanvasProps = props;
    return <div data-testid="viewer-canvas" />;
  },
}));

vi.mock("@/lib/data/loadBundle", () => ({
  episodeVideoAssetUrl: (filename: string) => `/atlas-data/demo-v1/${filename}`,
  loadEpisodeVideos: vi.fn(async () => episodeVideos),
  loadTrajectories: vi.fn(async () =>
    decodeTrajectories(trajectoriesJson, manifest),
  ),
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
    expect(screen.getByText("schema v1.0")).toBeVisible();
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
    expect(
      screen.getByText(
        "Synchronized episode video is not included in this bundle.",
      ),
    ).toBeVisible();
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
    expect(
      screen.queryByRole("group", { name: "Recorded raw gripper values" }),
    ).not.toBeInTheDocument();
  });

  it("threads matching required and optional episodes together", async () => {
    vi.mocked(loadTrajectories).mockResolvedValueOnce(
      trajectoriesWithOptionalState(),
    );
    render(<AtlasViewer />);
    fireEvent.click(screen.getByRole("button", { name: "Load playback" }));
    await screen.findByLabelText("Episode");

    expect(currentViewerCanvasProps().episode?.episodeId).toBe(0);
    expect(currentViewerCanvasProps().orientationEpisode?.episodeId).toBe(0);
    expect(currentViewerCanvasProps().recordedGripperEpisode?.episodeId).toBe(
      0,
    );
    expect(
      currentViewerCanvasProps().orientationEpisode
        ?.leftOrientationsXyzw[0],
    ).toEqual([0, 0, 0, 1]);
    const initialRawValues = screen.getByRole("group", {
      name: "Recorded raw gripper values",
    });
    expect(within(initialRawValues).getByText("-0.5")).toBeVisible();
    expect(within(initialRawValues).getByText("2.25")).toBeVisible();
    expect(within(initialRawValues).getByText(/Symbolic display only/)).toHaveTextContent(
      "Values are raw and device-specific; physical jaw width is not calibrated, and open/closed polarity is not established.",
    );

    fireEvent.change(screen.getByLabelText("Timeline"), {
      target: { value: "1" },
    });
    const advancedRawValues = screen.getByRole("group", {
      name: "Recorded raw gripper values",
    });
    expect(within(advancedRawValues).getByText("-1.5")).toBeVisible();
    expect(within(advancedRawValues).getByText("3.25")).toBeVisible();

    fireEvent.change(screen.getByLabelText("Episode"), {
      target: { value: "1" },
    });
    expect(currentViewerCanvasProps().episode?.episodeId).toBe(1);
    expect(currentViewerCanvasProps().orientationEpisode?.episodeId).toBe(1);
    expect(currentViewerCanvasProps().recordedGripperEpisode?.episodeId).toBe(
      1,
    );
    expect(
      currentViewerCanvasProps().orientationEpisode
        ?.leftOrientationsXyzw[0],
    ).toEqual([1, 0, 0, 0]);
    const switchedRawValues = screen.getByRole("group", {
      name: "Recorded raw gripper values",
    });
    expect(within(switchedRawValues).getByText("-10")).toBeVisible();
    expect(within(switchedRawValues).getByText("100")).toBeVisible();
  });

  it.each([
    ["orientation", "gripper"],
    ["gripper", "orientation"],
  ] as const)(
    "threads available %s data when %s is degraded",
    async (availableCapability, degradedCapability) => {
      vi.mocked(loadTrajectories).mockResolvedValueOnce(
        trajectoriesWithOptionalState(
          degradedCapability === "orientation" ? "degraded" : "available",
          degradedCapability === "gripper" ? "degraded" : "available",
        ),
      );
      render(<AtlasViewer />);
      fireEvent.click(screen.getByRole("button", { name: "Load playback" }));
      await screen.findByLabelText("Episode");

      expect(currentViewerCanvasProps().episode?.episodeId).toBe(0);
      expect(
        availableCapability === "orientation"
          ? currentViewerCanvasProps().orientationEpisode?.episodeId
          : currentViewerCanvasProps().recordedGripperEpisode?.episodeId,
      ).toBe(0);
      expect(
        degradedCapability === "orientation"
          ? currentViewerCanvasProps().orientationEpisode
          : currentViewerCanvasProps().recordedGripperEpisode,
      ).toBeNull();
      if (availableCapability === "gripper") {
        expect(
          screen.getByRole("group", {
            name: "Recorded raw gripper values",
          }),
        ).toBeVisible();
      } else {
        expect(
          screen.queryByRole("group", {
            name: "Recorded raw gripper values",
          }),
        ).not.toBeInTheDocument();
        expect(screen.getByText("Invalid gripper fixture.")).toHaveAttribute(
          "role",
          "note",
        );
      }
    },
  );

  it("exposes synchronized video and switches camera and episode sources", async () => {
    activeManifest = manifestWithVideos;
    render(<AtlasViewer />);

    expect(screen.getByText("schema v1.1")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Load playback" }));
    const video = await screen.findByLabelText(
      "Top camera synchronized episode video",
    );
    expect(video).toHaveAttribute(
      "src",
      "/atlas-data/demo-v1/media/episode-0/top.mp4",
    );
    expect(video).not.toHaveAttribute("controls");
    expect(video).toHaveAttribute("playsinline");
    expect(video).toHaveAttribute("preload", "metadata");

    fireEvent.change(screen.getByLabelText("Camera"), {
      target: { value: "left" },
    });
    expect(
      screen.getByLabelText("Left wrist camera synchronized episode video"),
    ).toHaveAttribute(
      "src",
      "/atlas-data/demo-v1/media/episode-0/left.mp4",
    );

    fireEvent.change(screen.getByLabelText("Episode"), {
      target: { value: "1" },
    });
    expect(
      screen.getByLabelText("Left wrist camera synchronized episode video"),
    ).toHaveAttribute(
      "src",
      "/atlas-data/demo-v1/media/episode-1/left.mp4",
    );
  });

  it("keeps trajectory controls when optional video metadata fails", async () => {
    activeManifest = manifestWithVideos;
    vi.mocked(loadEpisodeVideos).mockRejectedValueOnce(
      new Error("Invalid video metadata."),
    );
    render(<AtlasViewer />);

    fireEvent.click(screen.getByRole("button", { name: "Load playback" }));
    expect(await screen.findByLabelText("Timeline")).toBeVisible();
    expect(screen.getByRole("button", { name: "Play" })).toBeVisible();
    expect(
      await screen.findByText(/Synchronized episode video is unavailable/),
    ).toBeVisible();
  });
});
