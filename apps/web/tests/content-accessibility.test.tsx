import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v2/manifest.json";
import coverageJson from "@/public/atlas-data/demo-v2/coverage.json";
import trajectoriesJson from "@/public/atlas-data/demo-v2/trajectories.json";
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
  CoveragePayload,
  TrajectoryEpisode,
  TrajectoryEpisodeOrientations,
  TrajectoryEpisodeRecordedGripperValues,
  TrajectoryPayload,
} from "@/lib/atlas-schema/types";
import type { VoxelSelection } from "@/lib/data/radiusQuery";

const manifest = decodeManifest(manifestJson);
const manifestWithVideos = decodeManifest({
  ...manifestJson,
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
    minor: 2,
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
let activeCoverage: CoveragePayload = coverage;
let activePreparedArms = preparedArmsForTest;
let activeSelection: VoxelSelection | null = {
  arm: "left",
  voxelEntryIndex: 0,
  exportedCenter: Array.from(
    preparedArmsForTest[0].centers.slice(0, 3),
  ) as [number, number, number],
};
let activeRadius = 0.05;
let activeSpacing = manifest.coverage.armSpacing;
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

function rankedEpisodeRow(episodeId: number): HTMLElement {
  const ranking = screen.getByRole("list", {
    name: /Uncommon-space episode ranking/,
  });
  const row = within(ranking).getByText(`Episode ${episodeId}`).closest("li");
  if (row === null) throw new Error(`Episode ${episodeId} row was not rendered.`);
  return row;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  setSpacingMock.mockClear();
  activeManifest = manifest;
  activeCoverage = coverage;
  activePreparedArms = preparedArmsForTest;
  activeSelection = {
    arm: "left",
    voxelEntryIndex: 0,
    exportedCenter: Array.from(
      preparedArmsForTest[0].centers.slice(0, 3),
    ) as [number, number, number],
  };
  activeRadius = 0.05;
  activeSpacing = manifest.coverage.armSpacing;
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
      coverage: activeCoverage,
      preparedArms: activePreparedArms,
    },
  }),
}));

vi.mock("@/components/viewer/ViewerStore", () => ({
  useViewerStore: () => ({
    leftVisible: true,
    rightVisible: true,
    cameraResetToken: 0,
    metric: "visits",
    spacing: activeSpacing,
    radius: activeRadius,
    selection: activeSelection,
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
  episodeVideoAssetUrl: (filename: string) => `/atlas-data/demo-v2/${filename}`,
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
    expect(screen.getByText("demo-v2 / episodes 0–9")).toBeVisible();
    expect(screen.getByText("schema v1.2")).toBeVisible();
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

  it("derives stable bundle and episode labeling from coverage metadata", async () => {
    activeManifest = {
      ...manifest,
      bundleId: "pilot-selection",
    };
    activeSelection = null;
    const { rerender } = render(<AtlasViewer />);
    const expectedLabel = "pilot-selection / episodes 0–9";

    expect(screen.getByText(expectedLabel)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Load playback" }));
    expect(await screen.findByLabelText("Episode")).toHaveValue("0");
    expect(screen.getByText(expectedLabel)).toBeVisible();

    activeRadius = 0.3;
    activeSpacing = 1.2;
    rerender(<AtlasViewer />);
    expect(screen.getByText(expectedLabel)).toBeVisible();
  });

  it("presents an accessible global uncommon-space ranking by default", () => {
    render(<AtlasViewer />);

    expect(
      screen.getByRole("heading", { name: "Uncommon-space episodes" }),
    ).toBeVisible();
    expect(screen.getByLabelText("Episode scoring scope")).toHaveValue(
      "coverage",
    );
    const list = screen.getByRole("list", {
      name: "Uncommon-space episode ranking for entire coverage",
    });
    expect(within(list).getAllByRole("listitem")).toHaveLength(
      manifest.dataset.episodeCount,
    );
    const firstResult = within(list).getAllByRole("listitem")[0];
    expect(within(firstResult).getByText(/Episode \d+/)).toBeVisible();
    expect(
      within(firstResult).getByText(/Uncommonness \d+\.\d \/ 100/),
    ).toBeVisible();
    expect(
      within(firstResult).getByText("Arm-specific entries touched"),
    ).toBeVisible();
    expect(within(firstResult).getByText("Scoped entries touched")).toBeVisible();
    expect(
      within(firstResult).getByText("Distinct episodes represented"),
    ).toBeVisible();
    expect(screen.getByText(/Scores use distinct-episode frequency/)).toHaveTextContent(
      "Raw visit counts are a separate metric.",
    );
    const limitations = screen.getByText(
      /Scores describe only this exported coverage set/,
    );
    expect(limitations).toHaveTextContent(
      "not the full dataset or physical workspace generally",
    );
    expect(limitations).toHaveTextContent(
      "not a probability, percentile, task-quality judgment, or anomaly label",
    );
    expect(screen.getByText(/episodes ranked for entire coverage/)).toHaveAttribute(
      "aria-live",
      "polite",
    );
    expect(
      screen.getByRole("button", { name: "Check playback availability" }),
    ).toBeVisible();
    expect(screen.getAllByText("Playback availability not loaded.")).toHaveLength(
      manifest.dataset.episodeCount,
    );
    expect(loadTrajectories).not.toHaveBeenCalled();
  });

  it("checks ranked playback availability with one shared lazy request", async () => {
    render(<AtlasViewer />);
    fireEvent.click(
      screen.getByRole("button", { name: "Check playback availability" }),
    );

    expect(screen.getByText("Checking playback availability…")).toHaveAttribute(
      "aria-busy",
      "true",
    );
    expect(loadTrajectories).toHaveBeenCalledTimes(1);
    expect(
      await within(rankedEpisodeRow(0)).findByText("Playback available"),
    ).toBeVisible();
    const availableButton = within(rankedEpisodeRow(0)).getByRole("button", {
      name: "Open Episode 0 playback",
    });
    expect(availableButton).toHaveAttribute(
      "aria-describedby",
      "episode-0-playback-status",
    );
    expect(
      within(rankedEpisodeRow(2)).getByText(
        "Coverage evidence only — trajectory not exported.",
      ),
    ).toBeVisible();
    expect(
      within(rankedEpisodeRow(2)).queryByRole("button", {
        name: "Open Episode 2 playback",
      }),
    ).not.toBeInTheDocument();
  });

  it("shares one in-flight request between both activation actions", async () => {
    let resolveTrajectories!: (payload: TrajectoryPayload) => void;
    vi.mocked(loadTrajectories).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveTrajectories = resolve;
      }),
    );
    render(<AtlasViewer />);
    const checkButton = screen.getByRole("button", {
      name: "Check playback availability",
    });
    const loadButton = screen.getByRole("button", { name: "Load playback" });

    fireEvent.click(checkButton);
    fireEvent.click(loadButton);
    fireEvent.click(checkButton);
    expect(loadTrajectories).toHaveBeenCalledTimes(1);

    resolveTrajectories(positionOnlyTrajectories);
    expect(await screen.findByLabelText("Episode")).toHaveValue("0");
  });

  it("opens the exact ranked episode and preserves speed and loop", async () => {
    render(<AtlasViewer />);
    fireEvent.click(
      screen.getByRole("button", { name: "Check playback availability" }),
    );
    await within(rankedEpisodeRow(1)).findByText("Playback available");

    fireEvent.change(screen.getByLabelText("Playback speed"), {
      target: { value: "2" },
    });
    fireEvent.click(screen.getByLabelText("Loop playback"));
    fireEvent.change(screen.getByLabelText("Timeline"), {
      target: { value: "10" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    fireEvent.click(
      within(rankedEpisodeRow(1)).getByRole("button", {
        name: "Open Episode 1 playback",
      }),
    );

    expect(screen.getByLabelText("Episode")).toHaveValue("1");
    expect(screen.getByLabelText("Episode")).toHaveFocus();
    expect(screen.getByLabelText("Timeline")).toHaveValue("0");
    expect(screen.getByRole("button", { name: "Play" })).toBeVisible();
    expect(screen.getByLabelText("Playback speed")).toHaveValue("2");
    expect(screen.getByLabelText("Loop playback")).toBeChecked();
    expect(currentViewerCanvasProps().episode?.episodeId).toBe(1);
    expect(currentViewerCanvasProps().orientationEpisode?.episodeId).toBe(1);
    expect(currentViewerCanvasProps().recordedGripperEpisode?.episodeId).toBe(
      1,
    );
  });

  it("keeps rankings honest after loading failure and permits retry", async () => {
    vi.mocked(loadTrajectories)
      .mockRejectedValueOnce(new Error("Trajectory fixture failed."))
      .mockResolvedValueOnce(positionOnlyTrajectories);
    render(<AtlasViewer />);
    fireEvent.click(
      screen.getByRole("button", { name: "Check playback availability" }),
    );

    expect(await screen.findByText("Trajectory fixture failed.")).toHaveAttribute(
      "role",
      "alert",
    );
    expect(
      screen.getAllByText("Playback availability could not be loaded."),
    ).toHaveLength(manifest.dataset.episodeCount);
    expect(
      screen.getByRole("list", {
        name: "Uncommon-space episode ranking for entire coverage",
      }),
    ).toBeVisible();
    expect(
      screen.queryByText("Coverage evidence only — trajectory not exported."),
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Retry playback availability" }),
    );
    expect(await screen.findByLabelText("Episode")).toHaveValue("0");
    expect(loadTrajectories).toHaveBeenCalledTimes(2);
  });

  it("disables radius scoring and provides help before voxel selection", () => {
    activeSelection = null;
    render(<AtlasViewer />);

    expect(screen.getByRole("option", { name: "Selected radius" })).toBeDisabled();
    expect(
      screen.getByText(/Select an occupied voxel to score episodes within a radius/),
    ).toBeVisible();
  });

  it("updates local uncommon scores and returns safely to global scope", async () => {
    activeRadius = 0;
    const { rerender } = render(<AtlasViewer />);
    const scope = screen.getByLabelText("Episode scoring scope");
    fireEvent.change(scope, { target: { value: "radius" } });
    const initialSummary = screen.getByText(/episodes? ranked for selected radius/)
      .textContent;

    activeRadius = 0.3;
    rerender(<AtlasViewer />);
    expect(screen.getByLabelText("Episode scoring scope")).toHaveValue("radius");
    expect(screen.getByText(/episodes? ranked for selected radius/).textContent).not.toBe(
      initialSummary,
    );

    activeSelection = null;
    rerender(<AtlasViewer />);
    await waitFor(() =>
      expect(screen.getByLabelText("Episode scoring scope")).toHaveValue(
        "coverage",
      ),
    );
    expect(screen.getByRole("option", { name: "Selected radius" })).toBeDisabled();
  });

  it("shows an honest empty selected-radius state", () => {
    activePreparedArms = preparedArmsForTest.map((arm) => ({
      ...arm,
      centers: new Float32Array(),
      visits: new Uint32Array(),
      episodeCounts: new Uint32Array(),
      instanceLookup: [],
    }));
    render(<AtlasViewer />);
    fireEvent.change(screen.getByLabelText("Episode scoring scope"), {
      target: { value: "radius" },
    });

    expect(
      screen.getByText("No episode evidence exists in the current radius."),
    ).toBeVisible();
    expect(
      screen.queryByRole("list", {
        name: "Uncommon-space episode ranking for selected radius",
      }),
    ).not.toBeInTheDocument();
  });

  it("explains the defined zero score for one coverage episode", () => {
    activeManifest = {
      ...manifest,
      dataset: {
        ...manifest.dataset,
        episodeIds: [0],
        episodeCount: 1,
      },
    };
    activeCoverage = {
      schema: coverage.schema,
      arms: (["left", "right"] as const).map((arm) => ({
        arm,
        toolLink: "tool0",
        voxelIndices: [[0, 0, 0]],
        visitCounts: [1],
        episodeCounts: [1],
        episodeIdOffsets: [0, 1],
        episodeIds: [0],
        statistics: {
          voxelEntryCount: 1,
          minimumVisitCount: 1,
          maximumVisitCount: 1,
          minimumEpisodeCount: 1,
          maximumEpisodeCount: 1,
        },
      })),
    };
    activePreparedArms = prepareCoverage(activeManifest, activeCoverage);
    activeSelection = null;
    render(<AtlasViewer />);

    expect(
      screen.getByText(/Relative uncommonness is unavailable with one coverage episode/),
    ).toHaveTextContent("scores are defined as zero");
    expect(screen.getByText("Uncommonness 0.0 / 100")).toBeVisible();
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
    expect(await screen.findByLabelText("Episode")).toHaveValue("0");
    expect(loadTrajectories).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Play" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Restart" })).toBeVisible();
    expect(screen.getByLabelText("Timeline")).toBeVisible();
    expect(screen.getByLabelText("Playback speed")).toBeVisible();
    expect(screen.getByLabelText("Loop playback")).toBeVisible();
    expect(
      screen.getByRole("group", { name: "Recorded raw gripper values" }),
    ).toBeVisible();
    expect(currentViewerCanvasProps().orientationEpisode?.episodeId).toBe(0);
    expect(currentViewerCanvasProps().recordedGripperEpisode?.episodeId).toBe(
      0,
    );
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

    expect(screen.getByText("schema v1.2")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Load playback" }));
    const video = await screen.findByLabelText(
      "Top camera synchronized episode video",
    );
    expect(video).toHaveAttribute(
      "src",
      "/atlas-data/demo-v2/media/episode-0/top.mp4",
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
      "/atlas-data/demo-v2/media/episode-0/left.mp4",
    );

    fireEvent.change(screen.getByLabelText("Episode"), {
      target: { value: "1" },
    });
    expect(
      screen.getByLabelText("Left wrist camera synchronized episode video"),
    ).toHaveAttribute(
      "src",
      "/atlas-data/demo-v2/media/episode-1/left.mp4",
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
