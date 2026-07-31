import { afterEach, describe, expect, it, vi } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v2/manifest.json";
import coverageJson from "@/public/atlas-data/demo-v2/coverage.json";
import trajectoriesJson from "@/public/atlas-data/demo-v2/trajectories.json";
import { decodeManifest } from "@/lib/atlas-schema/validate";
import {
  loadDemoBundle,
  loadDemoBundleForBenchmark,
  loadEpisodeVideos,
  loadTrajectories,
  resolveBundleBase,
} from "@/lib/data/loadBundle";

const episodeVideosJson = {
  schema: {
    name: "lerobot-state-atlas.browser-data",
    major: 1,
    minor: 2,
  },
  defaultCameraId: "top",
  cameras: [
    {
      cameraId: "top",
      datasetFeature: "observation.images.top",
      label: "Top camera",
      width: 224,
      height: 224,
    },
  ],
  episodes: [
    {
      episodeId: 0,
      videos: [
        {
          cameraId: "top",
          filename: "media/episode-000000/top.mp4",
          mimeType: "video/mp4",
          fromTimestampSeconds: 0,
          toTimestampSeconds: 10.3,
          byteSize: 1234,
          sha256: "a".repeat(64),
        },
      ],
    },
  ],
};

function manifestWithEpisodeVideos() {
  return decodeManifest({
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
}

describe("bundle loading", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("uses demo-v2 by default and validates local bundle overrides", () => {
    expect(resolveBundleBase(undefined)).toBe("/atlas-data/demo-v2");
    expect(resolveBundleBase("/atlas-data/__local-benchmark__")).toBe(
      "/atlas-data/__local-benchmark__",
    );
    expect(resolveBundleBase("/atlas-data/__local-benchmark__/")).toBe(
      "/atlas-data/__local-benchmark__",
    );
  });

  it.each([
    "",
    "https://example.com/atlas-data/pilot",
    "//example.com/atlas-data/pilot",
    "/tmp/pilot",
    "/atlas-data/../demo-v2",
    "/atlas-data/pilot\\bundle",
    "/atlas-data/pilot?version=1",
    "/atlas-data/pilot#fragment",
  ])("rejects unsafe bundle override %j", (override) => {
    expect(() => resolveBundleBase(override)).toThrow(
      /safe root-relative \/atlas-data\/ path/,
    );
  });

  it("applies one override to initial and lazy payload requests", async () => {
    vi.stubEnv(
      "NEXT_PUBLIC_ATLAS_BUNDLE_BASE",
      "/atlas-data/__local-benchmark__/",
    );
    const urls: string[] = [];
    const fetcher = vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      urls.push(url);
      const payload = url.endsWith("manifest.json")
        ? manifestJson
        : url.endsWith("coverage.json")
          ? coverageJson
          : trajectoriesJson;
      return new Response(JSON.stringify(payload), { status: 200 });
    }) as unknown as typeof fetch;

    const bundle = await loadDemoBundle(fetcher, "development");
    await loadTrajectories(bundle.manifest, fetcher, "development");

    expect(urls).toEqual([
      "/atlas-data/__local-benchmark__/manifest.json",
      "/atlas-data/__local-benchmark__/coverage.json",
      "/atlas-data/__local-benchmark__/trajectories.json",
    ]);
    expect(urls.some((url) => url.includes("episode-videos"))).toBe(false);
  });

  it("loads development manifest and coverage with no-store", async () => {
    const urls: string[] = [];
    const options: Array<RequestInit | undefined> = [];
    const fetcher = vi.fn(async (
      input: string | URL | Request,
      init?: RequestInit,
    ) => {
      const url = String(input);
      urls.push(url);
      options.push(init);
      return new Response(
        JSON.stringify(url.endsWith("manifest.json") ? manifestJson : coverageJson),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as unknown as typeof fetch;

    const bundle = await loadDemoBundle(fetcher, "development");

    expect(urls).toEqual([
      "/atlas-data/demo-v2/manifest.json",
      "/atlas-data/demo-v2/coverage.json",
    ]);
    expect(bundle.preparedArms).toHaveLength(2);
    expect(options).toEqual([{ cache: "no-store" }, { cache: "no-store" }]);
    expect(urls.some((url) => url.includes("trajectories"))).toBe(false);
    expect(urls.some((url) => url.includes("episode-videos"))).toBe(false);
  });

  it("measures the existing manifest, coverage, and preparation boundaries", async () => {
    const times = [0, 5, 10, 17, 20, 29];
    const fetcher = vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      return new Response(
        JSON.stringify(url.endsWith("manifest.json") ? manifestJson : coverageJson),
        { status: 200 },
      );
    }) as unknown as typeof fetch;

    const result = await loadDemoBundleForBenchmark(
      fetcher,
      "development",
      () => times.shift()!,
    );

    expect(result.data.preparedArms).toHaveLength(2);
    expect(result.durations).toEqual({
      manifestLoadMilliseconds: 5,
      coverageLoadMilliseconds: 7,
      coveragePreparationMilliseconds: 9,
    });
  });

  it("does not disable browser caching for production bundle requests", async () => {
    const requests: Array<{
      url: string;
      init: RequestInit | undefined;
    }> = [];
    const fetcher = vi.fn(async (
      input: string | URL | Request,
      init?: RequestInit,
    ) => {
      const url = String(input);
      requests.push({ url, init });
      return new Response(
        JSON.stringify(url.endsWith("manifest.json") ? manifestJson : coverageJson),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as unknown as typeof fetch;

    await loadDemoBundle(fetcher, "production");

    expect(requests.map(({ init }) => init)).toEqual([undefined, undefined]);
    expect(requests.some(({ url }) => url.includes("trajectories"))).toBe(false);
    expect(
      requests.some(({ url }) => url.includes("episode-videos")),
    ).toBe(false);
  });

  it("reports a useful HTTP error", async () => {
    const fetcher = vi.fn(async () => new Response(null, { status: 404 })) as
      unknown as typeof fetch;

    await expect(loadDemoBundle(fetcher)).rejects.toThrow(
      /Unable to load atlas manifest/,
    );
  });

  it("lazily requests the optional trajectory payload on activation", async () => {
    const fetcher = vi.fn(async () =>
      new Response(JSON.stringify(trajectoriesJson), { status: 200 }),
    ) as unknown as typeof fetch;
    const result = await loadTrajectories(
      decodeManifest(manifestJson),
      fetcher,
      "development",
    );
    expect(fetcher).toHaveBeenCalledWith(
      "/atlas-data/demo-v2/trajectories.json",
      { cache: "no-store" },
    );
    expect(result.episodes.map((episode) => episode.episodeId)).toEqual([0, 1]);
    expect(result.orientation.status).toBe("available");
    expect(result.gripper.status).toBe("available");
  });

  it("handles a missing optional trajectory payload", async () => {
    const fetcher = vi.fn(async () => new Response(null, { status: 404 })) as
      unknown as typeof fetch;
    await expect(
      loadTrajectories(decodeManifest(manifestJson), fetcher),
    ).rejects.toThrow(/Unable to load trajectory payload/);
  });

  it("lazily requests episode-video metadata with the trajectory cache policy", async () => {
    const fetcher = vi.fn(async () =>
      new Response(JSON.stringify(episodeVideosJson), { status: 200 }),
    ) as unknown as typeof fetch;

    const result = await loadEpisodeVideos(
      manifestWithEpisodeVideos(),
      fetcher,
      "development",
    );

    expect(fetcher).toHaveBeenCalledWith(
      "/atlas-data/demo-v2/episode-videos.json",
      { cache: "no-store" },
    );
    expect(result.defaultCameraId).toBe("top");
  });

  it("reports missing and invalid optional episode-video metadata", async () => {
    const fetcher = vi.fn();
    await expect(
      loadEpisodeVideos(decodeManifest(manifestJson), fetcher),
    ).rejects.toThrow(/does not include synchronized episode video/);
    expect(fetcher).not.toHaveBeenCalled();

    const invalidFetcher = vi.fn(async () =>
      new Response(JSON.stringify({ ...episodeVideosJson, cameras: [] }), {
        status: 200,
      }),
    ) as unknown as typeof fetch;
    await expect(
      loadEpisodeVideos(manifestWithEpisodeVideos(), invalidFetcher),
    ).rejects.toThrow(/must contain cameras/);
  });

  it.each(["orientation", "gripper"] as const)(
    "loads video metadata after degraded %s trajectory state",
    async (degradedCapability) => {
      const manifest = decodeManifest({
        ...manifestJson,
        schema: { ...manifestJson.schema, minor: 2 },
        trajectoryState: {
          orientation: {
            available: true,
            representation: "unit-quaternion",
            componentOrder: ["x", "y", "z", "w"],
            frame: "canonical-shared-world",
            samplePolicy: "recorded-sample",
          },
          gripper: {
            available: true,
            leftSourceComponent: "left_gripper.pos",
            rightSourceComponent: "right_gripper.pos",
            valueSemantics: "raw-device-specific-unproven",
            physicalJawWidthCalibrated: false,
            polarityEstablished: false,
            visualizationGeometryCalibrated: false,
          },
        },
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
      const trajectoryPayload = {
        schema: {
          name: "lerobot-state-atlas.browser-data",
          major: 1,
          minor: 2,
        },
        episodes: [
          {
            episodeId: 0,
            frameIndices: [0],
            timestampsSeconds: [0],
            leftPositionsXyz: [[0, 0.4, 0]],
            rightPositionsXyz: [[0, -0.4, 0]],
            leftOrientationsXyzw: [[0, 0, 0, 1]],
            ...(degradedCapability === "orientation"
              ? {}
              : { rightOrientationsXyzw: [[0, 0, 0, 1]] }),
            leftRecordedGripperValues: [-0.5],
            ...(degradedCapability === "gripper"
              ? {}
              : { rightRecordedGripperValues: [2.5] }),
          },
        ],
      };
      const videoPayload = {
        ...episodeVideosJson,
        schema: { ...episodeVideosJson.schema, minor: 2 },
      };
      const urls: string[] = [];
      const fetcher = vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        urls.push(url);
        return new Response(
          JSON.stringify(
            url.endsWith("trajectories.json")
              ? trajectoryPayload
              : videoPayload,
          ),
          { status: 200 },
        );
      }) as unknown as typeof fetch;

      const trajectories = await loadTrajectories(
        manifest,
        fetcher,
        "development",
      );
      expect(trajectories.episodes).toHaveLength(1);
      expect(trajectories[degradedCapability].status).toBe("degraded");
      expect(
        trajectories[
          degradedCapability === "orientation" ? "gripper" : "orientation"
        ].status,
      ).toBe("available");
      const videos = await loadEpisodeVideos(
        manifest,
        fetcher,
        "development",
      );

      expect(videos.defaultCameraId).toBe("top");
      expect(urls).toEqual([
        "/atlas-data/demo-v2/trajectories.json",
        "/atlas-data/demo-v2/episode-videos.json",
      ]);
      expect(urls.some((url) => url.includes("enhanced"))).toBe(false);
    },
  );
});
