import { describe, expect, it, vi } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v1/manifest.json";
import coverageJson from "@/public/atlas-data/demo-v1/coverage.json";
import trajectoriesJson from "@/public/atlas-data/demo-v1/trajectories.json";
import { decodeManifest } from "@/lib/atlas-schema/validate";
import {
  loadDemoBundle,
  loadEpisodeVideos,
  loadTrajectories,
} from "@/lib/data/loadBundle";

const episodeVideosJson = {
  schema: {
    name: "lerobot-state-atlas.browser-data",
    major: 1,
    minor: 1,
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
}

describe("bundle loading", () => {
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
      "/atlas-data/demo-v1/manifest.json",
      "/atlas-data/demo-v1/coverage.json",
    ]);
    expect(bundle.preparedArms).toHaveLength(2);
    expect(options).toEqual([{ cache: "no-store" }, { cache: "no-store" }]);
    expect(urls.some((url) => url.includes("trajectories"))).toBe(false);
    expect(urls.some((url) => url.includes("episode-videos"))).toBe(false);
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
      "/atlas-data/demo-v1/trajectories.json",
      { cache: "no-store" },
    );
    expect(result.episodes.map((episode) => episode.episodeId)).toEqual([0, 1]);
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
      "/atlas-data/demo-v1/episode-videos.json",
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
});
