import { describe, expect, it, vi } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v1/manifest.json";
import coverageJson from "@/public/atlas-data/demo-v1/coverage.json";
import trajectoriesJson from "@/public/atlas-data/demo-v1/trajectories.json";
import { decodeManifest } from "@/lib/atlas-schema/validate";
import { loadDemoBundle, loadTrajectories } from "@/lib/data/loadBundle";

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
});
