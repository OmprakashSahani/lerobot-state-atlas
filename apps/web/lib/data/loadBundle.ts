import type {
  AtlasData,
  AtlasManifest,
  EpisodeVideoPayload,
  TrajectoryPayload,
} from "@/lib/atlas-schema/types";
import {
  AtlasDataError,
  decodeCoverage,
  decodeEpisodeVideos,
  decodeManifest,
  decodeTrajectories,
} from "@/lib/atlas-schema/validate";
import { prepareCoverage } from "@/lib/data/prepareCoverage";
import {
  atlasFetchOptions,
  type RuntimeEnvironment,
} from "@/lib/data/cachePolicy";

export const DEFAULT_BUNDLE_BASE = "/atlas-data/demo-v2";

export function resolveBundleBase(
  override = process.env.NEXT_PUBLIC_ATLAS_BUNDLE_BASE,
): string {
  if (override === undefined) return DEFAULT_BUNDLE_BASE;
  if (
    !override.startsWith("/atlas-data/") ||
    override.startsWith("//") ||
    override.includes("\\") ||
    override.includes("?") ||
    override.includes("#")
  ) {
    throw new AtlasDataError(
      "NEXT_PUBLIC_ATLAS_BUNDLE_BASE must be a safe root-relative /atlas-data/ path.",
    );
  }
  const normalized = override.endsWith("/") ? override.slice(0, -1) : override;
  const segments = normalized.split("/").slice(2);
  if (
    segments.length === 0 ||
    segments.some(
      (segment) =>
        segment === "" ||
        segment === "." ||
        segment === ".." ||
        !/^[A-Za-z0-9._-]+$/.test(segment),
    )
  ) {
    throw new AtlasDataError(
      "NEXT_PUBLIC_ATLAS_BUNDLE_BASE must be a safe root-relative /atlas-data/ path.",
    );
  }
  return normalized;
}

export function episodeVideoAssetUrl(filename: string): string {
  return `${resolveBundleBase()}/${filename}`;
}

async function jsonResponse(response: Response, label: string): Promise<unknown> {
  if (!response.ok) {
    throw new AtlasDataError(
      `Unable to load ${label} (${response.status} ${response.statusText}).`,
    );
  }
  try {
    return await response.json();
  } catch {
    throw new AtlasDataError(`${label} is not valid JSON.`);
  }
}

export async function loadTrajectories(
  manifest: AtlasManifest,
  fetcher: typeof fetch = fetch,
  environment: RuntimeEnvironment = process.env.NODE_ENV as RuntimeEnvironment,
): Promise<TrajectoryPayload> {
  const reference = manifest.payloads.find(
    (payload) => payload.kind === "trajectories",
  );
  if (!reference) {
    throw new AtlasDataError("This atlas bundle does not include trajectory playback.");
  }
  const response = await fetcher(
    `${resolveBundleBase()}/${reference.filename}`,
    atlasFetchOptions(environment),
  );
  return decodeTrajectories(
    await jsonResponse(response, "trajectory payload"),
    manifest,
  );
}

export async function loadEpisodeVideos(
  manifest: AtlasManifest,
  fetcher: typeof fetch = fetch,
  environment: RuntimeEnvironment = process.env.NODE_ENV as RuntimeEnvironment,
): Promise<EpisodeVideoPayload> {
  const reference = manifest.payloads.find(
    (payload) => payload.kind === "episode-videos",
  );
  if (!reference) {
    throw new AtlasDataError(
      "This atlas bundle does not include synchronized episode video.",
    );
  }
  const response = await fetcher(
    `${resolveBundleBase()}/${reference.filename}`,
    atlasFetchOptions(environment),
  );
  return decodeEpisodeVideos(
    await jsonResponse(response, "episode-video payload"),
  );
}

export interface BenchmarkLoadDurations {
  manifestLoadMilliseconds: number;
  coverageLoadMilliseconds: number;
  coveragePreparationMilliseconds: number;
}

interface BundleLoadResult {
  data: AtlasData;
  durations: BenchmarkLoadDurations;
}

async function loadBundle(
  fetcher: typeof fetch = fetch,
  environment: RuntimeEnvironment = process.env.NODE_ENV as RuntimeEnvironment,
  now: () => number,
): Promise<BundleLoadResult> {
  const fetchOptions = atlasFetchOptions(environment);
  const bundleBase = resolveBundleBase();
  const manifestStarted = now();
  const manifestResponse = await fetcher(
    `${bundleBase}/manifest.json`,
    fetchOptions,
  );
  const manifest = decodeManifest(
    await jsonResponse(manifestResponse, "atlas manifest"),
  );
  const manifestLoadMilliseconds = now() - manifestStarted;
  const reference = manifest.payloads.find(
    (payload) => payload.kind === "coverage",
  );
  if (!reference) {
    throw new AtlasDataError("Atlas manifest does not reference coverage data.");
  }
  const coverageStarted = now();
  const coverageResponse = await fetcher(
    `${bundleBase}/${reference.filename}`,
    fetchOptions,
  );
  const coverage = decodeCoverage(
    await jsonResponse(coverageResponse, "coverage payload"),
  );
  const coverageLoadMilliseconds = now() - coverageStarted;
  const preparationStarted = now();
  const preparedArms = prepareCoverage(manifest, coverage);
  const coveragePreparationMilliseconds = now() - preparationStarted;
  return {
    data: { manifest, coverage, preparedArms },
    durations: {
      manifestLoadMilliseconds,
      coverageLoadMilliseconds,
      coveragePreparationMilliseconds,
    },
  };
}

export async function loadDemoBundle(
  fetcher: typeof fetch = fetch,
  environment: RuntimeEnvironment = process.env.NODE_ENV as RuntimeEnvironment,
): Promise<AtlasData> {
  return (await loadBundle(fetcher, environment, performance.now.bind(performance)))
    .data;
}

export function loadDemoBundleForBenchmark(
  fetcher: typeof fetch = fetch,
  environment: RuntimeEnvironment = process.env.NODE_ENV as RuntimeEnvironment,
  now: () => number = performance.now.bind(performance),
): Promise<BundleLoadResult> {
  return loadBundle(fetcher, environment, now);
}
