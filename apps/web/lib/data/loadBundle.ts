import type { AtlasData, AtlasManifest, TrajectoryPayload } from "@/lib/atlas-schema/types";
import {
  AtlasDataError,
  decodeCoverage,
  decodeManifest,
  decodeTrajectories,
} from "@/lib/atlas-schema/validate";
import { prepareCoverage } from "@/lib/data/prepareCoverage";
import {
  atlasFetchOptions,
  type RuntimeEnvironment,
} from "@/lib/data/cachePolicy";

const BUNDLE_BASE = "/atlas-data/demo-v1";

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
    `${BUNDLE_BASE}/${reference.filename}`,
    atlasFetchOptions(environment),
  );
  return decodeTrajectories(await jsonResponse(response, "trajectory payload"));
}

export async function loadDemoBundle(
  fetcher: typeof fetch = fetch,
  environment: RuntimeEnvironment = process.env.NODE_ENV as RuntimeEnvironment,
): Promise<AtlasData> {
  const fetchOptions = atlasFetchOptions(environment);
  const manifestResponse = await fetcher(
    `${BUNDLE_BASE}/manifest.json`,
    fetchOptions,
  );
  const manifest = decodeManifest(
    await jsonResponse(manifestResponse, "atlas manifest"),
  );
  const reference = manifest.payloads.find(
    (payload) => payload.kind === "coverage",
  );
  if (!reference) {
    throw new AtlasDataError("Atlas manifest does not reference coverage data.");
  }
  const coverageResponse = await fetcher(
    `${BUNDLE_BASE}/${reference.filename}`,
    fetchOptions,
  );
  const coverage = decodeCoverage(
    await jsonResponse(coverageResponse, "coverage payload"),
  );
  return {
    manifest,
    coverage,
    preparedArms: prepareCoverage(manifest, coverage),
  };
}
