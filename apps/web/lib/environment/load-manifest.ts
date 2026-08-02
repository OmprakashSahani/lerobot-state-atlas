import type { AvailableEnvironmentManifest } from "./types";
import { decodeEnvironmentManifest } from "./validate";
import { MAX_ENVIRONMENT_MANIFEST_BYTES, MAX_ENVIRONMENT_ASSET_BYTES, MAX_ENVIRONMENT_SPLATS } from "./limits";
import { assertSafeFinalResponseUrl, resolveLocalAssetPath, validateLocalManifestPath } from "./path-safety";

export class EnvironmentLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EnvironmentLoadError";
  }
}

export async function readResponseBytes(response: Response, cap: number): Promise<Uint8Array> {
  if (!response.ok) throw new EnvironmentLoadError(`Environment request failed with HTTP ${response.status}.`);
  const declared = response.headers.get("content-length");
  if (declared !== null) {
    const length = Number(declared);
    if (!Number.isSafeInteger(length) || length < 0 || length > cap) {
      throw new EnvironmentLoadError("Environment Content-Length exceeds the allowed limit.");
    }
  }
  if (!response.body) throw new EnvironmentLoadError("Environment response has no readable body.");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > cap) {
        await reader.cancel();
        throw new EnvironmentLoadError("Environment response exceeded the streaming byte limit.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  if (declared !== null && total !== Number(declared)) {
    throw new EnvironmentLoadError("Environment Content-Length does not match actual bytes.");
  }
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

export interface LoadedLocalManifest {
  manifest: AvailableEnvironmentManifest;
  assetPath: string;
}

export async function loadLocalEnvironmentManifest(
  manifestPath: string,
  origin: string,
  signal: AbortSignal,
  fetcher: typeof fetch = fetch,
): Promise<LoadedLocalManifest> {
  const path = validateLocalManifestPath(manifestPath);
  const response = await fetcher(new URL(path, origin), { signal, redirect: "error" });
  assertSafeFinalResponseUrl(response, path, origin);
  const bytes = await readResponseBytes(response, MAX_ENVIRONMENT_MANIFEST_BYTES);
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new EnvironmentLoadError("Environment manifest is not valid UTF-8 JSON.");
  }
  const manifest = decodeEnvironmentManifest(value);
  if (manifest.status !== "available") throw new EnvironmentLoadError("Local synthetic environment is not available.");
  if (manifest.provenance.sourceKind !== "synthetic-test" || manifest.provenance.reconstructionClaim !== false) {
    throw new EnvironmentLoadError("Local environment must have truthful synthetic provenance.");
  }
  if (manifest.asset.byteSize > MAX_ENVIRONMENT_ASSET_BYTES || manifest.asset.splatCount > MAX_ENVIRONMENT_SPLATS) {
    throw new EnvironmentLoadError("Local environment exceeds spike limits.");
  }
  return { manifest, assetPath: resolveLocalAssetPath(path, manifest.asset.filename) };
}
