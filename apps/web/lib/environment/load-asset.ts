import type { AvailableEnvironmentManifest } from "./types";
import { MAX_ENVIRONMENT_ASSET_BYTES } from "./limits";
import { assertSafeFinalResponseUrl } from "./path-safety";
import { EnvironmentLoadError, readResponseBytes } from "./load-manifest";

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export async function loadVerifiedEnvironmentAsset(
  assetPath: string,
  manifest: AvailableEnvironmentManifest,
  origin: string,
  signal: AbortSignal,
  fetcher: typeof fetch = fetch,
  onPhase?: (phase: "asset-loaded" | "verifying") => void,
): Promise<Uint8Array> {
  const response = await fetcher(new URL(assetPath, origin), { signal, redirect: "error" });
  assertSafeFinalResponseUrl(response, assetPath, origin);
  const bytes = await readResponseBytes(response, MAX_ENVIRONMENT_ASSET_BYTES);
  onPhase?.("asset-loaded");
  if (bytes.byteLength !== manifest.asset.byteSize) {
    throw new EnvironmentLoadError("Environment asset size does not match its manifest.");
  }
  onPhase?.("verifying");
  if ((await sha256Hex(bytes)) !== manifest.asset.sha256) {
    throw new EnvironmentLoadError("Environment asset checksum does not match its manifest.");
  }
  return bytes;
}
