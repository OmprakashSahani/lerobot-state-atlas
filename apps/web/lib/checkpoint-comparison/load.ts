import { decodeCheckpointComparison } from "./validate";

function safeBundlePath(value: string): string {
  const hasSingleTrailingSlash = value.endsWith("/") && !value.endsWith("//");
  const normalized = hasSingleTrailingSlash ? value.slice(0, -1) : value;
  const segments = normalized.split("/").slice(1);
  if (
    !normalized ||
    normalized === "/" ||
    !normalized.startsWith("/") ||
    normalized.startsWith("//") ||
    value.endsWith("//") ||
    normalized.includes("://") ||
    normalized.includes("\\") ||
    normalized.includes("?") ||
    normalized.includes("#") ||
    segments.some((segment) => segment === ".." || segment === "." || segment.length === 0)
  ) {
    throw new Error("Comparison bundle URL must be a safe same-origin absolute path.");
  }
  return normalized;
}

async function digest(bytes: Uint8Array): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, "0")).join("");
}

export async function loadCheckpointComparison(bundlePath: string, signal?: AbortSignal) {
  const base = safeBundlePath(bundlePath);
  const manifestResponse = await fetch(`${base}/manifest.json`, { cache: "no-store", signal });
  if (!manifestResponse.ok) throw new Error(`Comparison manifest request failed (${manifestResponse.status}).`);
  const manifest = await manifestResponse.json();
  const payload = manifest?.payloads?.[0];
  const filename = payload?.filename;
  if (typeof filename !== "string" || !filename || filename.includes("/") || filename.includes("\\") || filename === "." || filename === "..") throw new Error("Comparison payload filename is unsafe.");
  const plansResponse = await fetch(`${base}/${filename}`, { cache: "no-store", signal });
  if (!plansResponse.ok) throw new Error(`Comparison plans request failed (${plansResponse.status}).`);
  const bytes = new Uint8Array(await plansResponse.arrayBuffer());
  if (bytes.byteLength !== payload.byteSize) throw new Error("Comparison plans byte count does not match the manifest.");
  if ((await digest(bytes)) !== payload.sha256) throw new Error("Comparison plans checksum does not match the manifest.");
  let plans: unknown;
  try {
    plans = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (error) {
    throw new Error(`Comparison plans are not valid UTF-8 JSON: ${String(error)}`);
  }
  return decodeCheckpointComparison(manifest, plans);
}
