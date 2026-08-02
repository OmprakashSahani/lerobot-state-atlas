import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const SYNTHETIC_LABEL = "Synthetic test environment — not a real reconstruction";
export const FRACTIONAL_BITS = 12;
export const EXPECTED_SPLAT_COUNT = 163;
export const EXPECTED_SHA256 = "4fb67ec298debc9ca0d5923f283427a4af442ad9a1dc8a6d487e898654f17b98";

function planeAndLandmarks() {
  const splats = [];
  for (let y = -5; y <= 5; y += 1) {
    for (let x = -5; x <= 5; x += 1) {
      splats.push({ p: [x * 0.1, y * 0.1, -0.04], c: [0.28, 0.35, 0.42], s: [0.065, 0.065, 0.008], q: [0, 0, 0, 1], a: 0.42 });
    }
  }
  const axes = [
    { axis: 0, color: [0.95, 0.08, 0.08] },
    { axis: 1, color: [0.08, 0.95, 0.12] },
    { axis: 2, color: [0.08, 0.2, 0.98] },
  ];
  for (const { axis, color } of axes) {
    for (let step = 1; step <= 12; step += 1) {
      const p = [0, 0, 0];
      p[axis] = step * 0.055;
      splats.push({ p, c: color, s: [0.025, 0.025, 0.025], q: [0, 0, 0, 1], a: 0.96 });
    }
  }
  const half = Math.sin(Math.PI / 8);
  const wide = Math.cos(Math.PI / 8);
  splats.push(
    { p: [0.22, 0.22, 0.15], c: [1, 0.45, 0.05], s: [0.16, 0.025, 0.045], q: [0, 0, half, wide], a: 0.75 },
    { p: [-0.22, 0.18, 0.2], c: [0.55, 0.1, 0.95], s: [0.03, 0.14, 0.05], q: [half, 0, 0, wide], a: 0.62 },
    { p: [0, -0.22, 0.17], c: [0.05, 0.8, 0.95], s: [0.11, 0.035, 0.035], q: [0, half, 0, wide], a: 0.84 },
    { p: [0.1, -0.08, 0.28], c: [1, 0.86, 0.12], s: [0.09, 0.09, 0.025], q: [0, 0, 0, 1], a: 0.38 },
    { p: [0.1, -0.08, 0.23], c: [0.12, 0.52, 1], s: [0.06, 0.06, 0.03], q: [0, 0, 0, 1], a: 0.9 },
    { p: [-0.35, -0.3, 0.12], c: [1, 0.15, 0.55], s: [0.045, 0.1, 0.025], q: [0, 0, half, wide], a: 0.55 },
  );
  return splats;
}

function write24(view, offset, value) {
  const integer = Math.max(-0x7fffff, Math.min(0x7fffff, Math.round(value * (1 << FRACTIONAL_BITS))));
  view.setUint8(offset, integer & 0xff);
  view.setUint8(offset + 1, (integer >> 8) & 0xff);
  view.setUint8(offset + 2, (integer >> 16) & 0xff);
}

function scaledRgb(value) {
  const shC0 = 0.28209479177387814;
  return Math.max(0, Math.min(255, Math.round(((value - 0.5) / (shC0 / 0.15) + 0.5) * 255)));
}

function compressedQuaternion(input) {
  const norm = Math.hypot(...input);
  const q = input.map((value) => value / norm);
  let largest = 0;
  for (let index = 1; index < 4; index += 1) if (Math.abs(q[index]) > Math.abs(q[largest])) largest = index;
  const negate = q[largest] < 0 ? 1 : 0;
  let packed = largest;
  for (let index = 0; index < 4; index += 1) {
    if (index === largest) continue;
    const negative = (q[index] < 0 ? 1 : 0) ^ negate;
    const magnitude = Math.floor(511 * (Math.abs(q[index]) / Math.SQRT1_2) + 0.5);
    packed = (packed << 10) | (negative << 9) | magnitude;
  }
  return packed >>> 0;
}

export function generateSyntheticSpz() {
  const splats = planeAndLandmarks();
  if (splats.length !== EXPECTED_SPLAT_COUNT) throw new Error("Synthetic fixture count changed unexpectedly.");
  const count = splats.length;
  const output = new Uint8Array(16 + count * 20);
  const view = new DataView(output.buffer);
  view.setUint32(0, 0x5053474e, true);
  view.setUint32(4, 3, true);
  view.setUint32(8, count, true);
  view.setUint8(12, 0);
  view.setUint8(13, FRACTIONAL_BITS);
  view.setUint8(14, 1);
  view.setUint8(15, 0);
  splats.forEach((splat, index) => {
    write24(view, 16 + index * 9, splat.p[0]);
    write24(view, 16 + index * 9 + 3, splat.p[1]);
    write24(view, 16 + index * 9 + 6, splat.p[2]);
    view.setUint8(16 + count * 9 + index, Math.round(splat.a * 255));
    const rgb = 16 + count * 10 + index * 3;
    view.setUint8(rgb, scaledRgb(splat.c[0]));
    view.setUint8(rgb + 1, scaledRgb(splat.c[1]));
    view.setUint8(rgb + 2, scaledRgb(splat.c[2]));
    const scale = 16 + count * 13 + index * 3;
    splat.s.forEach((value, component) => view.setUint8(scale + component, Math.max(0, Math.min(255, Math.round((Math.log(value) + 10) * 16)))));
    view.setUint32(16 + count * 16 + index * 4, compressedQuaternion(splat.q), true);
  });
  return new Uint8Array(gzipSync(output, { level: 9, mtime: 0 }));
}

export function syntheticManifest(bytes) {
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  return {
    schema: { name: "lerobot-state-atlas.environment-layer", major: 1, minor: 0 },
    environmentId: "local-synthetic-spz-spike",
    label: SYNTHETIC_LABEL,
    status: "available",
    provenance: { sourceKind: "synthetic-test", description: `${SYNTHETIC_LABEL}. Application-owned deterministic fixture.`, reconstructionClaim: false },
    coordinateFrame: "canonical-shared-world",
    alignment: { translationXyz: [0, 0, 0], rotationXyzw: [0, 0, 0, 1], uniformScale: 1, calibrated: false, disclosure: "Synthetic fixture alignment is illustrative and is not calibrated to robot data." },
    bounds: { minimumXyz: [-0.6, -0.6, -0.06], maximumXyz: [0.72, 0.6, 0.35] },
    asset: { filename: "synthetic-environment.spz", format: "spz", mimeType: "application/octet-stream", byteSize: bytes.byteLength, sha256, splatCount: EXPECTED_SPLAT_COUNT },
  };
}

export async function stageSyntheticFixture(outputRoot) {
  const scriptDirectory = dirname(fileURLToPath(import.meta.url));
  const approvedRoot = resolve(scriptDirectory, "../public/environment-data/__local-synthetic__");
  const requestedRoot = resolve(outputRoot ?? approvedRoot);
  if (requestedRoot !== approvedRoot) throw new Error(`Refusing to write outside ${approvedRoot}.`);
  const bytes = generateSyntheticSpz();
  const manifest = syntheticManifest(bytes);
  if (EXPECTED_SHA256 !== "TO_BE_REPLACED" && manifest.asset.sha256 !== EXPECTED_SHA256) throw new Error("Synthetic fixture checksum changed unexpectedly.");
  await mkdir(approvedRoot, { recursive: true });
  await writeFile(resolve(approvedRoot, "synthetic-environment.spz"), bytes);
  await writeFile(resolve(approvedRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(`${bytes.byteLength} bytes ${manifest.asset.sha256} ${EXPECTED_SPLAT_COUNT} splats\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await stageSyntheticFixture(process.argv[2]);
