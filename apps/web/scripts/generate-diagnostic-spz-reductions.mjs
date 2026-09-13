import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import { Matrix4, Quaternion, Vector3 } from "three";

const FRACTIONAL_BITS = 12;
const CROP_LOW = 0.025;
const CROP_HIGH = 0.975;
const MIN_OPACITY = 0.05;
const MAX_SCALE = 0.05;
const MAX_SPLATS = 100_000;
const HYBRID_IMPORTANCE_SPLATS = 80_000;
const VOXEL_SIZE_METRES = 0.011;
const MAX_PROJECTED_AREA = 64;
const MAX_FRUSTUM_MARGIN_PIXELS = 512;
const scriptDirectory = dirname(fileURLToPath(import.meta.url));

const DEFAULT_PLY = resolve(
  scriptDirectory,
  "../../../.cache/gaussian-splat/omprakash-new/eval-export/splat/splat.ply",
);
const DEFAULT_CAMERAS = resolve(
  scriptDirectory,
  "../public/environment-data/__local-real__/registered-cameras.json",
);
const DEFAULT_OUTPUT_DIRECTORY = resolve(
  scriptDirectory,
  "../public/environment-data/__local-real__",
);

const VARIANTS = {
  uniform: {
    fileName: "diagnostic-uniform-100k.spz",
    label: "Uniform sample SPZ",
  },
  cameraAware: {
    fileName: "diagnostic-camera-aware-100k.spz",
    label: "Camera-aware SPZ",
  },
  hybrid: {
    fileName: "diagnostic-hybrid-100k.spz",
    label: "Hybrid SPZ",
  },
};

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function quantile(sorted, fraction) {
  if (sorted.length === 0) {
    throw new Error("Cannot compute a quantile from an empty array.");
  }
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function distributionQuantiles(values) {
  const sorted = Float64Array.from(values).sort();
  return {
    p10: quantile(sorted, 0.1),
    p50: quantile(sorted, 0.5),
    p90: quantile(sorted, 0.9),
    p99: quantile(sorted, 0.99),
  };
}

function compareQuantiles(candidate, selected) {
  return Object.fromEntries(
    Object.keys(candidate).map((key) => [
      key,
      {
        delta: selected[key] - candidate[key],
        ratio: candidate[key] === 0 ? null : selected[key] / candidate[key],
      },
    ]),
  );
}

export function deterministicHash(value) {
  let hash = (value + 0x9e3779b9) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 16), 0x21f0aaad);
  hash = Math.imul(hash ^ (hash >>> 15), 0x735a2d97);
  return (hash ^ (hash >>> 15)) >>> 0;
}

export function selectUniformCandidateOffsets(sourceIndices, budget) {
  if (budget >= sourceIndices.length) {
    return Array.from(sourceIndices, (_, index) => index);
  }
  return Array.from(sourceIndices, (_, index) => index)
    .sort((left, right) => {
      const hashDifference =
        deterministicHash(sourceIndices[left]) -
        deterministicHash(sourceIndices[right]);
      return hashDifference || sourceIndices[left] - sourceIndices[right];
    })
    .slice(0, budget);
}

export function cameraAwareImportance({
  visibleCameraCount,
  cameraCount,
  meanFootprintContribution,
}) {
  if (visibleCameraCount <= 0 || cameraCount <= 0) return 0;
  const coverage = Math.min(1, visibleCameraCount / cameraCount);
  const footprint = Math.max(0, Math.min(1, meanFootprintContribution));
  return Math.sqrt(coverage) * (0.35 + 0.65 * footprint);
}

export function parseDiagnosticPlyHeader(buffer) {
  const marker = Buffer.from("end_header\n", "ascii");
  const markerIndex = buffer.indexOf(marker);
  if (markerIndex < 0) throw new Error("PLY end_header marker was not found.");
  const dataOffset = markerIndex + marker.length;
  const header = buffer.subarray(0, dataOffset).toString("ascii");
  if (!header.includes("format binary_little_endian 1.0")) {
    throw new Error("Only binary_little_endian PLY files are supported.");
  }
  const vertexMatch = header.match(/^element vertex (\d+)$/m);
  if (!vertexMatch) throw new Error("PLY vertex count was not found.");
  const lines = header.split(/\r?\n/);
  const properties = [];
  let insideVertex = false;
  for (const line of lines) {
    if (line.startsWith("element vertex ")) {
      insideVertex = true;
      continue;
    }
    if (insideVertex && line.startsWith("element ")) break;
    if (!insideVertex) continue;
    const match = line.match(/^property float ([A-Za-z0-9_]+)$/);
    if (match) properties.push(match[1]);
  }
  const required = [
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
  ];
  for (const name of required) {
    if (!properties.includes(name)) {
      throw new Error(`Required PLY property "${name}" is missing.`);
    }
  }
  return {
    dataOffset,
    vertexCount: Number(vertexMatch[1]),
    properties,
    propertyIndex: Object.fromEntries(
      properties.map((name, index) => [name, index]),
    ),
  };
}

function covarianceFromQuaternionAndScales(w, x, y, z, sx, sy, sz) {
  const norm = Math.hypot(w, x, y, z);
  if (!Number.isFinite(norm) || norm <= 1e-12) {
    throw new Error("Encountered an invalid zero-length quaternion.");
  }
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;
  const r00 = 1 - 2 * (y * y + z * z);
  const r01 = 2 * (x * y - z * w);
  const r02 = 2 * (x * z + y * w);
  const r10 = 2 * (x * y + z * w);
  const r11 = 1 - 2 * (x * x + z * z);
  const r12 = 2 * (y * z - x * w);
  const r20 = 2 * (x * z - y * w);
  const r21 = 2 * (y * z + x * w);
  const r22 = 1 - 2 * (x * x + y * y);
  const sx2 = sx * sx;
  const sy2 = sy * sy;
  const sz2 = sz * sz;
  return [
    r00 * r00 * sx2 + r01 * r01 * sy2 + r02 * r02 * sz2,
    r00 * r10 * sx2 + r01 * r11 * sy2 + r02 * r12 * sz2,
    r00 * r20 * sx2 + r01 * r21 * sy2 + r02 * r22 * sz2,
    r10 * r10 * sx2 + r11 * r11 * sy2 + r12 * r12 * sz2,
    r10 * r20 * sx2 + r11 * r21 * sy2 + r12 * r22 * sz2,
    r20 * r20 * sx2 + r21 * r21 * sy2 + r22 * r22 * sz2,
  ];
}

export function buildDiagnosticCandidates(input, ply) {
  const { dataOffset, vertexCount, properties, propertyIndex } = ply;
  const floatsPerVertex = properties.length;
  const bytesPerVertex = floatsPerVertex * 4;
  const expectedBytes = dataOffset + vertexCount * bytesPerVertex;
  if (input.byteLength !== expectedBytes) {
    throw new Error(
      `Unexpected PLY size. Expected ${expectedBytes} bytes, got ${input.byteLength}.`,
    );
  }
  const view = new DataView(
    input.buffer,
    input.byteOffset + dataOffset,
    vertexCount * bytesPerVertex,
  );
  const value = (vertex, propertyName) =>
    view.getFloat32(
      vertex * bytesPerVertex + propertyIndex[propertyName] * 4,
      true,
    );
  const xs = new Float32Array(vertexCount);
  const ys = new Float32Array(vertexCount);
  const zs = new Float32Array(vertexCount);
  for (let index = 0; index < vertexCount; index += 1) {
    xs[index] = value(index, "x");
    ys[index] = value(index, "y");
    zs[index] = value(index, "z");
  }
  const sortedX = Float32Array.from(xs).sort();
  const sortedY = Float32Array.from(ys).sort();
  const sortedZ = Float32Array.from(zs).sort();
  const boundsMin = [
    quantile(sortedX, CROP_LOW),
    quantile(sortedY, CROP_LOW),
    quantile(sortedZ, CROP_LOW),
  ];
  const boundsMax = [
    quantile(sortedX, CROP_HIGH),
    quantile(sortedY, CROP_HIGH),
    quantile(sortedZ, CROP_HIGH),
  ];
  const sourceIndices = new Uint32Array(vertexCount);
  const positions = new Float32Array(vertexCount * 3);
  const opacities = new Float32Array(vertexCount);
  const maxScales = new Float32Array(vertexCount);
  const covariances = new Float32Array(vertexCount * 6);
  let count = 0;
  const rejected = { spatial: 0, opacity: 0, scale: 0 };
  for (let index = 0; index < vertexCount; index += 1) {
    const x = xs[index];
    const y = ys[index];
    const z = zs[index];
    if (
      x < boundsMin[0] ||
      x > boundsMax[0] ||
      y < boundsMin[1] ||
      y > boundsMax[1] ||
      z < boundsMin[2] ||
      z > boundsMax[2]
    ) {
      rejected.spatial += 1;
      continue;
    }
    const opacity = sigmoid(value(index, "opacity"));
    if (!Number.isFinite(opacity) || opacity < MIN_OPACITY) {
      rejected.opacity += 1;
      continue;
    }
    const sx = Math.exp(value(index, "scale_0"));
    const sy = Math.exp(value(index, "scale_1"));
    const sz = Math.exp(value(index, "scale_2"));
    const maxScale = Math.max(sx, sy, sz);
    if (!Number.isFinite(maxScale) || maxScale > MAX_SCALE) {
      rejected.scale += 1;
      continue;
    }
    const covariance = covarianceFromQuaternionAndScales(
      value(index, "rot_0"),
      value(index, "rot_1"),
      value(index, "rot_2"),
      value(index, "rot_3"),
      sx,
      sy,
      sz,
    );
    sourceIndices[count] = index;
    positions[count * 3] = x;
    positions[count * 3 + 1] = y;
    positions[count * 3 + 2] = z;
    opacities[count] = opacity;
    maxScales[count] = maxScale;
    covariances.set(covariance, count * 6);
    count += 1;
  }
  return {
    boundsMin,
    boundsMax,
    count,
    covariances: covariances.subarray(0, count * 6),
    maxScales: maxScales.subarray(0, count),
    opacities: opacities.subarray(0, count),
    positions: positions.subarray(0, count * 3),
    rejected,
    sourceIndices: sourceIndices.subarray(0, count),
    value,
  };
}

export function convertDiagnosticCameras(dataset) {
  if (dataset.schemaVersion !== 1 || dataset.cameras?.length !== 210) {
    throw new Error("Expected the staged 210-camera diagnostic dataset.");
  }
  const [row0, row1, row2] = dataset.source.transform;
  const dataparserTransform = new Matrix4().set(
    ...row0,
    ...row1,
    ...row2,
    0,
    0,
    0,
    1,
  );
  return dataset.cameras.map((camera) => {
    const [w, x, y, z] = camera.quaternionWxyz;
    const worldToCamera = new Matrix4().compose(
      new Vector3(...camera.translation),
      new Quaternion(x, y, z, w),
      new Vector3(1, 1, 1),
    );
    const cameraToColmapWorld = worldToCamera
      .invert()
      .multiply(new Matrix4().makeScale(1, -1, -1));
    const cameraToWorld = dataparserTransform
      .clone()
      .multiply(cameraToColmapWorld);
    cameraToWorld.setPosition(
      new Vector3()
        .setFromMatrixPosition(cameraToWorld)
        .multiplyScalar(dataset.source.scale),
    );
    return {
      ...camera,
      worldToCamera: cameraToWorld.invert().elements.slice(),
    };
  });
}

function quadraticForm(covariances, offset, x, y, z) {
  const xx = covariances[offset];
  const xy = covariances[offset + 1];
  const xz = covariances[offset + 2];
  const yy = covariances[offset + 3];
  const yz = covariances[offset + 4];
  const zz = covariances[offset + 5];
  return (
    xx * x * x +
    yy * y * y +
    zz * z * z +
    2 * (xy * x * y + xz * x * z + yz * y * z)
  );
}

function bilinearForm(covariances, offset, ax, ay, az, bx, by, bz) {
  const xx = covariances[offset];
  const xy = covariances[offset + 1];
  const xz = covariances[offset + 2];
  const yy = covariances[offset + 3];
  const yz = covariances[offset + 4];
  const zz = covariances[offset + 5];
  return (
    ax * (xx * bx + xy * by + xz * bz) +
    ay * (xy * bx + yy * by + yz * bz) +
    az * (xz * bx + yz * by + zz * bz)
  );
}

function scoreCandidates(candidates, cameras) {
  const scores = new Float32Array(candidates.count);
  const visibleCounts = new Uint16Array(candidates.count);
  const meanContributions = new Float32Array(candidates.count);
  const logAreaDenominator = Math.log1p(MAX_PROJECTED_AREA);
  for (let index = 0; index < candidates.count; index += 1) {
    const px = candidates.positions[index * 3];
    const py = candidates.positions[index * 3 + 1];
    const pz = candidates.positions[index * 3 + 2];
    const covarianceOffset = index * 6;
    let visible = 0;
    let contributionSum = 0;
    for (const camera of cameras) {
      const matrix = camera.worldToCamera;
      const cameraX =
        matrix[0] * px + matrix[4] * py + matrix[8] * pz + matrix[12];
      const cameraY =
        matrix[1] * px + matrix[5] * py + matrix[9] * pz + matrix[13];
      const cameraZ =
        matrix[2] * px + matrix[6] * py + matrix[10] * pz + matrix[14];
      const depth = -cameraZ;
      if (depth <= 0.01 || depth >= 1e10) continue;
      const pixelX = camera.fx * (cameraX / depth) + camera.cx;
      const pixelY = camera.cy - camera.fy * (cameraY / depth);
      if (
        pixelX < -MAX_FRUSTUM_MARGIN_PIXELS ||
        pixelX > camera.width + MAX_FRUSTUM_MARGIN_PIXELS ||
        pixelY < -MAX_FRUSTUM_MARGIN_PIXELS ||
        pixelY > camera.height + MAX_FRUSTUM_MARGIN_PIXELS
      ) {
        continue;
      }
      const depthSquared = depth * depth;
      const gradientCameraUx = camera.fx / depth;
      const gradientCameraUz = (camera.fx * cameraX) / depthSquared;
      const gradientCameraVy = -camera.fy / depth;
      const gradientCameraVz = (-camera.fy * cameraY) / depthSquared;
      const gux =
        matrix[0] * gradientCameraUx + matrix[2] * gradientCameraUz;
      const guy =
        matrix[4] * gradientCameraUx + matrix[6] * gradientCameraUz;
      const guz =
        matrix[8] * gradientCameraUx + matrix[10] * gradientCameraUz;
      const gvx =
        matrix[1] * gradientCameraVy + matrix[2] * gradientCameraVz;
      const gvy =
        matrix[5] * gradientCameraVy + matrix[6] * gradientCameraVz;
      const gvz =
        matrix[9] * gradientCameraVy + matrix[10] * gradientCameraVz;
      const varianceU = Math.max(
        0,
        quadraticForm(candidates.covariances, covarianceOffset, gux, guy, guz),
      );
      const varianceV = Math.max(
        0,
        quadraticForm(candidates.covariances, covarianceOffset, gvx, gvy, gvz),
      );
      const covarianceUv = bilinearForm(
        candidates.covariances,
        covarianceOffset,
        gux,
        guy,
        guz,
        gvx,
        gvy,
        gvz,
      );
      const eigenDiscriminant = Math.sqrt(
        Math.max(
          0,
          (varianceU - varianceV) ** 2 + 4 * covarianceUv * covarianceUv,
        ),
      );
      const maximumVariance =
        (varianceU + varianceV + eigenDiscriminant) / 2;
      const margin = Math.min(
        MAX_FRUSTUM_MARGIN_PIXELS,
        3 * Math.sqrt(Math.max(0, maximumVariance)),
      );
      if (
        pixelX < -margin ||
        pixelX > camera.width + margin ||
        pixelY < -margin ||
        pixelY > camera.height + margin
      ) {
        continue;
      }
      const determinant = Math.max(
        0,
        varianceU * varianceV - covarianceUv * covarianceUv,
      );
      const projectedArea = Math.PI * Math.sqrt(determinant);
      const alphaArea = Math.min(
        MAX_PROJECTED_AREA,
        candidates.opacities[index] * projectedArea,
      );
      contributionSum += Math.log1p(alphaArea) / logAreaDenominator;
      visible += 1;
    }
    const meanContribution = visible === 0 ? 0 : contributionSum / visible;
    visibleCounts[index] = visible;
    meanContributions[index] = meanContribution;
    scores[index] = cameraAwareImportance({
      visibleCameraCount: visible,
      cameraCount: cameras.length,
      meanFootprintContribution: meanContribution,
    });
  }
  return { meanContributions, scores, visibleCounts };
}

function rankCameraAwareOffsets(candidates, scoring) {
  return Array.from(candidates.sourceIndices, (_, index) => index)
    .sort((left, right) => {
      const scoreDifference = scoring.scores[right] - scoring.scores[left];
      return scoreDifference || candidates.sourceIndices[left] - candidates.sourceIndices[right];
    });
}

export function selectScaleStratifiedCameraAwareOffsets(
  sourceIndices,
  maxScales,
  scores,
  budget,
  strata = 100,
) {
  if (budget >= sourceIndices.length) {
    return Array.from(sourceIndices, (_, index) => index);
  }
  const scaleOrdered = Array.from(sourceIndices, (_, index) => index).sort(
    (left, right) =>
      maxScales[left] - maxScales[right] ||
      sourceIndices[left] - sourceIndices[right],
  );
  const selected = [];
  for (let stratum = 0; stratum < strata; stratum += 1) {
    const start = Math.floor((sourceIndices.length * stratum) / strata);
    const end = Math.floor((sourceIndices.length * (stratum + 1)) / strata);
    const quotaStart = Math.floor((budget * stratum) / strata);
    const quotaEnd = Math.floor((budget * (stratum + 1)) / strata);
    const quota = quotaEnd - quotaStart;
    const ranked = scaleOrdered.slice(start, end).sort((left, right) => {
      const scoreDifference = scores[right] - scores[left];
      return scoreDifference || sourceIndices[left] - sourceIndices[right];
    });
    selected.push(...ranked.slice(0, quota));
  }
  return selected;
}

function voxelKey(candidates, index) {
  const x = Math.floor(
    (candidates.positions[index * 3] - candidates.boundsMin[0]) /
      VOXEL_SIZE_METRES,
  );
  const y = Math.floor(
    (candidates.positions[index * 3 + 1] - candidates.boundsMin[1]) /
      VOXEL_SIZE_METRES,
  );
  const z = Math.floor(
    (candidates.positions[index * 3 + 2] - candidates.boundsMin[2]) /
      VOXEL_SIZE_METRES,
  );
  return `${x},${y},${z}`;
}

function selectHybridOffsets(candidates, cameraRanked, budget) {
  const importanceCount = Math.min(HYBRID_IMPORTANCE_SPLATS, budget);
  const selected = cameraRanked.slice(0, importanceCount);
  const selectedFlags = new Uint8Array(candidates.count);
  const coveredCells = new Set();
  for (const index of selected) {
    selectedFlags[index] = 1;
    coveredCells.add(voxelKey(candidates, index));
  }
  const representativePerUncoveredCell = new Map();
  for (let index = 0; index < candidates.count; index += 1) {
    if (selectedFlags[index]) continue;
    const key = voxelKey(candidates, index);
    if (coveredCells.has(key)) continue;
    const previous = representativePerUncoveredCell.get(key);
    if (
      previous === undefined ||
      deterministicHash(candidates.sourceIndices[index]) <
        deterministicHash(candidates.sourceIndices[previous])
    ) {
      representativePerUncoveredCell.set(key, index);
    }
  }
  const coverageCandidates = Array.from(representativePerUncoveredCell.values()).sort(
    (left, right) =>
      deterministicHash(candidates.sourceIndices[left]) -
        deterministicHash(candidates.sourceIndices[right]) ||
      candidates.sourceIndices[left] - candidates.sourceIndices[right],
  );
  for (const index of coverageCandidates) {
    if (selected.length >= budget) break;
    selected.push(index);
    selectedFlags[index] = 1;
  }
  if (selected.length < budget) {
    for (const index of cameraRanked) {
      if (selected.length >= budget) break;
      if (!selectedFlags[index]) {
        selected.push(index);
        selectedFlags[index] = 1;
      }
    }
  }
  return selected;
}

function writeSigned24(view, offset, value) {
  const scaled = Math.round(value * (1 << FRACTIONAL_BITS));
  if (scaled < -0x800000 || scaled > 0x7fffff) {
    throw new Error(
      `Position ${value} cannot be represented with ${FRACTIONAL_BITS} fractional bits.`,
    );
  }
  const encoded = scaled & 0xffffff;
  view.setUint8(offset, encoded & 0xff);
  view.setUint8(offset + 1, (encoded >>> 8) & 0xff);
  view.setUint8(offset + 2, (encoded >>> 16) & 0xff);
}

function encodeDcCoefficient(value) {
  return clampByte((value * 0.15 + 0.5) * 255);
}

function compressedQuaternion(x, y, z, w) {
  const norm = Math.hypot(x, y, z, w);
  if (!Number.isFinite(norm) || norm <= 1e-12) {
    throw new Error("Encountered an invalid zero-length quaternion.");
  }
  const quaternion = [x / norm, y / norm, z / norm, w / norm];
  let largest = 0;
  for (let index = 1; index < 4; index += 1) {
    if (Math.abs(quaternion[index]) > Math.abs(quaternion[largest])) largest = index;
  }
  const negate = quaternion[largest] < 0 ? 1 : 0;
  let packed = largest;
  for (let index = 0; index < 4; index += 1) {
    if (index === largest) continue;
    const negative = (quaternion[index] < 0 ? 1 : 0) ^ negate;
    const magnitude = Math.max(
      0,
      Math.min(
        511,
        Math.floor(511 * (Math.abs(quaternion[index]) / Math.SQRT1_2) + 0.5),
      ),
    );
    packed = (packed << 10) | (negative << 9) | magnitude;
  }
  return packed >>> 0;
}

export function encodeDiagnosticSpz(
  selectedOffsets,
  candidates,
  maximumSplats = MAX_SPLATS,
) {
  const selectedSourceIndices = selectedOffsets
    .map((offset) => candidates.sourceIndices[offset])
    .sort((left, right) => left - right);
  const count = selectedSourceIndices.length;
  if (count < 1 || count > maximumSplats) {
    throw new Error(`Invalid diagnostic selection count ${count}.`);
  }
  const raw = new Uint8Array(16 + count * 20);
  const output = new DataView(raw.buffer);
  output.setUint32(0, 0x5053474e, true);
  output.setUint32(4, 3, true);
  output.setUint32(8, count, true);
  output.setUint8(12, 0);
  output.setUint8(13, FRACTIONAL_BITS);
  output.setUint8(14, 0);
  output.setUint8(15, 0);
  for (let outputIndex = 0; outputIndex < count; outputIndex += 1) {
    const sourceIndex = selectedSourceIndices[outputIndex];
    writeSigned24(output, 16 + outputIndex * 9, candidates.value(sourceIndex, "x"));
    writeSigned24(
      output,
      16 + outputIndex * 9 + 3,
      candidates.value(sourceIndex, "y"),
    );
    writeSigned24(
      output,
      16 + outputIndex * 9 + 6,
      candidates.value(sourceIndex, "z"),
    );
    output.setUint8(
      16 + count * 9 + outputIndex,
      clampByte(sigmoid(candidates.value(sourceIndex, "opacity")) * 255),
    );
    const rgbOffset = 16 + count * 10 + outputIndex * 3;
    output.setUint8(rgbOffset, encodeDcCoefficient(candidates.value(sourceIndex, "f_dc_0")));
    output.setUint8(
      rgbOffset + 1,
      encodeDcCoefficient(candidates.value(sourceIndex, "f_dc_1")),
    );
    output.setUint8(
      rgbOffset + 2,
      encodeDcCoefficient(candidates.value(sourceIndex, "f_dc_2")),
    );
    const scaleOffset = 16 + count * 13 + outputIndex * 3;
    output.setUint8(
      scaleOffset,
      clampByte((candidates.value(sourceIndex, "scale_0") + 10) * 16),
    );
    output.setUint8(
      scaleOffset + 1,
      clampByte((candidates.value(sourceIndex, "scale_1") + 10) * 16),
    );
    output.setUint8(
      scaleOffset + 2,
      clampByte((candidates.value(sourceIndex, "scale_2") + 10) * 16),
    );
    const w = candidates.value(sourceIndex, "rot_0");
    const x = candidates.value(sourceIndex, "rot_1");
    const y = candidates.value(sourceIndex, "rot_2");
    const z = candidates.value(sourceIndex, "rot_3");
    output.setUint32(
      16 + count * 16 + outputIndex * 4,
      compressedQuaternion(x, y, z, w),
      true,
    );
  }
  return new Uint8Array(gzipSync(raw, { level: 9, mtime: 0 }));
}

function selectedValues(values, selectedOffsets) {
  return Float64Array.from(selectedOffsets, (index) => values[index]);
}

function variantMetadata({
  key,
  selectedOffsets,
  bytes,
  candidates,
  candidateDistribution,
  scoring,
  inputPath,
  cameraPath,
  cameraCount,
}) {
  const opacity = distributionQuantiles(
    selectedValues(candidates.opacities, selectedOffsets),
  );
  const maxScale = distributionQuantiles(
    selectedValues(candidates.maxScales, selectedOffsets),
  );
  const visibleCameras = distributionQuantiles(
    selectedValues(scoring.visibleCounts, selectedOffsets),
  );
  const importanceScore = distributionQuantiles(
    selectedValues(scoring.scores, selectedOffsets),
  );
  const methods = {
    uniform:
      "Deterministic 32-bit hash ranking of all viable candidates; independent of position, opacity, scale, and camera evidence.",
    cameraAware:
      "Proportional max-scale-percentile allocation, ranked within each percentile stratum by 210-camera importance: square-root multi-view frustum coverage multiplied by a capped logarithmic opacity-weighted projected covariance footprint.",
    hybrid:
      "80,000 highest camera-aware candidates plus 20,000 deterministic representatives from 11 mm cells not already covered by the importance selection.",
  };
  return {
    schemaVersion: 1,
    developmentOnly: true,
    variant: key,
    label: VARIANTS[key].label,
    asset: VARIANTS[key].fileName,
    input: resolve(inputPath),
    registeredCameras: resolve(cameraPath),
    selectionMethod: methods[key],
    selectedSplats: selectedOffsets.length,
    candidateSplats: candidates.count,
    cameraCount,
    compressedBytes: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    spz: {
      version: 3,
      shDegree: 0,
      fractionalBits: FRACTIONAL_BITS,
      flags: 0,
    },
    filter: {
      cropQuantiles: [CROP_LOW, CROP_HIGH],
      cropBounds: {
        minimumXyz: candidates.boundsMin,
        maximumXyz: candidates.boundsMax,
      },
      minimumOpacity: MIN_OPACITY,
      maximumScale: MAX_SCALE,
      rejected: candidates.rejected,
    },
    cameraImportance: {
      coordinateSpace: "Nerfstudio reconstruction coordinates only",
      projectedCovariance: true,
      occlusionAware: false,
      maximumProjectedAlphaArea: MAX_PROJECTED_AREA,
      maximumFrustumMarginPixels: MAX_FRUSTUM_MARGIN_PIXELS,
      formula:
        "sqrt(visibleCameraCount / 210) * (0.35 + 0.65 * mean(log1p(min(64, opacity * projectedCovarianceArea)) / log1p(64)))",
    },
    distributions: {
      candidate: candidateDistribution,
      selected: { opacity, maxScale, visibleCameras, importanceScore },
      selectedVsCandidate: {
        opacity: compareQuantiles(candidateDistribution.opacity, opacity),
        maxScale: compareQuantiles(candidateDistribution.maxScale, maxScale),
      },
    },
  };
}

export async function generateDiagnosticReductions({
  inputPath = DEFAULT_PLY,
  cameraPath = DEFAULT_CAMERAS,
  outputDirectory = DEFAULT_OUTPUT_DIRECTORY,
} = {}) {
  const [input, cameraBytes] = await Promise.all([
    readFile(resolve(inputPath)),
    readFile(resolve(cameraPath), "utf8"),
  ]);
  const ply = parseDiagnosticPlyHeader(input);
  const candidates = buildDiagnosticCandidates(input, ply);
  if (candidates.count <= MAX_SPLATS) {
    throw new Error("Diagnostic reduction requires more than 100,000 candidates.");
  }
  const cameras = convertDiagnosticCameras(JSON.parse(cameraBytes));
  process.stdout.write(
    `Scoring ${candidates.count.toLocaleString()} viable Gaussians across ${cameras.length} captured cameras...\n`,
  );
  const scoring = scoreCandidates(candidates, cameras);
  const uniform = selectUniformCandidateOffsets(
    candidates.sourceIndices,
    MAX_SPLATS,
  );
  const cameraRanked = rankCameraAwareOffsets(candidates, scoring);
  const cameraAware = selectScaleStratifiedCameraAwareOffsets(
    candidates.sourceIndices,
    candidates.maxScales,
    scoring.scores,
    MAX_SPLATS,
  );
  const hybrid = selectHybridOffsets(candidates, cameraRanked, MAX_SPLATS);
  const candidateDistribution = {
    opacity: distributionQuantiles(candidates.opacities),
    maxScale: distributionQuantiles(candidates.maxScales),
    visibleCameras: distributionQuantiles(scoring.visibleCounts),
    importanceScore: distributionQuantiles(scoring.scores),
  };
  const selections = { uniform, cameraAware, hybrid };
  const summaries = [];
  await mkdir(resolve(outputDirectory), { recursive: true });
  for (const [key, selectedOffsets] of Object.entries(selections)) {
    const bytes = encodeDiagnosticSpz(selectedOffsets, candidates);
    const metadata = variantMetadata({
      key,
      selectedOffsets,
      bytes,
      candidates,
      candidateDistribution,
      scoring,
      inputPath,
      cameraPath,
      cameraCount: cameras.length,
    });
    const outputPath = resolve(outputDirectory, VARIANTS[key].fileName);
    await Promise.all([
      writeFile(outputPath, bytes),
      writeFile(`${outputPath}.json`, `${JSON.stringify(metadata, null, 2)}\n`),
    ]);
    summaries.push(metadata);
    process.stdout.write(
      `${VARIANTS[key].label}: ${metadata.selectedSplats.toLocaleString()} splats, ${metadata.compressedBytes.toLocaleString()} bytes, ${metadata.sha256}\n`,
    );
  }
  await writeFile(
    resolve(outputDirectory, "diagnostic-reductions.json"),
    `${JSON.stringify({ schemaVersion: 1, developmentOnly: true, variants: summaries }, null, 2)}\n`,
  );
  return summaries;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await generateDiagnosticReductions({
    inputPath: process.argv[2] ? resolve(process.argv[2]) : DEFAULT_PLY,
    cameraPath: process.argv[3] ? resolve(process.argv[3]) : DEFAULT_CAMERAS,
    outputDirectory: process.argv[4]
      ? resolve(process.argv[4])
      : DEFAULT_OUTPUT_DIRECTORY,
  });
}
