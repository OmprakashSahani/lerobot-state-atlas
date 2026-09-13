import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { gzipSync } from "node:zlib";

const FRACTIONAL_BITS = 12;
const CROP_LOW = 0.025;
const CROP_HIGH = 0.975;
const MIN_OPACITY = 0.05;
const MAX_SCALE = 0.05;
const VOXEL_SIZE_METRES = 0.011;
const MAX_SPLATS = 100_000;


function usage() {
  console.error(
    "Usage: node scripts/convert-nerfstudio-ply-to-spz.mjs <input.ply> <output.spz>",
  );
  process.exit(2);
}

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

function quantile(sorted, q) {
  if (sorted.length === 0) {
    throw new Error("Cannot compute a quantile from an empty array.");
  }

  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);

  if (lower === upper) return sorted[lower];

  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
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
  // SPZ v3 stores SH coefficients around 0.5 using a 0.15 scale.
  return clampByte((value * 0.15 + 0.5) * 255);
}

function compressedQuaternion(x, y, z, w) {
  const norm = Math.hypot(x, y, z, w);

  if (!Number.isFinite(norm) || norm <= 1e-12) {
    throw new Error("Encountered an invalid zero-length quaternion.");
  }

  const q = [x / norm, y / norm, z / norm, w / norm];

  let largest = 0;
  for (let index = 1; index < 4; index += 1) {
    if (Math.abs(q[index]) > Math.abs(q[largest])) largest = index;
  }

  const negate = q[largest] < 0 ? 1 : 0;
  let packed = largest;

  for (let index = 0; index < 4; index += 1) {
    if (index === largest) continue;

    const negative = (q[index] < 0 ? 1 : 0) ^ negate;
    const magnitude = Math.max(
      0,
      Math.min(
        511,
        Math.floor(
          511 * (Math.abs(q[index]) / Math.SQRT1_2) + 0.5,
        ),
      ),
    );

    packed = (packed << 10) | (negative << 9) | magnitude;
  }

  return packed >>> 0;
}

function parsePlyHeader(buffer) {
  const marker = Buffer.from("end_header\n", "ascii");
  const markerIndex = buffer.indexOf(marker);

  if (markerIndex < 0) {
    throw new Error("PLY end_header marker was not found.");
  }

  const dataOffset = markerIndex + marker.length;
  const header = buffer.subarray(0, dataOffset).toString("ascii");

  if (!header.includes("format binary_little_endian 1.0")) {
    throw new Error("Only binary_little_endian PLY files are supported.");
  }

  const vertexMatch = header.match(/^element vertex (\d+)$/m);
  if (!vertexMatch) {
    throw new Error("PLY vertex count was not found.");
  }

  const vertexCount = Number(vertexMatch[1]);

  const lines = header.split(/\r?\n/);
  const properties = [];
  let insideVertex = false;

  for (const line of lines) {
    if (line.startsWith("element vertex ")) {
      insideVertex = true;
      continue;
    }

    if (insideVertex && line.startsWith("element ")) {
      break;
    }

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
    vertexCount,
    properties,
    propertyIndex: Object.fromEntries(
      properties.map((name, index) => [name, index]),
    ),
  };
}

async function convert(inputPath, outputPath) {
  const input = await readFile(inputPath);
  const {
    dataOffset,
    vertexCount,
    properties,
    propertyIndex,
  } = parsePlyHeader(input);

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

  function value(vertex, propertyName) {
    const property = propertyIndex[propertyName];
    return view.getFloat32(
      vertex * bytesPerVertex + property * 4,
      true,
    );
  }

  const xs = new Float32Array(vertexCount);
  const ys = new Float32Array(vertexCount);
  const zs = new Float32Array(vertexCount);

  for (let index = 0; index < vertexCount; index += 1) {
    xs[index] = value(index, "x");
    ys[index] = value(index, "y");
    zs[index] = value(index, "z");
  }

  const sortedX = Float32Array.from(xs);
  const sortedY = Float32Array.from(ys);
  const sortedZ = Float32Array.from(zs);

  sortedX.sort();
  sortedY.sort();
  sortedZ.sort();

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

  console.log("Input splats:", vertexCount);
  console.log("Crop min:", boundsMin);
  console.log("Crop max:", boundsMax);

  const bestPerCell = new Map();

  let spatialRejected = 0;
  let opacityRejected = 0;
  let scaleRejected = 0;
  let candidates = 0;

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
      spatialRejected += 1;
      continue;
    }

    const opacity = sigmoid(value(index, "opacity"));

    if (!Number.isFinite(opacity) || opacity < MIN_OPACITY) {
      opacityRejected += 1;
      continue;
    }

    const scale0 = Math.exp(value(index, "scale_0"));
    const scale1 = Math.exp(value(index, "scale_1"));
    const scale2 = Math.exp(value(index, "scale_2"));
    const maxScale = Math.max(scale0, scale1, scale2);

    if (!Number.isFinite(maxScale) || maxScale > MAX_SCALE) {
      scaleRejected += 1;
      continue;
    }

    candidates += 1;

    const cellX = Math.floor((x - boundsMin[0]) / VOXEL_SIZE_METRES);
    const cellY = Math.floor((y - boundsMin[1]) / VOXEL_SIZE_METRES);
    const cellZ = Math.floor((z - boundsMin[2]) / VOXEL_SIZE_METRES);

    const key = `${cellX},${cellY},${cellZ}`;

    const voxelCenterX =
      boundsMin[0] + (cellX + 0.5) * VOXEL_SIZE_METRES;
    const voxelCenterY =
      boundsMin[1] + (cellY + 0.5) * VOXEL_SIZE_METRES;
    const voxelCenterZ =
      boundsMin[2] + (cellZ + 0.5) * VOXEL_SIZE_METRES;

    const distanceSquared =
      (x - voxelCenterX) ** 2 +
      (y - voxelCenterY) ** 2 +
      (z - voxelCenterZ) ** 2;

    const previous = bestPerCell.get(key);

    if (
      previous === undefined ||
      distanceSquared < previous.distanceSquared ||
      (distanceSquared === previous.distanceSquared &&
        maxScale < previous.maxScale) ||
      (distanceSquared === previous.distanceSquared &&
        maxScale === previous.maxScale &&
        opacity > previous.opacity) ||
      (distanceSquared === previous.distanceSquared &&
        maxScale === previous.maxScale &&
        opacity === previous.opacity &&
        index < previous.index)
    ) {
      bestPerCell.set(key, {
        index,
        distanceSquared,
        maxScale,
        opacity,
      });
    }
  }

  const selected = Array.from(
    bestPerCell.values(),
    (entry) => entry.index,
  ).sort((a, b) => a - b);

  console.log("Spatially rejected:", spatialRejected);
  console.log("Opacity rejected:", opacityRejected);
  console.log("Scale rejected:", scaleRejected);
  console.log("Candidates:", candidates);
  console.log("Occupied voxels / selected splats:", selected.length);

  if (selected.length < 1) {
    throw new Error("No splats survived conversion.");
  }

  if (selected.length > MAX_SPLATS) {
    throw new Error(
      `Selected ${selected.length} splats, exceeding the ${MAX_SPLATS} web limit.`,
    );
  }

  const count = selected.length;

  // SPZ v3 SH-degree-0 layout:
  //
  // header       16 bytes
  // positions     9 bytes × count
  // alpha         1 byte  × count
  // SH DC/RGB     3 bytes × count
  // scales        3 bytes × count
  // rotations     4 bytes × count
  //
  // Total = 16 + 20 * count.
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
    const sourceIndex = selected[outputIndex];

    const x = value(sourceIndex, "x");
    const y = value(sourceIndex, "y");
    const z = value(sourceIndex, "z");

    const positionOffset = 16 + outputIndex * 9;

    writeSigned24(output, positionOffset, x);
    writeSigned24(output, positionOffset + 3, y);
    writeSigned24(output, positionOffset + 6, z);

    const opacity = sigmoid(value(sourceIndex, "opacity"));

    output.setUint8(
      16 + count * 9 + outputIndex,
      clampByte(opacity * 255),
    );

    const rgbOffset = 16 + count * 10 + outputIndex * 3;

    output.setUint8(
      rgbOffset,
      encodeDcCoefficient(value(sourceIndex, "f_dc_0")),
    );
    output.setUint8(
      rgbOffset + 1,
      encodeDcCoefficient(value(sourceIndex, "f_dc_1")),
    );
    output.setUint8(
      rgbOffset + 2,
      encodeDcCoefficient(value(sourceIndex, "f_dc_2")),
    );

    const scaleOffset = 16 + count * 13 + outputIndex * 3;

    output.setUint8(
      scaleOffset,
      clampByte((value(sourceIndex, "scale_0") + 10) * 16),
    );
    output.setUint8(
      scaleOffset + 1,
      clampByte((value(sourceIndex, "scale_1") + 10) * 16),
    );
    output.setUint8(
      scaleOffset + 2,
      clampByte((value(sourceIndex, "scale_2") + 10) * 16),
    );

    // Nerfstudio/gsplat PLY rotation order is WXYZ.
    // The existing SPZ encoder works with XYZW.
    const w = value(sourceIndex, "rot_0");
    const qx = value(sourceIndex, "rot_1");
    const qy = value(sourceIndex, "rot_2");
    const qz = value(sourceIndex, "rot_3");

    output.setUint32(
      16 + count * 16 + outputIndex * 4,
      compressedQuaternion(qx, qy, qz, w),
      true,
    );
  }

  const compressed = new Uint8Array(
    gzipSync(raw, {
      level: 9,
      mtime: 0,
    }),
  );

  const sha256 = createHash("sha256")
    .update(compressed)
    .digest("hex");

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, compressed);

  const summary = {
    input: resolve(inputPath),
    output: resolve(outputPath),
    inputSplats: vertexCount,
    candidateSplats: candidates,
    selectedSplats: count,
    cropQuantiles: [CROP_LOW, CROP_HIGH],
    cropBounds: {
      minimumXyz: boundsMin,
      maximumXyz: boundsMax,
    },
    minimumOpacity: MIN_OPACITY,
    maximumScale: MAX_SCALE,
    voxelSizeMetres: VOXEL_SIZE_METRES,
    fractionalBits: FRACTIONAL_BITS,
    shDegree: 0,
    rawBytes: raw.byteLength,
    compressedBytes: compressed.byteLength,
    sha256,
  };

  await writeFile(
    `${outputPath}.json`,
    `${JSON.stringify(summary, null, 2)}\n`,
  );

  console.log("");
  console.log("Conversion complete");
  console.log("Selected splats:", count);
  console.log("Raw SPZ bytes:", raw.byteLength);
  console.log("Compressed SPZ bytes:", compressed.byteLength);
  console.log("SHA-256:", sha256);
  console.log("SPZ:", resolve(outputPath));
  console.log("Metadata:", resolve(`${outputPath}.json`));
}

const [, , inputArg, outputArg] = process.argv;

if (!inputArg || !outputArg) usage();

await convert(resolve(inputArg), resolve(outputArg));
