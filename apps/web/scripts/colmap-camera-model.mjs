import { readFile } from "node:fs/promises";

const CAMERA_MODELS = new Map([
  [0, { name: "SIMPLE_PINHOLE", parameterCount: 3 }],
  [1, { name: "PINHOLE", parameterCount: 4 }],
  [2, { name: "SIMPLE_RADIAL", parameterCount: 4 }],
  [3, { name: "RADIAL", parameterCount: 5 }],
  [4, { name: "OPENCV", parameterCount: 8 }],
  [5, { name: "OPENCV_FISHEYE", parameterCount: 8 }],
  [6, { name: "FULL_OPENCV", parameterCount: 12 }],
  [7, { name: "FOV", parameterCount: 5 }],
  [8, { name: "SIMPLE_RADIAL_FISHEYE", parameterCount: 4 }],
  [9, { name: "RADIAL_FISHEYE", parameterCount: 5 }],
  [10, { name: "THIN_PRISM_FISHEYE", parameterCount: 12 }],
]);

class BinaryCursor {
  constructor(buffer, source) {
    this.buffer = buffer;
    this.source = source;
    this.offset = 0;
  }

  ensure(byteCount) {
    if (this.offset + byteCount > this.buffer.length) {
      throw new Error(`Unexpected end of COLMAP binary: ${this.source}`);
    }
  }

  uint64() {
    this.ensure(8);
    const value = this.buffer.readBigUInt64LE(this.offset);
    this.offset += 8;
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`COLMAP integer exceeds JavaScript's safe range: ${value}`);
    }
    return Number(value);
  }

  int32() {
    this.ensure(4);
    const value = this.buffer.readInt32LE(this.offset);
    this.offset += 4;
    return value;
  }

  float64() {
    this.ensure(8);
    const value = this.buffer.readDoubleLE(this.offset);
    this.offset += 8;
    if (!Number.isFinite(value)) {
      throw new Error(`Non-finite COLMAP value in ${this.source}`);
    }
    return value;
  }

  nullTerminatedString() {
    const end = this.buffer.indexOf(0, this.offset);
    if (end < 0) {
      throw new Error(`Unterminated COLMAP image name in ${this.source}`);
    }
    const value = this.buffer.toString("utf8", this.offset, end);
    this.offset = end + 1;
    return value;
  }
}

export async function readColmapCamerasBinary(path) {
  const cursor = new BinaryCursor(await readFile(path), path);
  const cameraCount = cursor.uint64();
  const cameras = new Map();

  for (let index = 0; index < cameraCount; index += 1) {
    const cameraId = cursor.int32();
    const modelId = cursor.int32();
    const model = CAMERA_MODELS.get(modelId);
    if (!model) throw new Error(`Unsupported COLMAP camera model ID ${modelId}.`);
    const width = cursor.uint64();
    const height = cursor.uint64();
    const parameters = Array.from(
      { length: model.parameterCount },
      () => cursor.float64(),
    );
    cameras.set(cameraId, {
      cameraId,
      model: model.name,
      width,
      height,
      parameters,
    });
  }

  if (cursor.offset !== cursor.buffer.length) {
    throw new Error(`Trailing bytes in COLMAP camera model: ${path}`);
  }
  return cameras;
}

export async function readColmapImagesBinary(path) {
  const cursor = new BinaryCursor(await readFile(path), path);
  const imageCount = cursor.uint64();
  const images = [];

  for (let index = 0; index < imageCount; index += 1) {
    const colmapImageId = cursor.int32();
    const quaternionWxyz = Array.from({ length: 4 }, () => cursor.float64());
    const translation = Array.from({ length: 3 }, () => cursor.float64());
    const cameraId = cursor.int32();
    const sourceImage = cursor.nullTerminatedString();
    const pointCount = cursor.uint64();
    const pointBytes = pointCount * 24;
    cursor.ensure(pointBytes);
    cursor.offset += pointBytes;
    images.push({
      colmapImageId,
      cameraId,
      sourceImage,
      quaternionWxyz,
      translation,
    });
  }

  if (cursor.offset !== cursor.buffer.length) {
    throw new Error(`Trailing bytes in COLMAP image model: ${path}`);
  }
  return images.sort((left, right) => left.colmapImageId - right.colmapImageId);
}

function pinholeIntrinsics(camera) {
  if (camera.model === "PINHOLE" && camera.parameters.length === 4) {
    const [fx, fy, cx, cy] = camera.parameters;
    return { fx, fy, cx, cy };
  }
  if (camera.model === "SIMPLE_PINHOLE" && camera.parameters.length === 3) {
    const [focalLength, cx, cy] = camera.parameters;
    return { fx: focalLength, fy: focalLength, cx, cy };
  }
  throw new Error(
    `The Spark camera diagnostic requires undistorted PINHOLE cameras, received ${camera.model}.`,
  );
}

export async function parseRegisteredColmapCameras({
  camerasPath,
  imagesPath,
  evalInterval,
}) {
  const cameraModels = await readColmapCamerasBinary(camerasPath);
  const images = await readColmapImagesBinary(imagesPath);

  return images.map((image, registeredIndex) => {
    const camera = cameraModels.get(image.cameraId);
    if (!camera) {
      throw new Error(
        `COLMAP image ${image.colmapImageId} references missing camera ${image.cameraId}.`,
      );
    }
    return {
      registeredIndex,
      colmapImageId: image.colmapImageId,
      sourceImage: image.sourceImage,
      cameraId: image.cameraId,
      cameraModel: camera.model,
      width: camera.width,
      height: camera.height,
      ...pinholeIntrinsics(camera),
      quaternionWxyz: image.quaternionWxyz,
      translation: image.translation,
      evalIndex:
        registeredIndex % evalInterval === 0
          ? registeredIndex / evalInterval
          : null,
    };
  });
}
