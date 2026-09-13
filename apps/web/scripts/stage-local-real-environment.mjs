import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const MAX_ASSET_BYTES = 8 * 1024 * 1024;
const MAX_SPLATS = 250_000;
export const PROMOTED_ASSET = Object.freeze({
  variant: "uniform250k",
  selectedSplats: 250_000,
  compressedBytes: 3_990_074,
  sha256: "f90b5ed3d1c672f62e25fca1e6acf9534901f9e8c8c80ff1a6c6b874c1d7386d",
  selectionMethod:
    "Deterministic 32-bit hash sample of the viable cropped/filtered candidate distribution.",
});
export const PROMOTED_CAMERA_DATA = Object.freeze({
  filename: "registered-cameras.json",
  byteSize: 130_912,
  sha256: "49077a800713ef38070fd4ec547f73840373b914f548a0bc0e0029362a74f60a",
  registeredCameraCount: 210,
  dominantCameraCount: 205,
  detachedColmapImageIds: [197, 198, 199, 200, 201],
});

function fail(message) {
  throw new Error(message);
}

function finiteVector3(value, name) {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    value.some((component) => !Number.isFinite(component))
  ) {
    fail(`${name} must contain exactly three finite numbers.`);
  }
  return value;
}

export function validateMetadata(metadata, bytes) {
  if (!Number.isInteger(metadata.selectedSplats) || metadata.selectedSplats < 1) {
    fail("metadata.selectedSplats must be a positive integer.");
  }

  if (metadata.selectedSplats > MAX_SPLATS) {
    fail(`SPZ contains more than ${MAX_SPLATS} permitted splats.`);
  }

  const spz = metadata.spz ?? metadata;
  if (spz.version !== 3) {
    fail("Only SPZ version 3 is permitted by the current browser environment.");
  }

  if (spz.shDegree !== 0) {
    fail("Only SH degree 0 is permitted by the current browser spike.");
  }

  if (
    !Number.isInteger(spz.fractionalBits) ||
    spz.fractionalBits < 8 ||
    spz.fractionalBits > 16
  ) {
    fail("SPZ fractional bits must be between 8 and 16.");
  }

  if (spz.flags !== 0) {
    fail("The promoted workcell asset must use SPZ flags 0.");
  }

  if (bytes.byteLength > MAX_ASSET_BYTES) {
    fail(`SPZ exceeds the ${MAX_ASSET_BYTES}-byte browser spike limit.`);
  }

  if (metadata.compressedBytes !== bytes.byteLength) {
    fail("Converter metadata byte size does not match the SPZ asset.");
  }

  const sha256 = createHash("sha256").update(bytes).digest("hex");

  if (metadata.sha256 !== sha256) {
    fail("Converter metadata SHA-256 does not match the SPZ asset.");
  }

  if (
    metadata.developmentOnly !== true ||
    metadata.variant !== PROMOTED_ASSET.variant ||
    metadata.selectedSplats !== PROMOTED_ASSET.selectedSplats ||
    metadata.compressedBytes !== PROMOTED_ASSET.compressedBytes ||
    metadata.sha256 !== PROMOTED_ASSET.sha256 ||
    metadata.selectionMethod !== PROMOTED_ASSET.selectionMethod
  ) {
    fail("Input is not the reviewed deterministic Uniform 250k asset.");
  }

  const minimumXyz = finiteVector3(
    (metadata.viabilityFilter?.cropBounds ?? metadata.cropBounds)?.minimumXyz,
    "metadata.cropBounds.minimumXyz",
  );

  const maximumXyz = finiteVector3(
    (metadata.viabilityFilter?.cropBounds ?? metadata.cropBounds)?.maximumXyz,
    "metadata.cropBounds.maximumXyz",
  );

  if (minimumXyz.some((minimum, axis) => minimum >= maximumXyz[axis])) {
    fail("Crop bounds are not ordered.");
  }

  return {
    sha256,
    minimumXyz,
    maximumXyz,
  };
}

export function validateCameraData(cameraBytes) {
  if (cameraBytes.byteLength !== PROMOTED_CAMERA_DATA.byteSize) {
    fail("Registered-camera data byte size does not match the reviewed file.");
  }
  const sha256 = createHash("sha256").update(cameraBytes).digest("hex");
  if (sha256 !== PROMOTED_CAMERA_DATA.sha256) {
    fail("Registered-camera data SHA-256 does not match the reviewed file.");
  }

  let dataset;
  try {
    dataset = JSON.parse(cameraBytes.toString("utf8"));
  } catch {
    fail("Registered-camera data is not valid JSON.");
  }
  if (
    dataset.schemaVersion !== 1 ||
    !Array.isArray(dataset.cameras) ||
    dataset.cameras.length !== PROMOTED_CAMERA_DATA.registeredCameraCount ||
    dataset.cameras.some(
      (camera, index) => camera.registeredIndex !== index,
    )
  ) {
    fail("Registered-camera data does not contain the reviewed 210-camera set.");
  }
  const detached = dataset.cameras
    .filter((camera) =>
      PROMOTED_CAMERA_DATA.detachedColmapImageIds.includes(
        camera.colmapImageId,
      ),
    )
    .map((camera) => camera.colmapImageId)
    .sort((left, right) => left - right);
  if (
    detached.length !== PROMOTED_CAMERA_DATA.detachedColmapImageIds.length ||
    detached.some(
      (colmapImageId, index) =>
        colmapImageId !==
        PROMOTED_CAMERA_DATA.detachedColmapImageIds[index],
    )
  ) {
    fail("Registered-camera data does not preserve the reviewed detached subgroup.");
  }
  return { dataset, sha256 };
}

export async function stageRealEnvironment(spzPath, metadataPath, cameraPath) {
  if (!spzPath || !metadataPath || !cameraPath) {
    fail(
      "Usage: node scripts/stage-local-real-environment.mjs <input.spz> <metadata.json> <registered-cameras.json>",
    );
  }

  const scriptDirectory = dirname(fileURLToPath(import.meta.url));
  const approvedRoot = resolve(
    scriptDirectory,
    "../public/environment-data/__local-real__",
  );

  const bytes = await readFile(resolve(spzPath));
  const metadata = JSON.parse(await readFile(resolve(metadataPath), "utf8"));
  const cameraBytes = await readFile(resolve(cameraPath));

  const { sha256, minimumXyz, maximumXyz } = validateMetadata(
    metadata,
    bytes,
  );
  const { sha256: cameraSha256 } = validateCameraData(cameraBytes);

  const filename = "omprakash-workcell.spz";

  const manifest = {
    schema: {
      name: "lerobot-state-atlas.environment-layer",
      major: 1,
      minor: 0,
    },
    environmentId: "omprakash-workcell-real-scan",
    label: "Omprakash workcell — uncalibrated real scan",
    status: "available",
    provenance: {
      sourceKind: "real-scan",
      description:
        "Real Gaussian Splat reconstruction from the Omprakash workcell capture. Browser asset is the reviewed deterministic Uniform 250k sample of viable cropped, opacity-filtered, and scale-filtered Gaussians from the 30k Splatfacto reconstruction, encoded as SPZ v3 SH0.",
      reconstructionClaim: false,
    },
    coordinateFrame: "canonical-shared-world",
    alignment: {
      translationXyz: [0, 0, 0],
      rotationXyzw: [0, 0, 0, 1],
      uniformScale: 1,
      calibrated: false,
      disclosure:
        "Real reconstruction rendered in its current reconstruction coordinates only. No validated similarity transform to canonical robot-world coordinates has been established; alignment with robot trajectories must not be inferred.",
    },
    bounds: {
      minimumXyz,
      maximumXyz,
    },
    asset: {
      filename,
      format: "spz",
      mimeType: "application/octet-stream",
      byteSize: bytes.byteLength,
      sha256,
      splatCount: metadata.selectedSplats,
    },
  };

  await mkdir(approvedRoot, { recursive: true });

  await writeFile(resolve(approvedRoot, filename), bytes);
  await writeFile(
    resolve(approvedRoot, PROMOTED_CAMERA_DATA.filename),
    cameraBytes,
  );
  await writeFile(
    resolve(approvedRoot, "manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );

  process.stdout.write(
    [
      `staged: ${resolve(approvedRoot, filename)}`,
      `manifest: ${resolve(approvedRoot, "manifest.json")}`,
      `cameras: ${resolve(approvedRoot, PROMOTED_CAMERA_DATA.filename)}`,
      `bytes: ${bytes.byteLength}`,
      `splats: ${metadata.selectedSplats}`,
      `sha256: ${sha256}`,
      `camera sha256: ${cameraSha256}`,
    ].join("\n") + "\n",
  );
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [, , spzPath, metadataPath, cameraPath] = process.argv;
  await stageRealEnvironment(spzPath, metadataPath, cameraPath);
}
