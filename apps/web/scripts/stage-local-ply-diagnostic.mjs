import {
  lstat,
  mkdir,
  open,
  readFile,
  readlink,
  rename,
  stat,
  symlink,
  writeFile,
} from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

import { parseRegisteredColmapCameras } from "./colmap-camera-model.mjs";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const defaultSource = resolve(
  scriptDirectory,
  "../../../.cache/gaussian-splat/omprakash-new/eval-export/splat/splat.ply",
);
const destination = resolve(
  scriptDirectory,
  "../public/environment-data/__local-real__/diagnostic-original.ply",
);
const defaultColmapModel = resolve(
  scriptDirectory,
  "../../../.cache/gaussian-splat/omprakash-new/colmap-4fps/gs-input-model0/sparse",
);
const defaultDataparserTransform = resolve(
  scriptDirectory,
  "../../../.cache/gaussian-splat/omprakash-new/github-30k/outputs-final/omprakash-new/splatfacto/full-30000/dataparser_transforms.json",
);
const cameraDestination = resolve(
  scriptDirectory,
  "../public/environment-data/__local-real__/registered-cameras.json",
);
const defaultEvalRenders = resolve(
  scriptDirectory,
  "../../../.cache/gaussian-splat/omprakash-new/eval-export/eval-renders",
);
const evalRenderDestination = resolve(
  scriptDirectory,
  "../public/environment-data/__local-real__/eval-renders",
);
const EVAL_INTERVAL = 8;

async function inspectPly(source) {
  const sourceStats = await stat(source).catch(() => null);
  if (!sourceStats?.isFile()) {
    throw new Error(`PLY diagnostic source is not a file: ${source}`);
  }

  const handle = await open(source, "r");
  try {
    const prefix = Buffer.alloc(64 * 1024);
    const { bytesRead } = await handle.read(prefix, 0, prefix.length, 0);
    const text = prefix.subarray(0, bytesRead).toString("ascii");
    const endHeader = text.indexOf("end_header\n");
    if (endHeader < 0) {
      throw new Error("PLY header does not end within the first 64 KiB.");
    }
    const header = text.slice(0, endHeader + "end_header\n".length);
    if (
      !header.startsWith("ply\n") ||
      !header.includes("format binary_little_endian 1.0\n")
    ) {
      throw new Error("Expected a binary little-endian PLY 1.0 source.");
    }
    const vertexMatch = header.match(/^element vertex (\d+)$/m);
    if (!vertexMatch) throw new Error("PLY vertex count is missing.");
    return Number(vertexMatch[1]);
  } finally {
    await handle.close();
  }
}

async function ensureDiagnosticSymlink(destinationPath, sourcePath, type) {
  const existing = await lstat(destinationPath).catch(() => null);
  if (existing) {
    if (!existing.isSymbolicLink()) {
      throw new Error(
        `Refusing to replace non-symlink diagnostic destination: ${destinationPath}`,
      );
    }
    const existingTarget = resolve(
      dirname(destinationPath),
      await readlink(destinationPath),
    );
    if (existingTarget !== sourcePath) {
      throw new Error(
        `Refusing to replace diagnostic symlink to a different source: ${existingTarget}`,
      );
    }
    return;
  }
  await symlink(relative(dirname(destinationPath), sourcePath), destinationPath, type);
}

async function calculateEvalRenderPsnr(path, expectedWidth, expectedHeight) {
  const { data, info } = await sharp(path)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.width !== expectedWidth * 2 || info.height !== expectedHeight) {
    throw new Error(
      `Unexpected Nerfstudio eval montage dimensions ${info.width}x${info.height}: ${path}`,
    );
  }
  let squaredError = 0;
  for (let y = 0; y < expectedHeight; y += 1) {
    for (let x = 0; x < expectedWidth; x += 1) {
      const leftOffset = (y * info.width + x) * info.channels;
      const rightOffset = (y * info.width + x + expectedWidth) * info.channels;
      for (let channel = 0; channel < 3; channel += 1) {
        const difference = data[leftOffset + channel] - data[rightOffset + channel];
        squaredError += difference * difference;
      }
    }
  }
  const meanSquaredError =
    squaredError / (expectedWidth * expectedHeight * 3);
  return 10 * Math.log10((255 * 255) / meanSquaredError);
}

async function stagePlyDiagnostic(
  sourceArgument,
  colmapModelArgument,
  dataparserTransformArgument,
  evalRendersArgument,
) {
  const source = resolve(sourceArgument ?? defaultSource);
  const colmapModel = resolve(colmapModelArgument ?? defaultColmapModel);
  const dataparserTransformPath = resolve(
    dataparserTransformArgument ?? defaultDataparserTransform,
  );
  const evalRenders = resolve(evalRendersArgument ?? defaultEvalRenders);
  const splatCount = await inspectPly(source);
  const cameras = await parseRegisteredColmapCameras({
    camerasPath: resolve(colmapModel, "cameras.bin"),
    imagesPath: resolve(colmapModel, "images.bin"),
    evalInterval: EVAL_INTERVAL,
  });
  const dataparser = JSON.parse(
    await readFile(dataparserTransformPath, "utf8"),
  );
  if (cameras.length !== 210) {
    throw new Error(`Expected 210 registered cameras, received ${cameras.length}.`);
  }
  if (!Array.isArray(dataparser.transform) || !Number.isFinite(dataparser.scale)) {
    throw new Error("Invalid Nerfstudio dataparser transform metadata.");
  }
  await mkdir(dirname(destination), { recursive: true });
  await ensureDiagnosticSymlink(destination, source, "file");
  await ensureDiagnosticSymlink(evalRenderDestination, evalRenders, "dir");

  for (const camera of cameras) {
    camera.evalPsnr =
      camera.evalIndex === null
        ? null
        : await calculateEvalRenderPsnr(
            resolve(
              evalRenders,
              `eval_img_${String(camera.evalIndex).padStart(4, "0")}.png`,
            ),
            camera.width,
            camera.height,
          );
  }

  const cameraDataset = {
    schemaVersion: 1,
    source: {
      colmapModel: "gs-input-model0/sparse",
      dataparserTransform: "full-30000/dataparser_transforms.json",
      evalInterval: EVAL_INTERVAL,
      transform: dataparser.transform,
      scale: dataparser.scale,
    },
    cameras,
  };
  const cameraTemporary = `${cameraDestination}.tmp`;
  await writeFile(cameraTemporary, `${JSON.stringify(cameraDataset, null, 2)}\n`);
  await rename(cameraTemporary, cameraDestination);

  process.stdout.write(
    [
      `source: ${source}`,
      `local URL: /environment-data/__local-real__/diagnostic-original.ply`,
      `splats: ${splatCount}`,
      `registered cameras: ${cameras.length}`,
      `camera data: /environment-data/__local-real__/registered-cameras.json`,
      "isolated viewer: /diagnostics/spark-ply",
      "integrated viewer: /viewer/demo?sparkEnvironmentSource=ply",
    ].join("\n") + "\n",
  );
}

await stagePlyDiagnostic(
  process.argv[2],
  process.argv[3],
  process.argv[4],
  process.argv[5],
);
