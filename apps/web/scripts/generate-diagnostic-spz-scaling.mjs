import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildDiagnosticCandidates,
  distributionQuantiles,
  encodeDiagnosticSpz,
  parseDiagnosticPlyHeader,
  selectUniformCandidateOffsets,
} from "./generate-diagnostic-spz-reductions.mjs";

const EXISTING_ASSET_LIMIT_BYTES = 8 * 1024 * 1024;
const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const DEFAULT_PLY = resolve(
  scriptDirectory,
  "../../../.cache/gaussian-splat/omprakash-new/eval-export/splat/splat.ply",
);
const DEFAULT_OUTPUT_DIRECTORY = resolve(
  scriptDirectory,
  "../public/environment-data/__local-real__",
);

export const DIAGNOSTIC_SCALING_VARIANTS = [
  {
    key: "uniform150k",
    label: "Uniform 150k SPZ",
    fileName: "diagnostic-uniform-150k.spz",
    count: 150_000,
    population: "viable",
    selectionMethod:
      "Deterministic 32-bit hash sample of the viable cropped/filtered candidate distribution.",
  },
  {
    key: "uniform200k",
    label: "Uniform 200k SPZ",
    fileName: "diagnostic-uniform-200k.spz",
    count: 200_000,
    population: "viable",
    selectionMethod:
      "Deterministic 32-bit hash sample of the viable cropped/filtered candidate distribution.",
  },
  {
    key: "uniform250k",
    label: "Uniform 250k SPZ",
    fileName: "diagnostic-uniform-250k.spz",
    count: 250_000,
    population: "viable",
    selectionMethod:
      "Deterministic 32-bit hash sample of the viable cropped/filtered candidate distribution.",
  },
  {
    key: "uniform300k",
    label: "Uniform 300k SPZ",
    fileName: "diagnostic-uniform-300k.spz",
    count: 300_000,
    population: "viable",
    selectionMethod:
      "Deterministic 32-bit hash sample of the viable cropped/filtered candidate distribution.",
  },
  {
    key: "candidateFull",
    label: "Candidate-full SPZ",
    fileName: "diagnostic-candidate-full.spz",
    count: null,
    population: "viable",
    selectionMethod:
      "All viable candidates after the existing crop, minimum-opacity, and maximum-scale filters; no post-filter reduction.",
  },
  {
    key: "fullOriginal",
    label: "Full-original SPZ",
    fileName: "diagnostic-full-original.spz",
    count: null,
    population: "original",
    selectionMethod:
      "All original PLY Gaussians with no crop, opacity, scale, voxel, or sampling reduction.",
  },
];

const PROTECTED_ASSETS = new Set([
  "omprakash-workcell.spz",
  "diagnostic-uniform-100k.spz",
  "diagnostic-camera-aware-100k.spz",
  "diagnostic-hybrid-100k.spz",
]);

export function assertDiagnosticScalingOutputName(fileName) {
  if (PROTECTED_ASSETS.has(fileName)) {
    throw new Error(`Refusing to overwrite protected diagnostic asset ${fileName}.`);
  }
  if (!/^diagnostic-[a-z0-9-]+\.spz$/.test(fileName)) {
    throw new Error(`Invalid diagnostic scaling asset name ${fileName}.`);
  }
  return fileName;
}

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function buildOriginalPopulation(vertexCount, value) {
  const sourceIndices = new Uint32Array(vertexCount);
  const opacities = new Float32Array(vertexCount);
  const maxScales = new Float32Array(vertexCount);
  for (let index = 0; index < vertexCount; index += 1) {
    sourceIndices[index] = index;
    opacities[index] = sigmoid(value(index, "opacity"));
    maxScales[index] = Math.max(
      Math.exp(value(index, "scale_0")),
      Math.exp(value(index, "scale_1")),
      Math.exp(value(index, "scale_2")),
    );
  }
  return {
    count: vertexCount,
    sourceIndices,
    opacities,
    maxScales,
    value,
  };
}

function selectedValues(values, selectedOffsets) {
  return Float64Array.from(selectedOffsets, (index) => values[index]);
}

function summarizeDistribution(population, selectedOffsets) {
  return {
    opacity: distributionQuantiles(
      selectedValues(population.opacities, selectedOffsets),
    ),
    maxScale: distributionQuantiles(
      selectedValues(population.maxScales, selectedOffsets),
    ),
  };
}

function selectionForVariant(variant, viable, original) {
  const population = variant.population === "original" ? original : viable;
  const count = variant.count ?? population.count;
  if (count > population.count) {
    throw new Error(
      `${variant.label} requests ${count} splats from ${population.count} available.`,
    );
  }
  const selectedOffsets =
    count === population.count
      ? Array.from(population.sourceIndices, (_, index) => index)
      : selectUniformCandidateOffsets(population.sourceIndices, count);
  return { population, selectedOffsets };
}

export async function generateDiagnosticScalingAssets({
  inputPath = DEFAULT_PLY,
  outputDirectory = DEFAULT_OUTPUT_DIRECTORY,
  variantKeys,
  overwriteExisting = true,
} = {}) {
  const input = await readFile(resolve(inputPath));
  const ply = parseDiagnosticPlyHeader(input);
  const viable = buildDiagnosticCandidates(input, ply);
  const original = buildOriginalPopulation(ply.vertexCount, viable.value);
  const summaries = [];
  const selectedVariants = variantKeys
    ? DIAGNOSTIC_SCALING_VARIANTS.filter((variant) =>
        variantKeys.includes(variant.key),
      )
    : DIAGNOSTIC_SCALING_VARIANTS;
  if (
    selectedVariants.length === 0 ||
    (variantKeys && selectedVariants.length !== new Set(variantKeys).size)
  ) {
    throw new Error("Unknown or empty diagnostic scaling variant selection.");
  }
  await mkdir(resolve(outputDirectory), { recursive: true });

  for (const variant of selectedVariants) {
    assertDiagnosticScalingOutputName(variant.fileName);
    const { population, selectedOffsets } = selectionForVariant(
      variant,
      viable,
      original,
    );
    const bytes = encodeDiagnosticSpz(
      selectedOffsets,
      population,
      original.count,
    );
    const rawBytes = 16 + selectedOffsets.length * 20;
    const metadata = {
      schemaVersion: 1,
      developmentOnly: true,
      variant: variant.key,
      label: variant.label,
      asset: variant.fileName,
      input: resolve(inputPath),
      sourcePopulation: variant.population,
      inputSplats: original.count,
      viableCandidateSplats: viable.count,
      selectedSplats: selectedOffsets.length,
      selectionMethod: variant.selectionMethod,
      rawBytes,
      compressedBytes: bytes.byteLength,
      compressedMiB: bytes.byteLength / (1024 * 1024),
      existingAssetSizeLimitBytes: EXISTING_ASSET_LIMIT_BYTES,
      underExistingAssetSizeLimit:
        bytes.byteLength <= EXISTING_ASSET_LIMIT_BYTES,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      spz: {
        version: 3,
        shDegree: 0,
        fractionalBits: 12,
        flags: 0,
      },
      distributions: summarizeDistribution(population, selectedOffsets),
      viabilityFilter:
        variant.population === "viable"
          ? {
              cropQuantiles: [0.025, 0.975],
              cropBounds: {
                minimumXyz: viable.boundsMin,
                maximumXyz: viable.boundsMax,
              },
              minimumOpacity: 0.05,
              maximumScale: 0.05,
              rejected: viable.rejected,
            }
          : null,
    };
    const outputPath = resolve(outputDirectory, variant.fileName);
    if (!overwriteExisting) {
      const existing = await readFile(outputPath).catch(() => null);
      if (existing) {
        throw new Error(`Refusing to overwrite existing asset ${outputPath}.`);
      }
    }
    await Promise.all([
      writeFile(outputPath, bytes),
      writeFile(`${outputPath}.json`, `${JSON.stringify(metadata, null, 2)}\n`),
    ]);
    summaries.push(metadata);
    process.stdout.write(
      `${variant.label}: ${metadata.selectedSplats.toLocaleString()} splats, ${metadata.compressedBytes.toLocaleString()} bytes (${metadata.compressedMiB.toFixed(3)} MiB), ${metadata.sha256}\n`,
    );
  }

  let aggregateSummaries = summaries;
  if (selectedVariants.length !== DIAGNOSTIC_SCALING_VARIANTS.length) {
    const existingAggregate = await readFile(
      resolve(outputDirectory, "diagnostic-scaling.json"),
      "utf8",
    )
      .then((contents) => JSON.parse(contents).variants ?? [])
      .catch(() => []);
    const byVariant = new Map(
      [...existingAggregate, ...summaries].map((summary) => [
        summary.variant,
        summary,
      ]),
    );
    aggregateSummaries = DIAGNOSTIC_SCALING_VARIANTS.flatMap((variant) => {
      const summary = byVariant.get(variant.key);
      return summary ? [summary] : [];
    });
  }
  await writeFile(
    resolve(outputDirectory, "diagnostic-scaling.json"),
    `${JSON.stringify({ schemaVersion: 1, developmentOnly: true, variants: aggregateSummaries }, null, 2)}\n`,
  );
  return summaries;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const arguments_ = process.argv.slice(2);
  const onlyIndex = arguments_.indexOf("--only");
  const variantKeys =
    onlyIndex < 0 ? undefined : arguments_[onlyIndex + 1]?.split(",");
  if (onlyIndex >= 0) arguments_.splice(onlyIndex, 2);
  if (arguments_.length > 2 || (onlyIndex >= 0 && !variantKeys?.length)) {
    throw new Error(
      "Usage: node scripts/generate-diagnostic-spz-scaling.mjs [input.ply] [output-directory] [--only variant-key]",
    );
  }
  await generateDiagnosticScalingAssets({
    inputPath: arguments_[0] ? resolve(arguments_[0]) : DEFAULT_PLY,
    outputDirectory: arguments_[1]
      ? resolve(arguments_[1])
      : DEFAULT_OUTPUT_DIRECTORY,
    variantKeys,
    overwriteExisting: variantKeys === undefined,
  });
}
