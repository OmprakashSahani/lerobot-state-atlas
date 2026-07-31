import { cp, mkdir, rename, rm, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const defaultWebRoot = resolve(scriptDirectory, "..");

export async function stageLocalBenchmark(sourceArgument, webRoot = defaultWebRoot) {
  if (!sourceArgument) {
    throw new Error("Usage: npm run benchmark:stage -- SOURCE_BUNDLE_DIRECTORY");
  }
  const source = resolve(sourceArgument);
  const sourceStats = await stat(source).catch(() => null);
  if (!sourceStats?.isDirectory()) {
    throw new Error(`Benchmark source is not a directory: ${source}`);
  }
  for (const filename of ["manifest.json", "coverage.json"]) {
    const requiredFile = await stat(resolve(source, filename)).catch(() => null);
    if (!requiredFile?.isFile()) {
      throw new Error(`Benchmark source is missing required ${filename}.`);
    }
  }

  const destination = resolve(
    webRoot,
    "public",
    "atlas-data",
    "__local-benchmark__",
  );
  const temporary = `${destination}.staging-${process.pid}`;
  const backup = `${destination}.previous-${process.pid}`;
  await mkdir(dirname(destination), { recursive: true });
  await rm(temporary, { recursive: true, force: true });
  await rm(backup, { recursive: true, force: true });
  try {
    await cp(source, temporary, { recursive: true });
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }

  let movedPrevious = false;
  try {
    try {
      await rename(destination, backup);
      movedPrevious = true;
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
    await rename(temporary, destination);
    await rm(backup, { recursive: true, force: true });
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    if (movedPrevious) await rename(backup, destination);
    throw error;
  }

  return {
    source,
    destination,
    environmentValue: "/atlas-data/__local-benchmark__",
  };
}

async function main() {
  const result = await stageLocalBenchmark(process.argv[2]);
  console.log(`Staged benchmark bundle from: ${result.source}`);
  console.log(`Staged benchmark bundle to:   ${result.destination}`);
  console.log(
    `Set NEXT_PUBLIC_ATLAS_BUNDLE_BASE=${result.environmentValue}`,
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
