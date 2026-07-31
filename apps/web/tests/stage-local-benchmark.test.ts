import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

// @ts-expect-error The dependency-free development script intentionally has no TS build.
import { stageLocalBenchmark } from "@/scripts/stage-local-benchmark.mjs";

const temporaryRoots: string[] = [];

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "atlas-benchmark-test-"));
  temporaryRoots.push(root);
  return root;
}

async function sourceBundle(root: string, marker: string): Promise<string> {
  const source = join(root, `source-${marker}`);
  await mkdir(source, { recursive: true });
  await writeFile(join(source, "manifest.json"), `manifest-${marker}`);
  await writeFile(join(source, "coverage.json"), `coverage-${marker}`);
  await mkdir(join(source, "nested"));
  await writeFile(join(source, "nested", "payload.txt"), marker);
  return source;
}

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) =>
      rm(root, { recursive: true, force: true }),
    ),
  );
});

describe("local benchmark staging", () => {
  it("copies a complete bundle into the fixed local-only destination", async () => {
    const root = await temporaryRoot();
    const webRoot = join(root, "web");
    const source = await sourceBundle(root, "first");

    const result = await stageLocalBenchmark(source, webRoot);

    expect(result.environmentValue).toBe("/atlas-data/__local-benchmark__");
    expect(await readFile(join(result.destination, "manifest.json"), "utf8"))
      .toBe("manifest-first");
    expect(await readFile(join(result.destination, "nested", "payload.txt"), "utf8"))
      .toBe("first");
  });

  it("replaces only the previous staged directory", async () => {
    const root = await temporaryRoot();
    const webRoot = join(root, "web");
    const atlasRoot = join(webRoot, "public", "atlas-data");
    for (const demo of ["demo-v1", "demo-v2"]) {
      await mkdir(join(atlasRoot, demo), { recursive: true });
      await writeFile(join(atlasRoot, demo, "sentinel.txt"), demo);
    }
    const first = await sourceBundle(root, "first");
    const second = await sourceBundle(root, "second");
    const result = await stageLocalBenchmark(first, webRoot);
    await writeFile(join(result.destination, "stale.txt"), "stale");

    await stageLocalBenchmark(second, webRoot);

    await expect(access(join(result.destination, "stale.txt"))).rejects.toThrow();
    expect(await readFile(join(result.destination, "manifest.json"), "utf8"))
      .toBe("manifest-second");
    expect(await readFile(join(atlasRoot, "demo-v1", "sentinel.txt"), "utf8"))
      .toBe("demo-v1");
    expect(await readFile(join(atlasRoot, "demo-v2", "sentinel.txt"), "utf8"))
      .toBe("demo-v2");
  });

  it("rejects missing, non-directory, and incomplete sources", async () => {
    const root = await temporaryRoot();
    const webRoot = join(root, "web");
    const file = join(root, "file.json");
    await writeFile(file, "{}");
    const incomplete = join(root, "incomplete");
    await mkdir(incomplete);
    await writeFile(join(incomplete, "manifest.json"), "{}");

    await expect(stageLocalBenchmark(undefined, webRoot)).rejects.toThrow(/Usage/);
    await expect(stageLocalBenchmark(join(root, "missing"), webRoot)).rejects.toThrow(
      /not a directory/,
    );
    await expect(stageLocalBenchmark(file, webRoot)).rejects.toThrow(
      /not a directory/,
    );
    await expect(stageLocalBenchmark(incomplete, webRoot)).rejects.toThrow(
      /missing required coverage.json/,
    );
  });
});
