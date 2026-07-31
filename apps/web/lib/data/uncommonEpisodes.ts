import type { CoverageArm, CoveragePayload } from "@/lib/atlas-schema/types";

export interface CoverageEntryReference {
  arm: "left" | "right";
  voxelEntryIndex: number;
}

export interface UncommonEpisodeScore {
  episodeId: number;
  score: number;
  touchedEntryCount: number;
  scopeEntryShare: number;
  minimumDistinctEpisodeCount: number;
  maximumDistinctEpisodeCount: number;
}

export interface ScoreUncommonEpisodesOptions {
  coverage: CoveragePayload;
  episodeCount: number;
  allowedEpisodeIds: readonly number[];
  scope?: readonly CoverageEntryReference[];
}

interface ValidatedEntry {
  reference: CoverageEntryReference;
  episodeIds: readonly number[];
}

interface EpisodeAccumulator {
  raritySum: number;
  touchedEntryCount: number;
  minimumDistinctEpisodeCount: number;
  maximumDistinctEpisodeCount: number;
}

function validateEpisodeCount(episodeCount: number): void {
  if (!Number.isInteger(episodeCount) || episodeCount <= 0) {
    throw new Error("Coverage episode count must be a positive integer.");
  }
}

function allowedEpisodeUniverse(episodeIds: readonly number[]): Set<number> {
  const result = new Set<number>();
  for (const episodeId of episodeIds) {
    if (!Number.isInteger(episodeId) || episodeId < 0) {
      throw new Error("Allowed episode IDs must be non-negative integers.");
    }
    if (result.has(episodeId)) {
      throw new Error(`Allowed episode ID ${episodeId} is duplicated.`);
    }
    result.add(episodeId);
  }
  return result;
}

function validateArmEntries(
  arm: CoverageArm,
  allowedEpisodeIds: ReadonlySet<number>,
  episodeCount: number,
): ValidatedEntry[] {
  const entryCount = arm.voxelIndices.length;
  const offsets = arm.episodeIdOffsets;
  if (offsets.length !== entryCount + 1) {
    throw new Error(
      `${arm.arm} coverage CSR offsets must contain one offset per entry plus one.`,
    );
  }
  if (offsets[0] !== 0) {
    throw new Error(`${arm.arm} coverage CSR offsets must start at zero.`);
  }

  for (let index = 0; index < offsets.length; index += 1) {
    if (!Number.isInteger(offsets[index]) || offsets[index] < 0) {
      throw new Error(
        `${arm.arm} coverage CSR offset ${index} must be a non-negative integer.`,
      );
    }
    if (index > 0 && offsets[index] < offsets[index - 1]) {
      throw new Error(`${arm.arm} coverage CSR offsets must be monotonic.`);
    }
  }
  if (offsets.at(-1) !== arm.episodeIds.length) {
    throw new Error(
      `${arm.arm} coverage CSR offsets must end at the episode-ID length.`,
    );
  }

  return arm.voxelIndices.map((_, voxelEntryIndex) => {
    const start = offsets[voxelEntryIndex];
    const end = offsets[voxelEntryIndex + 1];
    const distinctEpisodeCount = end - start;
    if (distinctEpisodeCount < 1) {
      throw new Error(
        `${arm.arm} coverage entry ${voxelEntryIndex} must contain at least one episode ID.`,
      );
    }
    if (distinctEpisodeCount > episodeCount) {
      throw new Error(
        `${arm.arm} coverage entry ${voxelEntryIndex} distinct episode count ${distinctEpisodeCount} exceeds coverage episode count ${episodeCount}.`,
      );
    }
    if (
      arm.episodeCounts.length !== entryCount ||
      arm.episodeCounts[voxelEntryIndex] !== distinctEpisodeCount
    ) {
      throw new Error(
        `${arm.arm} coverage entry ${voxelEntryIndex} episode count disagrees with its CSR range.`,
      );
    }

    const episodeIds = arm.episodeIds.slice(start, end);
    const seen = new Set<number>();
    for (const episodeId of episodeIds) {
      if (!Number.isInteger(episodeId)) {
        throw new Error(
          `${arm.arm} coverage entry ${voxelEntryIndex} episode IDs must be integers.`,
        );
      }
      if (!allowedEpisodeIds.has(episodeId)) {
        throw new Error(
          `${arm.arm} coverage entry ${voxelEntryIndex} contains episode ID ${episodeId} outside the allowed episode universe.`,
        );
      }
      if (seen.has(episodeId)) {
        throw new Error(
          `${arm.arm} coverage entry ${voxelEntryIndex} contains duplicate episode ID ${episodeId}.`,
        );
      }
      seen.add(episodeId);
    }

    return {
      reference: { arm: arm.arm, voxelEntryIndex },
      episodeIds,
    };
  });
}

function selectScope(
  entriesByArm: ReadonlyMap<"left" | "right", readonly ValidatedEntry[]>,
  scope: readonly CoverageEntryReference[] | undefined,
): ValidatedEntry[] {
  if (scope === undefined) {
    return [
      ...(entriesByArm.get("left") ?? []),
      ...(entriesByArm.get("right") ?? []),
    ];
  }

  const selected: ValidatedEntry[] = [];
  const seen = new Set<string>();
  for (const reference of scope) {
    if (reference.arm !== "left" && reference.arm !== "right") {
      throw new Error(`Unknown coverage arm: ${String(reference.arm)}.`);
    }
    if (!Number.isInteger(reference.voxelEntryIndex)) {
      throw new Error("Scoped voxel entry indices must be integers.");
    }
    const key = `${reference.arm}:${reference.voxelEntryIndex}`;
    if (seen.has(key)) {
      throw new Error(`Duplicate scoped coverage entry: ${key}.`);
    }
    seen.add(key);
    const entry = entriesByArm.get(reference.arm)?.[reference.voxelEntryIndex];
    if (entry === undefined) {
      throw new Error(`Scoped coverage entry is out of range: ${key}.`);
    }
    selected.push(entry);
  }
  return selected;
}

export function scoreUncommonEpisodes({
  coverage,
  episodeCount,
  allowedEpisodeIds,
  scope,
}: ScoreUncommonEpisodesOptions): UncommonEpisodeScore[] {
  validateEpisodeCount(episodeCount);
  const allowed = allowedEpisodeUniverse(allowedEpisodeIds);
  const entriesByArm = new Map<"left" | "right", readonly ValidatedEntry[]>();
  for (const arm of coverage.arms) {
    if (entriesByArm.has(arm.arm)) {
      throw new Error(`Coverage contains duplicate ${arm.arm} arm data.`);
    }
    entriesByArm.set(
      arm.arm,
      validateArmEntries(arm, allowed, episodeCount),
    );
  }
  const entries = selectScope(entriesByArm, scope);
  if (entries.length === 0) return [];

  const accumulators = new Map<number, EpisodeAccumulator>();
  for (const entry of entries) {
    const distinctEpisodeCount = entry.episodeIds.length;
    const rarity =
      episodeCount <= 1
        ? 0
        : Math.log(episodeCount / distinctEpisodeCount) /
          Math.log(episodeCount);
    for (const episodeId of entry.episodeIds) {
      const accumulator = accumulators.get(episodeId) ?? {
        raritySum: 0,
        touchedEntryCount: 0,
        minimumDistinctEpisodeCount: distinctEpisodeCount,
        maximumDistinctEpisodeCount: distinctEpisodeCount,
      };
      accumulator.raritySum += rarity;
      accumulator.touchedEntryCount += 1;
      accumulator.minimumDistinctEpisodeCount = Math.min(
        accumulator.minimumDistinctEpisodeCount,
        distinctEpisodeCount,
      );
      accumulator.maximumDistinctEpisodeCount = Math.max(
        accumulator.maximumDistinctEpisodeCount,
        distinctEpisodeCount,
      );
      accumulators.set(episodeId, accumulator);
    }
  }

  return Array.from(accumulators, ([episodeId, accumulator]) => ({
    episodeId,
    score: accumulator.raritySum / accumulator.touchedEntryCount,
    touchedEntryCount: accumulator.touchedEntryCount,
    scopeEntryShare: accumulator.touchedEntryCount / entries.length,
    minimumDistinctEpisodeCount: accumulator.minimumDistinctEpisodeCount,
    maximumDistinctEpisodeCount: accumulator.maximumDistinctEpisodeCount,
  })).sort(
    (left, right) =>
      right.score - left.score ||
      right.touchedEntryCount - left.touchedEntryCount ||
      left.episodeId - right.episodeId,
  );
}
