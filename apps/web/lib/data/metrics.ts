import type { PreparedVoxelArm } from "@/lib/atlas-schema/types";

export type CoverageMetric = "visits" | "log-visits" | "episodes";

export const metricLabels: Record<CoverageMetric, string> = {
  visits: "Visits",
  "log-visits": "Log visits",
  episodes: "Distinct episodes",
};

export function metricValue(
  arm: PreparedVoxelArm,
  index: number,
  metric: CoverageMetric,
): number {
  if (metric === "visits") return arm.visits[index];
  if (metric === "log-visits") return Math.log1p(arm.visits[index]);
  return arm.episodeCounts[index];
}

export function metricDomain(
  arms: PreparedVoxelArm[],
  metric: CoverageMetric,
): [number, number] {
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const arm of arms) {
    for (let index = 0; index < arm.visits.length; index += 1) {
      const value = metricValue(arm, index, metric);
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  return [minimum, maximum];
}
