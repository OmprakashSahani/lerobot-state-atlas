"use client";

import { useMemo, useState } from "react";

import type {
  CoveragePayload,
  TrajectoryPayload,
} from "@/lib/atlas-schema/types";
import type {
  RadiusQueryResult,
  VoxelSelection,
} from "@/lib/data/radiusQuery";
import { scoreUncommonEpisodes } from "@/lib/data/uncommonEpisodes";

export type TrajectoryState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: TrajectoryPayload };

export interface EpisodeAnalysisPanelProps {
  coverage: CoveragePayload;
  episodeCount: number;
  episodeIds: readonly number[];
  radiusResult: RadiusQueryResult | null;
  selection: VoxelSelection | null;
  trajectories: TrajectoryState;
  onCheckPlayback: () => void;
  onOpenPlayback: (episodeId: number) => void;
}

export function EpisodeAnalysisPanel({
  coverage,
  episodeCount,
  episodeIds,
  radiusResult,
  selection,
  trajectories,
  onCheckPlayback,
  onOpenPlayback,
}: EpisodeAnalysisPanelProps) {
  const [radiusScopeSelection, setRadiusScopeSelection] =
    useState<VoxelSelection | null>(null);
  const usesRadiusScope =
    selection !== null && radiusScopeSelection === selection;

  const radiusMatches = radiusResult?.matches ?? [];
  const radiusScopeKey = radiusMatches
    .map((match) => `${match.arm}:${match.voxelEntryIndex}`)
    .join("|");
  const scores = useMemo(
    () =>
      scoreUncommonEpisodes({
        coverage,
        episodeCount,
        allowedEpisodeIds: episodeIds,
        scope: usesRadiusScope ? radiusMatches : undefined,
      }),
    // Entry identities fully determine local scores; geometry-only query changes
    // with the same matches must not repeat the CSR traversal.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [coverage, episodeCount, episodeIds, radiusScopeKey, usesRadiusScope],
  );
  const scopeLabel = usesRadiusScope ? "selected radius" : "entire coverage";
  const availableEpisodeIds = useMemo(
    () =>
      trajectories.status === "ready"
        ? new Set(trajectories.data.episodes.map((episode) => episode.episodeId))
        : null,
    [trajectories],
  );

  return (
    <aside
      aria-labelledby="episode-analysis-heading"
      className="episode-analysis-panel"
    >
      <header className="episode-analysis-header">
        <p className="eyebrow">Coverage evidence</p>
        <h2 id="episode-analysis-heading">Episode analysis</h2>
      </header>
      <section
        aria-labelledby="uncommon-episodes-heading"
        className="uncommon-episodes"
      >
        <div className="episode-analysis-context">
          <div className="section-title-row">
            <h3 id="uncommon-episodes-heading">Uncommon-space episodes</h3>
            <span>{usesRadiusScope ? "Local" : "Global"}</span>
          </div>
          <p className="control-help uncommon-disclosure">
            Scores use distinct-episode frequency across arm-specific voxel
            entries. Higher scores mean an episode reached entries shared with
            fewer exported coverage episodes. Raw visit counts are a separate
            metric.
          </p>
          <p className="control-help uncommon-disclosure">
            Scores describe only this exported coverage set, not the full
            dataset or physical workspace generally. A score is not a
            probability, percentile, task-quality judgment, or anomaly label.
          </p>
          <label className="field-label" htmlFor="uncommon-episode-scope">
            Episode scoring scope
          </label>
          <select
            id="uncommon-episode-scope"
            value={usesRadiusScope ? "radius" : "coverage"}
            onChange={(event) =>
              setRadiusScopeSelection(
                event.target.value === "radius" ? selection : null,
              )
            }
          >
            <option value="coverage">Entire coverage</option>
            <option value="radius" disabled={selection === null}>
              Selected radius
            </option>
          </select>
          {selection === null ? (
            <small className="control-help">
              Select an occupied voxel to score episodes within a radius.
            </small>
          ) : null}
          {usesRadiusScope && radiusResult ? (
            <p className="episode-analysis-radius-context">
              Radius {radiusResult.radius.toFixed(3)} m ·{" "}
              {radiusResult.entryCount.toLocaleString()} arm-specific{" "}
              {radiusResult.entryCount === 1 ? "entry" : "entries"}
            </p>
          ) : null}
          {episodeCount === 1 ? (
            <p className="uncommon-special-state" role="note">
              Relative uncommonness is unavailable with one coverage episode;
              scores are defined as zero.
            </p>
          ) : null}
          <p
            className="uncommon-result-summary"
            aria-atomic="true"
            aria-live="polite"
          >
            {scores.length} {scores.length === 1 ? "episode" : "episodes"}{" "}
            ranked for {scopeLabel}.
          </p>
          {trajectories.status === "idle" ? (
            <button
              className="compact-button uncommon-playback-check"
              type="button"
              onClick={onCheckPlayback}
            >
              Check playback availability
            </button>
          ) : null}
          {trajectories.status === "loading" ? (
            <p className="uncommon-playback-state" aria-busy="true">
              Checking playback availability…
            </p>
          ) : null}
          {trajectories.status === "error" ? (
            <button
              className="compact-button uncommon-playback-check"
              type="button"
              onClick={onCheckPlayback}
            >
              Retry playback availability
            </button>
          ) : null}
        </div>
        <div className="episode-analysis-results">
          {scores.length === 0 && usesRadiusScope ? (
            <p className="uncommon-special-state">
              No episode evidence exists in the current radius.
            </p>
          ) : (
            <ol
              className="uncommon-episode-list"
              aria-label={`Uncommon-space episode ranking for ${scopeLabel}`}
            >
              {scores.map((result) => (
                <li key={result.episodeId}>
                  <div className="uncommon-episode-heading">
                    <strong>Episode {result.episodeId}</strong>
                    <span>
                      Uncommonness {(result.score * 100).toFixed(1)} / 100
                    </span>
                  </div>
                  <dl>
                    <div>
                      <dt>Arm-specific entries touched</dt>
                      <dd>{result.touchedEntryCount.toLocaleString()}</dd>
                    </div>
                    <div>
                      <dt>Scoped entries touched</dt>
                      <dd>{(result.scopeEntryShare * 100).toFixed(1)}%</dd>
                    </div>
                    <div>
                      <dt>Distinct episodes represented</dt>
                      <dd>
                        {result.minimumDistinctEpisodeCount}–
                        {result.maximumDistinctEpisodeCount} per entry
                      </dd>
                    </div>
                  </dl>
                  <div className="uncommon-playback-availability">
                    {trajectories.status === "idle" ? (
                      <p>Playback availability not loaded.</p>
                    ) : null}
                    {trajectories.status === "loading" ? (
                      <p>Playback availability loading.</p>
                    ) : null}
                    {trajectories.status === "error" ? (
                      <p>Playback availability could not be loaded.</p>
                    ) : null}
                    {availableEpisodeIds?.has(result.episodeId) ? (
                      <>
                        <p id={`episode-${result.episodeId}-playback-status`}>
                          Playback available
                        </p>
                        <button
                          aria-describedby={`episode-${result.episodeId}-playback-status`}
                          className="compact-button"
                          type="button"
                          onClick={() => onOpenPlayback(result.episodeId)}
                        >
                          Open Episode {result.episodeId} playback
                        </button>
                      </>
                    ) : null}
                    {availableEpisodeIds !== null &&
                    !availableEpisodeIds.has(result.episodeId) ? (
                      <p>Coverage evidence only — trajectory not exported.</p>
                    ) : null}
                  </div>
                </li>
              ))}
            </ol>
          )}
        </div>
      </section>
    </aside>
  );
}
