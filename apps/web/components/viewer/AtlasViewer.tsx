"use client";

import { useEffect, useRef, useState } from "react";

import type { TrajectoryPayload } from "@/lib/atlas-schema/types";
import { metricDomain, metricLabels, type CoverageMetric } from "@/lib/data/metrics";
import { loadTrajectories } from "@/lib/data/loadBundle";
import { queryRadius } from "@/lib/data/radiusQuery";
import {
  advancePlayback,
  formatPlaybackStatus,
  type PlaybackState,
} from "@/lib/playback/controller";
import { ViewerCanvas } from "./ViewerCanvas";
import { useAtlasData } from "./AtlasDataProvider";
import { useViewerStore } from "./ViewerStore";

type TrajectoryState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: TrajectoryPayload };

const metricDescriptions: Record<CoverageMetric, string> = {
  visits: "Raw arm-specific tool-point visits",
  "log-visits": "log1p of raw arm-specific visits",
  episodes: "Exact episodes represented by voxel CSR",
};

function formatMetric(value: number, metric: CoverageMetric) {
  return metric === "log-visits" ? value.toFixed(2) : value.toLocaleString();
}

export function AtlasViewer() {
  const atlas = useAtlasData();
  const viewer = useViewerStore();
  const [trajectories, setTrajectories] = useState<TrajectoryState>({
    status: "idle",
  });
  const [episodeId, setEpisodeId] = useState<number | null>(null);
  const [playback, setPlayback] = useState<PlaybackState>({
    frame: 0,
    playing: false,
    speed: 1,
    loop: false,
  });
  const previousTime = useRef<number | null>(null);

  useEffect(() => {
    if (atlas.status === "ready" && viewer.spacing === 0.8) {
      viewer.setSpacing(atlas.data.manifest.coverage.armSpacing);
    }
    // Initialize once from the exported baseline; runtime changes are user-owned.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atlas.status]);

  const activatePlayback = () => {
    if (atlas.status !== "ready" || trajectories.status !== "idle") return;
    setTrajectories({ status: "loading" });
    loadTrajectories(atlas.data.manifest)
      .then((data) => {
        setTrajectories({ status: "ready", data });
        setEpisodeId(data.episodes[0].episodeId);
      })
      .catch((error: unknown) =>
        setTrajectories({
          status: "error",
          message: error instanceof Error ? error.message : "Trajectory playback failed to load.",
        }),
      );
  };

  const episode =
    trajectories.status === "ready"
      ? trajectories.data.episodes.find((item) => item.episodeId === episodeId) ?? null
      : null;

  useEffect(() => {
    if (!playback.playing || !episode || atlas.status !== "ready") {
      previousTime.current = null;
      return;
    }
    let animationFrame = 0;
    const tick = (now: number) => {
      const previous = previousTime.current ?? now;
      previousTime.current = now;
      setPlayback((state) =>
        advancePlayback(
          state,
          (now - previous) / 1000,
          atlas.data.manifest.dataset.fps,
          episode.frameIndices.length,
        ),
      );
      animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationFrame);
  }, [atlas, episode, playback.playing]);

  if (atlas.status === "loading") {
    return <div className="viewer-status" role="status">Loading pinned atlas data…</div>;
  }
  if (atlas.status === "error") {
    return (
      <section className="viewer-fallback" role="alert">
        <p className="eyebrow">Data load failed</p>
        <h1>The demo bundle could not be opened.</h1>
        <p>{atlas.message}</p>
        <button className="button button-secondary" onClick={() => window.location.reload()} type="button">Retry</button>
      </section>
    );
  }

  const { manifest, preparedArms, coverage } = atlas.data;
  const domain = metricDomain(preparedArms, viewer.metric);
  const radiusResult = viewer.selection
    ? queryRadius(
        preparedArms,
        coverage,
        viewer.selection,
        viewer.radius,
        viewer.spacing,
        manifest.coverage.armSpacing,
      )
    : null;
  const selectedCoverage = viewer.selection
    ? coverage.arms[viewer.selection.arm === "left" ? 0 : 1]
    : null;
  const selectedVisits =
    viewer.selection && selectedCoverage
      ? selectedCoverage.visitCounts[viewer.selection.voxelEntryIndex]
      : 0;
  const selectedEpisodes =
    viewer.selection && selectedCoverage
      ? selectedCoverage.episodeIdOffsets[viewer.selection.voxelEntryIndex + 1] -
        selectedCoverage.episodeIdOffsets[viewer.selection.voxelEntryIndex]
      : 0;
  const frameIndex = episode
    ? Math.min(episode.frameIndices.length - 1, Math.floor(playback.frame))
    : 0;

  return (
    <div className="viewer-shell">
      <section className="viewer-stage" aria-label="Interactive workspace scene">
        <ViewerCanvas data={atlas.data} episode={episode} playbackFrame={playback.frame} />
        <div className="scene-badge"><span className="live-dot" aria-hidden="true" />Canonical shared world</div>
        <p className="scene-help">Click a voxel to query · Drag to orbit · Scroll to zoom</p>
      </section>
      <aside className="viewer-panel" aria-label="Viewer controls and metadata">
        <div className="panel-heading">
          <div><p className="eyebrow">Demo / episodes 0–9</p><h1>Workspace coverage</h1></div>
          <span className="schema-chip">schema v1.0</span>
        </div>

        <section className="control-section" aria-labelledby="metric-heading">
          <div className="section-title-row"><h2 id="metric-heading">Coverage metric</h2><span>{metricLabels[viewer.metric]}</span></div>
          <label className="field-label" htmlFor="coverage-metric">Metric</label>
          <select
            id="coverage-metric"
            value={viewer.metric}
            onChange={(event) => viewer.setMetric(event.target.value as CoverageMetric)}
          >
            <option value="visits">Visits</option>
            <option value="log-visits">Log visits</option>
            <option value="episodes">Distinct episodes</option>
          </select>
          <small className="control-help">{metricDescriptions[viewer.metric]}</small>
          <div className="legend" aria-label={`${metricLabels[viewer.metric]} color range`}>
            <div className="legend-gradient" aria-hidden="true" />
            <div><span>{formatMetric(domain[0], viewer.metric)}</span><span>{formatMetric(domain[1], viewer.metric)}</span></div>
          </div>
        </section>

        <section className="control-section" aria-labelledby="scene-heading">
          <div className="section-title-row"><h2 id="scene-heading">Scene</h2><button className="compact-button" type="button" onClick={viewer.resetCamera}>Reset camera</button></div>
          <label className="layer-toggle"><input checked={viewer.leftVisible} onChange={() => viewer.toggleArm("left")} type="checkbox" /><span className="arm-dot arm-dot-left" aria-hidden="true" />Left arm entries<strong>{preparedArms[0].visits.length.toLocaleString()}</strong></label>
          <label className="layer-toggle"><input checked={viewer.rightVisible} onChange={() => viewer.toggleArm("right")} type="checkbox" /><span className="arm-dot arm-dot-right" aria-hidden="true" />Right arm entries<strong>{preparedArms[1].visits.length.toLocaleString()}</strong></label>
          <label className="layer-toggle simple-toggle"><input checked={viewer.autoRotate} onChange={(event) => viewer.setAutoRotate(event.target.checked)} type="checkbox" />Auto rotate</label>
          <label className="field-label" htmlFor="arm-spacing">Provisional arm spacing: {viewer.spacing.toFixed(2)} m</label>
          <input id="arm-spacing" type="range" min="0.2" max="1.4" step="0.02" value={viewer.spacing} onChange={(event) => viewer.setSpacing(Number(event.target.value))} />
          <button className="compact-button" type="button" onClick={() => viewer.setSpacing(manifest.coverage.armSpacing)}>Restore manifest spacing</button>
        </section>

        <section className="control-section" aria-labelledby="query-heading">
          <div className="section-title-row"><h2 id="query-heading">Radius query</h2>{viewer.selection ? <button className="compact-button" type="button" onClick={viewer.clearSelection}>Clear selection</button> : null}</div>
          <label className="field-label" htmlFor="query-radius">Query radius: {viewer.radius.toFixed(3)} m</label>
          <input id="query-radius" type="range" min="0" max="0.3" step="0.005" value={viewer.radius} onChange={(event) => viewer.setRadius(Number(event.target.value))} />
          {radiusResult ? (
            <div className="query-result" role="status" aria-live="polite">
              <strong>{radiusResult.selectedArm} arm voxel selected</strong>
              <span>Center: {radiusResult.center.map((value) => value.toFixed(3)).join(", ")} m</span>
              <span>Radius: {radiusResult.radius.toFixed(3)} m</span>
              <span>Arm-specific entries: {radiusResult.entryCount.toLocaleString()}</span>
              <span>Tool-point visits: {radiusResult.toolPointVisits.toLocaleString()}</span>
              <span>Left / right visits: {radiusResult.leftVisits.toLocaleString()} / {radiusResult.rightVisits.toLocaleString()}</span>
              <span>Exact episode union: {radiusResult.distinctEpisodeCount.toLocaleString()}</span>
              <span>Selected voxel: {selectedVisits.toLocaleString()} raw visits · {selectedEpisodes.toLocaleString()} exact episodes</span>
            </div>
          ) : <p className="control-help">Select an occupied voxel in the scene.</p>}
        </section>

        <section className="control-section" aria-labelledby="playback-heading">
          <div className="section-title-row"><h2 id="playback-heading">Trajectory playback</h2><span>Optional payload</span></div>
          {trajectories.status === "idle" ? <button className="compact-button" type="button" onClick={activatePlayback}>Load playback</button> : null}
          {trajectories.status === "loading" ? <p role="status">Loading trajectories…</p> : null}
          {trajectories.status === "error" ? <p role="alert">{trajectories.message}</p> : null}
          {trajectories.status === "ready" && episode ? (
            <div className="playback-controls">
              <label className="field-label" htmlFor="episode-selector">Episode</label>
              <select id="episode-selector" value={episode.episodeId} onChange={(event) => { setEpisodeId(Number(event.target.value)); setPlayback((state) => ({ ...state, frame: 0, playing: false })); }}>
                {trajectories.data.episodes.map((item) => <option key={item.episodeId} value={item.episodeId}>Episode {item.episodeId}</option>)}
              </select>
              <div className="button-row">
                <button className="compact-button" type="button" onClick={() => setPlayback((state) => ({ ...state, playing: !state.playing }))}>{playback.playing ? "Pause" : "Play"}</button>
                <button className="compact-button" type="button" onClick={() => setPlayback((state) => ({ ...state, frame: 0, playing: false }))}>Restart</button>
              </div>
              <label className="field-label" htmlFor="playback-timeline">Timeline</label>
              <input id="playback-timeline" type="range" min="0" max={episode.frameIndices.length - 1} step="1" value={frameIndex} onChange={(event) => setPlayback((state) => ({ ...state, frame: Number(event.target.value), playing: false }))} />
              <span className="playback-status">
                {formatPlaybackStatus(episode, playback.frame)}
              </span>
              <label className="field-label" htmlFor="playback-speed">Playback speed</label>
              <select id="playback-speed" value={playback.speed} onChange={(event) => setPlayback((state) => ({ ...state, speed: Number(event.target.value) }))}>
                <option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option>
              </select>
              <label className="layer-toggle simple-toggle"><input type="checkbox" checked={playback.loop} onChange={(event) => setPlayback((state) => ({ ...state, loop: event.target.checked }))} />Loop playback</label>
            </div>
          ) : null}
        </section>

        <section className="control-section metadata-grid" aria-label="Dataset metadata">
          <div><span>Dataset frames</span><strong>{manifest.totals.datasetFrameCount.toLocaleString()}</strong></div>
          <div><span>Tool-point visits</span><strong>{manifest.totals.toolPointVisitCount.toLocaleString()}</strong></div>
          <div><span>Arm voxel entries</span><strong>{manifest.totals.armVoxelEntryCount.toLocaleString()}</strong></div>
          <div><span>Shared grid cells</span><strong>{manifest.totals.uniqueSharedGridCellCount.toLocaleString()}</strong></div>
        </section>
        <section className="control-section detail-list" aria-label="Coordinate metadata">
          <div><span>Dataset</span><strong>{manifest.dataset.repositoryId}</strong></div><div><span>Robot</span><strong>{manifest.robot.modelName}</strong></div><div><span>Voxel edge</span><strong>{manifest.coverage.voxelSize.toFixed(2)} m</strong></div><div><span>Frame</span><strong>Right-handed · metres</strong></div>
        </section>
        <section className="control-section detail-list" aria-label="Data provenance">
          <div><span>Requested dataset ref</span><strong>{manifest.dataset.requestedRevision}</strong></div><div><span>Resolved HF commit</span><strong title={manifest.dataset.resolvedRevision}>{manifest.dataset.resolvedRevision.slice(0, 12)}…</strong></div><div><span>LeRobot metadata</span><strong>{manifest.dataset.lerobotCodebaseVersion}</strong></div><div><span>LeRobot package</span><strong>{manifest.dataset.lerobotPackageVersion}</strong></div><div><span>Repository HEAD</span><strong title={manifest.exporter.repositoryHeadCommit}>{manifest.exporter.repositoryHeadCommit.slice(0, 12)}…</strong></div>
        </section>
        {manifest.exporter.workingTreeDirty ? <div className="source-warning" role="note"><strong>Uncommitted exporter source</strong><p>{manifest.exporter.sourceDescription}</p></div> : null}
        <div className="spacing-warning" role="note"><strong>Provisional geometry</strong><p>{manifest.coverage.spacingDisclosure}</p></div>
      </aside>
    </div>
  );
}
