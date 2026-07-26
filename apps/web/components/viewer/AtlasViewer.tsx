"use client";

import { ViewerCanvas } from "./ViewerCanvas";
import { useAtlasData } from "./AtlasDataProvider";
import { useViewerStore } from "./ViewerStore";

export function AtlasViewer() {
  const atlas = useAtlasData();
  const viewer = useViewerStore();

  if (atlas.status === "loading") {
    return (
      <div className="viewer-status" role="status" aria-live="polite">
        <span className="status-pulse" aria-hidden="true" />
        Loading pinned atlas data…
      </div>
    );
  }
  if (atlas.status === "error") {
    return (
      <section className="viewer-fallback" role="alert">
        <p className="eyebrow">Data load failed</p>
        <h1>The demo bundle could not be opened.</h1>
        <p>{atlas.message}</p>
        <button
          className="button button-secondary"
          onClick={() => window.location.reload()}
          type="button"
        >
          Retry
        </button>
      </section>
    );
  }

  const { manifest, preparedArms } = atlas.data;
  const minimumVisit = Math.min(
    ...preparedArms.map((arm) => arm.minimumVisitCount),
  );
  const maximumVisit = Math.max(
    ...preparedArms.map((arm) => arm.maximumVisitCount),
  );

  return (
    <div className="viewer-shell">
      <section className="viewer-stage" aria-label="Interactive workspace scene">
        <ViewerCanvas data={atlas.data} />
        <div className="scene-badge">
          <span className="live-dot" aria-hidden="true" />
          Canonical shared world
        </div>
        <p className="scene-help">
          Drag to orbit · Shift-drag to pan · Scroll to zoom
        </p>
      </section>
      <aside className="viewer-panel" aria-label="Viewer controls and metadata">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Demo / episodes 0–9</p>
            <h1>Workspace coverage</h1>
          </div>
          <span className="schema-chip">schema v1.0</span>
        </div>

        <section className="control-section" aria-labelledby="metric-heading">
          <div className="section-title-row">
            <h2 id="metric-heading">Metric</h2>
            <span>Raw tool visits</span>
          </div>
          <div className="metric-card" aria-label="Selected metric: Visits">
            <span className="metric-swatch" aria-hidden="true" />
            <div>
              <strong>Visits</strong>
              <small>Arm-specific tool points per occupied voxel</small>
            </div>
          </div>
          <div className="legend" aria-label="Visit count color range">
            <div className="legend-gradient" aria-hidden="true" />
            <div>
              <span>{minimumVisit.toLocaleString()}</span>
              <span>{maximumVisit.toLocaleString()}</span>
            </div>
          </div>
        </section>

        <section className="control-section" aria-labelledby="layers-heading">
          <div className="section-title-row">
            <h2 id="layers-heading">Robot-data layers</h2>
            <button
              className="compact-button"
              type="button"
              onClick={viewer.resetCamera}
            >
              Reset camera
            </button>
          </div>
          <label className="layer-toggle">
            <input
              checked={viewer.leftVisible}
              onChange={() => viewer.toggleArm("left")}
              type="checkbox"
            />
            <span className="arm-dot arm-dot-left" aria-hidden="true" />
            Left arm entries
            <strong>
              {preparedArms[0].visits.length.toLocaleString()}
            </strong>
          </label>
          <label className="layer-toggle">
            <input
              checked={viewer.rightVisible}
              onChange={() => viewer.toggleArm("right")}
              type="checkbox"
            />
            <span className="arm-dot arm-dot-right" aria-hidden="true" />
            Right arm entries
            <strong>
              {preparedArms[1].visits.length.toLocaleString()}
            </strong>
          </label>
        </section>

        <section className="control-section metadata-grid" aria-label="Dataset metadata">
          <div>
            <span>Dataset frames</span>
            <strong>{manifest.totals.datasetFrameCount.toLocaleString()}</strong>
          </div>
          <div>
            <span>Tool-point visits</span>
            <strong>{manifest.totals.toolPointVisitCount.toLocaleString()}</strong>
          </div>
          <div>
            <span>Arm voxel entries</span>
            <strong>{manifest.totals.armVoxelEntryCount.toLocaleString()}</strong>
          </div>
          <div>
            <span>Shared grid cells</span>
            <strong>
              {manifest.totals.uniqueSharedGridCellCount.toLocaleString()}
            </strong>
          </div>
        </section>

        <section className="control-section detail-list" aria-label="Coordinate metadata">
          <div>
            <span>Dataset</span>
            <strong>{manifest.dataset.repositoryId}</strong>
          </div>
          <div>
            <span>Robot</span>
            <strong>{manifest.robot.modelName}</strong>
          </div>
          <div>
            <span>Voxel edge</span>
            <strong>{manifest.coverage.voxelSize.toFixed(2)} m</strong>
          </div>
          <div>
            <span>Frame</span>
            <strong>Right-handed · metres</strong>
          </div>
        </section>

        <section
          className="control-section detail-list"
          aria-label="Data provenance"
        >
          <div>
            <span>Requested dataset ref</span>
            <strong>{manifest.dataset.requestedRevision}</strong>
          </div>
          <div>
            <span>Resolved HF commit</span>
            <strong title={manifest.dataset.resolvedRevision}>
              {manifest.dataset.resolvedRevision.slice(0, 12)}…
            </strong>
          </div>
          <div>
            <span>LeRobot metadata</span>
            <strong>{manifest.dataset.lerobotCodebaseVersion}</strong>
          </div>
          <div>
            <span>LeRobot package</span>
            <strong>{manifest.dataset.lerobotPackageVersion}</strong>
          </div>
          <div>
            <span>Repository HEAD</span>
            <strong title={manifest.exporter.repositoryHeadCommit}>
              {manifest.exporter.repositoryHeadCommit.slice(0, 12)}…
            </strong>
          </div>
        </section>

        {manifest.exporter.workingTreeDirty ? (
          <div className="source-warning" role="note">
            <strong>Uncommitted exporter source</strong>
            <p>{manifest.exporter.sourceDescription}</p>
          </div>
        ) : null}

        <div className="spacing-warning" role="note">
          <strong>Provisional geometry</strong>
          <p>{manifest.coverage.spacingDisclosure}</p>
        </div>
      </aside>
    </div>
  );
}
