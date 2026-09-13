import type { EnvironmentCapability } from "@/lib/environment/types";
import type { LocalEnvironmentController } from "@/lib/environment/use-local-environment";
import {
  analyzeResourceLifecycle,
  formatSignedResourceDelta,
  INTEGRATED_ENVIRONMENT_SOURCES,
  INTEGRATED_SOURCE_SWITCH_CYCLE,
  type IntegratedEnvironmentPerformanceMetrics,
  type IntegratedEnvironmentSource,
  type SparkResourceLifecycleEvent,
} from "@/lib/environment/integrated-performance-diagnostic";

export interface IntegratedPerformanceDiagnosticControls {
  enabled: boolean;
  source: IntegratedEnvironmentSource;
  metrics: IntegratedEnvironmentPerformanceMetrics | null;
  lifecycleEvents: SparkResourceLifecycleEvent[];
  cycle: { running: boolean; sequenceIndex: number };
  canRunCycle: boolean;
  onSourceChange(source: IntegratedEnvironmentSource): void;
  onRunCycle(): void;
}

export function EnvironmentStatus({
  capability,
  local,
  diagnostic,
}: {
  capability: EnvironmentCapability;
  local?: LocalEnvironmentController;
  diagnostic?: IntegratedPerformanceDiagnosticControls;
}) {
  const phaseLabels: Partial<
    Record<LocalEnvironmentController["phase"], string>
  > = {
    idle: "Available — not requested",
    "loading-manifest": "Loading manifest",
    "loading-asset": "Loading asset",
    "verifying-checksum": "Verifying checksum",
    "inspecting-spz": "Inspecting SPZ",
    "initializing-renderer": "Initializing renderer",
    "ready-visible": "Environment visible",
    "ready-hidden": "Environment hidden",
    unloading: "Unloading",
    error: "Unavailable after local load error",
    "unsupported-webgl2": "Unsupported WebGL2",
    "mobile-refusal": "Desktop-only spike",
  };

  const localEnabled = local && local.phase !== "unavailable";
  const manifest = local?.request?.manifest;
  const lifecycleResult = analyzeResourceLifecycle(
    diagnostic?.lifecycleEvents ?? [],
  );
  const steadyResourceEvents = (diagnostic?.lifecycleEvents ?? []).filter(
    (event) => event.stage === "steady-state" && event.rendererInfo,
  );
  const lifecycleStageLabel = (stage: string) =>
    stage === "after-old-disposal-frame" ||
    stage === "after-new-attachment-frame"
      ? `${stage} (pre-render useFrame sample)`
      : stage;

  return (
    <section
      className="control-section environment-status"
      aria-labelledby="environment-heading"
    >
      <div className="section-title-row">
        <h2 id="environment-heading">Environment</h2>
        <span>Independent layer</span>
      </div>

      <dl>
        <div>
          <dt>Current state</dt>
          <dd>Analytical grid active</dd>
        </div>
        <div>
          <dt>Gaussian Splat status</dt>
          <dd>
            {localEnabled
              ? phaseLabels[local.phase]
              : capability.status === "available"
                ? "Available"
                : "Unavailable"}
          </dd>
        </div>
      </dl>

      {!localEnabled && capability.status === "unavailable" ? (
        <p role="note">
          {capability.reason} No real reconstruction or calibrated environment
          alignment is claimed. The robot workspace viewer remains fully
          available.
        </p>
      ) : null}

      {localEnabled ? (
        <div className="local-environment-controls">
          <strong>{manifest?.label ?? "Local environment configured"}</strong>

          {manifest ? (
            <p role="note">{manifest.provenance.description}</p>
          ) : (
            <p role="note">
              The environment layer is optional and remains independent from
              the analytical grid, robot trajectory, and dataset calculations.
            </p>
          )}

          {local?.disclosure ? <p role="note">{local.disclosure}</p> : null}

          {diagnostic?.enabled ? (
            <div className="environment-performance-diagnostic">
              <label htmlFor="integrated-environment-source">
                Development performance source
              </label>
              <select
                id="integrated-environment-source"
                value={diagnostic.source}
                onChange={(event) =>
                  diagnostic.onSourceChange(
                    event.target.value as IntegratedEnvironmentSource,
                  )
                }
              >
                {Object.values(INTEGRATED_ENVIRONMENT_SOURCES).map((source) => (
                  <option key={source.source} value={source.source}>
                    {source.label}
                  </option>
                ))}
              </select>
              <small>
                Render-only diagnostic. Production validation and analytical
                scene data are unchanged.
              </small>

              {diagnostic.metrics ? (
                <dl aria-label="Environment performance measurements">
                  <div>
                    <dt>Active source</dt>
                    <dd>
                      {
                        INTEGRATED_ENVIRONMENT_SOURCES[
                          diagnostic.metrics.source
                        ].label
                      }
                    </dd>
                  </div>
                  <div>
                    <dt>Decoded splats</dt>
                    <dd>
                      {diagnostic.metrics.decodedSplatCount.toLocaleString()}
                    </dd>
                  </div>
                  <div>
                    <dt>Loaded asset</dt>
                    <dd>
                      {diagnostic.metrics.assetBytes.toLocaleString()} bytes (
                      {(
                        diagnostic.metrics.assetBytes /
                        (1024 * 1024)
                      ).toFixed(3)}{" "}
                      MiB)
                    </dd>
                  </div>
                  <div>
                    <dt>Mesh initialization</dt>
                    <dd>
                      {diagnostic.metrics.meshInitializationMilliseconds.toFixed(
                        1,
                      )}{" "}
                      ms
                    </dd>
                  </div>
                  <div>
                    <dt>First ready</dt>
                    <dd>
                      {diagnostic.metrics.firstReadyMilliseconds.toFixed(1)} ms
                    </dd>
                  </div>
                  <div>
                    <dt>Frame window</dt>
                    <dd>
                      {diagnostic.metrics.frame
                        ? `${diagnostic.metrics.frame.averageMilliseconds.toFixed(2)} ms avg / ${diagnostic.metrics.frame.framesPerSecond.toFixed(1)} FPS`
                        : "Collecting…"}
                    </dd>
                  </div>
                  <div>
                    <dt>Frame classification</dt>
                    <dd>
                      {diagnostic.metrics.initializationFrameCount} init /{" "}
                      {diagnostic.metrics.stabilizationFrameCount} stabilizing
                      / {diagnostic.metrics.stabilizedFrameCount} measured
                    </dd>
                  </div>
                  <div>
                    <dt>Frame p50 / p90 / p95 / p99</dt>
                    <dd>
                      {diagnostic.metrics.frame
                        ? `${diagnostic.metrics.frame.p50Milliseconds.toFixed(2)} / ${diagnostic.metrics.frame.p90Milliseconds.toFixed(2)} / ${diagnostic.metrics.frame.p95Milliseconds.toFixed(2)} / ${diagnostic.metrics.frame.p99Milliseconds.toFixed(2)} ms`
                        : "Collecting…"}
                    </dd>
                  </div>
                  <div>
                    <dt>Frame max</dt>
                    <dd>
                      {diagnostic.metrics.frame
                        ? `${diagnostic.metrics.frame.maxMilliseconds.toFixed(2)} ms`
                        : "Collecting…"}
                    </dd>
                  </div>
                  <div>
                    <dt>Tail frames &gt;20 / 25 ms</dt>
                    <dd>
                      {diagnostic.metrics.frame
                        ? `${diagnostic.metrics.frame.over20Milliseconds} / ${diagnostic.metrics.frame.over25Milliseconds}`
                        : "Collecting…"}
                    </dd>
                  </div>
                  <div>
                    <dt>Tail frames &gt;33.3 / 50 ms</dt>
                    <dd>
                      {diagnostic.metrics.frame
                        ? `${diagnostic.metrics.frame.over33Point3Milliseconds} / ${diagnostic.metrics.frame.over50Milliseconds}`
                        : "Collecting…"}
                    </dd>
                  </div>
                  <div>
                    <dt>Three info.memory</dt>
                    <dd>
                      {diagnostic.metrics.rendererInfo
                        ? `${diagnostic.metrics.rendererInfo.geometries} geometries / ${diagnostic.metrics.rendererInfo.textures} textures / ${diagnostic.metrics.rendererInfo.programs ?? "?"} programs`
                        : "Unavailable"}
                    </dd>
                  </div>
                  <div>
                    <dt>Render primitives</dt>
                    <dd>
                      {diagnostic.metrics.rendererInfo
                        ? `${diagnostic.metrics.rendererInfo.renderCalls} calls / ${diagnostic.metrics.rendererInfo.triangles.toLocaleString()} triangles / ${diagnostic.metrics.rendererInfo.points.toLocaleString()} points / ${diagnostic.metrics.rendererInfo.lines.toLocaleString()} lines`
                        : "Unavailable"}
                    </dd>
                  </div>
                </dl>
              ) : (
                <p role="status">Load the environment to collect measurements.</p>
              )}
              <small>
                Three.js resource values are object/program counts, not GPU or
                system-memory byte estimates. Frame statistics cover the whole
                integrated viewer over a rolling window of up to 120 frames.
              </small>

              <button
                className="compact-button"
                disabled={!diagnostic.canRunCycle || diagnostic.cycle.running}
                onClick={diagnostic.onRunCycle}
                type="button"
              >
                {diagnostic.cycle.running
                  ? `Running switch cycle (${diagnostic.cycle.sequenceIndex + 1}/${INTEGRATED_SOURCE_SWITCH_CYCLE.length})`
                  : "Run source-switch cycle"}
              </button>
              {!diagnostic.canRunCycle && !diagnostic.cycle.running ? (
                <small>
                  Load and stabilize the current 86k source before running the
                  controlled cycle.
                </small>
              ) : null}

              {diagnostic.lifecycleEvents.length > 0 ? (
                <div className="environment-resource-lifecycle">
                  <strong>Resource lifecycle checkpoints</strong>
                  <ol>
                    {diagnostic.lifecycleEvents.map((event, index) => (
                      <li key={`${event.switchId}:${event.stage}:${index}`}>
                        <span>
                          #{event.switchId} {event.fromSource} → {event.toSource}{" "}
                          · {lifecycleStageLabel(event.stage)}
                        </span>
                        <strong>
                          {event.rendererInfo
                            ? `G${event.rendererInfo.geometries} T${event.rendererInfo.textures} P${event.rendererInfo.programs ?? "?"}`
                            : "renderer info unavailable"}
                        </strong>
                      </li>
                    ))}
                  </ol>
                  {steadyResourceEvents.length > 0 ? (
                    <small>
                      Steady sequence:{" "}
                      {steadyResourceEvents
                        .map((event) => {
                          const info = event.rendererInfo;
                          return info
                            ? `G${info.geometries} T${info.textures} P${info.programs ?? "?"}`
                            : "unavailable";
                        })
                        .join(" → ")}
                    </small>
                  ) : null}
                  {lifecycleResult.status !== "pending" &&
                  !diagnostic.cycle.running ? (
                    <small role="status">
                      Baseline result: {lifecycleResult.status}; Δ geometries{" "}
                      {formatSignedResourceDelta(
                        lifecycleResult.geometryDelta,
                      )}
                      , Δ textures{" "}
                      {formatSignedResourceDelta(lifecycleResult.textureDelta)},
                      Δ programs{" "}
                      {lifecycleResult.programDelta === null
                        ? "unavailable"
                        : formatSignedResourceDelta(
                            lifecycleResult.programDelta,
                          )}
                      .
                    </small>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : null}

          {local.phase === "mobile-refusal" ? (
            <p role="note">
              Loading is intentionally disabled on mobile devices for this
              desktop-only spike.
            </p>
          ) : null}

          {local.phase === "error" && local.error ? (
            <p role="alert">{local.error}</p>
          ) : null}

          <div className="environment-actions">
            {local.phase === "idle" ||
            local.phase === "unsupported-webgl2" ? (
              <button
                className="compact-button"
                disabled={local.phase === "unsupported-webgl2"}
                onClick={local.load}
                type="button"
              >
                Load environment
              </button>
            ) : null}

            {local.phase === "error" ? (
              <button
                className="compact-button"
                onClick={local.retry}
                type="button"
              >
                Retry
              </button>
            ) : null}

            {local.phase === "ready-visible" ? (
              <button
                className="compact-button"
                onClick={local.hide}
                type="button"
              >
                Hide
              </button>
            ) : null}

            {local.phase === "ready-hidden" ? (
              <button
                className="compact-button"
                onClick={local.show}
                type="button"
              >
                Show
              </button>
            ) : null}

            {local.request ? (
              <button
                className="compact-button"
                onClick={local.unload}
                type="button"
              >
                Unload
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  );
}
