import type { EnvironmentCapability } from "@/lib/environment/types";
import type { LocalEnvironmentController } from "@/lib/environment/use-local-environment";

export function EnvironmentStatus({
  capability,
  local,
}: {
  capability: EnvironmentCapability;
  local?: LocalEnvironmentController;
}) {
  const phaseLabels: Partial<Record<LocalEnvironmentController["phase"], string>> = {
    idle: "Synthetic available — not requested",
    "loading-manifest": "Loading manifest",
    "loading-asset": "Loading asset",
    "verifying-checksum": "Verifying checksum",
    "inspecting-spz": "Inspecting SPZ",
    "initializing-renderer": "Initializing renderer",
    "ready-visible": "Synthetic environment visible",
    "ready-hidden": "Synthetic environment hidden",
    unloading: "Unloading",
    error: "Unavailable after local load error",
    "unsupported-webgl2": "Unsupported WebGL2",
    "mobile-refusal": "Desktop-only spike",
  };
  const localEnabled = local && local.phase !== "unavailable";
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
          <dd>{localEnabled ? phaseLabels[local.phase] : capability.status === "available" ? "Available" : "Unavailable"}</dd>
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
          <strong>Synthetic test environment — not a real reconstruction</strong>
          <p role="note">
            This local compatibility fixture is uncalibrated. The analytical grid and robot viewer remain independent and fully available.
          </p>
          {local.disclosure ? <p role="note">{local.disclosure}</p> : null}
          {local.phase === "mobile-refusal" ? (
            <p role="note">Loading is intentionally disabled on mobile devices for this desktop-only spike.</p>
          ) : null}
          {local.phase === "error" && local.error ? <p role="alert">{local.error}</p> : null}
          <div className="environment-actions">
            {local.phase === "idle" || local.phase === "unsupported-webgl2" ? <button className="compact-button" disabled={local.phase === "unsupported-webgl2"} onClick={local.load} type="button">Load synthetic environment</button> : null}
            {local.phase === "error" ? <button className="compact-button" onClick={local.retry} type="button">Retry</button> : null}
            {local.phase === "ready-visible" ? <button className="compact-button" onClick={local.hide} type="button">Hide</button> : null}
            {local.phase === "ready-hidden" ? <button className="compact-button" onClick={local.show} type="button">Show</button> : null}
            {local.request ? <button className="compact-button" onClick={local.unload} type="button">Unload</button> : null}
          </div>
        </div>
      ) : null}
    </section>
  );
}
