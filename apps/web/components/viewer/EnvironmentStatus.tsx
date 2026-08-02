import type { EnvironmentCapability } from "@/lib/environment/types";

export function EnvironmentStatus({
  capability,
}: {
  capability: EnvironmentCapability;
}) {
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
          <dd>{capability.status === "available" ? "Available" : "Unavailable"}</dd>
        </div>
      </dl>
      {capability.status === "unavailable" ? (
        <p role="note">
          {capability.reason} No real reconstruction or calibrated environment
          alignment is claimed. The robot workspace viewer remains fully
          available.
        </p>
      ) : null}
    </section>
  );
}
