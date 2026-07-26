import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "Methodology" };

export default function MethodologyPage() {
  return (
    <article className="document-page">
      <p className="eyebrow">Reproducible methodology</p>
      <h1>From recorded state to shared-world coverage.</h1>
      <p className="document-lede">
        The web viewer does not run LeRobot, PyTorch, URDF parsing, or forward
        kinematics. It reads a validated, versioned export produced by the
        project&apos;s Python package.
      </p>

      <div className="pipeline" aria-label="Data processing pipeline">
        <span>LeRobot state</span>
        <span aria-hidden="true">→</span>
        <span>Forward kinematics</span>
        <span aria-hidden="true">→</span>
        <span>World transforms</span>
        <span aria-hidden="true">→</span>
        <span>Voxel coverage</span>
        <span aria-hidden="true">→</span>
        <span>Validated bundle</span>
      </div>

      <section>
        <h2>Coordinate frame</h2>
        <p>
          Positions use metres in a right-handed canonical shared-world frame.
          Each tool position is computed in its local follower
          <code> base_link </code> frame, then transformed as{" "}
          <code>p_world = R · p_local + translation</code>. Rotation uses{" "}
          <code>Rz(yaw) · Ry(pitch) · Rx(roll)</code>.
        </p>
        <p>
          The demo places the left base at world Y +0.4 m and the right base at
          world Y −0.4 m. The resulting 0.8 m separation is provisional,
          configurable, and not a claim about calibrated robot geometry.
        </p>
      </section>

      <section>
        <h2>Counting without ambiguity</h2>
        <dl className="definition-list">
          <div>
            <dt>Dataset frames</dt>
            <dd>Recorded timesteps selected from the source dataset.</dd>
          </div>
          <div>
            <dt>Tool-point visits</dt>
            <dd>
              Left and right visits summed. A single dataset frame can
              contribute two tool points.
            </dd>
          </div>
          <div>
            <dt>Arm-specific voxel entries</dt>
            <dd>
              Occupied voxels retained separately per arm, even when both arms
              share one integer grid cell.
            </dd>
          </div>
          <div>
            <dt>Unique shared grid cells</dt>
            <dd>
              Distinct integer voxel indices after both arms enter the
              canonical world frame.
            </dd>
          </div>
        </dl>
      </section>

      <section>
        <h2>Exact episode identity</h2>
        <p>
          Each occupied arm voxel stores sorted episode identities in a compact
          CSR representation. Future radius queries can therefore union exact
          source episode IDs rather than approximate them from aggregate
          counts.
        </p>
      </section>

      <div className="document-actions">
        <Link className="button button-primary" href="/viewer/demo">
          Inspect the demo
        </Link>
        <a
          className="text-link"
          href="https://github.com/OmprakashSahani/lerobot-state-atlas"
          rel="noreferrer"
          target="_blank"
        >
          Reproduce from source ↗
        </a>
      </div>
    </article>
  );
}
