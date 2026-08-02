import Link from "next/link";

const stats = [
  ["10", "episodes"],
  ["5,124", "dataset frames"],
  ["10,248", "tool-point visits"],
  ["1,205", "shared grid cells"],
];

export default function HomePage() {
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Robot workspace intelligence</p>
          <h1>See where a dual-arm dataset actually moves.</h1>
          <p className="hero-lede">
            LeRobot State Atlas converts recorded robot state into shared-world
            tool trajectories and exact voxel coverage—before a single byte
            reaches the browser.
          </p>
          <div className="hero-actions">
            <Link className="button button-primary" href="/viewer/demo">
              Open shared-world viewer
            </Link>
            <Link className="button button-secondary" href="/methodology">
              Read the methodology
            </Link>
            <Link className="button button-secondary" href="/capture-guide">
              Plan a workspace capture
            </Link>
            <Link className="button button-secondary" href="/checkpoint-comparison">
              Compare checkpoints
            </Link>
          </div>
          <p className="calibration-note">
            The demo&apos;s 0.8 m arm spacing is a configurable visualization
            assumption, not calibrated physical geometry.
          </p>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="atlas-orbit atlas-orbit-one" />
          <div className="atlas-orbit atlas-orbit-two" />
          <div className="atlas-axis atlas-axis-x" />
          <div className="atlas-axis atlas-axis-y" />
          <div className="voxel-cloud voxel-cloud-left" />
          <div className="voxel-cloud voxel-cloud-right" />
          <span className="visual-label label-world">SHARED WORLD / METRES</span>
          <span className="visual-label label-left">LEFT TOOL</span>
          <span className="visual-label label-right">RIGHT TOOL</span>
        </div>
      </section>

      <section className="stat-strip" aria-label="Demo dataset summary">
        {stats.map(([value, label]) => (
          <div key={label}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </section>

      <section className="product-section">
        <div>
          <p className="eyebrow">Precomputed by design</p>
          <h2>Robotics computation stays in Python. Interaction stays fast.</h2>
        </div>
        <div className="feature-grid">
          <article>
            <span>01</span>
            <h3>Pinned inputs</h3>
            <p>
              Requested dataset ref, resolved Hub commit, URDF identity,
              transforms, and export checksums travel with every browser bundle.
            </p>
          </article>
          <article>
            <span>02</span>
            <h3>Exact semantics</h3>
            <p>
              Arm-specific entries, shared grid cells, and dual-arm tool visits
              remain distinct instead of collapsing into one vague count.
            </p>
          </article>
          <article>
            <span>03</span>
            <h3>Layered rendering</h3>
            <p>
              Environment and robot-data layers share one coordinate system,
              leaving a clean extension point for scanned environments.
            </p>
          </article>
        </div>
      </section>
    </>
  );
}
