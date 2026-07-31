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
          CSR representation. Radius queries and uncommon-space scoring can
          therefore use exact source episode IDs rather than approximate them
          from aggregate counts.
        </p>
      </section>

      <section>
        <h2>Uncommon-space episode scoring</h2>
        <p>
          Uncommon-space exploration ranks recorded episodes by how uncommon
          their reached workspace entries are within this exported coverage
          set. It is an exploration aid, not an anomaly detector. It does not
          measure task quality, success, usefulness, safety, or physical
          novelty.
        </p>
        <p>
          Scoring uses distinct-episode incidence from the exact CSR episode
          identities. Left and right arm voxel entries remain separate
          analytical entries, even when their integer voxel coordinates match.
          Raw tool-point visit counts are a separate metric and do not affect
          uncommonness scores.
        </p>

        <h3>Formula in words and notation</h3>
        <dl className="definition-list">
          <div>
            <dt>E</dt>
            <dd>The number of episodes in the exported coverage set.</dd>
          </div>
          <div>
            <dt>c_v</dt>
            <dd>
              The number of distinct exported coverage episodes represented in
              arm-specific voxel entry v.
            </dd>
          </div>
          <div>
            <dt>V_i,S</dt>
            <dd>
              The unique arm-specific entries in scope S whose exact CSR
              identities contain episode i.
            </dd>
          </div>
        </dl>
        <p>
          When E is greater than one, entry rarity is{" "}
          <code>r(v) = ln(E / c_v) / ln(E)</code>. In words, this is the natural
          logarithm of total coverage episodes divided by the entry&apos;s distinct
          episode count, normalized by the natural logarithm of the total. When
          E is one or less, <code>r(v) = 0</code>.
        </p>
        <p>
          Episode uncommonness is{" "}
          <code>U(i,S) = mean of r(v) over v in V_i,S</code>. The displayed score
          is <code>100 × U(i,S)</code>, normalized to 0–100. It is not a
          probability or percentile. The evidence count is displayed
          separately, and averaging prevents breadth or episode length alone
          from dominating the score.
        </p>

        <h3>Scope and deterministic ordering</h3>
        <p>
          Entire coverage is the default scope. Selected radius uses the exact
          arm-specific entry identities returned by the shared-world radius
          query. Runtime arm spacing never changes an entry&apos;s rarity, but it
          may change which entries geometrically fall inside the radius. Arm
          visibility does not change the analytical scope.
        </p>
        <p>
          Results are ordered first by score descending, then by arm-specific
          entries touched descending, and finally by episode ID ascending.
        </p>

        <h3>Playback and limitations</h3>
        <p>
          Coverage scoring requires no trajectory load. Playback availability
          is checked lazily from the existing trajectory payload, without an
          additional initial request. An episode absent from that payload
          remains valid coverage evidence and is labelled coverage-only. The
          viewer does not infer a synthetic trajectory, optional state, or
          video. Orientation and raw gripper data match only the selected
          exported trajectory episode.
        </p>
        <p>
          Scores describe only the episodes and voxelization in this exported
          bundle. They do not establish rarity in the full source dataset,
          future recordings, production behavior, or the physical workspace
          generally. Results depend on voxel size, selected radius, episode
          selection, robot model, forward kinematics, and shared-world
          transforms. A high score supported by few entries should be read
          together with its displayed touched-entry count. With one exported
          coverage episode, relative uncommonness is unavailable and scores are
          defined as zero.
        </p>
        <p>
          Scores are derived client-side from existing coverage CSR data. No
          schema revision or additional initial network request was required,
          and work remains linear in relevant CSR membership for the current
          bundle. This does not claim full-dataset scale support; that remains a
          later roadmap phase.
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
