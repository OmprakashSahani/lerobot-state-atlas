import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Phone capture guide",
  description:
    "A tool-neutral field guide for recording a robot workspace for later Gaussian Splat reconstruction and calibration.",
};

const contents = [
  ["workflows", "Choose the recording workflow"],
  ["prerequisites", "Prerequisites and safety"],
  ["preparation", "Prepare the workspace"],
  ["phone-settings", "Set up the phone"],
  ["capture", "Capture the workspace"],
  ["surfaces", "Lighting and difficult surfaces"],
  ["alignment", "Scale, robot pose, and alignment"],
  ["records", "Organize files and metadata"],
  ["validation", "Reconstruct and validate"],
  ["failures", "Failure cases and recovery"],
  ["privacy", "Privacy and governance"],
  ["evidence", "Evidence gate"],
  ["acceptance", "Acceptance checklist"],
] as const;

export default function CaptureGuidePage() {
  return (
    <article className="document-page">
      <p className="eyebrow">Phone reconstruction field guide</p>
      <h1>Capture a robot workspace for later reconstruction.</h1>
      <p className="document-lede">
        LeRobot State Atlas does not contain a validated real scan, calibrated
        alignment, or production reconstruction today. This guide describes
        how to collect source material and evidence for a later, tool-neutral
        Gaussian Splat workflow; completing it does not make an environment
        production-ready.
      </p>

      <nav className="pipeline" aria-label="Capture guide contents">
        {contents.map(([id, label]) => (
          <a key={id} href={`#${id}`}>
            {label}
          </a>
        ))}
      </nav>

      <section id="workflows">
        <h2>Choose the recording workflow</h2>
        <div>
          <h3>Environment capture</h3>
          <p>
            Record a static workspace slowly and deliberately, with repeated
            viewpoints and stable scene conditions. These passes are source
            material for reconstruction and do not need to share the robot
            dataset timeline.
          </p>
          <h3>Synchronized robot-dataset recording</h3>
          <p>
            Record robot activity only when camera identity, timestamps, clock
            relationships, and drift are documented against the dataset. This
            supports time-aligned observation, not static scene coverage.
          </p>
          <p>
            Phone video alone does not establish synchronization or alignment
            to the robot shared-world coordinate frame. A video can be useful
            in either workflow, but it is not evidence that their clocks or
            coordinate systems agree.
          </p>
        </div>
      </section>

      <section id="prerequisites">
        <h2>Prerequisites and safety</h2>
        <div>
          <ul>
            <li>Use a charged phone with enough storage and a clean rear lens.</li>
            <li>Confirm that original files and metadata can be transferred without messaging-app recompression.</li>
            <li>Prepare measured references, a pass log, and a safe walking route before recording.</li>
            <li>Keep emergency-stop access clear and account for cables, trip hazards, pinch points, and restricted areas.</li>
            <li>Use a spotter when the site procedure or visibility requires one.</li>
          </ul>
          <p>
            Prefer a powered-down or safely inhibited robot for environment
            passes. Do not enter the operating envelope of a moving robot or
            trade safe separation for a camera angle. Recording robot motion
            requires the site&apos;s approved operating and synchronization procedure.
          </p>
        </div>
      </section>

      <section id="preparation">
        <h2>Prepare the workspace</h2>
        <div>
          <ul>
            <li>Remove accidental clutter while preserving fixed geometry relevant to the workspace.</li>
            <li>Keep objects, chairs, cables, doors, and robot components stationary across passes.</li>
            <li>Add removable visual texture only when it is safe, non-damaging, and documented.</li>
            <li>Place measured scale references and calibration targets where several viewpoints can see them.</li>
            <li>Record every object that was covered, removed, moved, or added for capture.</li>
          </ul>
        </div>
      </section>

      <section id="phone-settings">
        <h2>Set up the phone</h2>
        <div>
          <p>
            Use one rear camera and lens consistently. Select a resolution and
            frame rate the phone can sustain, keep orientation fixed, and lock
            focus, exposure, and white balance when the device permits. Record
            the actual settings rather than assuming defaults.
          </p>
          <p>
            Avoid digital zoom, lens switching, filters, portrait or cinematic
            effects, and modes that introduce changing geometry or exposure.
            Stabilization behavior varies by device and must be documented and
            checked rather than assumed beneficial. Preserve the original files.
          </p>
        </div>
      </section>

      <section id="capture">
        <h2>Capture the workspace</h2>
        <div>
          <ol>
            <li>Begin with a wide establishing pass showing the whole workspace and its fixed surroundings.</li>
            <li>Walk a continuous perimeter with substantial overlap and gradual changes in viewpoint.</li>
            <li>Translate around objects instead of standing in one place and only panning; reconstruction needs parallax.</li>
            <li>Move slowly enough to avoid motion blur, abrupt turns, and skipped surfaces.</li>
            <li>Make upper, middle, and lower passes, then targeted passes for safe-to-reach occluded areas.</li>
            <li>Repeat critical geometry from more than one distance and direction without changing the scene.</li>
          </ol>
          <p>
            If recording is interrupted, the lens changes, an object moves, or
            lighting changes materially, log the event and restart the affected
            pass. More footage is not a substitute for consistent, overlapping
            viewpoints.
          </p>
        </div>
      </section>

      <section id="surfaces">
        <h2>Lighting and difficult surfaces</h2>
        <div>
          <p>
            Use stable, diffuse illumination. Avoid flicker, moving shadows,
            direct glare, and exposure changes between passes. Monitor screens,
            mirrors, polished metal, glass, transparent containers, repeating
            patterns, and featureless surfaces as likely failure regions.
          </p>
          <p>
            When safe, cover or temporarily remove reflective or transparent
            objects and document the intervention. Otherwise expect holes,
            floaters, doubled surfaces, or unstable geometry. A visually
            convincing splat is not proof of geometric accuracy.
          </p>
        </div>
      </section>

      <section id="alignment">
        <h2>Scale, robot pose, and shared-world alignment</h2>
        <div>
          <p>
            Record exact dimensions and units for multiple visible references
            where practical. Scale recovery is not frame alignment: a correctly
            sized reconstruction may still have the wrong translation,
            rotation, origin, or handedness.
          </p>
          <p>
            Keep the robot in one documented, stationary pose during
            environment passes. Preserve its identity and joint-state or
            equivalent pose record. Capture fixed robot-base landmarks and
            calibration targets that can also be measured in the robot&apos;s frame.
          </p>
          <p>
            Alignment to <code>canonical-shared-world</code> requires a
            documented calibration method, measured correspondences or an
            equivalent traceable procedure, a fitted transform, and validation
            against independent points. Seeing the robot in a phone video does
            not supply that transform.
          </p>
        </div>
      </section>

      <section id="records">
        <h2>Organize files and metadata</h2>
        <div>
          <p>Keep immutable originals separate from working files:</p>
          <ul>
            <li><code>originals/</code> — untouched phone files and hashes</li>
            <li><code>passes/</code> — pass inventory, route, purpose, and events</li>
            <li><code>calibration/</code> — target images, measurements, correspondences, and transforms</li>
            <li><code>reconstruction/</code> — tool versions, settings, logs, and output hashes</li>
            <li><code>validation/</code> — thresholds, measurements, overlays, decisions, and limitations</li>
          </ul>
          <p>
            Record session ID, date, time zone, operator, location description,
            device and software versions, lens, resolution, frame rate,
            orientation, focus and exposure behavior, filenames, sizes,
            durations, hashes, lighting, scene changes, scale references, robot
            identity and pose source, safety state, privacy actions, and known
            failures. Do not call timestamps synchronized unless their clock
            relationship and drift were measured.
          </p>
        </div>
      </section>

      <section id="validation">
        <h2>Reconstruct and validate</h2>
        <div>
          <p>
            Record the reconstruction tool, version, settings, inputs, logs,
            and output hashes. Inspect tracking continuity, coverage, bounds,
            holes, floaters, doubled surfaces, and difficult materials. Compare
            reconstructed distances with measured references that were not all
            consumed by reconstruction or calibration.
          </p>
          <p>
            Before fitting or judging alignment, predeclare task-specific
            acceptance thresholds and their rationale. Report transform
            residuals and validate distributed points that were not used to fit
            the transform. Retain visual overlays at robot bases and workspace
            landmarks alongside the numerical results. This guide does not
            prescribe universal distance or angle tolerances.
          </p>
        </div>
      </section>

      <section id="failures">
        <h2>Failure cases and recovery</h2>
        <div>
          <dl className="definition-list">
            <div><dt>Blur or tracking loss</dt><dd>Slow down, improve stable lighting, and repeat the pass with smoother translation.</dd></div>
            <div><dt>Holes or floaters</dt><dd>Add overlapping viewpoints and address occlusion, glare, transparency, or featureless regions.</dd></div>
            <div><dt>Doubled geometry</dt><dd>Check for moved objects, people, shadows, exposure changes, and inconsistent passes; recapture a static scene.</dd></div>
            <div><dt>Wrong scale or alignment</dt><dd>Recheck units, reference measurements, frame definitions, correspondences, and transform conventions.</dd></div>
            <div><dt>Missing provenance</dt><dd>Recover originals and records if possible; otherwise document the gap and reject the real-scan claim.</dd></div>
          </dl>
        </div>
      </section>

      <section id="privacy">
        <h2>Privacy and governance</h2>
        <div>
          <p>
            Obtain permission to record the location and identifiable people.
            Exclude people, screens, badges, paperwork, addresses, network
            identifiers, and confidential equipment where possible. A
            reconstruction can preserve details that are easy to miss in one frame.
          </p>
          <p>
            Prefer prevention during capture. If redaction is required, retain
            a record of what changed, apply the governing access and retention
            policy to originals, and verify that derived outputs do not reveal
            the removed information.
          </p>
        </div>
      </section>

      <section id="evidence">
        <h2>Evidence gate for documented-real-scan</h2>
        <div>
          <p>
            A manifest may use <code>reconstructionClaim: &quot;documented-real-scan&quot;</code>{" "}
            only when all of the following evidence is retained and reviewable:
          </p>
          <ul>
            <li>Immutable original capture inventory with file hashes, capture metadata, and pass log.</li>
            <li>Real-scene provenance and a record of every material scene intervention.</li>
            <li>Reconstruction inputs, tool and version, settings, logs, and output artifact hashes.</li>
            <li>Scale-reference measurements, units, uncertainty where known, and images showing placement.</li>
            <li>Robot identity, pose evidence, shared-world frame definitions, and transform conventions.</li>
            <li>Calibration method, raw measurements or correspondences, fitted transform, and provenance of the result.</li>
            <li>Predeclared task-specific thresholds, quantitative residuals, and independent validation points not used to fit the transform.</li>
            <li>Distributed visual overlays, known limitations, failed regions, and unresolved discrepancies.</li>
            <li>A traceable reviewer identity, review date, and accept or reject decision.</li>
          </ul>
          <p>
            <code>reconstructionClaim</code> must remain <code>false</code>{" "}
            whenever required evidence is missing, contradictory, or outside
            predeclared tolerances. Passing this documentation gate does not by
            itself approve the asset for production rendering or robot control.
          </p>
        </div>
      </section>

      <section id="acceptance">
        <h2>Final acceptance checklist</h2>
        <div>
          <h3>Capture and records</h3>
          <ul>
            <li>All required surfaces have stable, overlapping coverage from multiple passes.</li>
            <li>Originals, hashes, settings, pass events, safety state, and scene changes are retained.</li>
            <li>Scale references, robot pose, and calibration targets are visible and measured.</li>
          </ul>
          <h3>Reconstruction and alignment</h3>
          <ul>
            <li>Known failure regions and interventions are documented.</li>
            <li>Predeclared task-specific thresholds are recorded before acceptance is judged.</li>
            <li>Scale checks, calibration residuals, independent validation points, and overlays meet those thresholds.</li>
          </ul>
          <h3>Governance and claim</h3>
          <ul>
            <li>Privacy, permission, access, and retention requirements are satisfied.</li>
            <li>A traceable reviewer identity, date, and decision are retained.</li>
            <li>The claim remains false unless every required evidence item is complete and consistent.</li>
            <li>No claim is based only on the existence of a phone video or visually convincing splat.</li>
          </ul>
        </div>
      </section>

      <div className="document-actions" aria-label="Related links">
        <Link className="button button-primary" href="/methodology">
          Read the methodology
        </Link>
        <Link className="button button-secondary" href="/viewer/demo">
          View the unavailable environment state
        </Link>
        <a
          className="text-link"
          href="https://github.com/OmprakashSahani/lerobot-state-atlas/blob/main/docs/environment-layer.md"
          rel="noreferrer"
          target="_blank"
        >
          Read the environment-layer contract ↗
        </a>
      </div>
    </article>
  );
}
