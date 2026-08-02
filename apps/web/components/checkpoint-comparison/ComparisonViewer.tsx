"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";

import { computeVisibleTrajectoryBounds } from "@/lib/checkpoint-comparison/sceneBounds";
import type { CheckpointComparisonData } from "@/lib/checkpoint-comparison/types";

const ComparisonScene = dynamic(() => import("./ComparisonScene").then((value) => value.ComparisonScene), { ssr: false });

const COMPONENTS = ["L J1", "L J2", "L J3", "L J4", "L J5", "L J6", "L grip", "R J1", "R J2", "R J3", "R J4", "R J5", "R J6", "R grip"];

export function ComparisonViewer({ data }: { data: CheckpointComparisonData }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [visibility, setVisibility] = useState<[boolean, boolean]>([true, true]);
  const [showPaths, setShowPaths] = useState(true);
  const [showMarkers, setShowMarkers] = useState(true);
  const [showOrientations, setShowOrientations] = useState(false);
  const [showGrippers, setShowGrippers] = useState(false);
  const [fitRequest, setFitRequest] = useState(0);
  const projection = data.plans.trajectoryProjection;
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setStep((value) => (value + 1) % 50), 20 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed]);
  const violations = useMemo(() => projection.available ? projection.plans.flatMap((plan) => plan.jointLimitViolations) : [], [projection]);
  const sceneBounds = useMemo(
    () => projection.available ? computeVisibleTrajectoryBounds(projection, visibility) : null,
    [projection, visibility],
  );
  return (
    <div className="comparison-shell">
      <header className="comparison-header">
        <div><p className="eyebrow">Generated policy plans · derived FK visualization</p><h1>Base π0.5 vs Fine-tuned π0.5</h1><p>{data.manifest.observation.observationId} · {data.manifest.dataset.repositoryId}</p></div>
        <div className="comparison-legend" aria-label="Policy legend"><span><i className="legend-base" />Base π0.5 — solid</span><span><i className="legend-fine" />Fine-tuned π0.5 — alternate</span></div>
      </header>
      <section className="comparison-controls" aria-label="Comparison playback controls">
        <div className="comparison-control-group comparison-control-group--playback" role="group" aria-label="Playback and view">
          <button type="button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? "Pause comparison" : "Play comparison"}>{playing ? "Pause" : "Play"}</button>
          <button type="button" onClick={() => { setPlaying(false); setStep(0); }}>Reset</button>
          <button type="button" onClick={() => setFitRequest((value) => value + 1)} aria-label="Fit view to visible trajectories" disabled={!projection.available}>Fit view</button>
          <label className="comparison-step-control">Step <input aria-label="Comparison step" type="range" min="0" max="49" value={step} onChange={(event) => setStep(Number(event.target.value))} /></label>
          <output>{step + 1}/50 · {data.plans.plans[0].relativeTimesSeconds[step].toFixed(2)} s</output>
          <label>Speed <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select></label>
        </div>
        <div className="comparison-control-group comparison-control-group--display" role="group" aria-label="Policy and glyph visibility">
          <label><input type="checkbox" checked={visibility[0]} onChange={(event) => setVisibility([event.target.checked, visibility[1]])} /> Base π0.5</label>
          <label><input type="checkbox" checked={visibility[1]} onChange={(event) => setVisibility([visibility[0], event.target.checked])} /> Fine-tuned π0.5</label>
          <label><input type="checkbox" checked={showPaths} onChange={(event) => setShowPaths(event.target.checked)} /> Paths</label>
          <label><input type="checkbox" checked={showMarkers} onChange={(event) => setShowMarkers(event.target.checked)} /> Current markers</label>
          <label><input type="checkbox" checked={showOrientations} onChange={(event) => setShowOrientations(event.target.checked)} /> XYZW orientation glyphs</label>
          <label><input type="checkbox" checked={showGrippers} onChange={(event) => setShowGrippers(event.target.checked)} /> Symbolic raw gripper glyphs</label>
        </div>
      </section>
      <div className="comparison-grid">
        <section className="comparison-stage" aria-label="Trajectory projection">
          {projection.available && sceneBounds ? (
            <ComparisonScene projection={projection} step={step} visibility={visibility} showPaths={showPaths} showMarkers={showMarkers} showOrientations={showOrientations} showGrippers={showGrippers} bounds={sceneBounds} fitRequest={fitRequest} />
          ) : !projection.available ? (
            <div className="comparison-unavailable"><p className="eyebrow">Projection unavailable</p><h2>No 3D trajectory was fabricated.</h2><p>{projection.reason}</p></div>
          ) : null}
        </section>
        <aside className="comparison-panel">
          <section><h2>Absolute position targets</h2><p>Authoritative postprocessed model actions. These are generated plans, not recorded ground-truth actions.</p><div className="action-table" role="table" aria-label="Current 14-component action comparison">{COMPONENTS.map((name, index) => <div role="row" key={name}><span role="cell">{name}</span><code role="cell">{data.plans.plans[0].actions[step][index].toFixed(4)}</code><code role="cell">{data.plans.plans[1].actions[step][index].toFixed(4)}</code></div>)}</div></section>
          {projection.available && <><section><h2>Projection provenance</h2><dl><div><dt>Robot / tool</dt><dd>{projection.robot.robotModelName} / {projection.robot.targetLink}</dd></div><div><dt>Frame</dt><dd>{projection.robot.outputCoordinateFrame}</dd></div><div><dt>FK</dt><dd>{projection.robot.fkImplementationId} {projection.robot.fkImplementationVersion}</dd></div><div><dt>Rotations</dt><dd>unit quaternion XYZW</dd></div><div><dt>Joint limits</dt><dd>{projection.jointLimitPolicy}</dd></div></dl></section><section className="comparison-warning"><h2>Geometry limitations</h2><p>Arm transforms calibrated: <strong>{String(projection.robot.calibratedArmTransforms)}</strong>. Generated gripper targets are raw device-specific unproven values—not physical jaw widths or calibrated openings. Derived paths are not a hardware-safety or executability claim.</p></section>{violations.length > 0 && <section className="comparison-warning" role="alert"><h2>{violations.length} generated joint-limit warning{violations.length === 1 ? "" : "s"}</h2><p>Targets were preserved without clipping under allow-with-recorded-violations.</p></section>}</>}
        </aside>
      </div>
    </div>
  );
}
