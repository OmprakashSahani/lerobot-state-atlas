# Environment layer contract

The environment-layer v1.0 contract describes an optional, render-only scene
layer for LeRobot State Atlas. It does not change or extend the browser-data
bundle that provides robot workspace coverage and recorded trajectory data.

The production demo currently exposes this as an intentionally unavailable
capability. It does not fetch an environment manifest or asset. The analytical
grid remains active, and the absence of an environment does not limit coverage,
trajectory playback, orientation and symbolic gripper glyphs, synchronized
media, radius queries, or episode analysis.

## Schema identity

Environment manifests use:

```json
{
  "name": "lerobot-state-atlas.environment-layer",
  "major": 1,
  "minor": 0
}
```

The structural schema is `schemas/environment-layer-v1.schema.json`. The web
validator additionally enforces exact fields, finite numeric values, unit XYZW
quaternions, ordered non-empty bounds, safe single-segment asset filenames,
lowercase SHA-256 metadata, and truthful synthetic provenance.

## Independence from robot analytics

An environment is visual context only. It must not participate in:

- voxel coverage or metric domains;
- trajectory playback or recorded sample selection;
- voxel selection or radius queries;
- uncommon-space episode scoring;
- orientation or gripper glyph semantics;
- synchronized media timing;
- browser-data totals or analytical scene bounds;
- checkpoint comparison.

Runtime arm spacing transforms robot layers only. Environment alignment is an
immutable transform into the right-handed, metre-based
`canonical-shared-world` frame with positive Z up.

## Availability and truthfulness

An unavailable environment is an intentional capability state, not a download
or application failure. No reconstruction or calibrated alignment may be
claimed without a validated real scan, documented provenance, and calibration
evidence.

The public [phone capture guide](/capture-guide) documents environment capture,
calibration evidence, validation, and the evidence gate for
`reconstructionClaim: "documented-real-scan"`. That gate is documented policy,
not a schema-enforced evidence object: environment-layer v1.0 represents the
claim but does not embed or validate its supporting evidence. The production
unavailable state and all synthetic-fixture restrictions below remain unchanged.

Synthetic manifests are permitted only in test fixtures. They must use
`sourceKind: "synthetic-test"`, set `reconstructionClaim` to `false`, and say
that they are synthetic in their description. They must never be referenced by
the production demo or presented as real reconstruction evidence.

### Real-capture feasibility result

A private phone video was processed locally with COLMAP at 4 FPS. Of 259
extracted frames, 227 unique frames registered across four sparse models. The
main model registered 167 images with 68,630 points, 282,087 observations, a
mean track length of 4.110258, and a COLMAP-reported mean reprojection error of
0.852467 px. Visual inspection showed a coherent workcell and plausible camera
path.

COLMAP successfully undistorted the main model's 167 images into a 111 MB
training dataset. A later Nerfstudio Splatfacto 30k run produced a
466,363-Gaussian reconstruction for development evaluation. This does not
establish a calibrated transform to robot-world coordinates.

The private video, extracted frames, databases, sparse models, and undistorted
dataset remain under ignored `.cache/`. Production remains unavailable, and the
immutable `demo-v1` and `demo-v2` bundles are unchanged.

### Real-scan browser representation decision

Development diagnostics compared the original 466,363-splat Nerfstudio PLY and
multiple SPZ reductions through the same Spark renderer settings and exact
captured-camera transforms. At the known-good eval 0 / COLMAP 1 /
`frame_0003.jpg` view (about 28.06 dB), the original PLY rendered coherently
while the old 86,954-splat, one-representative-per-voxel SPZ lost major robot,
tray, cylindrical, and boundary detail. Uniform 250k remained strong across
eval 0 / COLMAP 1, eval 1 / COLMAP 9, and eval 2 / COLMAP 17. The weak eval 8 /
COLMAP 65 view was also weak in the original PLY, and 300k improved it only
slightly over 250k. This separates reduction loss from reconstruction weakness.

The reviewed browser candidate is the existing deterministic uniform sample of
the viable cropped, opacity-filtered, and scale-filtered distribution:

- 250,000 splats;
- 3,990,074 compressed bytes (3.805 MiB), below the unchanged 8 MiB limit;
- SHA-256
  `f90b5ed3d1c672f62e25fca1e6acf9534901f9e8c8c80ff1a6c6b874c1d7386d`;
- SPZ v3, SH0, 12 fractional bits, flags 0;
- deterministic 32-bit hash selection, not spatial one-per-voxel collapse.

The integrated viewer used 30 discarded stabilization frames followed by a
120-frame measurement window. Representative stabilized results from the same
browser and scene were:

| Source | Bytes | Mean / FPS | p50 | p95 | Frames >33.3 ms | Frames >50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform 250k | 3,990,074 | 21.64 ms / 46.2 | 18.70 ms | 37.90 ms | 19 | 1 |
| Uniform 300k | 4,785,949 | 24.43 ms / 40.9 | 17.00 ms | 50.70 ms | 34 | 10 |

The 300k representation was only slightly better visually but had materially
worse frame-time tails; a later stabilized 300k window remained near 49 ms p95.
Uniform 250k is therefore the reviewed quality/performance candidate. This is
evidence for the new 250,000-splat validation ceiling, not evidence that every
250k asset or device will meet a performance target.

The source-switch lifecycle leak found during profiling was fixed by disposing
the `SplatMesh`, `SparkRenderer`, renderer geometry, and renderer material.
Controlled switches now return to a steady `G10 T7 P6`, with final deltas of
`+0` geometries, `+0` textures, and `+0` programs. Development-only exact-camera
and integrated-performance diagnostics remain available for regression work.

Arbitrary orbit views remain visibly poor even for the original PLY because
they leave the supported captured-camera manifold. The viewing constraint below
addresses that display risk; reconstruction-to-robot-world calibration remains
separate future work. The real-scan manifest must keep
`alignment.calibrated: false`, `reconstructionClaim: false`, and its explicit
no-calibration disclosure.

### Supported captured-view policy

When an uncalibrated real-scan environment is visible, the shared viewer camera
uses a captured-view rail instead of unrestricted orbit. The rail is built from
the same 210 COLMAP poses and the already validated COLMAP-to-Nerfstudio camera
conversion used by the exact-camera diagnostic. Camera centers are connected
with the existing four-times-median-nearest-neighbor radius (about 0.1593 in
reconstruction coordinates), which produces one dominant 205-camera component
and one detached five-camera component.

Only the dominant component participates in normal viewing. Its cameras are
ordered by captured source-frame number. Users can move with Previous view,
Next view, or an integer view slider, so every settled production view uses a
captured position, orientation, vertical field of view, principal point, and
near/far projection. Reset returns to the known-good COLMAP image 1 /
`frame_0003.jpg` pose. Auto-rotate and OrbitControls input are unavailable while
the Gaussian is visible; hiding or unloading the Gaussian restores the normal
analytical-scene orbit camera. If the registered-camera file cannot be loaded or
does not reproduce the reviewed 210 = 205 + 5 topology, the environment stays
locked to the known-good fallback pose instead of enabling free orbit.
The browser accepts the camera-support file only at its same-origin approved
path, under its exact 130,912-byte size and reviewed SHA-256
`49077a800713ef38070fd4ec547f73840373b914f548a0bc0e0029362a74f60a`.

The five detached cameras—COLMAP IDs 197 through 201—are excluded. Their small,
disconnected capture segment is not assumed to have support equivalent to the
dominant trajectory. Adding it later requires separate visual evidence and an
explicit policy review.

For engineering inspection, development mode accepts
`?environmentViewingDebug=1`. It reports the current reconstruction-space camera
position, nearest dominant captured camera, positional distance, orientation
difference, and inside/outside classification. The positional diagnostic uses
the same component radius; its orientation tolerance is 1.25 times the p95
direction change between locally connected, source-adjacent dominant cameras
(with a five-degree minimum and twenty-degree ceiling). Normal users do not see
this telemetry.

These constraints describe where the reconstruction has captured-view support;
they are not physical geometry and do not map camera centers into robot-world
coordinates. The existing environment render transform is applied only to place
the captured camera in the shared render scene. It does not calibrate the
Gaussian, alter analytical coordinates, or justify a robot/environment spatial
claim. Reconstruction-to-robot-world calibration remains separate work.

## Desktop-only local renderer spike

The repository contains a compatibility spike pinned to
`@sparkjsdev/spark@2.1.0`. It is disabled in production and cannot change the
production demo's unavailable capability. Development activation requires
`NEXT_PUBLIC_LOCAL_ENVIRONMENT_MANIFEST` to name a canonical manifest below
`/environment-data/__local-synthetic__/` or
`/environment-data/__local-real__/`. Conservative mobile user-agent signals
refuse the spike before a manifest request or Spark import; this is not a
permanent device classifier.

Nothing is requested until **Load environment** is selected. The
application fetches the manifest under a 64 KiB cap with no redirects, validates
environment-layer v1.0 semantics, and resolves its single-segment asset inside
the same directory. It streams the SPZ under an 8 MiB cap, checks declared and
actual byte counts, verifies lowercase SHA-256, and only then starts bounded
gzip header inspection. Spark receives only verified bytes; its URL, stream,
paging, and LOD loading are disabled.

The spike accepts only SPZ v3 with spherical-harmonic degree 0, 1 through
250,000 splats, a zero reserved byte, fractional bits from 8 through 16, and the
basic `0x01` antialias flag. The LOD flag and unknown flags are rejected. Header,
manifest, and decoded counts must agree. The fractional-bit range includes the
generator's fixed value of 12 while avoiding unsafe shifts and extreme
quantization.

Header preflight is not complete semantic validation of every Gaussian. Spark
performs final decoding. Production remains blocked pending decoded finite/range
validation, broader device profiling, deployable packaging for the promoted
asset and camera-support data, and renderer compatibility evidence.

The reviewed real-scan asset is staged without regenerating it:

```bash
cd apps/web
npm run environment:stage:real
```

That command accepts only the reviewed Uniform 250k metadata, exact asset
identity, and exact 210-camera support-data identity. It writes
`omprakash-workcell.spz`, `manifest.json`, and `registered-cameras.json` through
the normal guarded staging path. The diagnostic comparison assets remain
available.

### Deterministic synthetic fixture

```bash
cd apps/web
npm run environment:stage
NEXT_PUBLIC_LOCAL_ENVIRONMENT_MANIFEST=/environment-data/__local-synthetic__/manifest.json npm run dev
```

The staging command writes only `manifest.json` and
`synthetic-environment.spz` beneath the ignored local synthetic root and refuses
a different output root. The application-owned fixture contains a reference
plane, colored axes, rotated anisotropic Gaussians, overlap, and varied opacity.
It is labelled **Synthetic test environment — not a real reconstruction** and
makes no calibration claim.

Hide preserves loaded GPU resources; Show reuses them. Unload and final teardown
explicitly dispose the Spark mesh and renderer. Fetches are abortable and
generation tokens prevent stale attachment. Spark's pooled decoder cannot be
cancelled and may finish after unmount, at which point the stale result is
disposed. Stable 2.1.0 also retains pooled workers/WASM and predates upstream
shared-renderer state fixes.

Manual Chrome testing confirmed that Spark 2.1.0 initializes its embedded WASM
through `fetch(data:application/wasm;base64,...)`. The local development policy
therefore adds `data:` narrowly to `connect-src`; production retains
`connect-src 'self'`. Blob workers execute under `worker-src 'self' blob:`.
No new `unsafe-eval` allowance and no `wasm-unsafe-eval` allowance were added.
The generated fixture root is excluded by both `.gitignore` and
`apps/web/.vercelignore`, and staging tests remove generated assets after use.

Production enablement still requires reconstruction-to-robot-world calibration
evidence, stronger decoded-value validation, worker/cancellation decisions,
broader device performance measurements, deployable asset/camera-data
packaging, CSP proof, and shared-canvas visual acceptance. The promoted
render-only asset and captured-view rail do not change analytical State Atlas
calculations or make the Gaussian environment fully production-ready.
