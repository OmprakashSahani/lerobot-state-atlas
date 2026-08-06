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
training dataset. This establishes sparse-reconstruction and dataset-preparation
feasibility only: no Gaussian Splat has been trained, validated, aligned to the
robot frame, or packaged for the renderer.

The private video, extracted frames, databases, sparse models, undistorted
dataset, and generated assets remain under ignored `.cache/`. Production remains
unavailable, and the immutable `demo-v1` and `demo-v2` bundles are unchanged.

## Desktop-only local renderer spike

The repository contains a compatibility spike pinned to
`@sparkjsdev/spark@2.1.0`. It is disabled in production and cannot change the
production demo's unavailable capability. Development activation requires
`NEXT_PUBLIC_LOCAL_ENVIRONMENT_MANIFEST` to name a canonical manifest below
`/environment-data/__local-synthetic__/`. Conservative mobile user-agent
signals refuse the spike before a manifest request or Spark import; this is not
a permanent device classifier.

Nothing is requested until **Load synthetic environment** is selected. The
application fetches the manifest under a 64 KiB cap with no redirects, validates
environment-layer v1.0 semantics, and resolves its single-segment asset inside
the same directory. It streams the SPZ under an 8 MiB cap, checks declared and
actual byte counts, verifies lowercase SHA-256, and only then starts bounded
gzip header inspection. Spark receives only verified bytes; its URL, stream,
paging, and LOD loading are disabled.

The spike accepts only SPZ v3 with spherical-harmonic degree 0, 1 through
100,000 splats, a zero reserved byte, fractional bits from 8 through 16, and the
basic `0x01` antialias flag. The LOD flag and unknown flags are rejected. Header,
manifest, and decoded counts must agree. The fractional-bit range includes the
generator's fixed value of 12 while avoiding unsafe shifts and extreme
quantization.

Header preflight is not complete semantic validation of every Gaussian. Spark
performs final decoding. Production remains blocked pending decoded finite/range
validation, profiling, and renderer compatibility evidence.

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

Production enablement still requires a validated real scan, provenance and
calibration evidence, stronger decoded-value validation, worker/cancellation
decisions, performance measurements, CSP proof, and shared-canvas visual and
cleanup acceptance.
