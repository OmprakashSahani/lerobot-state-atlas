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

Synthetic manifests are permitted only in test fixtures. They must use
`sourceKind: "synthetic-test"`, set `reconstructionClaim` to `false`, and say
that they are synthetic in their description. They must never be referenced by
the production demo or presented as real reconstruction evidence.

## Available manifests reserved for a later phase

The available branch records provenance, canonical alignment, bounds, and an
integrity-addressed static asset reference. Version 1 reserves SPZ as its single
asset format so that future loaders can reject unknown formats deterministically.
This contract does not implement a loader, parser, renderer, download policy,
WASM module, worker, or cross-origin fetch.

Before an available environment can be enabled in production, a later roadmap
phase must define and test byte and splat-count limits, same-origin path
resolution, bounded fetching, byte-size and SHA-256 verification, malformed
asset handling, device capability behavior, GPU cleanup, and mobile opt-in.
