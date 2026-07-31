# Full-dataset coverage scaling evidence

This record captures the first 100-episode export pilot for roadmap item 5.
It is operational evidence for deciding how to scale coverage export; it is
not a deployed public bundle and does not establish full-dataset scaling.

## Dataset and pilot configuration

| Item | Value |
| --- | ---: |
| Dataset repository | `DreamMachines/actuator_unboxing_4h_diverse` |
| Requested revision | `v3.0` |
| Resolved immutable revision | `e973df866c80f52884cc68355579043cab828e78` |
| Source episodes | 1,344 |
| Source frames | 696,107 |
| Source frame rate | 50 FPS |
| Coverage episodes | 0–99 |
| Selected episodes | 100 |
| Selected frames | 47,455 |
| Trajectory episodes | 0 and 1 |
| Trajectory samples | Episode 0: 515; episode 1: 445 |
| Voxel size | 0.02 m |
| Arm spacing | 0.8 m |
| Video | None |

The pilot output and measurement reports remained outside the repository and
outside all public bundle directories.

## Original batch-size comparison at `63f4783`

| Measurement | Batch size 32 | Batch size 8 |
| --- | ---: | ---: |
| Coverage aggregation | 6.660031835002883 s | 18.42487498599803 s |
| Total export | 13.387987123998755 s | 24.49947216700093 s |
| Peak process RSS | 902,606,848 bytes | 888,926,208 bytes |
| Final occupied entries | 5,154 | 5,154 |
| Final CSR incidence | 17,831 | 17,831 |
| Final raw visits | 94,910 | 94,910 |

Batch size 8 reduced the observed peak RSS by only 13,680,640 bytes while
making coverage aggregation approximately 2.77 times slower. Every generated
bundle file was byte-identical between the two batch sizes. Batch size 32
therefore remains the preferred pilot setting.

## Optimized exporter evidence at `b8b830f`

| Measurement | First optimized run | Same-commit repeat |
| --- | ---: | ---: |
| Coverage aggregation | 6.100260890001664 s | 6.117546312998456 s |
| Total export | 11.731494232000841 s | 11.83615061599994 s |
| Peak process RSS | 871,440,384 bytes | 870,293,504 bytes |

Relative to the original batch-size-32 run, the first optimized run reduced
observed peak RSS by 31,166,464 bytes, or approximately 29.72 MiB. Coverage
aggregation time improved by approximately 8.4%, and total export time improved
by approximately 12.4%.

Peak RSS is obtained from
`resource.getrusage(RUSAGE_SELF).ru_maxrss` and normalized to bytes where the
platform semantics are known. RSS and timings are process-, machine-, and
platform-dependent observations, not universal performance guarantees.

## Final analytical and artifact results

### Coverage totals

| Measurement | Left | Right | Total |
| --- | ---: | ---: | ---: |
| Occupied arm-specific entries | 3,136 | 2,018 | 5,154 |
| Voxel–episode CSR incidence | 9,829 | 8,002 | 17,831 |
| Raw tool-point visits | 47,455 | 47,455 | 94,910 |

The coverage contains 4,746 unique shared grid cells. Arm-specific entries
remain distinct even when left and right use the same integer grid coordinate.

### Artifacts

| File | Uncompressed bytes | Deterministic gzip level-9 bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `coverage.json` | 152,937 | 49,008 | `2e9eefd9cf42824c9ba131309fe549b1e422ea2032be4d6fd9b3f7679d4f7b46` |
| `trajectories.json` | 333,950 | 136,044 | `eb58ee7806fe1c3f1b682be8a03a2594ff53dd523f74202be90f1018baa5d476` |
| `manifest.json` (optimized same-commit export) | 3,169 | 1,708 | `9fa027d581336cbf247dcf43b4e305a572c62438c8cee27028b04fc692f1efb0` |
| **Total** | **490,056** | **186,760** | — |

`coverage.json` and `trajectories.json` remained byte-identical before and
after the exporter optimization. The pre- and post-optimization manifests
differed only because `repositoryHeadCommit` truthfully changed from `63f4783`
to `b8b830f`. Two optimized exports from `b8b830f` produced byte-identical
`coverage.json`, `trajectories.json`, and `manifest.json`. Their measurement
reports intentionally differed because reports contain run-specific timings,
RSS, platform details, and timestamps.

## Growth compared with `demo-v2`

| Measurement | `demo-v2` (10 episodes) | 100-episode pilot | Growth |
| --- | ---: | ---: | ---: |
| Frames | 5,124 | 47,455 | approximately 9.26× |
| Arm-specific entries | 1,224 | 5,154 | approximately 4.21× |
| CSR incidence | 1,717 | 17,831 | approximately 10.39× |
| `coverage.json` bytes | 26,564 | 152,937 | approximately 5.76× |

The trajectory selection remained episodes 0 and 1, so
`trajectories.json` remained approximately the same size. This single
contiguous prefix pilot does not prove linear scaling to the full dataset.

## Reproduction commands

These commands deliberately use replaceable `/tmp` paths. They must not be
redirected into `apps/web/public`. Run the original comparisons from commit
`63f4783` and the optimized repeat from commit `b8b830f`, with the pinned URDF
inputs shown in the main README.

Original batch-size-32 pilot:

```sh
uv run lerobot-state-atlas export-browser-data \
  DreamMachines/actuator_unboxing_4h_diverse \
  --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
  --urdf-upstream-identity .cache/robot-models/trlc-dk1/UPSTREAM_COMMIT \
  --dataset-revision v3.0 \
  --episode-start 0 --episode-end 99 \
  --trajectory-episode 0 --trajectory-episode 1 \
  --episode-batch-size 32 --voxel-size 0.02 --arm-spacing 0.8 \
  --bundle-id pilot-100-batch32 \
  --output /tmp/lerobot-state-atlas-pilot-100-bs32 \
  --measurement-report /tmp/lerobot-state-atlas-pilot-100-bs32-measurement.json
```

Batch-size-8 comparison:

```sh
uv run lerobot-state-atlas export-browser-data \
  DreamMachines/actuator_unboxing_4h_diverse \
  --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
  --urdf-upstream-identity .cache/robot-models/trlc-dk1/UPSTREAM_COMMIT \
  --dataset-revision v3.0 \
  --episode-start 0 --episode-end 99 \
  --trajectory-episode 0 --trajectory-episode 1 \
  --episode-batch-size 8 --voxel-size 0.02 --arm-spacing 0.8 \
  --bundle-id pilot-100-batch32 \
  --output /tmp/lerobot-state-atlas-pilot-100-bs8 \
  --measurement-report /tmp/lerobot-state-atlas-pilot-100-bs8-measurement.json
```

Optimized same-commit repeat (use the same bundle ID for direct byte
comparison between optimized runs):

```sh
uv run lerobot-state-atlas export-browser-data \
  DreamMachines/actuator_unboxing_4h_diverse \
  --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
  --urdf-upstream-identity .cache/robot-models/trlc-dk1/UPSTREAM_COMMIT \
  --dataset-revision v3.0 \
  --episode-start 0 --episode-end 99 \
  --trajectory-episode 0 --trajectory-episode 1 \
  --episode-batch-size 32 --voxel-size 0.02 --arm-spacing 0.8 \
  --bundle-id pilot-100-batch32 \
  --output /tmp/lerobot-state-atlas-pilot-100-optimized-repeat \
  --measurement-report /tmp/lerobot-state-atlas-pilot-100-optimized-repeat-measurement.json
```

Strict validation and byte comparisons:

```sh
uv run lerobot-state-atlas validate-browser-data \
  /tmp/lerobot-state-atlas-pilot-100-optimized-repeat
sha256sum /tmp/lerobot-state-atlas-pilot-100-optimized-repeat/*.json
cmp /tmp/lerobot-state-atlas-pilot-100-optimized/coverage.json \
  /tmp/lerobot-state-atlas-pilot-100-optimized-repeat/coverage.json
cmp /tmp/lerobot-state-atlas-pilot-100-optimized/trajectories.json \
  /tmp/lerobot-state-atlas-pilot-100-optimized-repeat/trajectories.json
cmp /tmp/lerobot-state-atlas-pilot-100-optimized/manifest.json \
  /tmp/lerobot-state-atlas-pilot-100-optimized-repeat/manifest.json
```

## Decision and next gate

- Retain episode batch size 32.
- Preserve exact CSR episode identities and separate arm-specific entries.
- Keep the small trajectory selection independent from coverage selection.
- Do not deploy the 100-episode pilot yet.
- Make frontend large-array hardening and dynamic episode-range labeling the
  next implementation commit.
- After frontend hardening, measure browser behavior before adding a public
  immutable pilot bundle.
- Keep 250-, 500-, and 1,344-episode exports gated on measured exporter and
  browser behavior.
