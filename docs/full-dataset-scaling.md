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

### Local browser measurement

Stage the existing optimized pilot into the ignored development-only bundle
path. This replaces only `__local-benchmark__`; it does not modify either
immutable demo:

```sh
cd apps/web
npm run benchmark:stage -- /tmp/lerobot-state-atlas-pilot-100-optimized
```

Run the local viewer with the staged bundle and opt-in timing probe:

```sh
NEXT_PUBLIC_ATLAS_BUNDLE_BASE=/atlas-data/__local-benchmark__ \
NEXT_PUBLIC_ATLAS_ENABLE_BENCHMARKS=1 \
npm run dev
```

For a local production-build measurement, use:

```sh
NEXT_PUBLIC_ATLAS_BUNDLE_BASE=/atlas-data/__local-benchmark__ \
NEXT_PUBLIC_ATLAS_ENABLE_BENCHMARKS=1 \
npm run build
```

The default remains `/atlas-data/demo-v2` when the bundle-base override is
absent. After the viewer finishes loading, inspect the structured report in
browser developer tools:

```js
window.__LEROBOT_STATE_ATLAS_BENCHMARK__
```

The automated report times manifest fetch/parse/validation, coverage
fetch/parse/validation, coverage preparation, global uncommon scoring, and
radius query plus local scoring at 0, 0.05, and 0.30 metres. It does not load
trajectories or video. It also does not measure GPU frame rate or the complete
network-user experience.

For each browser and build mode, manually record:

- decoded JavaScript heap after coverage preparation;
- Three.js instance upload cost;
- normal orbit frame rate after the scene settles;
- metric-switch latency;
- complete ranking DOM and layout behavior; and
- keyboard navigation, accessible names, and screen-reader behavior for the
  complete ranking.

### Measured 100-episode browser evidence

The browser gate was measured on 2026-07-31 with Chrome 150.0.0.0 on Windows
10 x64. The viewer used the local Next.js development server with DevTools
open, bundle base `/atlas-data/__local-benchmark__`, and bundle ID
`pilot-100-batch32`. The bundle contained 100 coverage episodes, 5,154
arm-specific entries, and 17,831 CSR incidences. The complete 100-episode
ranking was rendered.

#### Automated benchmark

| Operation | Duration | Result size |
| --- | ---: | ---: |
| Manifest load, JSON parse, and validation | 62.2 ms | — |
| Coverage load, JSON parse, and validation | 48.2 ms | — |
| Coverage preparation | 3.1 ms | 5,154 arm-specific entries |
| Global uncommon scoring | 9.5 ms | 100 ranked episodes |

| Radius | Matched entries | Ranked episodes | Radius query | Local uncommon scoring |
| ---: | ---: | ---: | ---: | ---: |
| 0 m | 1 | 1 | 1.7 ms | 4.0 ms |
| 0.05 m | 19 | 5 | 4.1 ms | 3.1 ms |
| 0.30 m | 2,363 | 100 | 1.8 ms | 7.8 ms |

These are local development-mode timings for the existing CPU operations. They
do not represent production delivery or network-user performance.

#### Manual browser observations

| Observation | Snapshot |
| --- | ---: |
| Idle frame rate | 60.0 FPS |
| Continuous-orbit frame rate | approximately 55.2 FPS |
| GPU raster | On |
| GPU memory used | approximately 6.1 MB |
| JavaScript heap | approximately 20.1 MB |
| Idle CPU | approximately 16.3% |
| Performance Monitor DOM nodes | 3,847 |
| Later `document.querySelectorAll("*")` count | 1,836 |

The two DOM-node counts came from different DevTools and runtime snapshots and
must not be compared as if they were identical measurements. The heap and DOM
graphs appeared stable during the observed idle interval. Idle and continuous
orbit rendering were visually responsive in this development-mode observation.

Five approximate measurements from metric switch until two animation frames
had completed were:

| Switch | Duration |
| --- | ---: |
| Visits → episodes | 374.5 ms |
| Episodes → visits | 314.1 ms |
| Visits → episodes | 297.0 ms |
| Episodes → visits | 328.1 ms |
| Visits → episodes | 336.9 ms |
| **Average** | **330.1 ms** |
| **Median** | **328.1 ms** |
| **Range** | **297.0–374.5 ms** |

The measured automated CPU operations were small relative to the approximately
330 ms metric-switch-to-paint observation. This record does not diagnose that
delay; doing so requires a proper browser Performance trace.

The ranking was found with the accessible label
`Uncommon-space episode ranking for entire coverage`. It contained 100 list
items. Its `scrollHeight` and `clientHeight` were both 14,554 px, demonstrating
that all 100 items were laid out and the ranking was not virtualized. The full
DOM ranking is acceptable at 100 episodes, but remains a concrete scaling risk
for 1,344 episodes.

An accidental `requestAnimation is not defined` console error during the
session came from a manually mistyped measurement snippet, not application
code.

The 100-episode pilot successfully loads, prepares, scores, queries, ranks, and
renders, so the 100-episode browser gate is **passed**. These observations do
not establish linear full-dataset scaling. DevTools, development mode, HMR,
hardware, viewport size, and Three.js continuous rendering all affect the
reported values. The automated probe also does not measure GPU frame rate or
the complete network-user experience.

- Retain episode batch size 32.
- Preserve exact CSR episode identities and separate arm-specific entries.
- Keep the small trajectory selection independent from coverage selection.
- Do not deploy the 100-episode pilot yet.
- Coverage min/max preparation no longer spreads payload-sized arrays into
  function arguments, and visible episode labeling now derives from the
  manifest selection.
- Introduce a dedicated Episode Analysis panel for scoring scope and ranking in
  the next implementation commit; this measurement does not implement it.
- Measure the revised layout again with 100 episodes.
- If that evidence remains acceptable, make a measured 250-episode export and
  local browser run the next dataset gate.
- Preserve staged growth: measure 250 episodes, then 500 episodes, then the full
  1,344 episodes only while exporter and browser evidence remains acceptable.
