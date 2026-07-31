# LeRobot State Atlas

Analyze and visualize state-space and tool-workspace coverage in LeRobot datasets.

LeRobot State Atlas loads robot state trajectories, applies forward kinematics using a robot URDF, computes voxelized workspace coverage, and produces static or interactive dual-arm visualizations.

## Features

- Inspect LeRobot dataset metadata without downloading videos.
- Load selected episodes and state components.
- Compute left and right tool trajectories using forward kinematics.
- Measure workspace coverage with configurable 3D voxel sizes.
- Generate static PNG workspace plots.
- Generate self-contained interactive Plotly HTML visualizations.
- Aggregate large episode ranges in bounded memory.
- Compare multiple episodes with separate trajectory colours.
- Hover over voxels to inspect coordinates and visit counts.
- Query a configurable radius around any aggregated workspace voxel.
- Rotate, zoom, and pan interactive 3D workspace views during playback.
- Automatically rotate the shared workspace while trajectories play.
- Play, pause, reset, and seek through trajectory playback.
- Scrub through frames with an interactive playback timeline.
- Animate trajectories using the dataset's native FPS.
- Preserve episode boundaries without drawing false connections.

## Requirements

- Python 3.12 or newer
- `uv`
- A compatible TRLC-DK1 follower URDF
- Access to the selected Hugging Face LeRobot dataset

## Installation

Clone the repository and install its dependencies:

    git clone https://github.com/OmprakashSahani/lerobot-state-atlas.git
    cd lerobot-state-atlas
    uv sync

Display the CLI help:

    uv run lerobot-state-atlas --help

## Inspect a dataset

Inspect metadata without loading video streams:

    uv run lerobot-state-atlas inspect \
      DreamMachines/actuator_unboxing_4h_diverse

The summary includes information such as:

- robot type
- dataset revision
- frames per second
- episode and frame counts
- total duration
- available features
- state component names and dimensions

## Generate a static workspace plot

Create a dual-arm PNG for one episode:

    uv run lerobot-state-atlas visualize-workspace \
      DreamMachines/actuator_unboxing_4h_diverse \
      --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
      --episode 0 \
      --voxel-size 0.02 \
      --output artifacts/episode-0-workspace.png

Combine multiple episodes by repeating `--episode`:

    uv run lerobot-state-atlas visualize-workspace \
      DreamMachines/actuator_unboxing_4h_diverse \
      --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
      --episode 0 \
      --episode 1 \
      --episode 2 \
      --voxel-size 0.02 \
      --output artifacts/episodes-0-1-2-workspace.png

## Generate an interactive workspace heatmap

Create a self-contained offline HTML visualization:

    uv run lerobot-state-atlas interactive-workspace \
      DreamMachines/actuator_unboxing_4h_diverse \
      --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
      --episode 0 \
      --episode 1 \
      --episode 2 \
      --voxel-size 0.02 \
      --arm-spacing 0.8 \
      --output artifacts/episodes-0-1-2-shared-workspace.html

The interactive visualization provides:

- one shared dual-arm 3D world-space scene
- configurable lateral arm-base spacing
- one coloured trajectory per episode and arm
- voxel colours based on raw visit frequency
- transformed world-coordinate and visit-count hover information
- rotation, zoom, and pan while playback continues
- synchronized automatic camera rotation during playback
- an Auto rotate On/Off control
- Play/Pause and Reset controls
- an interactive timeline for seeking to any frame
- automatic pause and resume behavior while scrubbing
- a live frame and elapsed-time display
- playback using the dataset's reported FPS

The visited-voxel heatmap remains visible while the trajectories are progressively drawn. The timeline can be scrubbed while playback is running or paused; playback resumes after release only when it was already running. Automatic rotation pauses during manual camera interaction and resumes from the updated camera position after release.

## Aggregate many episodes

For larger selections, generate a coverage-only heatmap while loading episodes in bounded batches:

    uv run lerobot-state-atlas aggregate-workspace \
      DreamMachines/actuator_unboxing_4h_diverse \
      --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
      --episode-start 0 \
      --episode-end 99 \
      --episode-batch-size 10 \
      --voxel-size 0.02 \
      --arm-spacing 0.8 \
      --output artifacts/episodes-0-99-aggregated-workspace.html

`--episode-end` is inclusive. The command:

- loads at most the configured number of episodes per batch
- computes left and right tool positions for each batch
- applies configurable arm-base transforms after forward kinematics
- voxelizes both arms in one shared world coordinate frame
- incrementally accumulates raw voxel visit counts
- counts the distinct episodes that entered each voxel
- preserves exact per-voxel episode identities for radius queries
- retains aggregate statistics instead of every trajectory point
- writes a self-contained interactive HTML heatmap

The generated heatmap includes a metric selector with three views:

- **Visits** — total dataset frames that entered each voxel
- **Log visits** — `log1p(visits)`, which compresses the dynamic range so rarely and moderately visited regions remain visible alongside heavily visited regions
- **Episode count** — number of distinct selected episodes that entered each voxel

Hover information shows the raw visit count, its `log1p` value, and the distinct episode count for each voxel. Each metric uses its own automatically adjusted colour range. The logarithmic transform affects only the displayed colours; the underlying raw visit counts remain unchanged.

The aggregated heatmap also provides a workspace radius query:

- click any occupied voxel to use its centre as the query centre
- enter a radius in metres
- inspect the number of arm-specific voxel entries whose centres lie inside the radius
- inspect total frame visits and separate left/right arm visit counts
- inspect the exact number of distinct selected episodes represented in the radius
- use **Clear query** to remove the current selection

Radius membership is based on Euclidean distance between voxel centres. In a shared dual-arm query, one dataset frame can contribute one tool-point visit for each arm, so the displayed **frame visits** are arm-specific tool-point visits rather than a deduplicated dataset-frame count.

The default shared-frame placement uses zero arm rotations and positions the arm bases laterally along the world Y axis:

- left base: `(0, +arm_spacing / 2, 0)`
- right base: `(0, -arm_spacing / 2, 0)`

The default `--arm-spacing 0.8` value is configurable. It is an approximate starting assumption rather than a confirmed robot calibration.

The aggregated view intentionally omits individual trajectory lines and playback. This keeps memory usage and HTML size practical when analysing tens, hundreds, or all dataset episodes.

A real validation using episodes 0 through 9 produced:

- 10 episodes processed across 4 bounded batches
- 5,124 dataset frames
- 10,248 dual-arm tool points
- 1,224 arm-specific occupied-voxel entries at a `0.020 m` voxel size
- a shared scene using `0.8 m` provisional lateral arm spacing
- responsive voxel-centred radius queries with exact distinct-episode counts
- a self-contained offline HTML visualization

## Coordinate frames

The `visualize-workspace` command retains separate left and right panels using their respective local `base_link` frames. Those static panels must not be interpreted as sharing one common world coordinate frame.

The `interactive-workspace` and `aggregate-workspace` commands apply configurable rigid arm-base transforms after forward kinematics and before voxelization. Both render the transformed left and right coverage in one shared world-space 3D scene. The interactive command also includes trajectory playback and automatic camera rotation.

The current default shared-frame convention places the bases laterally along world Y with zero roll, pitch, and yaw:

- left base: `(0, +arm_spacing / 2, 0)`
- right base: `(0, -arm_spacing / 2, 0)`

The spacing is configurable through `--arm-spacing`; its default `0.8 m` value is provisional and should not be treated as calibrated physical geometry.

## Example validation

The interactive workflow was validated using episodes 0, 1, and 2 from:

    DreamMachines/actuator_unboxing_4h_diverse

At a voxel size of `0.020 m` and provisional `0.8 m` lateral arm spacing, the generated visualization contained:

- 2,838 trajectory points across both arms
- 423 arm-specific occupied-voxel entries
- one shared dual-arm world-space scene
- playback at the dataset's native 50 FPS
- an interactive frame-seeking timeline with elapsed and total time
- synchronized automatic camera rotation
- manual rotation, zoom, and pan during playback
- a self-contained offline HTML file

## Development

Run the complete test suite:

    uv run pytest

Run linting:

    uv run ruff check .

Check formatting:

    uv run ruff format --check .

Check the Git diff for whitespace errors:

    git diff --check

## Browser-data export

The web application reads deterministic browser-data schema v1 bundles.
Schema v1.1 adds optional synchronized episode-video metadata while remaining
compatible with existing v1.0 bundles. The application does not run LeRobot,
PyTorch, URDF parsing, or forward kinematics at request time.

### Browser viewer features

The statically deployed Next.js viewer keeps the schema v1 bundle immutable and
loads only `manifest.json` followed by `coverage.json` on its initial path.
Opening trajectory playback lazily requests the optional `trajectories.json`;
missing or invalid optional trajectory data is reported without hiding coverage.
When a v1.1 manifest also references `episode-videos.json`, its metadata is
loaded only after trajectories load successfully. Referenced MP4 assets remain
ordinary bundle-relative static files and are requested by the video element
only for the active episode and camera. Missing or invalid optional video data
does not hide the 3D scene or trajectory controls.

Coverage can be coloured by three exact arm-specific metrics:

- **Visits** uses the authoritative raw voxel visit count.
- **Log visits** uses `log1p(raw visits)` for colour only and does not rewrite
  payload values.
- **Distinct episodes** is the exact number of episode IDs in the voxel's CSR
  range.

Each metric has one colour domain computed across both arms. Hiding an arm does
not change that domain. Selected-voxel details always retain raw visits and the
exact CSR episode count.

Clicking a voxel starts an exact shared-world radius query around its centre.
Membership uses Euclidean centre distance and includes arm-specific entries
from both arms, including exact centre matches at zero radius. Results distinguish
arm-specific entries from unique shared grid cells and report left, right, and
total **tool-point visits**. One dataset frame can contribute two tool-point
visits, one per arm. The distinct-episode result is the exact union of stored
CSR episode IDs across all matching entries.

#### Uncommon-space episode exploration

The viewer ranks recorded episodes by how uncommon their reached workspace
entries are within the exported coverage set. This is an exploration aid, not
an anomaly detector. It does not measure task quality, success, usefulness,
safety, or physical novelty.

Scoring uses distinct-episode incidence from the exact per-voxel CSR episode
identities. Left and right arm voxel entries remain separate analytical entries,
including when their integer voxel coordinates match. Raw tool-point visit
counts are a separate metric and do not affect uncommonness scores.

For a scoring scope, define:

- `E` as the number of exported coverage episodes;
- `c_v` as the number of distinct exported coverage episodes represented in
  arm-specific voxel entry `v`; and
- `V_i,S` as the unique arm-specific entries in scope `S` whose CSR identity
  range contains episode `i`.

For `E > 1`, entry rarity is:

    r(v) = ln(E / c_v) / ln(E)

For `E <= 1`, `r(v) = 0`. Episode uncommonness is the arithmetic mean of
`r(v)` over the entries in `V_i,S`:

    U(i,S) = mean(r(v) for v in V_i,S)

The viewer displays `100 × U(i,S)`, normalized to a 0–100 range. It is not a
probability or percentile. The touched-entry evidence count is shown separately.
Averaging prevents entry breadth or episode length alone from dominating the
score.

**Entire coverage** is the default scope. **Selected radius** uses the exact
entry identities returned by the shared-world radius query. Runtime arm spacing
never changes an entry's rarity, although it may change which entries
geometrically fall inside a selected radius. Hiding an arm does not alter the
analytical scope.

Results use deterministic ordering: score descending, then arm-specific entries
touched descending, then episode ID ascending. Coverage scoring requires no
trajectory load or additional initial request. Trajectory availability is
checked lazily from the existing optional payload. Episodes absent from that
payload remain valid coverage evidence and are labelled coverage-only; no
trajectory, optional state, or video is synthesized. Orientation and raw
gripper data continue to match only the selected exported trajectory episode.

Scores describe only the episodes and voxelization in the exported bundle. They
do not establish rarity in the full source dataset, future recordings,
production behavior, or the physical workspace generally. Results depend on
voxel size, selected radius, episode selection, robot model, forward kinematics,
and shared-world transforms. A high score with little evidence should be read
together with its touched-entry count. With one exported coverage episode,
relative uncommonness is unavailable and scores are defined as zero.

The scores are derived client-side from the existing coverage CSR data, so no
schema revision was required. Computation remains linear in the relevant CSR
membership for the current bundle. Full-dataset scaling is a separate future
roadmap phase and is not claimed by this feature.

Playback keeps both tool points synchronized, uses the validated dataset FPS,
and advances from elapsed wall-clock time with selectable speed rather than
advancing one dataset frame per browser render. Episode changes restart safely;
the timeline supports immediate scrubbing, restart, and optional looping while
coverage remains visible.

Optional episode video follows that same playback state; it does not introduce
a second timeline or native media controls. The current atlas sample timestamp
is mapped relative to the trajectory episode start and into the selected
camera's declared media interval. Play, pause, speed, episode changes, restart,
scrubbing, and loop state remain controlled by the atlas playback controls.

The viewer also supports OrbitControls-based automatic rotation and manual
orbit, pan, zoom, and camera reset. Runtime arm spacing is a display and query
configuration relative to the exported baseline: the left arm receives
`+(spacing - manifestSpacing) / 2` on world Y and the right arm receives the
negative delta. The same adjustment applies to coverage, base references,
selection queries, and trajectories. The manifest's current `0.8 m` baseline is
provisional configurable geometry, not calibrated geometry.

Gaussian-splat environments, scan upload, and calibration UI remain future
work. The environment layer is intentionally independent of coverage,
selection, and playback so those additions do not require coupling robotics
data to a future environment renderer.

Generate the pinned demo bundle:

    uv run lerobot-state-atlas export-browser-data \
      DreamMachines/actuator_unboxing_4h_diverse \
      --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
      --urdf-upstream-identity .cache/robot-models/trlc-dk1/UPSTREAM_COMMIT \
      --dataset-revision v3.0 \
      --episode-start 0 \
      --episode-end 9 \
      --trajectory-episode 0 \
      --trajectory-episode 1 \
      --episode-batch-size 4 \
      --voxel-size 0.02 \
      --arm-spacing 0.8 \
      --bundle-id demo-v2 \
      --output apps/web/public/atlas-data/demo-v2

For an explicitly measured export, add a report path outside the immutable
bundle directory:

    --measurement-report artifacts/demo-v2-export-measurement.json

The standalone measurement report records export-stage timings, best-effort
peak process RSS, per-batch coverage growth, and installed artifact sizes.
Timings and RSS are machine- and platform-dependent operational evidence. The
report is not referenced by the manifest, and enabling it does not change the
deterministic bundle files, payload checksums, or browser-data contract.

To package synchronized MP4 media into a new schema v1.2 bundle, also provide
both of these options:

    --episode-video-metadata /path/to/episode-videos.json \
    --episode-video-media-root /path/to/media-root

The options are paired: neither may be supplied alone. Every bundle-relative
media filename declared by the metadata is resolved beneath
`--episode-video-media-root`; for example, a declared
`media/episode-000000/top.mp4` resolves to
`/path/to/media-root/media/episode-000000/top.mp4`.

Validate any bundle and its payload checksums:

    uv run lerobot-state-atlas validate-browser-data \
      apps/web/public/atlas-data/demo-v2

The exporter first resolves `--dataset-revision` through the Hugging Face Hub
to a full immutable dataset commit SHA. That SHA is passed explicitly to every
LeRobot metadata and state read. The manifest records the requested ref and
resolved commit separately, along with the dataset metadata codebase version
and installed LeRobot package version.

The exporter writes deterministic JSON into a temporary sibling directory,
copies declared MP4 files there, validates the complete bundle, and atomically
installs it. Coverage, trajectory, and episode-video metadata payloads are
deterministic for identical pinned inputs. Validation checks every packaged
MP4 against its declared positive byte size and SHA-256 checksum. No timestamp
is stored. Manifest provenance records the repository HEAD separately from a
dirty-working-tree flag, so an uncommitted exporter is never represented as
fully identified by HEAD alone. The dirty check includes modified, staged, and
non-ignored untracked files; ignored output such as `node_modules` and `.next`
is excluded.

The deployed `demo-v2` bundle uses schema v1.2 and includes episodes 0 through
9 for coverage and episodes 0 and 1 for the optional, initially unloaded
trajectory payload. Its recorded trajectory state includes end-effector XYZW
unit quaternions and raw gripper values. The gripper values are device-specific:
physical jaw width is not calibrated, and open/closed polarity is not
established. `demo-v2` contains no episode-video metadata or MP4 files. Its
files are:

- `manifest.json`: 2,900 bytes uncompressed; 1,534 bytes gzip-compressed
- `coverage.json`: 26,564 bytes uncompressed; 7,652 bytes gzip-compressed
- `trajectories.json`: 333,950 bytes uncompressed; 136,377 bytes gzip-compressed
- total: 363,414 bytes uncompressed; 145,563 bytes gzip-compressed

Compressed sizes use gzip level 9 and may vary slightly with the deployment
CDN. The viewer loads only the manifest and coverage payload initially. The
previously deployed `demo-v1` remains immutable and was not replaced or
mutated. Private or gated dataset media must not be published without permission;
access tokens and gated remote URLs must never be embedded in a browser bundle.

## Web application

The production-oriented Next.js application lives in `apps/web` and remains
self-contained:

    cd apps/web
    npm ci
    npm run lint
    npm run typecheck
    npm test
    npm run build

Run it locally with:

    npm run dev

Then open `http://localhost:3000`, `/methodology`, or `/viewer/demo`.

For Vercel, import this Git repository and set the project root directory to
`apps/web`. No environment variables, API routes, Python runtime, dataset
download, or data-generation build step is required. Versioned atlas payloads
receive immutable cache headers; static routes receive practical security
headers from `next.config.ts`.

Production atlas bundle directories are immutable. Once a path such as
`/atlas-data/demo-v1/` has been deployed, its files must never be replaced or
mutated. Any manifest, schema, provenance, coverage, trajectory, episode-video
metadata, or media change must be published under a new bundle/version
directory and referenced by the application through that new URL. Development
responses use `no-store` so local schema and bundle iteration cannot reuse an
older immutable browser response.

## Current validation status

- 168 Python tests passed
- 15 frontend tests passed
- Ruff lint passed
- Ruff formatting check passed
- Frontend lint and TypeScript checks passed
- Next.js production build passed
- `git diff --check` passed

## License

See [LICENSE](LICENSE).
