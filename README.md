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

The web application reads deterministic browser-data schema v1.0 bundles. It
does not run LeRobot, PyTorch, URDF parsing, or forward kinematics at request
time.

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
      --bundle-id demo-v1 \
      --output apps/web/public/atlas-data/demo-v1

Validate any bundle and its payload checksums:

    uv run lerobot-state-atlas validate-browser-data \
      apps/web/public/atlas-data/demo-v1

The exporter first resolves `--dataset-revision` through the Hugging Face Hub
to a full immutable dataset commit SHA. That SHA is passed explicitly to every
LeRobot metadata and state read. The manifest records the requested ref and
resolved commit separately, along with the dataset metadata codebase version
and installed LeRobot package version.

The exporter writes deterministic JSON into a temporary sibling directory,
validates the complete bundle, and atomically installs it. Coverage and
trajectory payloads are deterministic for identical pinned dataset, URDF,
parameters, and exporter source. No timestamp is stored. Manifest provenance
records the repository HEAD separately from a dirty-working-tree flag, so an
uncommitted exporter is never represented as fully identified by HEAD alone.
The dirty check includes modified, staged, and non-ignored untracked files;
ignored output such as `node_modules` and `.next` is excluded.

The included demo uses episodes 0 through 9 for coverage and episodes 0 and 1
for the optional, currently unloaded trajectory payload. Its files are:

- `manifest.json`: 2,457 bytes uncompressed; 1,343 bytes gzip-compressed
- `coverage.json`: 26,564 bytes uncompressed; 7,638 bytes gzip-compressed
- `trajectories.json`: 139,152 bytes uncompressed; 57,461 bytes gzip-compressed
- total: 168,173 bytes uncompressed; 66,442 bytes gzip-compressed

Compressed sizes use gzip level 9 and may vary slightly with the deployment
CDN. The viewer loads only the manifest and coverage payload.

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
mutated. Any manifest, schema, provenance, coverage, or trajectory change must
be published under a new bundle/version directory and referenced by the
application through that new URL. Development responses use `no-store` so
local schema and bundle iteration cannot reuse an older immutable browser
response.

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
