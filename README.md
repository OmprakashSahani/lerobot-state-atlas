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
- Rotate, zoom, and pan each arm workspace independently.
- Play, pause, and reset trajectory playback.
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
      --output artifacts/episodes-0-1-2-workspace.html

The interactive visualization provides:

- separate left and right 3D panels
- one coloured trajectory per episode and arm
- voxel colours based on raw visit frequency
- coordinate and visit-count hover information
- independent rotation, zoom, and pan
- Play/Pause and Reset controls
- a live frame counter
- playback using the dataset's reported FPS

The visited-voxel heatmap remains visible while the trajectories are progressively drawn.

## Aggregate many episodes

For larger selections, generate a coverage-only heatmap while loading episodes in bounded batches:

    uv run lerobot-state-atlas aggregate-workspace \
      DreamMachines/actuator_unboxing_4h_diverse \
      --urdf .cache/robot-models/trlc-dk1/TRLC-DK1-Follower.urdf \
      --episode-start 0 \
      --episode-end 99 \
      --episode-batch-size 10 \
      --voxel-size 0.02 \
      --output artifacts/episodes-0-99-aggregated-workspace.html

`--episode-end` is inclusive. The command:

- loads at most the configured number of episodes per batch
- computes left and right tool positions for each batch
- incrementally accumulates raw voxel visit counts
- counts the distinct episodes that entered each voxel
- retains aggregate statistics instead of every trajectory point
- writes a self-contained interactive HTML heatmap

The generated heatmap includes a metric selector with three views:

- **Visits** — total dataset frames that entered each voxel
- **Log visits** — `log1p(visits)`, which compresses the dynamic range so rarely and moderately visited regions remain visible alongside heavily visited regions
- **Episode count** — number of distinct selected episodes that entered each voxel

Hover information shows the raw visit count, its `log1p` value, and the distinct episode count for each voxel. Each metric uses its own automatically adjusted colour range. The logarithmic transform affects only the displayed colours; the underlying raw visit counts remain unchanged.

The aggregated view intentionally omits individual trajectory lines and playback. This keeps memory usage and HTML size practical when analysing tens, hundreds, or all dataset episodes.

A real validation using episodes 0 through 9 produced:

- 10 episodes processed across 4 bounded batches
- 5,124 dataset frames
- 10,248 dual-arm tool points
- 1,224 occupied voxels at a `0.020 m` voxel size
- a 4.7 MB self-contained HTML visualization

## Coordinate frames

The left and right panels use their respective local `base_link` frames.

They must not be interpreted as sharing one common world coordinate frame. This note is displayed in both the CLI output and generated interactive HTML.

## Example validation

The interactive workflow was validated using episodes 0, 1, and 2 from:

    DreamMachines/actuator_unboxing_4h_diverse

At a voxel size of `0.020 m`, the generated visualization contained:

- 2,838 trajectory points across both arms
- 423 occupied voxels
- playback at the dataset's native 50 FPS
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

## Current validation status

- 138 tests passed
- Ruff lint passed
- Ruff formatting check passed
- `git diff --check` passed

## License

A project license has not yet been added.
