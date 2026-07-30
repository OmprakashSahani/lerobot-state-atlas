import json
from math import isfinite
from pathlib import Path, PurePosixPath

import typer
from rich.console import Console
from rich.table import Table

from lerobot_state_atlas.aggregation import (
    aggregate_workspace_coverages,
)
from lerobot_state_atlas.browser_data import (
    export_browser_data,
    validate_browser_data,
)
from lerobot_state_atlas.coverage import (
    compute_workspace_coverage,
)
from lerobot_state_atlas.dataset import load_dataset_summary
from lerobot_state_atlas.interactive import (
    save_interactive_workspace_coverage_heatmap,
    save_interactive_workspace_heatmap,
)
from lerobot_state_atlas.schema import DatasetSummary
from lerobot_state_atlas.state import load_state_batch
from lerobot_state_atlas.trajectory import (
    build_trlc_dk1_joint_component_map,
    compute_tool_trajectory,
)
from lerobot_state_atlas.transforms import (
    RigidTransform,
    transform_tool_trajectory,
)
from lerobot_state_atlas.urdf import load_robot_model
from lerobot_state_atlas.visualization import (
    save_workspace_plot,
)


app = typer.Typer(
    name="lerobot-state-atlas",
    help=(
        "Analyze state coverage, trajectories, and reset "
        "consistency in LeRobot datasets."
    ),
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.callback()
def callback() -> None:
    """Analyze LeRobot dataset state coverage and consistency."""


@app.command()
def version() -> None:
    """Display the installed application version."""
    console.print("lerobot-state-atlas 0.1.0")


def display_dataset_summary(summary: DatasetSummary) -> None:
    """Display a dataset summary in the terminal."""
    overview = Table(
        title="Dataset Overview",
        show_header=False,
    )
    overview.add_column("Field", style="bold")
    overview.add_column("Value")

    overview.add_row("Repository", summary.repo_id)
    overview.add_row("Requested revision", summary.requested_revision)
    overview.add_row("Resolved revision", summary.resolved_revision)
    overview.add_row(
        "LeRobot codebase version",
        summary.lerobot_codebase_version,
    )
    overview.add_row(
        "Robot type",
        summary.robot_type or "Unknown",
    )
    overview.add_row("FPS", f"{summary.fps:g}")
    overview.add_row(
        "Episodes",
        f"{summary.total_episodes:,}",
    )
    overview.add_row(
        "Frames",
        f"{summary.total_frames:,}",
    )
    overview.add_row(
        "Tasks",
        f"{summary.total_tasks:,}",
    )
    overview.add_row(
        "Duration",
        (f"{summary.total_duration_seconds / 3600:.2f} hours"),
    )

    console.print(overview)

    features = Table(title="Dataset Features")
    features.add_column("Feature", style="bold")
    features.add_column("Data type")
    features.add_column("Shape")
    features.add_column("Components")

    for feature in summary.features:
        shape = " × ".join(str(dimension) for dimension in feature.shape)
        components = (
            ", ".join(feature.component_names)
            if feature.component_names is not None
            else "—"
        )

        features.add_row(
            feature.name,
            feature.dtype,
            shape,
            components,
        )

    console.print(features)


def _state_component_names(
    summary: DatasetSummary,
) -> tuple[str, ...]:
    for feature in summary.features:
        if feature.name != "observation.state":
            continue

        if feature.component_names is None:
            raise ValueError("observation.state does not define component names.")

        return feature.component_names

    raise ValueError("Dataset does not define observation.state.")


@app.command("inspect")
def inspect_dataset(
    repo_id: str = typer.Argument(
        ...,
        help=("Hugging Face repository ID of the LeRobot dataset."),
    ),
) -> None:
    """Inspect LeRobot dataset metadata without loading videos."""
    try:
        console.print(f"Loading metadata for [bold]{repo_id}[/bold]...")
        summary = load_dataset_summary(repo_id)
    except Exception as error:
        console.print(f"[red]Failed to inspect dataset:[/red] {error}")
        raise typer.Exit(code=1) from error

    display_dataset_summary(summary)


@app.command("visualize-workspace")
def visualize_workspace(
    repo_id: str = typer.Argument(
        ...,
        help=("Hugging Face repository ID of the LeRobot dataset."),
    ),
    urdf_path: Path = typer.Option(
        ...,
        "--urdf",
        help="Path to the TRLC-DK1 follower URDF.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    episode: list[int] = typer.Option(
        [0],
        "--episode",
        "-e",
        help=(
            "Episode index to visualize. Repeat this option "
            "to combine multiple episodes."
        ),
    ),
    voxel_size: float = typer.Option(
        0.02,
        "--voxel-size",
        help="Workspace voxel edge length in metres.",
    ),
    output_path: Path = typer.Option(
        Path("workspace.png"),
        "--output",
        "-o",
        help="Destination PNG path.",
    ),
) -> None:
    """Generate a dual-arm tool workspace PNG."""
    try:
        if not episode:
            raise ValueError("At least one episode must be selected.")

        if any(index < 0 for index in episode):
            raise ValueError("Episode index must be nonnegative.")

        if len(set(episode)) != len(episode):
            raise ValueError("Episode indices must be unique.")

        if not isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("Voxel size must be finite and greater than zero.")

        console.print(f"Loading metadata for [bold]{repo_id}[/bold]...")
        summary = load_dataset_summary(repo_id)
        component_names = _state_component_names(summary)

        episode_text = ", ".join(str(index) for index in episode)
        episode_label = "episode" if len(episode) == 1 else "episodes"

        console.print(f"Loading {episode_label} [bold]{episode_text}[/bold]...")
        batch = load_state_batch(
            repo_id,
            episodes=episode,
            revision=summary.resolved_revision,
        )

        console.print(f"Loading robot model from [bold]{urdf_path}[/bold]...")
        model = load_robot_model(urdf_path)

        trajectories = tuple(
            compute_tool_trajectory(
                batch.states,
                component_names,
                model,
                build_trlc_dk1_joint_component_map(arm),
                arm=arm,
                episode_indices=batch.episode_indices,
            )
            for arm in ("left", "right")
        )

        coverages = tuple(
            compute_workspace_coverage(
                trajectory,
                voxel_size=voxel_size,
            )
            for trajectory in trajectories
        )

        result = save_workspace_plot(
            trajectories,
            output_path,
            coverages=coverages,
            title=(
                f"TRLC-DK1 Episode {episode_text} Tool Workspace"
                if len(episode) == 1
                else (f"TRLC-DK1 Episodes {episode_text} Tool Workspace")
            ),
        )
    except Exception as error:
        console.print(f"[red]Failed to visualize workspace:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"Saved workspace plot to [bold]{result.output_path}[/bold]")
    console.print(
        f"Plotted {result.num_points:,} points "
        f"across {result.num_trajectories} "
        "trajectories."
    )
    console.print(f"Voxel size: {result.voxel_size:.3f} m")
    console.print(
        "[yellow]Coordinate note:[/yellow] "
        "left and right panels use their respective "
        "local base_link frames."
    )


@app.command("interactive-workspace")
def interactive_workspace(
    repo_id: str = typer.Argument(
        ...,
        help=("Hugging Face repository ID of the LeRobot dataset."),
    ),
    urdf_path: Path = typer.Option(
        ...,
        "--urdf",
        help="Path to the TRLC-DK1 follower URDF.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    episode: list[int] = typer.Option(
        [0],
        "--episode",
        "-e",
        help=(
            "Episode index to visualize. Repeat this option "
            "to combine multiple episodes."
        ),
    ),
    voxel_size: float = typer.Option(
        0.02,
        "--voxel-size",
        help="Workspace voxel edge length in metres.",
    ),
    arm_spacing: float = typer.Option(
        0.8,
        "--arm-spacing",
        help=(
            "Lateral distance in metres between the left and right "
            "arm bases in the shared world frame."
        ),
    ),
    output_path: Path = typer.Option(
        Path("workspace-heatmap.html"),
        "--output",
        "-o",
        help="Destination interactive HTML path.",
    ),
) -> None:
    """Generate an interactive dual-arm workspace heatmap."""
    try:
        if not episode:
            raise ValueError("At least one episode must be selected.")

        if any(index < 0 for index in episode):
            raise ValueError("Episode index must be nonnegative.")

        if len(set(episode)) != len(episode):
            raise ValueError("Episode indices must be unique.")

        if not isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("Voxel size must be finite and greater than zero.")

        if not isfinite(arm_spacing) or arm_spacing <= 0.0:
            raise ValueError("Arm spacing must be finite and greater than zero.")

        console.print(f"Loading metadata for [bold]{repo_id}[/bold]...")
        summary = load_dataset_summary(repo_id)
        component_names = _state_component_names(summary)

        episode_text = ", ".join(str(index) for index in episode)
        episode_label = "episode" if len(episode) == 1 else "episodes"

        console.print(f"Loading {episode_label} [bold]{episode_text}[/bold]...")
        batch = load_state_batch(
            repo_id,
            episodes=episode,
            revision=summary.resolved_revision,
        )

        console.print(f"Loading robot model from [bold]{urdf_path}[/bold]...")
        model = load_robot_model(urdf_path)

        local_trajectories = tuple(
            compute_tool_trajectory(
                batch.states,
                component_names,
                model,
                build_trlc_dk1_joint_component_map(arm),
                arm=arm,
                episode_indices=batch.episode_indices,
            )
            for arm in ("left", "right")
        )

        half_spacing = arm_spacing / 2.0
        arm_transforms = {
            "left": RigidTransform(
                translation_xyz=(0.0, half_spacing, 0.0),
            ),
            "right": RigidTransform(
                translation_xyz=(0.0, -half_spacing, 0.0),
            ),
        }

        trajectories = tuple(
            transform_tool_trajectory(
                trajectory,
                arm_transforms[trajectory.arm],
            )
            for trajectory in local_trajectories
        )

        coverages = tuple(
            compute_workspace_coverage(
                trajectory,
                voxel_size=voxel_size,
                voxel_origin_xyz=(0.0, 0.0, 0.0),
            )
            for trajectory in trajectories
        )

        result = save_interactive_workspace_heatmap(
            trajectories,
            output_path,
            coverages=coverages,
            playback_fps=summary.fps,
            shared_space=True,
            title=(
                f"TRLC-DK1 Episode {episode_text} Shared Interactive Workspace Heatmap"
                if len(episode) == 1
                else (
                    f"TRLC-DK1 Episodes {episode_text} "
                    "Shared Interactive Workspace Heatmap"
                )
            ),
        )
    except Exception as error:
        console.print(f"[red]Failed to generate interactive workspace:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(
        f"Saved interactive workspace heatmap to [bold]{result.output_path}[/bold]"
    )
    console.print(
        f"Plotted {result.num_points:,} points "
        f"across {result.num_trajectories} trajectories."
    )
    console.print(f"Occupied voxels: {result.occupied_voxels:,}")
    console.print(f"Voxel size: {result.voxel_size:.3f} m")
    console.print(f"Arm base spacing: {arm_spacing:.3f} m")
    console.print(
        "[yellow]Coordinate note:[/yellow] "
        "left and right arms are shown in one shared world frame."
    )


@app.command("aggregate-workspace")
def aggregate_workspace(
    repo_id: str = typer.Argument(
        ...,
        help="Hugging Face repository ID of the LeRobot dataset.",
    ),
    urdf_path: Path = typer.Option(
        ...,
        "--urdf",
        help="Path to the TRLC-DK1 follower URDF.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    episode_start: int = typer.Option(
        0,
        "--episode-start",
        help="First episode index to aggregate.",
    ),
    episode_end: int = typer.Option(
        ...,
        "--episode-end",
        help="Last episode index to aggregate, inclusive.",
    ),
    episode_batch_size: int = typer.Option(
        32,
        "--episode-batch-size",
        help="Maximum number of episodes loaded in each batch.",
    ),
    voxel_size: float = typer.Option(
        0.02,
        "--voxel-size",
        help="Workspace voxel edge length in metres.",
    ),
    arm_spacing: float = typer.Option(
        0.8,
        "--arm-spacing",
        help=(
            "Lateral distance in metres between the left and right "
            "arm bases in the shared world frame."
        ),
    ),
    output_path: Path = typer.Option(
        Path("aggregated-workspace.html"),
        "--output",
        "-o",
        help="Destination interactive HTML path.",
    ),
) -> None:
    """Aggregate a large episode range into a workspace heatmap."""
    try:
        if episode_start < 0:
            raise ValueError("Episode start must be nonnegative.")

        if episode_end < 0:
            raise ValueError("Episode end must be nonnegative.")

        if episode_end < episode_start:
            raise ValueError(
                "Episode end must be greater than or equal to episode start."
            )

        if episode_batch_size <= 0:
            raise ValueError("Episode batch size must be greater than zero.")

        if not isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("Voxel size must be finite and greater than zero.")

        if not isfinite(arm_spacing) or arm_spacing <= 0.0:
            raise ValueError("Arm spacing must be finite and greater than zero.")

        console.print(f"Loading metadata for [bold]{repo_id}[/bold]...")
        summary = load_dataset_summary(repo_id)

        if episode_end >= summary.total_episodes:
            raise ValueError(
                "Episode end must be less than the dataset "
                f"episode count ({summary.total_episodes})."
            )

        component_names = _state_component_names(summary)
        episodes = tuple(
            range(
                episode_start,
                episode_end + 1,
            )
        )

        console.print(f"Loading robot model from [bold]{urdf_path}[/bold]...")
        model = load_robot_model(urdf_path)

        console.print(
            "Aggregating episodes "
            f"[bold]{episode_start}-{episode_end}[/bold] "
            f"in batches of {episode_batch_size}..."
        )

        half_spacing = arm_spacing / 2.0
        arm_transforms = {
            "left": RigidTransform(
                translation_xyz=(0.0, half_spacing, 0.0),
            ),
            "right": RigidTransform(
                translation_xyz=(0.0, -half_spacing, 0.0),
            ),
        }

        aggregation = aggregate_workspace_coverages(
            repo_id,
            episodes,
            component_names=component_names,
            model=model,
            voxel_size=voxel_size,
            episode_batch_size=episode_batch_size,
            arm_transforms=arm_transforms,
            revision=summary.resolved_revision,
        )

        result = save_interactive_workspace_coverage_heatmap(
            aggregation.coverages,
            output_path,
            title=(
                f"TRLC-DK1 Episodes {episode_start}-{episode_end} "
                "Shared Workspace Heatmap"
            ),
            shared_space=True,
        )
    except Exception as error:
        console.print(f"[red]Failed to aggregate workspace:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(
        f"Saved aggregated workspace heatmap to [bold]{result.output_path}[/bold]"
    )
    console.print(
        f"Aggregated {aggregation.num_episodes:,} episodes "
        f"across {aggregation.num_batches:,} batches."
    )
    console.print(
        f"Processed {aggregation.num_frames:,} dataset frames "
        f"and {result.num_points:,} dual-arm tool points."
    )
    console.print(f"Occupied voxels: {result.occupied_voxels:,}")
    console.print(f"Voxel size: {result.voxel_size:.3f} m")
    console.print(f"Arm base spacing: {arm_spacing:.3f} m")
    console.print(
        "[yellow]Coordinate note:[/yellow] "
        "left and right arms are shown in one shared world frame."
    )


def _episode_video_inputs(
    metadata_path: Path | None,
    media_root: Path | None,
) -> tuple[dict | None, dict[str, Path] | None]:
    if (metadata_path is None) != (media_root is None):
        raise ValueError(
            "--episode-video-metadata and --episode-video-media-root "
            "must be provided together."
        )

    if metadata_path is None or media_root is None:
        return None, None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Episode-video metadata must be valid JSON.") from error

    if not isinstance(payload, dict):
        raise ValueError("Episode-video metadata must be a JSON object.")

    try:
        sources = [
            source for episode in payload["episodes"] for source in episode["videos"]
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Episode-video metadata must contain episode video sources."
        ) from error

    media: dict[str, Path] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(
            source.get("filename"),
            str,
        ):
            raise ValueError("Each episode-video source must contain a filename.")

        filename = source["filename"]
        relative = PurePosixPath(filename)
        if (
            "\\" in filename
            or "://" in filename
            or filename.startswith("//")
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != filename
        ):
            raise ValueError(
                "Episode-video filenames must be safe bundle-relative POSIX paths."
            )

        if filename in media:
            raise ValueError("Episode-video media filenames must be unique.")

        media[filename] = media_root.joinpath(*relative.parts)

    return payload, media


@app.command("export-browser-data")
def export_browser_data_command(
    repo_id: str = typer.Argument(
        ...,
        help="Hugging Face repository ID of the LeRobot dataset.",
    ),
    urdf_path: Path = typer.Option(
        ...,
        "--urdf",
        help="Path to the pinned TRLC-DK1 follower URDF.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    urdf_upstream_identity_path: Path = typer.Option(
        ...,
        "--urdf-upstream-identity",
        help="Text file containing the pinned upstream URDF commit identity.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    dataset_revision: str | None = typer.Option(
        None,
        "--dataset-revision",
        help=(
            "Dataset tag, branch, or commit to resolve before export. "
            "Defaults to LeRobot's normal codebase-version ref."
        ),
    ),
    episode_start: int = typer.Option(
        0,
        "--episode-start",
        help="First coverage episode index.",
    ),
    episode_end: int = typer.Option(
        ...,
        "--episode-end",
        help="Last coverage episode index, inclusive.",
    ),
    trajectory_episode: list[int] = typer.Option(
        [],
        "--trajectory-episode",
        help="Episode to include in the optional trajectory payload. Repeatable.",
    ),
    episode_video_metadata_path: Path | None = typer.Option(
        None,
        "--episode-video-metadata",
        help=(
            "Optional v1.1 episode-video metadata JSON. Requires "
            "--episode-video-media-root."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    episode_video_media_root: Path | None = typer.Option(
        None,
        "--episode-video-media-root",
        help=(
            "Directory containing MP4 paths declared by the optional "
            "episode-video metadata."
        ),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    episode_batch_size: int = typer.Option(
        32,
        "--episode-batch-size",
        help="Maximum number of coverage episodes loaded per batch.",
    ),
    voxel_size: float = typer.Option(
        0.02,
        "--voxel-size",
        help="Workspace voxel edge length in metres.",
    ),
    arm_spacing: float = typer.Option(
        0.8,
        "--arm-spacing",
        help="Provisional lateral distance between arm bases in metres.",
    ),
    bundle_id: str = typer.Option(
        ...,
        "--bundle-id",
        help="Stable identifier for this browser-data bundle.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination browser-data directory.",
    ),
) -> None:
    """Export deterministic, versioned data for the browser viewer."""
    try:
        if episode_start < 0:
            raise ValueError("Episode start must be nonnegative.")
        if episode_end < episode_start:
            raise ValueError(
                "Episode end must be greater than or equal to episode start."
            )
        if episode_batch_size <= 0:
            raise ValueError("Episode batch size must be greater than zero.")
        if not isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("Voxel size must be finite and greater than zero.")
        if not isfinite(arm_spacing) or arm_spacing <= 0:
            raise ValueError("Arm spacing must be finite and greater than zero.")
        if any(value < 0 for value in trajectory_episode):
            raise ValueError("Trajectory episode indices must be nonnegative.")
        if len(set(trajectory_episode)) != len(trajectory_episode):
            raise ValueError("Trajectory episode indices must be unique.")

        episode_video_payload, episode_video_media = _episode_video_inputs(
            episode_video_metadata_path,
            episode_video_media_root,
        )

        upstream_identity = urdf_upstream_identity_path.read_text(
            encoding="utf-8"
        ).strip()
        if not upstream_identity:
            raise ValueError("URDF upstream identity must not be empty.")

        result = export_browser_data(
            repo_id,
            urdf_path=urdf_path,
            episodes=tuple(range(episode_start, episode_end + 1)),
            trajectory_episodes=tuple(sorted(trajectory_episode)),
            episode_batch_size=episode_batch_size,
            voxel_size=voxel_size,
            arm_spacing=arm_spacing,
            output_path=output_path,
            bundle_id=bundle_id,
            urdf_upstream_identity=upstream_identity,
            repository_path=Path.cwd(),
            dataset_revision=dataset_revision,
            episode_video_payload=episode_video_payload,
            episode_video_media=episode_video_media,
        )
    except Exception as error:
        console.print(f"[red]Failed to export browser data:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"Exported browser data to [bold]{result.output_path}[/bold]")
    console.print(f"Dataset frames: {result.dataset_frame_count:,}")
    console.print(f"Dual-arm tool-point visits: {result.tool_point_visit_count:,}")
    console.print(f"Arm-specific voxel entries: {result.arm_voxel_entry_count:,}")
    console.print(f"Unique shared grid cells: {result.unique_shared_grid_cell_count:,}")


@app.command("validate-browser-data")
def validate_browser_data_command(
    path: Path = typer.Argument(
        ...,
        help="Browser-data bundle directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Validate a browser-data v1 bundle and its checksums."""
    try:
        manifest = validate_browser_data(path)
    except Exception as error:
        console.print(f"[red]Invalid browser data:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(
        f"Valid browser-data bundle [bold]{manifest['bundleId']}[/bold] "
        f"(schema {manifest['schema']['major']}."
        f"{manifest['schema']['minor']})."
    )


def main() -> None:
    """Run the command-line application."""
    app()
