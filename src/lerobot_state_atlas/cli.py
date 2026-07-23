from math import isfinite
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lerobot_state_atlas.coverage import (
    compute_workspace_coverage,
)
from lerobot_state_atlas.dataset import load_dataset_summary
from lerobot_state_atlas.schema import DatasetSummary
from lerobot_state_atlas.state import load_state_batch
from lerobot_state_atlas.trajectory import (
    build_trlc_dk1_joint_component_map,
    compute_tool_trajectory,
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
    overview.add_row("Revision", summary.revision)
    overview.add_row(
        "LeRobot version",
        summary.codebase_version,
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
    episode: int = typer.Option(
        0,
        "--episode",
        "-e",
        help="Episode index to visualize.",
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
        if episode < 0:
            raise ValueError("Episode index must be nonnegative.")

        if not isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("Voxel size must be finite and greater than zero.")

        console.print(f"Loading metadata for [bold]{repo_id}[/bold]...")
        summary = load_dataset_summary(repo_id)
        component_names = _state_component_names(summary)

        console.print(f"Loading episode [bold]{episode}[/bold]...")
        batch = load_state_batch(
            repo_id,
            episodes=[episode],
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
            title=(f"TRLC-DK1 Episode {episode} Tool Workspace"),
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


def main() -> None:
    """Run the command-line application."""
    app()
