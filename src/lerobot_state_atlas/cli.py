import typer
from rich.console import Console
from rich.table import Table

from lerobot_state_atlas.dataset import load_dataset_summary
from lerobot_state_atlas.schema import DatasetSummary

app = typer.Typer(
    name="lerobot-state-atlas",
    help="Analyze state coverage, trajectories, and reset consistency in LeRobot datasets.",
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
    overview = Table(title="Dataset Overview", show_header=False)
    overview.add_column("Field", style="bold")
    overview.add_column("Value")

    overview.add_row("Repository", summary.repo_id)
    overview.add_row("Revision", summary.revision)
    overview.add_row("LeRobot version", summary.codebase_version)
    overview.add_row("Robot type", summary.robot_type or "Unknown")
    overview.add_row("FPS", f"{summary.fps:g}")
    overview.add_row("Episodes", f"{summary.total_episodes:,}")
    overview.add_row("Frames", f"{summary.total_frames:,}")
    overview.add_row("Tasks", f"{summary.total_tasks:,}")
    overview.add_row(
        "Duration",
        f"{summary.total_duration_seconds / 3600:.2f} hours",
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


@app.command("inspect")
def inspect_dataset(
    repo_id: str = typer.Argument(
        ...,
        help="Hugging Face repository ID of the LeRobot dataset.",
    ),
) -> None:
    """Inspect LeRobot dataset metadata without loading video frames."""
    try:
        console.print(f"Loading metadata for [bold]{repo_id}[/bold]...")
        summary = load_dataset_summary(repo_id)
    except Exception as error:
        console.print(f"[red]Failed to inspect dataset:[/red] {error}")
        raise typer.Exit(code=1) from error

    display_dataset_summary(summary)


def main() -> None:
    """Run the command-line application."""
    app()
