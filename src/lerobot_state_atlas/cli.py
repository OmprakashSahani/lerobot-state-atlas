import typer
from rich.console import Console

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


def main() -> None:
    """Run the command-line application."""
    app()
