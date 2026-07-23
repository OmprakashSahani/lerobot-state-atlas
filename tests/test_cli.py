from typer.testing import CliRunner

from lerobot_state_atlas.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Analyze state coverage" in result.stdout
    assert "version" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "lerobot-state-atlas 0.1.0" in result.stdout
