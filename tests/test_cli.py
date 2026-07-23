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


def test_inspect_command(monkeypatch) -> None:
    from lerobot_state_atlas.schema import DatasetSummary, FeatureSummary

    summary = DatasetSummary(
        repo_id="DreamMachines/example",
        revision="v3.0",
        codebase_version="v3.0",
        robot_type="bi_dk1_follower",
        fps=50.0,
        total_episodes=1344,
        total_frames=696107,
        total_tasks=1,
        total_duration_seconds=13922.14,
        features=(
            FeatureSummary(
                name="observation.state",
                dtype="float32",
                shape=(14,),
                component_names=("left_joint_1.pos", "right_joint_1.pos"),
            ),
        ),
    )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        lambda repo_id: summary,
    )

    result = runner.invoke(app, ["inspect", "DreamMachines/example"])

    assert result.exit_code == 0
    assert "DreamMachines/example" in result.stdout
    assert "bi_dk1_follower" in result.stdout
    assert "1,344" in result.stdout
    assert "696,107" in result.stdout
    assert "observation.state" in result.stdout


def test_inspect_command_reports_loading_error(monkeypatch) -> None:
    def raise_error(repo_id: str) -> None:
        raise RuntimeError(f"Unable to load {repo_id}")

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        raise_error,
    )

    result = runner.invoke(app, ["inspect", "missing/dataset"])

    assert result.exit_code == 1
    assert "Failed to inspect dataset" in result.stdout
    assert "Unable to load missing/dataset" in result.stdout
