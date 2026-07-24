from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from lerobot_state_atlas.cli import app
from lerobot_state_atlas.interactive import (
    InteractiveWorkspaceHeatmap,
)
from lerobot_state_atlas.schema import (
    DatasetSummary,
    FeatureSummary,
)
from lerobot_state_atlas.trajectory import ToolTrajectory
from lerobot_state_atlas.visualization import WorkspacePlot


runner = CliRunner()


def compact_output(output: str) -> str:
    return "".join(output.split())


def make_summary(
    *,
    component_names: tuple[str, ...] | None = (
        "left_joint_1.pos",
        "left_joint_2.pos",
        "left_joint_3.pos",
        "left_joint_4.pos",
        "left_joint_5.pos",
        "left_joint_6.pos",
        "left_gripper.pos",
        "right_joint_1.pos",
        "right_joint_2.pos",
        "right_joint_3.pos",
        "right_joint_4.pos",
        "right_joint_5.pos",
        "right_joint_6.pos",
        "right_gripper.pos",
    ),
) -> DatasetSummary:
    return DatasetSummary(
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
                component_names=component_names,
            ),
        ),
    )


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Analyze state coverage" in result.stdout
    assert "version" in result.stdout
    assert "visualize-workspace" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "lerobot-state-atlas 0.1.0" in result.stdout


def test_inspect_command(monkeypatch) -> None:
    summary = make_summary()

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        lambda repo_id: summary,
    )

    result = runner.invoke(
        app,
        ["inspect", "DreamMachines/example"],
    )

    assert result.exit_code == 0
    assert "DreamMachines/example" in result.stdout
    assert "bi_dk1_follower" in result.stdout
    assert "1,344" in result.stdout
    assert "696,107" in result.stdout
    assert "observation.state" in result.stdout


def test_inspect_command_reports_loading_error(
    monkeypatch,
) -> None:
    def raise_error(repo_id: str) -> None:
        raise RuntimeError(f"Unable to load {repo_id}")

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        raise_error,
    )

    result = runner.invoke(
        app,
        ["inspect", "missing/dataset"],
    )

    assert result.exit_code == 1
    assert "Failed to inspect dataset" in result.stdout
    assert "Unable to load missing/dataset" in result.stdout


def test_visualize_workspace_help() -> None:
    result = runner.invoke(
        app,
        ["visualize-workspace", "--help"],
    )

    assert result.exit_code == 0
    assert "--urdf" in result.stdout
    assert "--episode" in result.stdout
    assert "--voxel-size" in result.stdout
    assert "--output" in result.stdout


def test_visualize_workspace_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        "<robot name='test'/>",
        encoding="utf-8",
    )
    output_path = tmp_path / "workspace.png"

    summary = make_summary()
    states = torch.zeros(
        (3, 14),
        dtype=torch.float32,
    )
    batch = SimpleNamespace(
        states=states,
        episode_indices=torch.tensor(
            [3, 3, 3],
            dtype=torch.int64,
        ),
    )
    model = object()
    calls: dict[str, object] = {}

    def fake_load_summary(
        repo_id: str,
    ) -> DatasetSummary:
        calls["summary_repo_id"] = repo_id
        return summary

    def fake_load_batch(
        repo_id: str,
        episodes: list[int],
    ) -> SimpleNamespace:
        calls["batch_repo_id"] = repo_id
        calls["episodes"] = episodes
        return batch

    def fake_load_model(path: Path) -> object:
        calls["urdf_path"] = path
        return model

    def fake_compute_trajectory(
        state_values: torch.Tensor,
        component_names: tuple[str, ...],
        robot_model: object,
        joint_component_map: dict[str, str],
        *,
        arm: str,
        episode_indices: torch.Tensor,
    ) -> ToolTrajectory:
        calls[f"{arm}_components"] = component_names
        calls[f"{arm}_mapping"] = joint_component_map

        assert state_values is states
        assert robot_model is model
        assert torch.equal(
            episode_indices,
            batch.episode_indices,
        )

        offset = 0.0 if arm == "left" else 1.0
        return ToolTrajectory(
            arm=arm,
            link_name="tool0",
            positions=torch.tensor(
                [
                    [offset, 0.0, 0.0],
                    [offset, 0.1, 0.2],
                    [offset, 0.2, 0.3],
                ],
                dtype=torch.float64,
            ),
        )

    def fake_compute_coverage(
        trajectory: ToolTrajectory,
        *,
        voxel_size: float,
    ) -> SimpleNamespace:
        calls[f"{trajectory.arm}_voxel_size"] = voxel_size
        return SimpleNamespace(
            arm=trajectory.arm,
            voxel_size=voxel_size,
        )

    def fake_save_plot(
        trajectories: tuple[
            ToolTrajectory,
            ...,
        ],
        destination: Path,
        *,
        coverages: tuple[
            SimpleNamespace,
            ...,
        ],
        title: str,
    ) -> WorkspacePlot:
        calls["trajectories"] = trajectories
        calls["coverages"] = coverages
        calls["output_path"] = destination
        calls["title"] = title

        return WorkspacePlot(
            output_path=destination,
            num_trajectories=2,
            num_points=6,
            voxel_size=0.05,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        fake_load_summary,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_state_batch",
        fake_load_batch,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_robot_model",
        fake_load_model,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.compute_tool_trajectory",
        fake_compute_trajectory,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.compute_workspace_coverage",
        fake_compute_coverage,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.save_workspace_plot",
        fake_save_plot,
    )

    result = runner.invoke(
        app,
        [
            "visualize-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--episode",
            "3",
            "--voxel-size",
            "0.05",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert calls["summary_repo_id"] == "DreamMachines/example"
    assert calls["batch_repo_id"] == "DreamMachines/example"
    assert calls["episodes"] == [3]
    assert calls["urdf_path"] == urdf_path.resolve()
    assert calls["output_path"] == output_path
    assert calls["title"] == "TRLC-DK1 Episode 3 Tool Workspace"
    assert calls["left_voxel_size"] == 0.05
    assert calls["right_voxel_size"] == 0.05

    trajectories = calls["trajectories"]
    assert isinstance(trajectories, tuple)
    assert tuple(trajectory.arm for trajectory in trajectories) == ("left", "right")

    assert "Saved workspace plot" in result.stdout
    assert "workspace.png" in compact_output(result.stdout)
    assert "Plotted 6 points" in result.stdout
    assert "Voxel size: 0.050 m" in result.stdout
    assert "localbase_linkframes" in compact_output(result.stdout)


def test_visualize_workspace_rejects_negative_episode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()

    def unexpected_load(
        repo_id: str,
    ) -> DatasetSummary:
        raise AssertionError(f"Unexpected metadata load: {repo_id}")

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        unexpected_load,
    )

    result = runner.invoke(
        app,
        [
            "visualize-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--episode",
            "-1",
        ],
    )

    assert result.exit_code == 1
    assert "Episode index must be nonnegative" in result.stdout


@pytest.mark.parametrize(
    "voxel_size",
    ("0", "-0.1", "nan", "inf", "-inf"),
)
def test_visualize_workspace_rejects_voxel_size(
    monkeypatch,
    tmp_path: Path,
    voxel_size: str,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()

    def unexpected_load(
        repo_id: str,
    ) -> DatasetSummary:
        raise AssertionError(f"Unexpected metadata load: {repo_id}")

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        unexpected_load,
    )

    result = runner.invoke(
        app,
        [
            "visualize-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--voxel-size",
            voxel_size,
        ],
    )

    assert result.exit_code == 1
    assert "Voxel size must be finite and greater than zero" in result.stdout


def test_visualize_workspace_rejects_missing_components(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        lambda repo_id: make_summary(component_names=None),
    )

    result = runner.invoke(
        app,
        [
            "visualize-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
        ],
    )

    assert result.exit_code == 1
    assert "observation.statedoesnotdefinecomponentnames" in compact_output(
        result.stdout
    )


def test_visualize_workspace_reports_pipeline_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        lambda repo_id: make_summary(),
    )

    def raise_error(
        repo_id: str,
        episodes: list[int],
    ) -> None:
        raise RuntimeError("Episode download failed")

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_state_batch",
        raise_error,
    )

    result = runner.invoke(
        app,
        [
            "visualize-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
        ],
    )

    assert result.exit_code == 1
    assert "Failed to visualize workspace" in result.stdout
    assert "Episode download failed" in result.stdout


def test_visualize_workspace_accepts_multiple_episodes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        "<robot name='test'/>",
        encoding="utf-8",
    )
    output_path = tmp_path / "multi-episode-workspace.png"

    summary = make_summary()
    states = torch.zeros(
        (4, 14),
        dtype=torch.float32,
    )
    episode_indices = torch.tensor(
        [2, 2, 5, 5],
        dtype=torch.int64,
    )
    batch = SimpleNamespace(
        states=states,
        episode_indices=episode_indices,
    )
    model = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        lambda repo_id: summary,
    )

    def fake_load_batch(
        repo_id: str,
        episodes: list[int],
    ) -> SimpleNamespace:
        calls["episodes"] = episodes
        return batch

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_state_batch",
        fake_load_batch,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_robot_model",
        lambda path: model,
    )

    def fake_compute_trajectory(
        state_values: torch.Tensor,
        component_names: tuple[str, ...],
        robot_model: object,
        joint_component_map: dict[str, str],
        *,
        arm: str,
        episode_indices: torch.Tensor,
    ) -> ToolTrajectory:
        assert state_values is states
        assert robot_model is model
        assert torch.equal(
            episode_indices,
            batch.episode_indices,
        )

        offset = 0.0 if arm == "left" else 1.0

        return ToolTrajectory(
            arm=arm,
            link_name="tool0",
            positions=torch.tensor(
                [
                    [offset, 0.0, 0.0],
                    [offset, 0.1, 0.1],
                    [offset, 0.2, 0.2],
                    [offset, 0.3, 0.3],
                ],
                dtype=torch.float64,
            ),
            episode_indices=episode_indices,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.compute_tool_trajectory",
        fake_compute_trajectory,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.compute_workspace_coverage",
        lambda trajectory, *, voxel_size: SimpleNamespace(
            arm=trajectory.arm,
            voxel_size=voxel_size,
        ),
    )

    def fake_save_plot(
        trajectories: tuple[ToolTrajectory, ...],
        destination: Path,
        *,
        coverages: tuple[SimpleNamespace, ...],
        title: str,
    ) -> WorkspacePlot:
        calls["title"] = title

        assert all(trajectory.num_episodes == 2 for trajectory in trajectories)

        return WorkspacePlot(
            output_path=destination,
            num_trajectories=2,
            num_points=8,
            voxel_size=0.02,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.save_workspace_plot",
        fake_save_plot,
    )

    result = runner.invoke(
        app,
        [
            "visualize-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--episode",
            "2",
            "--episode",
            "5",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert calls["episodes"] == [2, 5]
    assert calls["title"] == ("TRLC-DK1 Episodes 2, 5 Tool Workspace")
    assert "Plotted 8 points" in result.stdout


def test_visualize_workspace_rejects_duplicate_episodes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()

    def unexpected_load(
        repo_id: str,
    ) -> DatasetSummary:
        raise AssertionError(f"Unexpected metadata load: {repo_id}")

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        unexpected_load,
    )

    result = runner.invoke(
        app,
        [
            "visualize-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--episode",
            "3",
            "--episode",
            "3",
        ],
    )

    assert result.exit_code == 1
    assert "Episode indices must be unique" in result.stdout


def test_interactive_workspace_help() -> None:
    result = runner.invoke(
        app,
        ["interactive-workspace", "--help"],
    )

    assert result.exit_code == 0
    assert "--urdf" in result.stdout
    assert "--episode" in result.stdout
    assert "--voxel-size" in result.stdout
    assert "--output" in result.stdout


def test_interactive_workspace_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        "<robot name='test'/>",
        encoding="utf-8",
    )
    output_path = tmp_path / "workspace-heatmap.html"

    summary = make_summary()
    states = torch.zeros(
        (4, 14),
        dtype=torch.float32,
    )
    episode_indices = torch.tensor(
        [2, 2, 5, 5],
        dtype=torch.int64,
    )
    batch = SimpleNamespace(
        states=states,
        episode_indices=episode_indices,
    )
    model = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        lambda repo_id: summary,
    )

    def fake_load_batch(
        repo_id: str,
        episodes: list[int],
    ) -> SimpleNamespace:
        calls["repo_id"] = repo_id
        calls["episodes"] = episodes
        return batch

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_state_batch",
        fake_load_batch,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_robot_model",
        lambda path: model,
    )

    def fake_compute_trajectory(
        state_values: torch.Tensor,
        component_names: tuple[str, ...],
        robot_model: object,
        joint_component_map: dict[str, str],
        *,
        arm: str,
        episode_indices: torch.Tensor,
    ) -> ToolTrajectory:
        assert state_values is states
        assert robot_model is model
        assert torch.equal(
            episode_indices,
            batch.episode_indices,
        )

        offset = 0.0 if arm == "left" else 1.0

        return ToolTrajectory(
            arm=arm,
            link_name="tool0",
            positions=torch.tensor(
                [
                    [offset, 0.0, 0.0],
                    [offset, 0.1, 0.1],
                    [offset, 0.2, 0.2],
                    [offset, 0.3, 0.3],
                ],
                dtype=torch.float64,
            ),
            episode_indices=episode_indices,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.compute_tool_trajectory",
        fake_compute_trajectory,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.compute_workspace_coverage",
        lambda trajectory, *, voxel_size: SimpleNamespace(
            arm=trajectory.arm,
            voxel_size=voxel_size,
        ),
    )

    def fake_save_heatmap(
        trajectories: tuple[ToolTrajectory, ...],
        destination: Path,
        *,
        coverages: tuple[SimpleNamespace, ...],
        title: str,
        playback_fps: float,
    ) -> InteractiveWorkspaceHeatmap:
        calls["trajectories"] = trajectories
        calls["coverages"] = coverages
        calls["output_path"] = destination
        calls["title"] = title
        calls["playback_fps"] = playback_fps

        return InteractiveWorkspaceHeatmap(
            output_path=destination,
            num_trajectories=2,
            num_points=8,
            occupied_voxels=6,
            voxel_size=0.02,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.save_interactive_workspace_heatmap",
        fake_save_heatmap,
    )

    result = runner.invoke(
        app,
        [
            "interactive-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--episode",
            "2",
            "--episode",
            "5",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert calls["repo_id"] == "DreamMachines/example"
    assert calls["episodes"] == [2, 5]
    assert calls["output_path"] == output_path
    assert calls["title"] == ("TRLC-DK1 Episodes 2, 5 Interactive Workspace Heatmap")
    assert calls["playback_fps"] == pytest.approx(50.0)
    assert "Saved interactive workspace heatmap" in result.stdout
    assert "Plotted 8 points" in result.stdout
    assert "Occupied voxels: 6" in result.stdout
    assert "Voxel size: 0.020 m" in result.stdout
    assert "localbase_linkframes" in compact_output(result.stdout)
