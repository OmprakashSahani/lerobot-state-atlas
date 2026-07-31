import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from click import unstyle
from click.testing import Result
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


def plain_output(result: Result) -> str:
    return unstyle(result.stdout)


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
        requested_revision="v3.0",
        resolved_revision="a" * 40,
        lerobot_codebase_version="v3.0",
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
    output = plain_output(result)

    assert result.exit_code == 0
    assert "Analyze state coverage" in output
    assert "version" in output
    assert "visualize-workspace" in compact_output(output)


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "lerobot-state-atlas 0.1.0" in plain_output(result)


def test_export_browser_data_help_includes_optional_inputs() -> None:
    result = runner.invoke(app, ["export-browser-data", "--help"])
    assert result.exit_code == 0

    from typer.main import get_command

    root_command = get_command(app)
    export_command = root_command.commands["export-browser-data"]
    option_names = {
        option
        for parameter in export_command.params
        for option in getattr(parameter, "opts", ())
    }

    assert "--dataset-revision" in option_names
    assert "--episode-video-metadata" in option_names
    assert "--episode-video-media-root" in option_names


def test_export_browser_data_forwards_episode_video_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='test'/>", encoding="utf-8")

    identity_path = tmp_path / "UPSTREAM_COMMIT"
    identity_path.write_text("upstream-commit\n", encoding="utf-8")

    media_root = tmp_path / "video-inputs"
    media_path = media_root / "media/episode-000000/top.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"test-video")

    payload = {
        "schema": {
            "name": "lerobot-state-atlas.browser-data",
            "major": 1,
            "minor": 1,
        },
        "defaultCameraId": "top",
        "cameras": [],
        "episodes": [
            {
                "episodeId": 0,
                "videos": [
                    {
                        "filename": "media/episode-000000/top.mp4",
                    }
                ],
            }
        ],
    }
    metadata_path = tmp_path / "episode-videos.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    calls: dict[str, object] = {}

    def fake_export(repo_id: str, **kwargs):
        calls["repo_id"] = repo_id
        calls.update(kwargs)
        return SimpleNamespace(
            output_path=tmp_path / "bundle",
            dataset_frame_count=1,
            tool_point_visit_count=2,
            arm_voxel_entry_count=1,
            unique_shared_grid_cell_count=1,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.export_browser_data",
        fake_export,
    )

    result = runner.invoke(
        app,
        [
            "export-browser-data",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--urdf-upstream-identity",
            str(identity_path),
            "--episode-end",
            "0",
            "--trajectory-episode",
            "0",
            "--episode-video-metadata",
            str(metadata_path),
            "--episode-video-media-root",
            str(media_root),
            "--bundle-id",
            "test-v1",
            "--output",
            str(tmp_path / "bundle"),
        ],
    )

    assert result.exit_code == 0
    assert calls["episode_video_payload"] == payload
    assert calls["episode_video_media"] == {
        "media/episode-000000/top.mp4": media_path.resolve()
    }


def test_export_browser_data_requires_both_video_options(
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='test'/>", encoding="utf-8")

    identity_path = tmp_path / "UPSTREAM_COMMIT"
    identity_path.write_text("upstream-commit\n", encoding="utf-8")

    metadata_path = tmp_path / "episode-videos.json"
    metadata_path.write_text('{"episodes": []}', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "export-browser-data",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--urdf-upstream-identity",
            str(identity_path),
            "--episode-end",
            "0",
            "--episode-video-metadata",
            str(metadata_path),
            "--bundle-id",
            "test-v1",
            "--output",
            str(tmp_path / "bundle"),
        ],
    )

    assert result.exit_code == 1
    assert "must be provided together" in plain_output(result)


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
    assert "DreamMachines/example" in plain_output(result)
    assert "bi_dk1_follower" in plain_output(result)
    assert "1,344" in plain_output(result)
    assert "696,107" in plain_output(result)
    assert "observation.state" in plain_output(result)


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
    assert "Failed to inspect dataset" in plain_output(result)
    assert "Unable to load missing/dataset" in plain_output(result)


def test_visualize_workspace_help() -> None:
    result = runner.invoke(
        app,
        ["visualize-workspace", "--help"],
    )
    output = compact_output(plain_output(result))

    assert result.exit_code == 0
    assert "--urdf" in output
    assert "--episode" in output
    assert "--voxel-size" in output
    assert "--output" in output


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
        *,
        revision: str,
    ) -> SimpleNamespace:
        calls["batch_repo_id"] = repo_id
        calls["episodes"] = episodes
        calls["revision"] = revision
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
            rotation_matrices=torch.eye(3, dtype=torch.float64)
            .expand(3, -1, -1)
            .clone(),
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

    assert "Saved workspace plot" in plain_output(result)
    assert "workspace.png" in compact_output(plain_output(result))
    assert "Plotted 6 points" in plain_output(result)
    assert "Voxel size: 0.050 m" in plain_output(result)
    assert "localbase_linkframes" in compact_output(plain_output(result))


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
    assert "Episode index must be nonnegative" in plain_output(result)


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
    assert "Voxel size must be finite and greater than zero" in plain_output(result)


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
        plain_output(result)
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
        *,
        revision: str,
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
    assert "Failed to visualize workspace" in plain_output(result)
    assert "Episode download failed" in plain_output(result)


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
        *,
        revision: str,
    ) -> SimpleNamespace:
        calls["episodes"] = episodes
        calls["revision"] = revision
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
            rotation_matrices=torch.eye(3, dtype=torch.float64)
            .expand(4, -1, -1)
            .clone(),
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
    assert "Plotted 8 points" in plain_output(result)


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
    assert "Episode indices must be unique" in plain_output(result)


def test_interactive_workspace_help() -> None:
    result = runner.invoke(
        app,
        ["interactive-workspace", "--help"],
    )
    output = compact_output(plain_output(result))

    assert result.exit_code == 0
    assert "--urdf" in output
    assert "--episode" in output
    assert "--voxel-size" in output
    assert "--arm-spacing" in output
    assert "--output" in output


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
        *,
        revision: str,
    ) -> SimpleNamespace:
        calls["repo_id"] = repo_id
        calls["episodes"] = episodes
        calls["revision"] = revision
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
            rotation_matrices=torch.eye(3, dtype=torch.float64)
            .expand(4, -1, -1)
            .clone(),
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

    def fake_compute_coverage(
        trajectory: ToolTrajectory,
        *,
        voxel_size: float,
        voxel_origin_xyz: tuple[float, float, float],
    ) -> SimpleNamespace:
        calls[f"{trajectory.arm}_coverage_positions"] = trajectory.positions.clone()
        calls[f"{trajectory.arm}_voxel_origin"] = voxel_origin_xyz

        return SimpleNamespace(
            arm=trajectory.arm,
            voxel_size=voxel_size,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.compute_workspace_coverage",
        fake_compute_coverage,
    )

    def fake_save_heatmap(
        trajectories: tuple[ToolTrajectory, ...],
        destination: Path,
        *,
        coverages: tuple[SimpleNamespace, ...],
        title: str,
        playback_fps: float,
        shared_space: bool,
    ) -> InteractiveWorkspaceHeatmap:
        calls["trajectories"] = trajectories
        calls["coverages"] = coverages
        calls["output_path"] = destination
        calls["title"] = title
        calls["playback_fps"] = playback_fps
        calls["shared_space"] = shared_space

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
            "--arm-spacing",
            "0.8",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert calls["repo_id"] == "DreamMachines/example"
    assert calls["episodes"] == [2, 5]
    assert calls["output_path"] == output_path
    assert calls["title"] == (
        "TRLC-DK1 Episodes 2, 5 Shared Interactive Workspace Heatmap"
    )
    assert calls["playback_fps"] == pytest.approx(50.0)
    assert calls["shared_space"] is True

    trajectories = calls["trajectories"]
    left_trajectory = trajectories[0]
    right_trajectory = trajectories[1]

    assert left_trajectory.positions[:, 1] == pytest.approx([0.4, 0.5, 0.6, 0.7])
    assert right_trajectory.positions[:, 1] == pytest.approx([-0.4, -0.3, -0.2, -0.1])
    assert calls["left_voxel_origin"] == (0.0, 0.0, 0.0)
    assert calls["right_voxel_origin"] == (0.0, 0.0, 0.0)

    assert "Saved interactive workspace heatmap" in plain_output(result)
    assert "Plotted 8 points" in plain_output(result)
    assert "Occupied voxels: 6" in plain_output(result)
    assert "Voxel size: 0.020 m" in plain_output(result)
    assert "Arm base spacing: 0.800 m" in plain_output(result)
    assert "sharedworldframe" in compact_output(plain_output(result))


def test_aggregate_workspace_help() -> None:
    result = runner.invoke(
        app,
        ["aggregate-workspace", "--help"],
    )
    output = compact_output(plain_output(result))

    assert result.exit_code == 0
    assert "--urdf" in output
    assert "--episode-start" in output
    assert "--episode-end" in output
    assert "--episode-batch-size" in output
    assert "--voxel-size" in output
    assert "--arm-spacing" in output
    assert "--output" in output


def test_aggregate_workspace_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        "<robot name='test'/>",
        encoding="utf-8",
    )
    output_path = tmp_path / "aggregated-workspace.html"

    summary = make_summary()
    model = object()
    coverages = (
        SimpleNamespace(
            arm="left",
            num_points=10,
            occupied_voxels=3,
            voxel_size=0.02,
        ),
        SimpleNamespace(
            arm="right",
            num_points=10,
            occupied_voxels=4,
            voxel_size=0.02,
        ),
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_dataset_summary",
        lambda repo_id: summary,
    )
    monkeypatch.setattr(
        "lerobot_state_atlas.cli.load_robot_model",
        lambda path: model,
    )

    def fake_aggregate(
        repo_id: str,
        episodes: tuple[int, ...],
        *,
        component_names: tuple[str, ...],
        model: object,
        voxel_size: float,
        episode_batch_size: int,
        arm_transforms: object,
        revision: str,
    ) -> SimpleNamespace:
        calls["repo_id"] = repo_id
        calls["episodes"] = episodes
        calls["component_names"] = component_names
        calls["model"] = model
        calls["voxel_size"] = voxel_size
        calls["episode_batch_size"] = episode_batch_size
        calls["arm_transforms"] = arm_transforms
        calls["revision"] = revision

        return SimpleNamespace(
            coverages=coverages,
            num_batches=2,
            num_episodes=4,
            num_frames=10,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.aggregate_workspace_coverages",
        fake_aggregate,
    )

    def fake_save_heatmap(
        workspace_coverages: tuple[SimpleNamespace, ...],
        destination: Path,
        *,
        title: str,
        shared_space: bool,
    ) -> InteractiveWorkspaceHeatmap:
        calls["coverages"] = workspace_coverages
        calls["output_path"] = destination
        calls["title"] = title
        calls["shared_space"] = shared_space

        return InteractiveWorkspaceHeatmap(
            output_path=destination,
            num_trajectories=0,
            num_points=20,
            occupied_voxels=7,
            voxel_size=0.02,
        )

    monkeypatch.setattr(
        "lerobot_state_atlas.cli.save_interactive_workspace_coverage_heatmap",
        fake_save_heatmap,
    )

    result = runner.invoke(
        app,
        [
            "aggregate-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--episode-start",
            "2",
            "--episode-end",
            "5",
            "--episode-batch-size",
            "2",
            "--arm-spacing",
            "0.8",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert calls["repo_id"] == "DreamMachines/example"
    assert calls["episodes"] == (2, 3, 4, 5)
    assert calls["model"] is model
    assert calls["voxel_size"] == pytest.approx(0.02)
    assert calls["episode_batch_size"] == 2

    arm_transforms = calls["arm_transforms"]

    assert arm_transforms["left"].translation_xyz == pytest.approx((0.0, 0.4, 0.0))
    assert arm_transforms["right"].translation_xyz == pytest.approx((0.0, -0.4, 0.0))
    assert arm_transforms["left"].rotation_rpy == pytest.approx((0.0, 0.0, 0.0))
    assert arm_transforms["right"].rotation_rpy == pytest.approx((0.0, 0.0, 0.0))

    assert calls["coverages"] is coverages
    assert calls["output_path"] == output_path
    assert calls["title"] == ("TRLC-DK1 Episodes 2-5 Shared Workspace Heatmap")
    assert calls["shared_space"] is True

    assert "Saved aggregated workspace heatmap" in plain_output(result)
    assert "Aggregated 4 episodes across 2 batches" in plain_output(result)
    assert "Processed 10 dataset frames" in plain_output(result)
    assert "20 dual-arm tool points" in plain_output(result)
    assert "Occupied voxels: 7" in plain_output(result)
    assert "Voxel size: 0.020 m" in plain_output(result)
    assert "Arm base spacing: 0.800 m" in plain_output(result)
    assert "sharedworldframe" in compact_output(plain_output(result))


@pytest.mark.parametrize(
    "arm_spacing",
    [
        "0",
        "-0.1",
        "nan",
        "inf",
    ],
)
def test_aggregate_workspace_rejects_invalid_arm_spacing(
    tmp_path: Path,
    arm_spacing: str,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        "<robot name='test'/>",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "aggregate-workspace",
            "DreamMachines/example",
            "--urdf",
            str(urdf_path),
            "--episode-end",
            "0",
            "--arm-spacing",
            arm_spacing,
        ],
    )

    assert result.exit_code == 1
    assert "Armspacingmustbefiniteandgreaterthanzero" in compact_output(
        plain_output(result)
    )
