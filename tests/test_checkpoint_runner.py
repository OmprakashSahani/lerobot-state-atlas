from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from lerobot_state_atlas.checkpoint_comparison import (
    CheckpointComparisonRunnerError,
    DeterministicTorchSettings,
    execute_checkpoint_comparison_run,
    install_checkpoint_comparison_run,
    load_checkpoint_comparison_runner_manifest,
    preflight_checkpoint_comparison_runner,
    resolve_checkpoint_comparison_runner_inputs,
    unavailable_policy_comparison_trajectory_result,
    validate_checkpoint_comparison,
)
from test_checkpoint_inference import bound_input, observation
from test_checkpoint_runner_manifest import _manifest
import lerobot_state_atlas.checkpoint_comparison.runner as runner_module


class FakeDependencies:
    def __init__(self, fail=None):
        self.events = []
        self.fail = fail

    def load_manifest(self, path):
        self.events.append("manifest")
        if self.fail == "manifest":
            raise ValueError("bad manifest")
        return SimpleNamespace(manifest_sha256="a" * 64)

    def resolve(self, manifest):
        self.events.append("resolution")
        if self.fail == "resolution":
            raise ValueError("unsafe input")
        return "resolved"

    def preflight(self, manifest, resolved):
        self.events.append("preflight")
        if self.fail == "preflight":
            raise ValueError("CUDA unavailable")
        return SimpleNamespace(passed=True)


def test_preflight_only_stops_before_execution_dependencies() -> None:
    dependencies = FakeDependencies()
    report = preflight_checkpoint_comparison_runner(
        "runner.json", dependencies=dependencies
    )
    assert report.passed
    assert dependencies.events == ["manifest", "resolution", "preflight"]


@pytest.mark.parametrize("phase", ["manifest", "resolution", "preflight"])
def test_preflight_errors_are_phase_tagged(phase: str) -> None:
    dependencies = FakeDependencies(phase)
    with pytest.raises(CheckpointComparisonRunnerError) as raised:
        preflight_checkpoint_comparison_runner("runner.json", dependencies=dependencies)
    assert raised.value.phase == phase


def test_fake_full_lifecycle_installs_deterministic_v1_1_run(
    tmp_path, monkeypatch
) -> None:
    path, _ = _manifest(
        tmp_path,
        lambda value: (
            value["projection"].update(
                mode="unavailable",
                unavailableReason="Synthetic test has no FK.",
                acknowledgeUncalibratedArmTransforms=False,
                acknowledgeRecordedLimitViolations=False,
            )
            or value["output"].update(replaceExisting=True)
        ),
    )
    source = observation()
    manifest = load_checkpoint_comparison_runner_manifest(path)
    resolved = resolve_checkpoint_comparison_runner_inputs(manifest)
    camera_paths = tuple(
        replace(
            camera,
            path=resolved.manifest_directory / f"inputs/camera-{index}.png",
        )
        for index, camera in enumerate(source.cameras)
    )
    for camera in camera_paths:
        camera.path.parent.mkdir(parents=True, exist_ok=True)
        camera.path.write_bytes(b"camera")
    source = replace(
        source,
        manifest_path=resolved.manifest_directory / manifest.observation_manifest.path,
        manifest_sha256=manifest.observation_manifest.sha256,
        manifest_byte_count=manifest.observation_manifest.byte_count,
        dataset=replace(
            source.dataset,
            repository_id=manifest.dataset.repository_id,
            revision=manifest.dataset.revision,
        ),
        cameras=camera_paths,
    )
    events = []
    inference_devices = []
    original_run_sequential = runner_module.run_sequential_policy_comparison

    def capture_run_sequential(*args, **kwargs):
        inference_devices.append((kwargs["noise_device"], kwargs["noise_dtype"]))
        return original_run_sequential(*args, **kwargs)

    monkeypatch.setattr(
        runner_module, "run_sequential_policy_comparison", capture_run_sequential
    )

    class Policy:
        def __init__(self, value, name):
            self.value = value
            self.name = name

        def predict_action_chunk(self, processed, *, noise, num_inference_steps=None):
            events.append(f"{self.name}:run")
            return torch.full((1, 50, 14), self.value)

    class Dependencies:
        load_manifest = staticmethod(load_checkpoint_comparison_runner_manifest)
        resolve = staticmethod(resolve_checkpoint_comparison_runner_inputs)

        def preflight(self, manifest, resolved):
            return SimpleNamespace(
                passed=True,
                manifest_sha256=manifest.manifest_sha256,
                requested_device="cuda:0",
                gpu_name="Fake GPU",
                compute_capability=(8, 0),
            )

        def execution_device(self, manifest):
            events.append("execution-device:cpu")
            return "cpu"

        def load_observation(self, resolved):
            events.append("observation")
            return source

        def prepare_cameras(self, value):
            events.append("cameras")
            return bound_input(value)

        def adapt_configuration(self, manifest, resolved):
            return SimpleNamespace(
                source_sha256="1" * 64,
                effective_sha256="2" * 64,
                transformations=(),
            )

        def verify_processors(self, manifest, resolved):
            compatibility = SimpleNamespace(
                source_preprocessor_sha256="3" * 64,
                effective_preprocessor_sha256="4" * 64,
                source_postprocessor_sha256="5" * 64,
                effective_postprocessor_sha256="6" * 64,
            )
            return SimpleNamespace(
                compatibility=compatibility,
                preprocessor_state=SimpleNamespace(sha256="7" * 64),
                postprocessor_state=SimpleNamespace(sha256="8" * 64),
                shared_for_policy_ids=("base-pi05", "fine-tuned-pi05"),
            )

        def verify_tokenizer(self, manifest, resolved, processors):
            return SimpleNamespace(
                repository_id="google/paligemma-3b-pt-224",
                directory_identity_sha256="9" * 64,
            )

        def load_tokenizer(self, verification):
            return object()

        def build_processors(self, *args, device):
            assert device == "cpu"
            return SimpleNamespace(
                preprocessor=lambda value: events.append("preprocess") or value,
                postprocessor=lambda value: value,
            )

        @contextmanager
        def deterministic_context(self, manifest):
            yield DeterministicTorchSettings(
                ":4096:8",
                True,
                True,
                False,
                False,
                False,
                "highest",
                "cuda:0",
                "bfloat16",
            )

        def policy_factory(self, policy_id, *args):
            @contextmanager
            def manager():
                events.append(f"{policy_id}:enter")
                try:
                    yield Policy(1.0 if policy_id == "base-pi05" else 2.0, policy_id)
                finally:
                    events.append(f"{policy_id}:exit")

            return manager

        def project(self, manifest, resolved, observation, inference):
            return unavailable_policy_comparison_trajectory_result(
                observation, inference, reason="Synthetic test has no FK."
            )

        def install(self, destination, manifest, plans, receipt, *, replace_existing):
            return install_checkpoint_comparison_run(
                destination,
                manifest,
                plans,
                receipt,
                replace_existing=replace_existing,
            )

    def reject_receipt_reread(path):
        pytest.fail(f"receipt must not reacquire the observation manifest: {path}")

    monkeypatch.setattr(
        runner_module, "read_stable_file_snapshot", reject_receipt_reread
    )
    result = execute_checkpoint_comparison_run(path, dependencies=Dependencies())
    assert events.index("base-pi05:exit") < events.index("fine-tuned-pi05:enter")
    assert "execution-device:cpu" in events
    assert inference_devices == [("cpu", torch.float32)]
    assert events.count("preprocess") == 1
    assert result.policy_order == ("base-pi05", "fine-tuned-pi05")
    assert not result.projection_available
    validate_checkpoint_comparison(result.comparison_directory)
    assert result.receipt_path.is_file()
    receipt_document = json.loads(result.receipt_path.read_bytes())
    assert receipt_document["observation"] == {
        "manifestByteCount": source.manifest_byte_count,
        "manifestSha256": source.manifest_sha256,
        "observationId": source.observation_id,
    }
    first_bytes = tuple(
        path.read_bytes()
        for path in (
            result.comparison_directory / "manifest.json",
            result.comparison_directory / "plans.json",
            result.receipt_path,
        )
    )
    repeated = execute_checkpoint_comparison_run(path, dependencies=Dependencies())
    assert inference_devices == [("cpu", torch.float32), ("cpu", torch.float32)]
    assert first_bytes == tuple(
        item.read_bytes()
        for item in (
            repeated.comparison_directory / "manifest.json",
            repeated.comparison_directory / "plans.json",
            repeated.receipt_path,
        )
    )


@pytest.mark.parametrize("field", ["sha256", "byte_count"])
def test_observation_snapshot_identity_must_match_runner_manifest_before_cameras(
    tmp_path, field
) -> None:
    path, _ = _manifest(tmp_path)
    manifest = load_checkpoint_comparison_runner_manifest(path)
    resolved = resolve_checkpoint_comparison_runner_inputs(manifest)
    source = observation()
    changes = {
        "manifest_path": resolved.manifest_directory
        / manifest.observation_manifest.path,
        "manifest_sha256": manifest.observation_manifest.sha256,
        "manifest_byte_count": manifest.observation_manifest.byte_count,
        "dataset": replace(
            source.dataset,
            repository_id=manifest.dataset.repository_id,
            revision=manifest.dataset.revision,
        ),
    }
    changes[f"manifest_{field}"] = (
        "f" * 64 if field == "sha256" else manifest.observation_manifest.byte_count + 1
    )
    source = replace(source, **changes)
    events = []

    class Dependencies:
        load_manifest = staticmethod(load_checkpoint_comparison_runner_manifest)
        resolve = staticmethod(resolve_checkpoint_comparison_runner_inputs)

        def preflight(self, manifest, resolved):
            events.append("preflight")
            return SimpleNamespace(passed=True)

        def load_observation(self, resolved):
            events.append("observation")
            return source

        def prepare_cameras(self, value):
            pytest.fail("camera preparation must not run after provenance mismatch")

    with pytest.raises(CheckpointComparisonRunnerError) as caught:
        execute_checkpoint_comparison_run(path, dependencies=Dependencies())
    assert caught.value.phase == "observation"
    assert field.replace("_", " ").upper().split()[0] in caught.value.message.upper()
    assert events == ["preflight", "observation"]


def test_primary_runner_error_survives_multiple_cleanup_failures(tmp_path) -> None:
    primary_cause = ValueError("model exploded")
    primary = CheckpointComparisonRunnerError("base-inference", "model exploded")
    primary.__cause__ = primary_cause
    paths = (tmp_path / "one", tmp_path / "two")
    failures = tuple(
        runner_module.RunnerCleanupFailure(
            resource=f"resource-{index}",
            exception_type="OSError",
            message=f"cleanup {index} failed",
            recoverable_path=path,
            manual_retry_possible=True,
            installed_output_remains_valid=False,
        )
        for index, path in enumerate(paths)
    )

    with pytest.raises(CheckpointComparisonRunnerError) as caught:
        runner_module._resolve_cleanup_outcome(primary, primary.__traceback__, failures)

    assert caught.value is primary
    assert caught.value.phase == "base-inference"
    assert caught.value.__cause__ is primary_cause
    assert caught.value.cleanup_failures == failures
    assert caught.value.recoverable_paths == paths


def test_cleanup_operations_continue_after_failures(tmp_path) -> None:
    events = []

    def fail(name):
        def operation():
            events.append(name)
            raise OSError(f"{name} failed")

        return operation

    paths = (tmp_path / "one", tmp_path / "two")
    for path in paths:
        path.mkdir()
    failures = runner_module._attempt_cleanups(
        tuple(
            (f"resource-{index}", path, fail(str(index)))
            for index, path in enumerate(paths)
        ),
        installed_output_remains_valid=True,
    )
    assert events == ["0", "1"]
    assert len(failures) == 2
    assert all(item.installed_output_remains_valid for item in failures)


def test_successful_work_with_cleanup_failure_raises_cleanup_phase(tmp_path) -> None:
    failure = runner_module.RunnerCleanupFailure(
        resource="temporary-run-directory",
        exception_type="OSError",
        message="busy",
        recoverable_path=tmp_path,
        manual_retry_possible=True,
        installed_output_remains_valid=True,
    )
    with pytest.raises(CheckpointComparisonRunnerError) as caught:
        runner_module._resolve_cleanup_outcome(None, None, (failure,))
    assert caught.value.phase == "cleanup"
    assert caught.value.cleanup_failures == (failure,)
    assert "installed run remains valid" in caught.value.message


def test_production_dependencies_preserve_manifest_processor_device(
    monkeypatch, tmp_path
) -> None:
    path, _ = _manifest(
        tmp_path,
        lambda document: document["runtime"].update(device="cuda:3"),
    )
    manifest = load_checkpoint_comparison_runner_manifest(path)
    captured = {}

    def build(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runner_module, "build_verified_pi05_processor_pair", build)
    dependencies = runner_module.DefaultCheckpointComparisonRunnerDependencies()
    assert dependencies.execution_device(manifest) == "cuda:3"
    dependencies.build_processors(object(), object(), object(), device="cuda:3")
    assert captured["device"] == "cuda:3"


@pytest.mark.parametrize("relationship", ["contains-camera", "inside-camera-source"])
def test_runner_rejects_camera_output_overlap_before_camera_preparation(
    tmp_path, relationship
) -> None:
    output_relative = "inputs/observation/generated-run"
    path, _ = _manifest(
        tmp_path,
        lambda document: document["output"].update(
            runDirectory=output_relative, replaceExisting=True
        ),
    )
    manifest = load_checkpoint_comparison_runner_manifest(path)
    resolved = resolve_checkpoint_comparison_runner_inputs(manifest)
    source = observation()
    if relationship == "contains-camera":
        camera_parent = resolved.output_run_directory
    else:
        camera_parent = resolved.output_run_directory.parent
    cameras = tuple(
        replace(camera, path=camera_parent / f"camera-{index}.png")
        for index, camera in enumerate(source.cameras)
    )
    for camera in cameras:
        camera.path.parent.mkdir(parents=True, exist_ok=True)
        camera.path.write_bytes(b"camera-source")
    sentinel = resolved.output_run_directory / "existing.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_bytes(b"preserve-me")
    source = replace(
        source,
        manifest_path=resolved.manifest_directory / manifest.observation_manifest.path,
        manifest_sha256=manifest.observation_manifest.sha256,
        manifest_byte_count=manifest.observation_manifest.byte_count,
        dataset=replace(
            source.dataset,
            repository_id=manifest.dataset.repository_id,
            revision=manifest.dataset.revision,
        ),
        cameras=cameras,
    )
    events = []

    class Dependencies:
        load_manifest = staticmethod(load_checkpoint_comparison_runner_manifest)
        resolve = staticmethod(resolve_checkpoint_comparison_runner_inputs)

        def preflight(self, manifest, resolved):
            events.append("preflight")
            return SimpleNamespace(passed=True)

        def load_observation(self, resolved):
            events.append("observation")
            return source

        def prepare_cameras(self, value):
            pytest.fail("camera preparation must not run after overlap rejection")

    with pytest.raises(CheckpointComparisonRunnerError) as caught:
        execute_checkpoint_comparison_run(path, dependencies=Dependencies())
    assert caught.value.phase == "resolution"
    assert "observation.cameras[0]" in str(caught.value)
    expected = (
        "output-contains-input"
        if relationship == "contains-camera"
        else "output-inside-input"
    )
    assert expected in str(caught.value)
    assert events == ["preflight", "observation"]
    assert sentinel.read_bytes() == b"preserve-me"
    assert tuple(camera.path.read_bytes() for camera in cameras) == (
        b"camera-source",
        b"camera-source",
        b"camera-source",
    )
    assert not any(tmp_path.rglob("*.previous-*"))


def test_runner_staging_candidate_is_rejected_before_creation_when_inside_output(
    monkeypatch, tmp_path
) -> None:
    path, _ = _manifest(
        tmp_path,
        lambda document: document["output"].update(replaceExisting=True),
    )
    manifest = load_checkpoint_comparison_runner_manifest(path)
    resolved = resolve_checkpoint_comparison_runner_inputs(manifest)
    resolved.output_run_directory.mkdir(parents=True)
    candidate = resolved.output_run_directory / (
        "lerobot-state-atlas-comparison-" + "1" * 32
    )
    monkeypatch.setattr(
        runner_module,
        "uuid4",
        lambda: SimpleNamespace(hex="1" * 32),
    )
    with pytest.raises(ValueError, match="output-inside-input"):
        runner_module._create_runner_staging_directory(
            resolved, (), temp_directory=resolved.output_run_directory
        )
    assert not candidate.exists()


def test_runner_staging_directory_uses_the_preflight_resolved_root(tmp_path) -> None:
    path, _ = _manifest(tmp_path / "manifest")
    manifest = load_checkpoint_comparison_runner_manifest(path)
    resolved = resolve_checkpoint_comparison_runner_inputs(manifest)
    staging_root = tmp_path / "injected-system-temp"
    staging_root.mkdir()
    temporary = runner_module._create_runner_staging_directory(
        resolved, (), staging_root=staging_root
    )
    try:
        assert temporary.path.parent == staging_root
    finally:
        temporary.cleanup()
