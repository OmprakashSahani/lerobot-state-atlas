import json
from pathlib import Path
import re

import pytest
from jsonschema import Draft202012Validator

from lerobot_state_atlas.checkpoint_comparison.artifact import (
    build_checkpoint_comparison_documents,
    install_checkpoint_comparison_bundle,
)
from lerobot_state_atlas.checkpoint_comparison.serialize import (
    deterministic_json_bytes,
    sha256_bytes,
)
from lerobot_state_atlas.checkpoint_comparison.validate import (
    CheckpointComparisonValidationError,
    validate_checkpoint_comparison,
)
import lerobot_state_atlas.checkpoint_comparison.validate as validate_module
from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
)


FIXTURE = Path(__file__).parent / "fixtures/checkpoint-comparison-v1"


def documents() -> tuple[dict, dict]:
    return (
        json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8")),
        json.loads((FIXTURE / "plans.json").read_text(encoding="utf-8")),
    )


def write_bundle(path: Path, manifest: dict, plans: dict) -> None:
    path.mkdir()
    manifest_bytes, plans_bytes = build_checkpoint_comparison_documents(manifest, plans)
    (path / "manifest.json").write_bytes(manifest_bytes)
    (path / "plans.json").write_bytes(plans_bytes)


def replace_payload_bytes(bundle: Path, content: bytes) -> None:
    (bundle / "plans.json").write_bytes(content)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payloads"][0]["byteSize"] = len(content)
    manifest["payloads"][0]["sha256"] = sha256_bytes(content)
    manifest_path.write_bytes(deterministic_json_bytes(manifest))


def test_valid_synthetic_fixture() -> None:
    manifest = validate_checkpoint_comparison(FIXTURE)
    assert manifest["bundleId"] == "synthetic-checkpoint-comparison-v1"
    assert manifest["noise"]["shape"] == [1, 50, 32]
    assert manifest["schema"]["major"] == 1
    assert manifest["schema"]["minor"] == 0


def test_manifest_is_acquired_once_and_snapshot_drives_parsing(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    real_snapshot = validate_module.read_stable_file_snapshot
    real_parse = validate_module._load_manifest_snapshot
    acquired: list[bytes] = []
    parsed: list[bytes] = []

    def acquire(path: Path) -> bytes:
        content = real_snapshot(path)
        if path.name == "manifest.json":
            acquired.append(content)
        return content

    def parse(content: bytes, filename: str):
        parsed.append(content)
        return real_parse(content, filename)

    monkeypatch.setattr(validate_module, "read_stable_file_snapshot", acquire)
    monkeypatch.setattr(validate_module, "_load_manifest_snapshot", parse)

    validate_checkpoint_comparison(bundle)

    assert len(acquired) == 1
    assert parsed == acquired


@pytest.mark.parametrize("dangling", [False, True])
def test_manifest_symlink_is_rejected_without_following_target(
    tmp_path: Path, dangling: bool
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    manifest_path = bundle / "manifest.json"
    target = tmp_path / "external-manifest.json"
    original = manifest_path.read_bytes()
    if not dangling:
        target.write_bytes(original)
    manifest_path.unlink()
    manifest_path.symlink_to(target)

    with pytest.raises(
        CheckpointComparisonValidationError, match="manifest.*symbolic link"
    ):
        validate_checkpoint_comparison(bundle)

    assert manifest_path.is_symlink()
    if not dangling:
        assert target.read_bytes() == original


def test_manifest_path_replacement_after_snapshot_does_not_change_validation(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    manifest_path = bundle / "manifest.json"
    original = manifest_path.read_bytes()
    replacement_document = json.loads(original)
    replacement_document["bundleId"] = "x" * len(replacement_document["bundleId"])
    replacement = deterministic_json_bytes(replacement_document)
    assert len(replacement) == len(original)
    real_snapshot = validate_module.read_stable_file_snapshot

    def acquire_then_replace(path: Path) -> bytes:
        content = real_snapshot(path)
        if path.name == "manifest.json":
            path.write_bytes(replacement)
        return content

    monkeypatch.setattr(
        validate_module, "read_stable_file_snapshot", acquire_then_replace
    )

    validated = validate_checkpoint_comparison(bundle)

    assert validated["bundleId"] == manifest["bundleId"]
    assert manifest_path.read_bytes() == replacement


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"\xff", "Invalid manifest UTF-8"), (b"{broken", "Invalid manifest JSON")],
)
def test_manifest_snapshot_decode_failures_are_precise(
    tmp_path: Path, content: bytes, message: str
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_bytes(content)

    with pytest.raises(CheckpointComparisonValidationError, match=message):
        validate_checkpoint_comparison(bundle)


def test_manifest_mutation_during_snapshot_acquisition_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)

    def changed(path: Path) -> bytes:
        raise StableFileSnapshotError(f"{path.name} changed while it was being read")

    monkeypatch.setattr(validate_module, "read_stable_file_snapshot", changed)
    with pytest.raises(
        CheckpointComparisonValidationError,
        match=r"manifest.*manifest\.json.*changed while it was being read",
    ):
        validate_checkpoint_comparison(bundle)


def test_manifest_missing_nonregular_and_unreadable_errors_are_precise(
    tmp_path: Path, monkeypatch
) -> None:
    missing_bundle = tmp_path / "missing"
    missing_bundle.mkdir()
    with pytest.raises(CheckpointComparisonValidationError, match="manifest.*missing"):
        validate_checkpoint_comparison(missing_bundle)

    directory_bundle = tmp_path / "directory"
    (directory_bundle / "manifest.json").mkdir(parents=True)
    with pytest.raises(
        CheckpointComparisonValidationError, match="manifest.*regular file"
    ):
        validate_checkpoint_comparison(directory_bundle)

    manifest, plans = documents()
    unreadable_bundle = tmp_path / "unreadable"
    write_bundle(unreadable_bundle, manifest, plans)

    def unreadable(path: Path) -> bytes:
        raise StableFileSnapshotError(f"could not read {path.name}")

    monkeypatch.setattr(validate_module, "read_stable_file_snapshot", unreadable)
    with pytest.raises(
        CheckpointComparisonValidationError,
        match=r"manifest.*manifest\.json.*could not read",
    ):
        validate_checkpoint_comparison(unreadable_bundle)


def test_manifest_and_payload_bytes_remain_unchanged_after_validation(
    tmp_path: Path,
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    before = {path: path.read_bytes() for path in bundle.iterdir()}

    validate_checkpoint_comparison(bundle)

    assert {path: path.read_bytes() for path in bundle.iterdir()} == before


@pytest.mark.parametrize(
    ("document", "field", "value", "expected_path"),
    [
        ("manifest", "major", True, "manifest.schema.major"),
        ("manifest", "minor", False, "manifest.schema.minor"),
        ("plans", "major", True, "plans.schema.major"),
        ("plans", "minor", False, "plans.schema.minor"),
    ],
)
def test_schema_versions_reject_booleans_with_precise_field_path(
    tmp_path: Path,
    document: str,
    field: str,
    value: bool,
    expected_path: str,
) -> None:
    manifest, plans = documents()
    target = manifest if document == "manifest" else plans
    target["schema"][field] = value
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)

    with pytest.raises(
        CheckpointComparisonValidationError,
        match=rf"^{expected_path} must be an integer\.$",
    ):
        validate_checkpoint_comparison(bundle)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_noise_shape_rejects_boolean_at_each_index(tmp_path: Path, index: int) -> None:
    manifest, plans = documents()
    manifest["noise"]["shape"][index] = True
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)

    with pytest.raises(
        CheckpointComparisonValidationError,
        match=rf"^manifest\.noise\.shape\[{index}\] must be an integer\.$",
    ):
        validate_checkpoint_comparison(bundle)


@pytest.mark.parametrize("value", [1.0, "1", None])
def test_noise_shape_rejects_other_non_integer_values(
    tmp_path: Path, value: object
) -> None:
    manifest, plans = documents()
    manifest["noise"]["shape"][0] = value
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)

    with pytest.raises(
        CheckpointComparisonValidationError,
        match=r"^manifest\.noise\.shape\[0\] must be an integer\.$",
    ):
        validate_checkpoint_comparison(bundle)


@pytest.mark.parametrize("shape", [[1, 50], [1, 50, 32, 14]])
def test_noise_shape_rejects_wrong_array_length(
    tmp_path: Path, shape: list[int]
) -> None:
    manifest, plans = documents()
    manifest["noise"]["shape"] = shape
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)

    with pytest.raises(
        CheckpointComparisonValidationError,
        match=r"^manifest\.noise\.shape must be \[1, 50, 32\]\.$",
    ):
        validate_checkpoint_comparison(bundle)


def test_valid_noise_shape_passes_strict_integer_validation(tmp_path: Path) -> None:
    manifest, plans = documents()
    manifest["noise"]["shape"] = [1, 50, 32]
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)

    validated = validate_checkpoint_comparison(bundle)
    assert validated["noise"]["shape"] == [1, 50, 32]


@pytest.mark.parametrize(
    ("document", "field", "expected_path"),
    [
        ("manifest", "actionDimension", "manifest.comparison.actionDimension"),
        ("manifest", "chunkLength", "manifest.comparison.chunkLength"),
        ("plans", "actionDimension", "plans.actionDimension"),
        ("plans", "chunkLength", "plans.chunkLength"),
    ],
)
def test_numeric_dimensions_reject_booleans_with_precise_field_path(
    tmp_path: Path,
    document: str,
    field: str,
    expected_path: str,
) -> None:
    manifest, plans = documents()
    target = manifest["comparison"] if document == "manifest" else plans
    target[field] = True
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)

    with pytest.raises(
        CheckpointComparisonValidationError,
        match=rf"^{expected_path} must be an integer\.$",
    ):
        validate_checkpoint_comparison(bundle)


def test_json_schema_represents_both_fixture_documents() -> None:
    schema_path = (
        Path(__file__).parents[1] / "schemas/checkpoint-comparison-v1.schema.json"
    )
    validator = Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    manifest, plans = documents()
    validator.validate(manifest)
    validator.validate(plans)


def test_deterministic_bytes_newline_and_key_order() -> None:
    manifest, plans = documents()
    first = build_checkpoint_comparison_documents(manifest, plans)
    second = build_checkpoint_comparison_documents(manifest, plans)
    assert first == second
    assert all(value.endswith(b"\n") for value in first)
    assert first[0].startswith(b'{"bundleId"')
    assert first[1].startswith(b'{"actionDimension"')
    with pytest.raises(ValueError, match="Out of range float values"):
        deterministic_json_bytes({"invalid": float("nan")})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest, plans: manifest["schema"].update(name="invalid"),
            "manifest.schema.name",
        ),
        (
            lambda manifest, plans: manifest["schema"].update(minor=1),
            "manifest.schema.minor",
        ),
        (
            lambda manifest, plans: manifest["dataset"].update(revision="ABC"),
            "manifest.dataset.revision",
        ),
        (
            lambda manifest, plans: manifest["policies"].reverse(),
            "manifest.policies[0].policyId",
        ),
        (lambda manifest, plans: manifest["policies"].pop(), "exactly two policies"),
        (
            lambda manifest, plans: manifest["policies"][0].update(
                label="First checkpoint"
            ),
            "manifest.policies[0].label",
        ),
        (
            lambda manifest, plans: manifest["noise"].update(shape=[1, 50, 14]),
            "manifest.noise.shape",
        ),
        (
            lambda manifest, plans: plans["plans"][0]["actions"][0].pop(),
            "plans.plans[0].actions[0]",
        ),
        (lambda manifest, plans: plans["plans"].reverse(), "plans.plans[0].policyId"),
        (
            lambda manifest, plans: plans.update(observationId="other"),
            "plans.observationId",
        ),
        (
            lambda manifest, plans: plans.update(actionDimension=13),
            "plans.actionDimension",
        ),
    ],
)
def test_strict_validation_failures(tmp_path: Path, mutation, message: str) -> None:
    manifest, plans = documents()
    mutation(manifest, plans)
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    with pytest.raises(CheckpointComparisonValidationError, match=re.escape(message)):
        validate_checkpoint_comparison(bundle)


def test_unsafe_payload_filename_is_precise(tmp_path: Path) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    saved = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    saved["payloads"][0]["filename"] = "../plans.json"
    (bundle / "manifest.json").write_bytes(deterministic_json_bytes(saved))
    with pytest.raises(CheckpointComparisonValidationError, match="filename"):
        validate_checkpoint_comparison(bundle)


@pytest.mark.parametrize("field", ["byteSize", "sha256"])
def test_payload_integrity_mismatch(tmp_path: Path, field: str) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    saved = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    saved["payloads"][0][field] = 1 if field == "byteSize" else "f" * 64
    (bundle / "manifest.json").write_bytes(deterministic_json_bytes(saved))
    with pytest.raises(CheckpointComparisonValidationError, match=field):
        validate_checkpoint_comparison(bundle)


def test_payload_symlink_escaping_bundle_is_rejected_without_modifying_target(
    tmp_path: Path,
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    external_payload = tmp_path / "external-plans.json"
    external_contents = (bundle / "plans.json").read_bytes()
    external_payload.write_bytes(external_contents)
    (bundle / "plans.json").unlink()
    (bundle / "plans.json").symlink_to(external_payload)

    with pytest.raises(
        CheckpointComparisonValidationError,
        match=r"manifest\.payloads\[0\]\.filename.*plans\.json.*outside",
    ):
        validate_checkpoint_comparison(bundle)

    assert external_payload.read_bytes() == external_contents


def test_internal_payload_symlink_is_rejected_explicitly(tmp_path: Path) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    internal_target = bundle / "stored-plans.json"
    (bundle / "plans.json").replace(internal_target)
    (bundle / "plans.json").symlink_to(internal_target.name)

    with pytest.raises(
        CheckpointComparisonValidationError,
        match=r"manifest\.payloads\[0\]\.filename.*plans\.json.*not a symlink",
    ):
        validate_checkpoint_comparison(bundle)


def test_normal_in_bundle_payload_remains_valid(tmp_path: Path) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    assert not (bundle / "plans.json").is_symlink()
    assert validate_checkpoint_comparison(bundle)["bundleId"] == manifest["bundleId"]


def test_payload_is_acquired_once_and_one_snapshot_drives_all_validation(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    real_snapshot = validate_module.read_stable_file_snapshot
    real_hash = validate_module.sha256_bytes
    real_load = validate_module._load_json_snapshot
    snapshots: list[bytes] = []
    hashed: list[bytes] = []
    parsed: list[bytes] = []

    def acquire(path: Path) -> bytes:
        content = real_snapshot(path)
        if path.name == "plans.json":
            snapshots.append(content)
        return content

    def hash_snapshot(content: bytes) -> str:
        hashed.append(content)
        return real_hash(content)

    def parse_snapshot(content: bytes, filename: str):
        parsed.append(content)
        return real_load(content, filename)

    monkeypatch.setattr(validate_module, "read_stable_file_snapshot", acquire)
    monkeypatch.setattr(validate_module, "sha256_bytes", hash_snapshot)
    monkeypatch.setattr(validate_module, "_load_json_snapshot", parse_snapshot)

    validate_checkpoint_comparison(bundle)

    assert len(snapshots) == 1
    assert hashed == snapshots
    assert parsed == snapshots


def test_payload_path_replacement_after_snapshot_cannot_mix_validation_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    payload_path = bundle / "plans.json"
    original = payload_path.read_bytes()
    replacement_document = json.loads(original)
    observation_id = replacement_document["observationId"]
    replacement_document["observationId"] = observation_id[:-1] + (
        "x" if observation_id[-1] != "x" else "y"
    )
    replacement = deterministic_json_bytes(replacement_document)
    assert len(replacement) == len(original)
    real_snapshot = validate_module.read_stable_file_snapshot

    def acquire_then_replace(path: Path) -> bytes:
        content = real_snapshot(path)
        if path.name == "plans.json":
            path.write_bytes(replacement)
        return content

    monkeypatch.setattr(
        validate_module, "read_stable_file_snapshot", acquire_then_replace
    )

    validated = validate_checkpoint_comparison(bundle)

    assert validated["bundleId"] == manifest["bundleId"]
    assert payload_path.read_bytes() == replacement


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\xff", "Invalid plans payload UTF-8"),
        (b"{broken", "Invalid plans payload JSON"),
    ],
)
def test_payload_snapshot_decode_failures_are_precise(
    tmp_path: Path, content: bytes, message: str
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    replace_payload_bytes(bundle, content)

    with pytest.raises(CheckpointComparisonValidationError, match=message):
        validate_checkpoint_comparison(bundle)


def test_payload_mutation_during_snapshot_acquisition_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)

    real_snapshot = validate_module.read_stable_file_snapshot

    def changed(path: Path) -> bytes:
        if path.name == "plans.json":
            raise StableFileSnapshotError("plans.json changed while it was being read")
        return real_snapshot(path)

    monkeypatch.setattr(validate_module, "read_stable_file_snapshot", changed)
    with pytest.raises(
        CheckpointComparisonValidationError,
        match=r"manifest\.payloads\[0\]\.filename.*plans\.json.*changed while it was being read",
    ):
        validate_checkpoint_comparison(bundle)


def test_nonfinite_action_is_rejected(tmp_path: Path) -> None:
    manifest, plans = documents()
    plans["plans"][1]["actions"][12][3] = float("inf")
    plans_bytes = json.dumps(plans, allow_nan=True).encode() + b"\n"
    manifest["payloads"] = [
        {
            "kind": "plans",
            "filename": "plans.json",
            "encoding": "json",
            "byteSize": len(plans_bytes),
            "sha256": sha256_bytes(plans_bytes),
        }
    ]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_bytes(deterministic_json_bytes(manifest))
    (bundle / "plans.json").write_bytes(plans_bytes)
    with pytest.raises(
        CheckpointComparisonValidationError, match=r"actions\[12\]\[3\].*finite"
    ):
        validate_checkpoint_comparison(bundle)


def test_recorded_ground_truth_consistency(tmp_path: Path) -> None:
    manifest, plans = documents()
    plans["recordedGroundTruth"] = {"available": False, "reason": None}
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    with pytest.raises(
        CheckpointComparisonValidationError, match="recordedGroundTruth.reason"
    ):
        validate_checkpoint_comparison(bundle)

    manifest, plans = documents()
    plans["recordedGroundTruth"] = {
        "available": True,
        "reason": None,
    }
    bundle = tmp_path / "bundle-available"
    write_bundle(bundle, manifest, plans)
    with pytest.raises(CheckpointComparisonValidationError, match="must include"):
        validate_checkpoint_comparison(bundle)


def test_cross_file_recorded_ground_truth_must_match(tmp_path: Path) -> None:
    manifest, plans = documents()
    manifest["recordedGroundTruth"]["reason"] = "Different reason."
    bundle = tmp_path / "bundle"
    write_bundle(bundle, manifest, plans)
    with pytest.raises(CheckpointComparisonValidationError, match="does not match"):
        validate_checkpoint_comparison(bundle)


def test_atomic_success_and_replacement(tmp_path: Path) -> None:
    manifest, plans = documents()
    destination = tmp_path / "comparison"
    result = install_checkpoint_comparison_bundle(destination, manifest, plans)
    assert result.output_path == destination.resolve()
    assert (
        validate_checkpoint_comparison(destination)["bundleId"] == manifest["bundleId"]
    )
    (destination / "obsolete.txt").write_text("old", encoding="utf-8")
    install_checkpoint_comparison_bundle(destination, manifest, plans)
    assert not (destination / "obsolete.txt").exists()
    assert not list(tmp_path.glob(".comparison.*"))


def test_existing_file_destination_is_rejected_without_filesystem_changes(
    tmp_path: Path,
) -> None:
    manifest, plans = documents()
    destination = tmp_path / "comparison"
    original_contents = b"existing comparison destination\x00\xff"
    destination.write_bytes(original_contents)
    entries_before = set(tmp_path.iterdir())

    with pytest.raises(ValueError, match="destination must be a directory path"):
        install_checkpoint_comparison_bundle(destination, manifest, plans)

    assert destination.is_file()
    assert destination.read_bytes() == original_contents
    assert set(tmp_path.iterdir()) == entries_before
    assert not list(tmp_path.glob(".comparison.*"))


def test_symlink_destination_is_rejected_without_touching_target(
    tmp_path: Path,
) -> None:
    manifest, plans = documents()
    target = tmp_path / "unrelated-target"
    target.mkdir()
    target_file = target / "important.bin"
    original_contents = b"unrelated data\x00\xff"
    target_file.write_bytes(original_contents)
    destination = tmp_path / "comparison"
    destination.symlink_to(target, target_is_directory=True)
    original_link_target = destination.readlink()
    entries_before = set(tmp_path.iterdir())

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        install_checkpoint_comparison_bundle(destination, manifest, plans)

    assert destination.is_symlink()
    assert destination.readlink() == original_link_target
    assert target.is_dir()
    assert target_file.read_bytes() == original_contents
    assert set(tmp_path.iterdir()) == entries_before
    assert not list(tmp_path.glob(".comparison.*"))


def test_dangling_symlink_destination_is_rejected_without_modification(
    tmp_path: Path,
) -> None:
    manifest, plans = documents()
    missing_target = tmp_path / "missing-target"
    destination = tmp_path / "comparison"
    destination.symlink_to(missing_target, target_is_directory=True)
    original_link_target = destination.readlink()
    entries_before = set(tmp_path.iterdir())

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        install_checkpoint_comparison_bundle(destination, manifest, plans)

    assert destination.is_symlink()
    assert destination.readlink() == original_link_target
    assert not missing_target.exists()
    assert set(tmp_path.iterdir()) == entries_before
    assert not list(tmp_path.glob(".comparison.*"))


def test_invalid_staged_bundle_leaves_no_partial_destination(tmp_path: Path) -> None:
    manifest, plans = documents()
    manifest["noise"]["shape"] = [1, 50, 14]
    destination = tmp_path / "comparison"
    with pytest.raises(CheckpointComparisonValidationError):
        install_checkpoint_comparison_bundle(destination, manifest, plans)
    assert not destination.exists()
    assert not list(tmp_path.glob(".comparison.*"))


def test_atomic_install_rolls_back_existing_destination(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    destination = tmp_path / "comparison"
    destination.mkdir()
    (destination / "sentinel.txt").write_text("preserve", encoding="utf-8")
    import lerobot_state_atlas.checkpoint_comparison.artifact as artifact

    real_replace = artifact.os.replace
    calls = 0

    def fail_install(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("install failed")
        return real_replace(source, target)

    monkeypatch.setattr(artifact.os, "replace", fail_install)
    with pytest.raises(OSError, match="install failed"):
        install_checkpoint_comparison_bundle(destination, manifest, plans)
    assert (destination / "sentinel.txt").read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob(".comparison.*"))


def test_backup_cleanup_failure_before_deletion_preserves_new_and_old_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    destination = tmp_path / "comparison"
    destination.mkdir()
    original_contents = b"original-bundle-bytes"
    (destination / "sentinel.bin").write_bytes(original_contents)
    import lerobot_state_atlas.checkpoint_comparison.artifact as artifact

    real_rmtree = artifact.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if ".previous-" in Path(path).name:
            raise OSError("simulated backup cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(artifact.shutil, "rmtree", fail_backup_cleanup)
    with pytest.raises(
        artifact.CheckpointComparisonInstallError,
        match=r"new bundle remains installed.*may be partial.*must not be blindly restored",
    ) as raised:
        install_checkpoint_comparison_bundle(destination, manifest, plans)

    assert (
        validate_checkpoint_comparison(destination)["bundleId"] == manifest["bundleId"]
    )
    assert not (destination / "sentinel.bin").exists()
    backups = list(tmp_path.glob(".comparison.previous-*"))
    assert len(backups) == 1
    assert (backups[0] / "sentinel.bin").read_bytes() == original_contents
    assert str(destination) in str(raised.value)
    assert str(backups[0]) in str(raised.value)


def test_install_failure_restore_failure_preserves_intact_backup(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    destination = tmp_path / "comparison"
    destination.mkdir()
    (destination / "sentinel.bin").write_bytes(b"original")
    import lerobot_state_atlas.checkpoint_comparison.artifact as artifact

    real_replace = artifact.os.replace
    install_attempted = False

    def fail_install_and_restore(source, target):
        nonlocal install_attempted
        if Path(target) == destination and ".previous-" not in Path(source).name:
            install_attempted = True
            raise OSError("simulated install failure")
        if ".previous-" in Path(source).name and Path(target) == destination:
            raise OSError("simulated restore failure")
        return real_replace(source, target)

    monkeypatch.setattr(artifact.os, "replace", fail_install_and_restore)
    with pytest.raises(OSError, match="restore failure"):
        install_checkpoint_comparison_bundle(destination, manifest, plans)

    assert install_attempted
    backups = list(tmp_path.glob(".comparison.previous-*"))
    assert len(backups) == 1
    assert (backups[0] / "sentinel.bin").read_bytes() == b"original"
    assert not destination.exists()


def test_partial_backup_cleanup_preserves_valid_destination_and_remainder(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    destination = tmp_path / "comparison"
    destination.mkdir()
    (destination / "deleted-first.bin").write_bytes(b"old-deleted")
    (destination / "remaining.bin").write_bytes(b"old-remaining")
    import lerobot_state_atlas.checkpoint_comparison.artifact as artifact

    real_rmtree = artifact.shutil.rmtree
    real_replace = artifact.os.replace
    operations = []

    def partially_remove_then_fail(path, *args, **kwargs):
        path = Path(path)
        operations.append(("rmtree", path))
        if ".previous-" in path.name:
            (path / "deleted-first.bin").unlink()
            raise OSError("simulated partial backup cleanup")
        return real_rmtree(path, *args, **kwargs)

    def record_replace(source, target):
        operations.append(("replace", Path(source), Path(target)))
        return real_replace(source, target)

    monkeypatch.setattr(artifact.shutil, "rmtree", partially_remove_then_fail)
    monkeypatch.setattr(artifact.os, "replace", record_replace)
    with pytest.raises(
        artifact.CheckpointComparisonInstallError,
        match=r"may be partial.*must not be blindly restored.*partial backup cleanup",
    ):
        install_checkpoint_comparison_bundle(destination, manifest, plans)

    assert (
        validate_checkpoint_comparison(destination)["bundleId"] == manifest["bundleId"]
    )
    destination_bytes = {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    }
    backups = list(tmp_path.glob(".comparison.previous-*"))
    assert len(backups) == 1
    assert not (backups[0] / "deleted-first.bin").exists()
    assert (backups[0] / "remaining.bin").read_bytes() == b"old-remaining"
    cleanup_index = next(
        index
        for index, operation in enumerate(operations)
        if operation[0] == "rmtree" and ".previous-" in operation[1].name
    )
    later = operations[cleanup_index + 1 :]
    assert not any(
        operation[0] == "replace"
        and (operation[1] == backups[0] or operation[2] == destination)
        for operation in later
    )
    assert not any(
        operation[0] == "rmtree" and operation[1] == destination for operation in later
    )
    assert {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    } == destination_bytes


def test_backup_deleted_then_cleanup_raises_preserves_valid_destination(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, plans = documents()
    destination = tmp_path / "comparison"
    destination.mkdir()
    (destination / "old.bin").write_bytes(b"old")
    import lerobot_state_atlas.checkpoint_comparison.artifact as artifact

    real_rmtree = artifact.shutil.rmtree

    def delete_then_fail(path, *args, **kwargs):
        path = Path(path)
        if ".previous-" in path.name:
            real_rmtree(path, *args, **kwargs)
            raise OSError("simulated post-delete cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(artifact.shutil, "rmtree", delete_then_fail)
    with pytest.raises(
        artifact.CheckpointComparisonInstallError,
        match=r"backup disappeared.*no backup recovery path remains",
    ) as raised:
        install_checkpoint_comparison_bundle(destination, manifest, plans)

    assert (
        validate_checkpoint_comparison(destination)["bundleId"] == manifest["bundleId"]
    )
    assert not list(tmp_path.glob(".comparison.previous-*"))
    assert "previous-" not in str(raised.value)


@pytest.mark.parametrize("demo", ["demo-v1", "demo-v2"])
def test_immutable_demo_destination_is_rejected(tmp_path: Path, demo: str) -> None:
    manifest, plans = documents()
    destination = tmp_path / "apps/web/public/atlas-data" / demo
    with pytest.raises(ValueError, match="immutable demo-v1 or demo-v2"):
        install_checkpoint_comparison_bundle(destination, manifest, plans)
    assert not destination.exists()
