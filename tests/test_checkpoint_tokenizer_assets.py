from dataclasses import replace
import os
from pathlib import Path
import socket

import pytest

from lerobot_state_atlas.checkpoint_comparison import (
    PI05TokenizerLoadingError,
    PI05TokenizerVerificationError,
    load_verified_local_pi05_tokenizer,
    verify_local_pi05_tokenizer_directory,
    verify_pi05_processor_assets,
)
from lerobot_state_atlas.checkpoint_comparison.local_files import (
    StableFileSnapshotError,
)
from lerobot_state_atlas.checkpoint_comparison.tokenizer_assets import (
    _canonical_identity,
)
from lerobot_state_atlas.checkpoint_comparison.runner_models import (
    TokenizerFileIdentity,
)
from test_checkpoint_processor_compatibility import runner_assets


def setup_tokenizer(tmp_path: Path, monkeypatch):
    manifest, resolved, _ = runner_assets(tmp_path, monkeypatch)
    processor = verify_pi05_processor_assets(manifest, resolved)
    root = next(
        entry.canonical_path
        for entry in resolved.inventory
        if entry.logical_input_id == "processors.tokenizerDirectory"
    )
    files = {
        "tokenizer_config.json": b'{"model_max_length": 200}',
        "tokenizer.json": b'{"version":"1.0"}',
        "nested/special_tokens_map.json": b'{"eos_token":"</s>"}',
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    identities = tuple(
        TokenizerFileIdentity(
            relative, len(content), __import__("hashlib").sha256(content).hexdigest()
        )
        for relative, content in sorted(files.items())
    )
    identity = _canonical_identity(identities)
    tokenizer_input = replace(
        manifest.processors.tokenizer_directory, identity_sha256=identity
    )
    manifest = replace(
        manifest,
        processors=replace(manifest.processors, tokenizer_directory=tokenizer_input),
    )
    inventory = tuple(
        replace(entry, expected_sha256=identity)
        if entry.logical_input_id == "processors.tokenizerDirectory"
        else entry
        for entry in resolved.inventory
    )
    resolved = replace(resolved, inventory=inventory)
    return manifest, resolved, processor, root, files


def test_deterministic_recursive_identity_and_single_acquisition(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, resolved, processor, root, files = setup_tokenizer(tmp_path, monkeypatch)
    calls = []
    result = verify_local_pi05_tokenizer_directory(
        manifest,
        resolved,
        processor,
        snapshot_reader=lambda path: calls.append(path) or path.read_bytes(),
    )
    repeated = verify_local_pi05_tokenizer_directory(manifest, resolved, processor)
    assert result == repeated
    assert tuple(item.relative_path for item in result.files) == tuple(sorted(files))
    assert calls == [root / relative for relative in sorted(files)]
    assert result.repository_id == "google/paligemma-3b-pt-224"
    assert result.layout_uncertainty


@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
def test_nested_symlinks_are_rejected(tmp_path: Path, monkeypatch, kind: str) -> None:
    manifest, resolved, processor, root, _ = setup_tokenizer(tmp_path, monkeypatch)
    link = root / f"bad-{kind}"
    if kind == "directory":
        link.symlink_to(root / "nested", target_is_directory=True)
    elif kind == "file":
        link.symlink_to(root / "tokenizer.json")
    else:
        link.symlink_to(root / "missing")
    with pytest.raises(PI05TokenizerVerificationError, match="symbolic-link"):
        verify_local_pi05_tokenizer_directory(manifest, resolved, processor)


def test_missing_layout_and_wrong_identity_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, resolved, processor, root, _ = setup_tokenizer(tmp_path, monkeypatch)
    (root / "tokenizer_config.json").unlink()
    with pytest.raises(PI05TokenizerVerificationError, match="tokenizer_config"):
        verify_local_pi05_tokenizer_directory(manifest, resolved, processor)
    (root / "tokenizer_config.json").write_bytes(b"{}")
    with pytest.raises(PI05TokenizerVerificationError, match="identitySha256"):
        verify_local_pi05_tokenizer_directory(manifest, resolved, processor)


def test_acquisition_mutation_is_rejected(tmp_path: Path, monkeypatch) -> None:
    manifest, resolved, processor, _, _ = setup_tokenizer(tmp_path, monkeypatch)

    def fail(path):
        raise StableFileSnapshotError("changed while read")

    with pytest.raises(PI05TokenizerVerificationError, match="changed while"):
        verify_local_pi05_tokenizer_directory(
            manifest, resolved, processor, snapshot_reader=fail
        )


def test_offline_local_loading_and_environment_restoration(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, resolved, processor, root, _ = setup_tokenizer(tmp_path, monkeypatch)
    verification = verify_local_pi05_tokenizer_directory(manifest, resolved, processor)
    calls = []

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, *args, **kwargs):
            return {}

    def loader(path, **kwargs):
        calls.append(
            (
                path,
                kwargs,
                os.environ["HF_HUB_OFFLINE"],
                os.environ["TRANSFORMERS_OFFLINE"],
            )
        )
        return Tokenizer()

    monkeypatch.setenv("HF_HUB_OFFLINE", "prior")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    tokenizer = load_verified_local_pi05_tokenizer(
        verification, tokenizer_loader=loader
    )
    assert callable(tokenizer)
    assert calls == [
        (root, {"local_files_only": True, "trust_remote_code": False}, "1", "1")
    ]
    assert os.environ["HF_HUB_OFFLINE"] == "prior"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_loading_reverifies_tree_and_rejects_incapable_tokenizer(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, resolved, processor, root, _ = setup_tokenizer(tmp_path, monkeypatch)
    verification = verify_local_pi05_tokenizer_directory(manifest, resolved, processor)
    (root / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(PI05TokenizerLoadingError, match="changed before loading"):
        load_verified_local_pi05_tokenizer(
            verification, tokenizer_loader=lambda *a, **k: object()
        )
    (root / "tokenizer.json").write_bytes(b'{"version":"1.0"}')
    with pytest.raises(PI05TokenizerLoadingError, match="must be callable"):
        load_verified_local_pi05_tokenizer(
            verification, tokenizer_loader=lambda *a, **k: object()
        )


def test_no_network_boundary(tmp_path: Path, monkeypatch) -> None:
    manifest, resolved, processor, _, _ = setup_tokenizer(tmp_path, monkeypatch)
    verification = verify_local_pi05_tokenizer_directory(manifest, resolved, processor)
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("network")
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, *args, **kwargs):
            return {}

    assert load_verified_local_pi05_tokenizer(
        verification, tokenizer_loader=lambda *a, **k: Tokenizer()
    )
