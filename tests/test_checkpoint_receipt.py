import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lerobot_state_atlas.checkpoint_comparison import (
    CheckpointComparisonReceiptError,
    build_checkpoint_comparison_run_receipt,
    validate_checkpoint_comparison_run_receipt,
)


def receipt() -> dict:
    sha = "a" * 64
    revision = "b" * 40
    return {
        "schema": {
            "name": "lerobot-state-atlas.checkpoint-comparison-run-receipt",
            "major": 1,
            "minor": 0,
        },
        "runnerManifestSha256": sha,
        "software": {
            "projectVersion": "0.1.0",
            "sourceIdentity": "dirty-tree",
            "pythonVersion": "3.12",
            "lerobotVersion": "0.6.0",
            "torchVersion": "2",
            "safetensorsVersion": "1",
            "pillowVersion": "1",
            "transformersVersion": "1",
        },
        "runtime": {
            "device": "cuda:0",
            "modelDtype": "bfloat16",
            "noiseDtype": "float32",
            "cudaRuntime": "13",
            "cudaDriver": "unknown",
            "gpuName": "fake",
            "computeCapability": [8, 0],
            "deterministicSettings": {
                "cublasWorkspaceConfig": ":4096:8",
                "deterministicAlgorithms": True,
                "cudnnDeterministic": True,
                "cudnnBenchmark": False,
                "cudaMatmulAllowTf32": False,
                "cudnnAllowTf32": False,
                "float32MatmulPrecision": "highest",
            },
        },
        "dataset": {"repositoryId": "DreamMachines/example", "revision": revision},
        "observation": {
            "manifestSha256": sha,
            "manifestByteCount": 1,
            "observationId": "sample",
        },
        "cameras": [
            {
                "featureName": name,
                "relativePath": f"camera/{index}.png",
                "byteCount": 1,
                "sha256": sha,
            }
            for index, name in enumerate(
                (
                    "observation.images.left_wrist",
                    "observation.images.right_wrist",
                    "observation.images.top",
                )
            )
        ],
        "checkpoints": [
            {
                "policyId": policy,
                "repositoryId": "local/model",
                "revision": revision,
                "byteCount": 1,
                "sha256": sha,
            }
            for policy in ("base-pi05", "fine-tuned-pi05")
        ],
        "modelConfiguration": {
            "sourceSha256": sha,
            "effectiveSha256": sha,
            "transformations": [],
        },
        "processors": {
            "sourcePreprocessorSha256": sha,
            "effectivePreprocessorSha256": sha,
            "preprocessorStateSha256": sha,
            "sourcePostprocessorSha256": sha,
            "effectivePostprocessorSha256": sha,
            "postprocessorStateSha256": sha,
            "sharedForPolicyIds": ["base-pi05", "fine-tuned-pi05"],
        },
        "tokenizer": {
            "repositoryId": "google/paligemma-3b-pt-224",
            "directoryIdentitySha256": sha,
        },
        "comparison": {
            "policyOrder": ["base-pi05", "fine-tuned-pi05"],
            "numInferenceSteps": 10,
            "noise": {
                "seed": 1,
                "shape": [1, 50, 32],
                "dtype": "float32",
                "sha256": sha,
                "generator": "torch.Generator",
            },
        },
        "projection": {
            "available": False,
            "actionInterpretationId": "pi05-postprocessed-absolute-position-targets",
            "actionInterpretationVersion": "1.0",
            "urdfSha256": sha,
            "fkImplementationId": "synthetic-fk",
            "calibratedArmTransforms": False,
            "calibratedGripperGeometry": False,
            "jointLimitPolicy": "reject",
            "jointLimitViolationCount": 0,
        },
        "artifact": {
            "schemaVersion": "1.1",
            "manifest": {"byteCount": 1, "sha256": sha},
            "plans": {"byteCount": 1, "sha256": sha},
        },
    }


def test_receipt_is_deterministic_and_has_final_newline() -> None:
    first = build_checkpoint_comparison_run_receipt(receipt())
    second = build_checkpoint_comparison_run_receipt(receipt())
    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first)["comparison"]["policyOrder"] == [
        "base-pi05",
        "fine-tuned-pi05",
    ]
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "schemas/checkpoint-comparison-run-receipt-v1.schema.json"
        ).read_text()
    )
    Draft202012Validator(schema).validate(json.loads(first))


def test_receipt_rejects_boolean_version_and_wrong_noise_shape() -> None:
    value = receipt()
    value["schema"]["major"] = True
    with pytest.raises(CheckpointComparisonReceiptError, match="schema.major"):
        validate_checkpoint_comparison_run_receipt(value)
    value = receipt()
    value["comparison"]["noise"]["shape"] = [True, 50, 32]
    with pytest.raises(CheckpointComparisonReceiptError, match="noise.shape"):
        validate_checkpoint_comparison_run_receipt(value)
