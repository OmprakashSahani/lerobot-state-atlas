"""Deterministic validation and serialization of comparison run receipts."""

from __future__ import annotations

import copy
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from lerobot_state_atlas.checkpoint_comparison.serialize import deterministic_json_bytes


RECEIPT_SCHEMA_NAME = "lerobot-state-atlas.checkpoint-comparison-run-receipt"


class CheckpointComparisonReceiptError(ValueError):
    """Raised when deterministic run provenance is incomplete or malformed."""


def _fail(path: str, message: str) -> None:
    raise CheckpointComparisonReceiptError(f"{path} {message}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object.")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string.")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be a non-boolean integer >= {minimum}.")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number.")
    result = float(value)
    if not isfinite(result):
        _fail(path, "must be a finite number.")
    return result


def _sha(value: Any, path: str) -> str:
    value = _text(value, path)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        _fail(path, "must be a lowercase 64-character SHA-256.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        _fail(path, f"is missing fields {missing!r}.")
    if extra:
        _fail(path, f"contains unsupported fields {extra!r}.")


def validate_checkpoint_comparison_run_receipt(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the strict receipt v1.0 structural and integrity contract."""
    value = _object(document, "receipt")
    expected = {
        "schema",
        "runnerManifestSha256",
        "software",
        "runtime",
        "dataset",
        "observation",
        "cameras",
        "checkpoints",
        "modelConfiguration",
        "processors",
        "tokenizer",
        "comparison",
        "projection",
        "artifact",
    }
    _exact_keys(value, expected, "receipt")
    schema = _object(value["schema"], "schema")
    _exact_keys(schema, {"name", "major", "minor"}, "schema")
    if schema["name"] != RECEIPT_SCHEMA_NAME:
        _fail("schema.name", f"must equal {RECEIPT_SCHEMA_NAME!r}.")
    if _integer(schema["major"], "schema.major") != 1:
        _fail("schema.major", "must equal 1.")
    if _integer(schema["minor"], "schema.minor") != 0:
        _fail("schema.minor", "must equal 0.")
    _sha(value["runnerManifestSha256"], "runnerManifestSha256")

    software = _object(value["software"], "software")
    _exact_keys(
        software,
        {
            "projectVersion",
            "sourceIdentity",
            "pythonVersion",
            "lerobotVersion",
            "torchVersion",
            "safetensorsVersion",
            "pillowVersion",
            "transformersVersion",
        },
        "software",
    )
    for key, item in software.items():
        _text(item, f"software.{key}")

    runtime = _object(value["runtime"], "runtime")
    _exact_keys(
        runtime,
        {
            "device",
            "modelDtype",
            "noiseDtype",
            "cudaRuntime",
            "cudaDriver",
            "gpuName",
            "computeCapability",
            "deterministicSettings",
        },
        "runtime",
    )
    for key in (
        "device",
        "modelDtype",
        "noiseDtype",
        "cudaRuntime",
        "cudaDriver",
        "gpuName",
    ):
        _text(runtime[key], f"runtime.{key}")
    if not runtime["device"].startswith("cuda:") or not runtime["device"][5:].isdigit():
        _fail("runtime.device", "must be an indexed CUDA device.")
    if runtime["modelDtype"] not in {"bfloat16", "float32"}:
        _fail("runtime.modelDtype", "must be bfloat16 or float32.")
    if runtime["noiseDtype"] != "float32":
        _fail("runtime.noiseDtype", "must equal float32.")
    capability = runtime["computeCapability"]
    if not isinstance(capability, list) or len(capability) != 2:
        _fail("runtime.computeCapability", "must contain exactly two integers.")
    for index, item in enumerate(capability):
        _integer(item, f"runtime.computeCapability[{index}]")
    settings = _object(
        runtime["deterministicSettings"], "runtime.deterministicSettings"
    )
    required_settings = {
        "cublasWorkspaceConfig": ":4096:8",
        "deterministicAlgorithms": True,
        "cudnnDeterministic": True,
        "cudnnBenchmark": False,
        "cudaMatmulAllowTf32": False,
        "cudnnAllowTf32": False,
        "float32MatmulPrecision": "highest",
    }
    _exact_keys(settings, set(required_settings), "runtime.deterministicSettings")
    for key, expected_setting in required_settings.items():
        if settings[key] != expected_setting or type(settings[key]) is not type(
            expected_setting
        ):
            _fail(
                f"runtime.deterministicSettings.{key}",
                f"must equal {expected_setting!r}.",
            )

    dataset = _object(value["dataset"], "dataset")
    _exact_keys(dataset, {"repositoryId", "revision"}, "dataset")
    _text(dataset["repositoryId"], "dataset.repositoryId")
    revision = _text(dataset["revision"], "dataset.revision")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        _fail("dataset.revision", "must be a lowercase 40-character commit SHA.")

    observation = _object(value["observation"], "observation")
    _exact_keys(
        observation,
        {"manifestSha256", "manifestByteCount", "observationId"},
        "observation",
    )
    _sha(observation["manifestSha256"], "observation.manifestSha256")
    _integer(
        observation["manifestByteCount"],
        "observation.manifestByteCount",
        minimum=1,
    )
    _text(observation["observationId"], "observation.observationId")

    cameras = value["cameras"]
    if not isinstance(cameras, list) or len(cameras) != 3:
        _fail("cameras", "must contain exactly three ordered cameras.")
    expected_cameras = (
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.top",
    )
    for index, camera_value in enumerate(cameras):
        camera = _object(camera_value, f"cameras[{index}]")
        _exact_keys(
            camera,
            {"featureName", "relativePath", "byteCount", "sha256"},
            f"cameras[{index}]",
        )
        if camera["featureName"] != expected_cameras[index]:
            _fail(
                f"cameras[{index}].featureName",
                f"must equal {expected_cameras[index]!r}.",
            )
        path = _text(camera["relativePath"], f"cameras[{index}].relativePath")
        if path.startswith("/") or ".." in Path(path).parts:
            _fail(f"cameras[{index}].relativePath", "must be bundle-relative.")
        _integer(camera["byteCount"], f"cameras[{index}].byteCount", minimum=1)
        _sha(camera["sha256"], f"cameras[{index}].sha256")

    checkpoints = value["checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        _fail("checkpoints", "must contain exactly two ordered checkpoint records.")
    for index, expected_id in enumerate(("base-pi05", "fine-tuned-pi05")):
        item = _object(checkpoints[index], f"checkpoints[{index}]")
        _exact_keys(
            item,
            {"policyId", "repositoryId", "revision", "byteCount", "sha256"},
            f"checkpoints[{index}]",
        )
        if item["policyId"] != expected_id:
            _fail(f"checkpoints[{index}].policyId", f"must equal {expected_id!r}.")
        _text(item["repositoryId"], f"checkpoints[{index}].repositoryId")
        revision = _text(item["revision"], f"checkpoints[{index}].revision")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            _fail(
                f"checkpoints[{index}].revision",
                "must be a lowercase 40-character commit SHA.",
            )
        _integer(item["byteCount"], f"checkpoints[{index}].byteCount", minimum=1)
        _sha(item["sha256"], f"checkpoints[{index}].sha256")

    configuration = _object(value["modelConfiguration"], "modelConfiguration")
    _exact_keys(
        configuration,
        {"sourceSha256", "effectiveSha256", "transformations"},
        "modelConfiguration",
    )
    _sha(configuration["sourceSha256"], "modelConfiguration.sourceSha256")
    _sha(configuration["effectiveSha256"], "modelConfiguration.effectiveSha256")
    transformations = configuration["transformations"]
    if not isinstance(transformations, list) or any(
        not isinstance(item, str) for item in transformations
    ):
        _fail("modelConfiguration.transformations", "must be an array of strings.")

    processors = _object(value["processors"], "processors")
    processor_hash_fields = {
        "sourcePreprocessorSha256",
        "effectivePreprocessorSha256",
        "preprocessorStateSha256",
        "sourcePostprocessorSha256",
        "effectivePostprocessorSha256",
        "postprocessorStateSha256",
    }
    _exact_keys(
        processors,
        processor_hash_fields | {"sharedForPolicyIds"},
        "processors",
    )
    for key in processor_hash_fields:
        _sha(processors[key], f"processors.{key}")
    if processors["sharedForPolicyIds"] != ["base-pi05", "fine-tuned-pi05"]:
        _fail(
            "processors.sharedForPolicyIds",
            "must contain the two policies in canonical order.",
        )

    tokenizer = _object(value["tokenizer"], "tokenizer")
    _exact_keys(tokenizer, {"repositoryId", "directoryIdentitySha256"}, "tokenizer")
    if tokenizer["repositoryId"] != "google/paligemma-3b-pt-224":
        _fail("tokenizer.repositoryId", "must identify google/paligemma-3b-pt-224.")
    _sha(
        tokenizer["directoryIdentitySha256"],
        "tokenizer.directoryIdentitySha256",
    )

    projection = _object(value["projection"], "projection")
    _exact_keys(
        projection,
        {
            "available",
            "actionInterpretationId",
            "actionInterpretationVersion",
            "urdfSha256",
            "fkImplementationId",
            "calibratedArmTransforms",
            "calibratedGripperGeometry",
            "jointLimitPolicy",
            "jointLimitViolationCount",
        },
        "projection",
    )
    if not isinstance(projection["available"], bool):
        _fail("projection.available", "must be boolean.")
    if (
        projection["actionInterpretationId"]
        != "pi05-postprocessed-absolute-position-targets"
    ):
        _fail("projection.actionInterpretationId", "is unsupported.")
    if projection["actionInterpretationVersion"] != "1.0":
        _fail("projection.actionInterpretationVersion", "must equal '1.0'.")
    _sha(projection["urdfSha256"], "projection.urdfSha256")
    _text(projection["fkImplementationId"], "projection.fkImplementationId")
    if not isinstance(projection["calibratedArmTransforms"], bool):
        _fail("projection.calibratedArmTransforms", "must be boolean.")
    if projection["calibratedGripperGeometry"] is not False:
        _fail("projection.calibratedGripperGeometry", "must be false.")
    if projection["jointLimitPolicy"] not in {
        "reject",
        "allow-with-recorded-violations",
    }:
        _fail("projection.jointLimitPolicy", "is unsupported.")
    _integer(
        projection["jointLimitViolationCount"],
        "projection.jointLimitViolationCount",
    )

    comparison = _object(value["comparison"], "comparison")
    _exact_keys(comparison, {"policyOrder", "numInferenceSteps", "noise"}, "comparison")
    if comparison["policyOrder"] != ["base-pi05", "fine-tuned-pi05"]:
        _fail("comparison.policyOrder", "must be Base π0.5 then Fine-tuned π0.5.")
    _integer(comparison["numInferenceSteps"], "comparison.numInferenceSteps", minimum=1)
    noise = _object(comparison["noise"], "comparison.noise")
    _exact_keys(
        noise, {"seed", "shape", "dtype", "sha256", "generator"}, "comparison.noise"
    )
    _integer(noise["seed"], "comparison.noise.seed")
    if noise["shape"] != [1, 50, 32] or any(
        isinstance(item, bool) for item in noise["shape"]
    ):
        _fail("comparison.noise.shape", "must equal [1, 50, 32] using integers.")
    _text(noise["dtype"], "comparison.noise.dtype")
    _sha(noise["sha256"], "comparison.noise.sha256")
    _text(noise["generator"], "comparison.noise.generator")

    artifact = _object(value["artifact"], "artifact")
    _exact_keys(artifact, {"schemaVersion", "manifest", "plans"}, "artifact")
    if artifact["schemaVersion"] != "1.1":
        _fail("artifact.schemaVersion", "must equal '1.1'.")
    for name in ("manifest", "plans"):
        item = _object(artifact[name], f"artifact.{name}")
        _exact_keys(item, {"byteCount", "sha256"}, f"artifact.{name}")
        _integer(item["byteCount"], f"artifact.{name}.byteCount", minimum=1)
        _sha(item["sha256"], f"artifact.{name}.sha256")

    # Verify JSON compatibility and reject non-finite values recursively.
    try:
        deterministic_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise CheckpointComparisonReceiptError(
            f"receipt is not deterministic JSON: {error}"
        ) from error
    return copy.deepcopy(dict(value))


def build_checkpoint_comparison_run_receipt(document: Mapping[str, Any]) -> bytes:
    """Validate and serialize a deterministic receipt with a final newline."""
    return deterministic_json_bytes(
        validate_checkpoint_comparison_run_receipt(document)
    )


def load_checkpoint_comparison_run_receipt(path: str | Path) -> dict[str, Any]:
    """Load a receipt for post-installation validation."""
    try:
        raw = Path(path).read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointComparisonReceiptError(
            f"run-receipt.json could not be loaded: {error}"
        ) from error
    return validate_checkpoint_comparison_run_receipt(document)
