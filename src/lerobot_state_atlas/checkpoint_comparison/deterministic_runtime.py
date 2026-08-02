"""Scoped deterministic Torch backend configuration for comparison execution."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Any, Callable, Iterator


class DeterministicTorchExecutionError(RuntimeError):
    """Raised when deterministic backend requirements cannot be applied or restored."""


@dataclass(frozen=True)
class DeterministicTorchSettings:
    cublas_workspace_config: str
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    float32_matmul_precision: str
    device: str
    model_dtype: str

    def as_items(self) -> tuple[tuple[str, str | bool], ...]:
        """Return deterministic receipt-ready settings without mutable mappings."""
        return (
            ("cublasWorkspaceConfig", self.cublas_workspace_config),
            ("deterministicAlgorithms", self.deterministic_algorithms),
            ("cudnnDeterministic", self.cudnn_deterministic),
            ("cudnnBenchmark", self.cudnn_benchmark),
            ("cudaMatmulAllowTf32", self.cuda_matmul_allow_tf32),
            ("cudnnAllowTf32", self.cudnn_allow_tf32),
            ("float32MatmulPrecision", self.float32_matmul_precision),
        )


Restoration = tuple[str, Callable[[], None]]


def _restore_settings(restorations: list[Restoration]) -> tuple[str, ...]:
    """Attempt every registered restoration in reverse setup order."""
    failures: list[str] = []
    for name, restore in reversed(restorations):
        try:
            restore()
        except Exception as error:
            failures.append(f"{name}: {error!r}")
    return tuple(failures)


def _restore_environment(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


@contextmanager
def deterministic_torch_execution(
    *, device: str, model_dtype: str, torch_module: Any | None = None
) -> Iterator[DeterministicTorchSettings]:
    """Apply and restore all supported process-global deterministic settings."""
    if not device.startswith("cuda:"):
        raise DeterministicTorchExecutionError(
            "deterministic execution requires indexed CUDA device."
        )
    if model_dtype not in {"bfloat16", "float32"}:
        raise DeterministicTorchExecutionError(
            "model_dtype must be bfloat16 or float32."
        )
    if torch_module is None:
        import torch as torch_module
    old_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        old_deterministic = torch_module.are_deterministic_algorithms_enabled()
        warn_only_getter = getattr(
            torch_module, "is_deterministic_algorithms_warn_only_enabled", None
        )
        old_warn_only = warn_only_getter() if callable(warn_only_getter) else None
        old_cudnn_deterministic = torch_module.backends.cudnn.deterministic
        old_cudnn_benchmark = torch_module.backends.cudnn.benchmark
        old_cuda_matmul_tf32 = torch_module.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch_module.backends.cudnn.allow_tf32
        old_matmul_precision = torch_module.get_float32_matmul_precision()
    except Exception as error:
        raise DeterministicTorchExecutionError(
            f"Could not capture prior Torch setting through its getter/API: {error}"
        ) from error

    restorations: list[Restoration] = []
    try:
        restorations.append(
            (
                "CUBLAS_WORKSPACE_CONFIG",
                lambda: _restore_environment("CUBLAS_WORKSPACE_CONFIG", old_env),
            )
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        def restore_deterministic_algorithms() -> None:
            if old_warn_only is None:
                torch_module.use_deterministic_algorithms(old_deterministic)
            else:
                torch_module.use_deterministic_algorithms(
                    old_deterministic, warn_only=old_warn_only
                )

        restorations.append(
            ("torch deterministic algorithms", restore_deterministic_algorithms)
        )
        if old_warn_only is None:
            torch_module.use_deterministic_algorithms(True)
        else:
            torch_module.use_deterministic_algorithms(True, warn_only=False)

        restorations.append(
            (
                "torch.backends.cudnn.deterministic",
                lambda: setattr(
                    torch_module.backends.cudnn,
                    "deterministic",
                    old_cudnn_deterministic,
                ),
            )
        )
        torch_module.backends.cudnn.deterministic = True

        restorations.append(
            (
                "torch.backends.cudnn.benchmark",
                lambda: setattr(
                    torch_module.backends.cudnn, "benchmark", old_cudnn_benchmark
                ),
            )
        )
        torch_module.backends.cudnn.benchmark = False

        restorations.append(
            (
                "torch.backends.cuda.matmul.allow_tf32",
                lambda: setattr(
                    torch_module.backends.cuda.matmul,
                    "allow_tf32",
                    old_cuda_matmul_tf32,
                ),
            )
        )
        torch_module.backends.cuda.matmul.allow_tf32 = False

        restorations.append(
            (
                "torch.backends.cudnn.allow_tf32",
                lambda: setattr(
                    torch_module.backends.cudnn, "allow_tf32", old_cudnn_tf32
                ),
            )
        )
        torch_module.backends.cudnn.allow_tf32 = False

        restorations.append(
            (
                "torch float32 matmul precision",
                lambda: torch_module.set_float32_matmul_precision(old_matmul_precision),
            )
        )
        torch_module.set_float32_matmul_precision("highest")
        settings = DeterministicTorchSettings(
            ":4096:8", True, True, False, False, False, "highest", device, model_dtype
        )
    except Exception as error:
        restoration_failures = _restore_settings(restorations)
        suffix = (
            f"; restoration failures: {', '.join(restoration_failures)}"
            if restoration_failures
            else ""
        )
        raise DeterministicTorchExecutionError(
            f"Could not apply deterministic Torch settings: {error}{suffix}"
        ) from error
    try:
        yield settings
    except BaseException as body_error:
        restoration_failures = _restore_settings(restorations)
        if restoration_failures:
            raise DeterministicTorchExecutionError(
                "Managed deterministic Torch body failed with "
                f"{body_error!r}; restoration failures: "
                f"{', '.join(restoration_failures)}"
            ) from body_error
        raise
    else:
        restoration_failures = _restore_settings(restorations)
        if restoration_failures:
            raise DeterministicTorchExecutionError(
                "Could not restore prior Torch settings: "
                f"{', '.join(restoration_failures)}"
            )
