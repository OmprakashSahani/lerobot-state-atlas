"""Strict context-managed factory for locally staged PI05 policy weights."""

from contextlib import contextmanager
import gc
from pathlib import Path
import sys
from typing import Any, Callable, Iterator, Mapping

import torch

from lerobot_state_atlas.checkpoint_comparison.checkpoint_staging import (
    load_staged_checkpoint_into_fresh_module,
    stage_runner_checkpoint,
)
from lerobot_state_atlas.checkpoint_comparison.models import PI05CompatibilityResult
from lerobot_state_atlas.checkpoint_comparison.policy_adapters import (
    PI05PredictActionChunkAdapter,
)
from lerobot_state_atlas.checkpoint_comparison.runner_models import (
    RunnerCheckpointInput,
    RunnerRuntimeConfiguration,
)


class PI05PolicyFactoryError(RuntimeError):
    """Raised when a local PI05 lifecycle cannot be completed safely."""


def _thaw(value: object) -> Any:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {item[0]: _thaw(item[1]) for item in value}
        return [_thaw(item) for item in value]
    return value


def _default_constructor(config_values: Mapping[str, Any]) -> object:
    import draccus
    from lerobot.policies.pi05 import PI05Config, PI05Policy

    config = draccus.decode(PI05Config, dict(config_values))
    return PI05Policy(config)


def _default_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def create_pi05_policy_factory(
    *,
    policy_id: str,
    effective_config: PI05CompatibilityResult,
    checkpoint_input: RunnerCheckpointInput,
    checkpoint_path: str | Path,
    runtime: RunnerRuntimeConfiguration,
    staging_parent: str | Path,
    policy_constructor: Callable[[Mapping[str, Any]], object] | None = None,
    checkpoint_stager: Callable[..., Any] | None = None,
    checkpoint_loader: Callable[..., Any] | None = None,
    cleanup_hooks: tuple[Callable[[], None], ...] | None = None,
) -> Callable[[], Any]:
    """Return a zero-argument context factory suitable for sequential inference."""
    if policy_id not in {"base-pi05", "fine-tuned-pi05"}:
        raise PI05PolicyFactoryError("policy_id must be base-pi05 or fine-tuned-pi05.")
    constructor = policy_constructor or _default_constructor
    stager = checkpoint_stager or stage_runner_checkpoint
    loader = checkpoint_loader or load_staged_checkpoint_into_fresh_module
    hooks = cleanup_hooks if cleanup_hooks is not None else (_default_cleanup,)
    checkpoint_kind = "base" if policy_id == "base-pi05" else "fine-tuned"

    @contextmanager
    def factory() -> Iterator[PI05PredictActionChunkAdapter]:
        config = _thaw(effective_config.effective_config)
        config["compile_model"] = False
        config["gradient_checkpointing"] = False
        config["device"] = "cpu"
        policy = None
        adapter = None
        try:
            with stager(
                checkpoint_path,
                expected_byte_count=checkpoint_input.byte_count,
                expected_sha256=checkpoint_input.sha256,
                staging_parent=staging_parent,
                checkpoint_kind=checkpoint_kind,
            ) as staged:
                policy = constructor(config)
                loader(
                    policy,
                    staged,
                    checkpoint_kind,
                    drop_unused_lm_head=effective_config.drop_unused_lm_head,
                )
                dtype = (
                    torch.bfloat16
                    if runtime.model_dtype == "bfloat16"
                    else torch.float32
                )
                try:
                    policy.to(device=runtime.device, dtype=dtype)
                    policy.eval()
                except Exception as error:
                    raise PI05PolicyFactoryError(
                        f"{policy_id} could not move to {runtime.device}/{runtime.model_dtype}: {error}"
                    ) from error
                adapter = PI05PredictActionChunkAdapter(policy)
                try:
                    yield adapter
                finally:
                    active_error = sys.exc_info()[1]
                    try:
                        adapter.release()
                    except Exception as release_error:
                        message = f"{policy_id} adapter release failed: {release_error}"
                        if active_error is not None:
                            message += f"; original failure: {active_error!r}"
                        raise PI05PolicyFactoryError(message) from active_error
                    finally:
                        adapter = None
                        policy = None
        except PI05PolicyFactoryError:
            raise
        except Exception as error:
            raise PI05PolicyFactoryError(
                f"{policy_id} lifecycle failed: {error}"
            ) from error
        finally:
            active_error = sys.exc_info()[1]
            adapter = None
            policy = None
            failures = []
            for hook in hooks:
                try:
                    hook()
                except Exception as error:
                    failures.append(repr(error))
            if failures:
                message = f"{policy_id} cleanup failed: {', '.join(failures)}"
                if active_error is not None:
                    message += f"; original failure: {active_error!r}"
                raise PI05PolicyFactoryError(message) from active_error

    return factory
