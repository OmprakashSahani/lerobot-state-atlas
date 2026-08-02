"""Small invocation adapters for locally constructed PI05-compatible policies."""

import inspect
from typing import Any


class PI05PolicyAdapterError(ValueError):
    """Raised when a policy cannot satisfy the explicit PI05 invocation contract."""


class PI05PredictActionChunkAdapter:
    """Translate the comparison API's step override to LeRobot's ``num_steps``."""

    def __init__(self, policy: Any) -> None:
        method = getattr(policy, "predict_action_chunk", None)
        if not callable(method):
            raise PI05PolicyAdapterError(
                "policy.predict_action_chunk must be callable."
            )
        self._policy = policy

    def release(self) -> None:
        """Irrevocably sever the adapter's reference to its policy."""
        self._policy = None

    def predict_action_chunk(
        self,
        processed_observation: Any,
        *,
        noise: Any,
        num_inference_steps: int | None = None,
    ) -> Any:
        policy = self._policy
        if policy is None:
            raise PI05PolicyAdapterError("PI05 policy adapter has been released.")
        if num_inference_steps is not None and (
            isinstance(num_inference_steps, bool)
            or not isinstance(num_inference_steps, int)
            or num_inference_steps <= 0
        ):
            raise PI05PolicyAdapterError(
                "num_inference_steps must be a positive integer when supplied."
            )
        method = policy.predict_action_chunk
        kwargs = {"noise": noise}
        if num_inference_steps is not None:
            kwargs["num_steps"] = num_inference_steps
        try:
            inspect.signature(method).bind(processed_observation, **kwargs)
        except (TypeError, ValueError) as error:
            raise PI05PolicyAdapterError(
                "policy.predict_action_chunk has an incompatible explicit-noise/num_steps signature: "
                f"{error}"
            ) from error
        return method(processed_observation, **kwargs)
