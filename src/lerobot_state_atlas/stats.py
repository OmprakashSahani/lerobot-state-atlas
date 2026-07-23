from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ComponentStatistics:
    """Statistics for one state component."""

    name: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    movement_range: float


@dataclass(frozen=True)
class StateStatistics:
    """Summary statistics for a batch of robot states."""

    num_frames: int
    state_dimension: int
    components: tuple[ComponentStatistics, ...]


def compute_state_statistics(
    states: Tensor,
    component_names: Sequence[str] | None = None,
) -> StateStatistics:
    """Compute per-component statistics for a two-dimensional state tensor."""
    if states.ndim != 2:
        raise ValueError("States must have shape (num_frames, state_dimension).")

    num_frames, state_dimension = states.shape

    if num_frames == 0:
        raise ValueError("States must contain at least one frame.")

    if component_names is None:
        names = tuple(f"state_{index}" for index in range(state_dimension))
    else:
        names = tuple(component_names)

    if len(names) != state_dimension:
        raise ValueError(
            "The number of component names must match the state dimension."
        )

    values = states.detach().to(dtype=torch.float64)

    minimums = values.min(dim=0).values
    maximums = values.max(dim=0).values
    means = values.mean(dim=0)
    standard_deviations = values.std(dim=0, correction=0)
    movement_ranges = maximums - minimums

    components = tuple(
        ComponentStatistics(
            name=names[index],
            minimum=float(minimums[index].item()),
            maximum=float(maximums[index].item()),
            mean=float(means[index].item()),
            standard_deviation=float(standard_deviations[index].item()),
            movement_range=float(movement_ranges[index].item()),
        )
        for index in range(state_dimension)
    )

    return StateStatistics(
        num_frames=int(num_frames),
        state_dimension=int(state_dimension),
        components=components,
    )
