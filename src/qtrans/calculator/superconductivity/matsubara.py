"""Matsubara grids and transition-temperature bracketing."""

from typing import Callable

import numpy as np

from .state import TcSearchResult


def positive_matsubara_frequencies(temperature, cutoff, max_points):
    """Return positive fermionic Matsubara frequencies up to ``cutoff``."""
    if temperature <= 0.0 or cutoff <= 0.0:
        raise ValueError("temperature and cutoff must be positive.")
    first = np.pi * temperature
    if first > cutoff:
        raise ValueError(
            "Matsubara cutoff {:.6e} is below the first frequency {:.6e}."
            .format(cutoff, first)
        )
    count = int(np.floor((cutoff / first - 1.0) / 2.0)) + 1
    if count > max_points:
        raise RuntimeError(
            "The requested grid has {} positive Matsubara points, exceeding "
            "max_matsubara_points={}. Increase temperature/cutoff limits or "
            "raise the explicit safety limit.".format(count, max_points)
        )
    return (2.0 * np.arange(count, dtype=float) + 1.0) * first


def _descending_temperature_grid(temperature_min, temperature_max, step):
    count = int(np.floor((temperature_max - temperature_min) / step))
    values = temperature_max - np.arange(count + 1, dtype=float) * step
    values = values[values >= temperature_min]
    if values.size == 0 or values[-1] > temperature_min * (1.0 + 1.0e-12):
        values = np.append(values, temperature_min)
    return values


def search_transition_temperature(
    eigenvalue: Callable[[float], float],
    temperature_min: float,
    temperature_max: float,
    temperature_step: float,
    tolerance: float,
):
    """Locate lambda_max(T)=1 using a descending scan followed by bisection."""
    temperatures = []
    eigenvalues = []

    def evaluate(temperature):
        value = float(eigenvalue(float(temperature)))
        if not np.isfinite(value):
            raise RuntimeError(
                "The linearized Eliashberg eigenvalue is not finite at T={:.6e}."
                .format(temperature)
            )
        temperatures.append(float(temperature))
        eigenvalues.append(value)
        return value

    grid = _descending_temperature_grid(
        temperature_min, temperature_max, temperature_step
    )
    high_temperature = float(grid[0])
    high_value = evaluate(high_temperature)
    if high_value >= 1.0:
        return TcSearchResult(
            tc=np.nan,
            status="above_temperature_range",
            temperatures=np.asarray(temperatures),
            eigenvalues=np.asarray(eigenvalues),
        )

    low_temperature = None
    low_value = None
    for candidate in grid[1:]:
        candidate_value = evaluate(float(candidate))
        if candidate_value >= 1.0:
            low_temperature = float(candidate)
            low_value = candidate_value
            break
        high_temperature = float(candidate)
        high_value = candidate_value

    if low_temperature is None:
        return TcSearchResult(
            tc=0.0,
            status="below_temperature_range",
            temperatures=np.asarray(temperatures),
            eigenvalues=np.asarray(eigenvalues),
        )

    while high_temperature - low_temperature > tolerance:
        midpoint = 0.5 * (low_temperature + high_temperature)
        midpoint_value = evaluate(midpoint)
        if midpoint_value >= 1.0:
            low_temperature = midpoint
            low_value = midpoint_value
        else:
            high_temperature = midpoint
            high_value = midpoint_value

    del low_value, high_value
    return TcSearchResult(
        tc=0.5 * (low_temperature + high_temperature),
        status="bracketed",
        temperatures=np.asarray(temperatures),
        eigenvalues=np.asarray(eigenvalues),
    )
