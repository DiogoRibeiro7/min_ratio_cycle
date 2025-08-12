"""Statistical analysis utilities for solver results."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Tuple

import numpy as np


def confidence_interval(
    values: Iterable[float], alpha: float = 0.05
) -> Tuple[float, float]:
    """Return a two-sided confidence interval for the sample mean.

    Parameters
    ----------
    values : Iterable[float]
        Sequence of observations.
    alpha : float, optional
        Significance level of the interval, by default ``0.05`` which
        corresponds to a 95% confidence interval.

    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds of the interval.

    Raises
    ------
    ValueError
        If fewer than two observations are provided.

    Notes
    -----
    A normal approximation is used.  For non-default ``alpha`` the critical
    value is estimated through Monte Carlo sampling.

    Examples
    --------
    >>> confidence_interval([1.0, 2.0, 3.0, 4.0])  # doctest: +SKIP
    (1.0..., 3.9...)
    """

    data = np.array(list(values), dtype=float)
    if data.size < 2:
        raise ValueError("At least two values are required")
    mean = float(data.mean())
    std = float(data.std(ddof=1))
    if alpha == 0.05:
        z = 1.96
    else:
        z = abs(
            np.percentile(np.random.standard_normal(10_000), [100 * (1 - alpha / 2)])[0]
        )
    margin = z * std / sqrt(data.size)
    return mean - margin, mean + margin


def convergence_rate(errors: Iterable[float]) -> float:
    """Estimate the average ratio of successive errors.

    Parameters
    ----------
    errors : Iterable[float]
        Ordered sequence of error magnitudes from an iterative algorithm.

    Returns
    -------
    float
        Mean of ``errors[i+1] / errors[i]``.

    Raises
    ------
    ValueError
        If fewer than two error values are supplied.
    """

    data = np.array(list(errors), dtype=float)
    if data.size < 2:
        raise ValueError("Need at least two error values")
    ratios = data[1:] / data[:-1]
    return float(ratios.mean())


def compare_solutions(values1: Iterable[float], values2: Iterable[float]) -> float:
    """Return Welch's t statistic comparing two samples.

    Parameters
    ----------
    values1 : Iterable[float]
        First sequence of observations.
    values2 : Iterable[float]
        Second sequence of observations.

    Returns
    -------
    float
        Welch's t statistic. ``math.inf`` is returned when the denominator is
        zero.

    Raises
    ------
    ValueError
        If either sample contains fewer than two values.

    See Also
    --------
    :func:`convergence_rate` : Estimate average convergence rate from errors.
    :func:`confidence_interval` : Compute a confidence interval for a sample.
    """

    x = np.array(list(values1), dtype=float)
    y = np.array(list(values2), dtype=float)
    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least two values in each sample")
    mean_diff = x.mean() - y.mean()
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    denom = sqrt(var_x / x.size + var_y / y.size)
    if denom == 0:
        return float("inf")
    return float(mean_diff / denom)
