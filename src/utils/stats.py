"""Statistics helpers: mean, std, 95% CI via t-distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class Summary:
    """Aggregate summary of a sample."""

    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    n: int


def summarise(values: Sequence[float]) -> Summary:
    """Return Summary(mean, std, 95% CI low/high, n) for a 1-D sample.

    Uses Student's t with df = n-1. For n == 1 the CI collapses to the point
    estimate (no interval available); for n == 0 returns all NaN.
    """
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        nan = float("nan")
        return Summary(mean=nan, std=nan, ci95_low=nan, ci95_high=nan, n=0)
    mean = float(arr.mean())
    if n == 1:
        return Summary(mean=mean, std=0.0, ci95_low=mean, ci95_high=mean, n=1)
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(n)
    margin = float(scipy_stats.t.ppf(0.975, df=n - 1)) * sem
    return Summary(
        mean=mean,
        std=std,
        ci95_low=mean - margin,
        ci95_high=mean + margin,
        n=n,
    )
