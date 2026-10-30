"""Bootstrap confidence intervals. With a few dozen questions, differences of a few points
between systems are often noise; paired intervals make that visible."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float], *, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a: list[float], b: list[float], **kwargs) -> tuple[float, float, float]:
    """Mean of a - b over the same questions, with a bootstrap interval. The difference is
    significant at the chosen level when the interval excludes zero."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci([x - y for x, y in zip(a, b, strict=True)], **kwargs)
