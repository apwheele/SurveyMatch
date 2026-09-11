"""Matplotlib helpers used by paper.qmd. Kept separate from the statistical
core in matching.py so the method has no plotting dependency."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.stats import rv_continuous, rv_discrete

from .matching import match_count_pmf

__all__ = ["plot_match_fit"]


def plot_match_fit(
    counts: NDArray,
    null_dist: rv_discrete | rv_continuous,
    v: int,
    tail: int | None = None,
    ax: Axes | None = None,
    label: str = "Fitted null",
    color: str = "black",
) -> Axes:
    """Bar chart of observed match-count proportions with a fitted null overlaid."""
    x, observed, pmf = match_count_pmf(counts, null_dist, v, tail=tail)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x, observed, color="0.75", width=0.9, label="Observed")
    ax.plot(x, pmf, color=color, linewidth=2, label=label)
    ax.set_xlabel("Number of matching questions")
    ax.set_ylabel("Proportion of pairs")
    ax.legend(frameon=False)
    return ax
