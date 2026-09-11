"""Simulate correlated categorical responses.

Used only to check FDR calibration under within-respondent question
correlation (see paper.qmd); simulate_responses in matching.py draws every
question independently, which is the case the exact Poisson-binomial null is
built for but real surveys rarely satisfy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["simulate_correlated_responses"]


def simulate_correlated_responses(
    n: int,
    v: int,
    category_probs: ArrayLike,
    tilt: ArrayLike,
    rho: float,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Simulate n respondents on v questions sharing one common trait z.

    Every question has the same baseline category distribution
    category_probs, but a respondent-level latent trait z ~ N(0,1) tilts the
    log-probabilities by rho * z * tilt before renormalizing, so rho controls
    how strongly a respondent's answers move together across questions. Each
    question is drawn independently given z, so rho=0 reproduces the
    independent case used elsewhere in the package.
    """
    base_p = np.asarray(category_probs, dtype=float)
    tilt = np.asarray(tilt, dtype=float)
    z = rng.normal(size=n)
    logits = np.log(base_p)[None, :] + rho * z[:, None] * tilt[None, :]
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    u = rng.uniform(size=(n, v))
    categories = (u[:, :, None] > cum[:, None, :]).sum(axis=2)
    return categories.astype(int)
