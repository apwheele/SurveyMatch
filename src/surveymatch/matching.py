"""Core routines for detecting near-duplicate survey responses.

The method: compare every pair of respondents on how many of v categorical
questions they answered identically, then test whether the resulting
distribution of match counts has a heavier-than-expected right tail. Because
every respondent takes part in n-1 comparisons, the C(n,2) pairwise match
counts are highly correlated, so the null distribution is estimated directly
from the empirical data (beta-binomial or normal) rather than assumed
independent, and the false discovery rate correction is chosen to tolerate
that dependence.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import betabinom, norm, rv_continuous, rv_discrete
from statsmodels.stats.multitest import multipletests

__all__ = [
    "simulate_responses",
    "expected_match_probs",
    "pairwise_match_counts",
    "pair_index",
    "poisson_binomial_null",
    "fit_beta_binomial_mom",
    "fit_beta_binomial_ml",
    "fit_normal_approx",
    "match_pvalues",
    "fdr_flag",
    "connected_components",
    "match_count_pmf",
    "null_fit_sse",
    "MatchResult",
    "flag_duplicate_pairs",
]


def simulate_responses(
    n: int, category_probs: list[ArrayLike], rng: np.random.Generator | None = None
) -> NDArray[np.int_]:
    """Simulate n respondents answering len(category_probs) questions.

    Each element of ``category_probs`` gives the marginal probability of each
    response category for one question; questions are drawn independently.
    Returns an (n, v) integer array of category codes.
    """
    if rng is None:
        rng = np.random.default_rng()
    columns = []
    for p in category_probs:
        p = np.asarray(p, dtype=float)
        columns.append(rng.choice(len(p), size=(n, 1), p=p))
    return np.concatenate(columns, axis=1)


def expected_match_probs(data: NDArray) -> NDArray[np.float64]:
    """Per-question probability that two independently sampled respondents match.

    For question c with observed category proportions p_ck, this is the
    Simpson collision probability sum_k p_ck^2. Summed over questions it gives
    the mean of the pairwise match-count distribution under independence.
    """
    n = data.shape[0]
    probs = np.empty(data.shape[1])
    for c in range(data.shape[1]):
        _, counts = np.unique(data[:, c], return_counts=True)
        probs[c] = ((counts / n) ** 2).sum()
    return probs


def poisson_binomial_null(question_match_probs: ArrayLike) -> rv_discrete:
    """Exact null distribution of the match count if questions were independent.

    Each question contributes an independent Bernoulli(theta_c) indicator for
    whether a pair matches on that question, where theta_c is the collision
    probability from expected_match_probs. Their sum is Poisson-binomial; the
    pmf is built by the standard O(v^2) convolution recursion. This is the
    theoretically "correct" null under independent questions, but real survey
    items are usually positively correlated, which overdisperses the observed
    match counts relative to this null (see the beta-binomial/normal fits).
    """
    probs = np.asarray(question_match_probs, dtype=float)
    v = len(probs)
    pmf = np.zeros(v + 1)
    pmf[0] = 1.0
    for p in probs:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= 1 - p
    return rv_discrete(name="poisson_binomial", values=(np.arange(v + 1), pmf))


def pairwise_match_counts(data: NDArray) -> NDArray[np.int_]:
    """Number of matching questions for every C(n,2) pair of rows in data.

    Pairs are ordered the same way as itertools.combinations(range(n), 2),
    which is what pair_index reproduces when mapping a flagged position back
    to a respondent pair.
    """
    n = data.shape[0]
    matches = [
        (data[r, :] == data[(r + 1) :, :]).sum(axis=1) for r in range(n - 1)
    ]
    return np.concatenate(matches, axis=0)


def pair_index(n: int, mask: NDArray[np.bool_]) -> NDArray[np.int_]:
    """Map a boolean mask over pairwise_match_counts back to (i, j) row pairs."""
    all_pairs = np.array(list(itertools.combinations(range(n), 2)))
    return all_pairs[mask]


def _mom_params(counts: NDArray, v: int) -> tuple[float, float] | None:
    """Method-of-moments (alpha, beta), or None if the sample is underdispersed
    relative to a binomial (Wikipedia parameterization)."""
    m1 = counts.mean()
    m2 = (counts.astype(float) ** 2).mean()
    d = v * (m2 / m1 - m1 - 1) + m1
    if d == 0:
        return None
    alpha = (v * m1 - m2) / d
    beta = ((v - m1) * (v - m2 / m1)) / d
    if alpha <= 0 or beta <= 0 or not (np.isfinite(alpha) and np.isfinite(beta)):
        return None
    return alpha, beta


def fit_beta_binomial_mom(counts: NDArray, v: int) -> rv_discrete:
    """Method-of-moments beta-binomial fit.

    Falls back to a normal approximation when the sample is underdispersed
    relative to a binomial, which yields a negative alpha or beta.
    """
    params = _mom_params(counts, v)
    if params is None:
        return norm(counts.mean(), counts.std())
    return betabinom(v, *params)


def _neg_log_lik(params: NDArray, counts: NDArray, v: int) -> float:
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf
    return -betabinom.logpmf(counts, v, alpha, beta).sum()


def fit_beta_binomial_ml(counts: NDArray, v: int) -> rv_discrete:
    """Maximum-likelihood beta-binomial fit.

    The method-of-moments estimator is unstable near the boundary of valid
    (alpha, beta), so this direct likelihood fit is the one used in the paper;
    method-of-moments is used only to pick a fast-converging starting point.
    """
    x0 = _mom_params(counts, v) or (1.0, 1.0)
    result = minimize(
        _neg_log_lik,
        x0=x0,
        args=(counts, v),
        method="Nelder-Mead",
        options={"xatol": 1e-2, "fatol": 1e-2},
    )
    alpha, beta = result.x
    return betabinom(v, alpha, beta)


def fit_normal_approx(counts: NDArray) -> rv_continuous:
    """Normal approximation to the match-count distribution, mean/sd matched."""
    return norm(counts.mean(), counts.std())


def match_pvalues(counts: NDArray, null_dist: rv_discrete | rv_continuous) -> NDArray[np.float64]:
    """Upper-tail p-value for each observed match count under null_dist.

    P(X >= k) = 1 - P(X <= k - 1) = 1 - cdf(k) + pmf(k) for a discrete null;
    using 1 - cdf(k) (as here) is the conservative, slightly smaller-tail
    convention used throughout the accompanying simulations.
    """
    return 1 - null_dist.cdf(counts)


def fdr_flag(
    pvalues: ArrayLike, alpha: float = 0.05, method: str = "fdr_bh"
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """False discovery rate correction; wraps statsmodels multipletests.

    Use method="fdr_bh" (Benjamini-Hochberg) when comparisons can be treated
    as independent or positively dependent, and "fdr_by" (Benjamini-Yekutieli)
    when the pairwise match counts are more strongly and arbitrarily
    correlated, as they are once every respondent appears in n-1 comparisons.
    """
    rejected, qvalues, _, _ = multipletests(pvalues, alpha=alpha, method=method)
    return rejected, qvalues


def connected_components(pairs: NDArray) -> list[list[int]]:
    """Group flagged pairs into connected components of shared respondents."""
    graph = nx.Graph()
    graph.add_edges_from(pairs.tolist())
    return [list(component) for component in nx.connected_components(graph)]


def match_count_pmf(
    counts: NDArray, null_dist: rv_discrete | rv_continuous, v: int, tail: int | None = None
) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
    """Observed proportions and fitted pmf over the support of counts, for plotting."""
    lo = max(int(np.floor(counts.min() * 0.95)), 0)
    hi = min(int(np.ceil(counts.max() * 1.05)), v)
    x = np.arange(lo, hi + 1)
    if tail:
        x = x[-tail:]
    observed = np.array([(counts == xv).mean() for xv in x])
    try:
        pmf = null_dist.pmf(x)
    except AttributeError:
        pmf = null_dist.cdf(x + 0.5) - null_dist.cdf(x - 0.5)
    return x, observed, pmf


def null_fit_sse(counts: NDArray, null_dist: rv_discrete | rv_continuous, v: int) -> float:
    """Sum of squared error between the observed histogram and a candidate null's pmf.

    A quick, distribution-agnostic way to choose between candidate nulls (e.g.
    beta-binomial vs. normal) when the beta-binomial's maximum-likelihood fit
    is visibly a poor match to skewed real data.
    """
    _, observed, pmf = match_count_pmf(counts, null_dist, v)
    return float(np.sum((observed - pmf) ** 2))


@dataclass
class MatchResult:
    """Output of flag_duplicate_pairs: the full pairwise test plus flagged groups."""

    counts: NDArray[np.int_]
    null_dist: rv_discrete | rv_continuous
    pvalues: NDArray[np.float64]
    qvalues: NDArray[np.float64]
    rejected: NDArray[np.bool_]
    flagged_pairs: NDArray[np.int_]
    components: list[list[int]]


def flag_duplicate_pairs(
    data: NDArray,
    alpha: float = 0.05,
    fdr_method: str = "fdr_by",
    null: str = "beta-binomial",
) -> MatchResult:
    """End-to-end pipeline: fit the null, test every pair, and group flags.

    null is one of "beta-binomial" (maximum likelihood) or "normal".
    fdr_method defaults to "fdr_by" (Benjamini-Yekutieli): because every
    respondent appears in n-1 of the C(n,2) tests, "fdr_bh" is measurably
    liberal even with independent questions (see the paper's calibration
    check), so the dependence-robust correction is the safer default.
    """
    n, v = data.shape
    counts = pairwise_match_counts(data)
    if null == "beta-binomial":
        null_dist = fit_beta_binomial_ml(counts, v)
    elif null == "normal":
        null_dist = fit_normal_approx(counts)
    else:
        raise ValueError(f"Unknown null distribution family: {null}")
    pvalues = match_pvalues(counts, null_dist)
    rejected, qvalues = fdr_flag(pvalues, alpha=alpha, method=fdr_method)
    flagged_pairs = pair_index(n, rejected)
    components = connected_components(flagged_pairs) if flagged_pairs.size else []
    return MatchResult(
        counts=counts,
        null_dist=null_dist,
        pvalues=pvalues,
        qvalues=qvalues,
        rejected=rejected,
        flagged_pairs=flagged_pairs,
        components=components,
    )
