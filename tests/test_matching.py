import itertools

import numpy as np
import pytest
from scipy.stats import binom

from surveymatch import (
    connected_components,
    expected_match_probs,
    fdr_flag,
    fit_beta_binomial_ml,
    fit_normal_approx,
    flag_duplicate_pairs,
    match_pvalues,
    null_fit_sse,
    pair_index,
    pairwise_match_counts,
    simulate_correlated_responses,
    simulate_responses,
)
from surveymatch.matching import poisson_binomial_null


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_simulate_responses_shape_and_support(rng):
    data = simulate_responses(500, [[0.5, 0.5], [0.2, 0.3, 0.5]], rng=rng)
    assert data.shape == (500, 2)
    assert set(np.unique(data[:, 0])) <= {0, 1}
    assert set(np.unique(data[:, 1])) <= {0, 1, 2}


def test_expected_match_probs_matches_analytic_bernoulli(rng):
    data = simulate_responses(200_000, [[0.5, 0.5]], rng=rng)
    probs = expected_match_probs(data)
    assert probs.shape == (1,)
    assert probs[0] == pytest.approx(0.5, abs=0.01)


def test_expected_match_probs_matches_analytic_skewed(rng):
    data = simulate_responses(200_000, [[0.7, 0.2, 0.1]], rng=rng)
    probs = expected_match_probs(data)
    assert probs[0] == pytest.approx(0.7**2 + 0.2**2 + 0.1**2, abs=0.01)


def test_pairwise_match_counts_small_example():
    data = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]])
    counts = pairwise_match_counts(data)
    # row0 vs row1: 2 matches, row0 vs row2: 1 match, row1 vs row2: 2 matches
    assert list(counts) == [2, 1, 2]


def test_pairwise_match_counts_length_is_n_choose_2(rng):
    n = 12
    data = simulate_responses(n, [[0.5, 0.5]] * 5, rng=rng)
    counts = pairwise_match_counts(data)
    assert counts.shape[0] == n * (n - 1) // 2


def test_pair_index_matches_itertools_combinations():
    n = 6
    all_pairs = np.array(list(itertools.combinations(range(n), 2)))
    mask = np.zeros(len(all_pairs), dtype=bool)
    mask[[0, 3, 5]] = True
    result = pair_index(n, mask)
    np.testing.assert_array_equal(result, all_pairs[[0, 3, 5]])


def test_beta_binomial_fit_recovers_mean(rng):
    v = 40
    data = simulate_responses(2000, [[0.6, 0.4]] * v, rng=rng)
    counts = pairwise_match_counts(data)
    fitted = fit_beta_binomial_ml(counts, v)
    assert fitted.mean() == pytest.approx(counts.mean(), rel=0.05)


def test_match_pvalues_are_in_unit_interval(rng):
    v = 20
    data = simulate_responses(300, [[0.5, 0.5]] * v, rng=rng)
    counts = pairwise_match_counts(data)
    fitted = fit_beta_binomial_ml(counts, v)
    pvalues = match_pvalues(counts, fitted)
    assert np.all(pvalues >= 0) and np.all(pvalues <= 1)


def test_fdr_flag_rejects_nothing_under_null(rng):
    uniform_pvalues = rng.uniform(size=5000)
    rejected, qvalues = fdr_flag(uniform_pvalues, alpha=0.01, method="fdr_bh")
    assert rejected.sum() <= 5000 * 0.02
    assert np.all((qvalues >= 0) & (qvalues <= 1))


def test_connected_components_groups_shared_members():
    pairs = np.array([[0, 1], [1, 2], [5, 6]])
    components = connected_components(pairs)
    components_as_sets = {frozenset(c) for c in components}
    assert frozenset({0, 1, 2}) in components_as_sets
    assert frozenset({5, 6}) in components_as_sets


def test_flag_duplicate_pairs_finds_injected_duplicate(rng):
    v = 60
    props = [np.array([0.9, 0.1])] * v
    data = simulate_responses(400, props, rng=rng)
    # Force one exact duplicate pair that would be extremely unlikely by chance.
    data[1] = data[0]
    result = flag_duplicate_pairs(data, alpha=0.05, fdr_method="fdr_bh")
    flagged_rows = set(result.flagged_pairs.flatten().tolist())
    assert {0, 1} <= flagged_rows


def test_poisson_binomial_null_matches_binomial_special_case():
    v = 25
    dist = poisson_binomial_null([0.5] * v)
    x = np.arange(v + 1)
    np.testing.assert_allclose(dist.pmf(x), binom(v, 0.5).pmf(x), atol=1e-10)


def test_poisson_binomial_null_mean_matches_sum_of_probs():
    probs = [0.7**2 + 0.2**2 + 0.1**2] * 10 + [0.5] * 5
    dist = poisson_binomial_null(probs)
    assert dist.mean() == pytest.approx(sum(probs), abs=1e-8)


def test_flag_duplicate_pairs_normal_null_runs(rng):
    v = 15
    data = simulate_responses(150, [[0.5, 0.5]] * v, rng=rng)
    result = flag_duplicate_pairs(data, null="normal")
    assert result.counts.shape[0] == 150 * 149 // 2


def test_null_fit_sse_prefers_correctly_specified_family(rng):
    v = 40
    # Highly skewed marginal -> overdispersed, left-skewed match counts that a
    # symmetric normal should fit worse than a beta-binomial.
    props = [np.array([0.92, 0.08])] * v
    data = simulate_responses(1500, props, rng=rng)
    counts = pairwise_match_counts(data)
    bb = fit_beta_binomial_ml(counts, v)
    normal = fit_normal_approx(counts)
    assert null_fit_sse(counts, bb, v) < null_fit_sse(counts, normal, v)


def test_simulate_correlated_responses_shape_and_support(rng):
    data = simulate_correlated_responses(
        200, 10, [0.6, 0.3, 0.1], tilt=[1.0, 0.0, -1.0], rho=0.8, rng=rng
    )
    assert data.shape == (200, 10)
    assert set(np.unique(data)) <= {0, 1, 2}


def test_simulate_correlated_responses_zero_rho_matches_marginals(rng):
    data = simulate_correlated_responses(
        50_000, 1, [0.7, 0.2, 0.1], tilt=[1.0, -0.3, -0.7], rho=0.0, rng=rng
    )
    _, counts = np.unique(data, return_counts=True)
    np.testing.assert_allclose(counts / data.shape[0], [0.7, 0.2, 0.1], atol=0.01)


def test_simulate_correlated_responses_increases_pairwise_match_rate(rng):
    v = 30
    kwargs = dict(n=800, v=v, category_probs=[0.6, 0.3, 0.1], tilt=[1.0, 0.0, -1.0])
    low = simulate_correlated_responses(rho=0.0, rng=np.random.default_rng(1), **kwargs)
    high = simulate_correlated_responses(rho=1.5, rng=np.random.default_rng(1), **kwargs)
    assert pairwise_match_counts(high).var() > pairwise_match_counts(low).var()
