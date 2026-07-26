"""Validation tests for portfolio decomposition and copula sampling.

Items these cover:

* Euler additivity — component VaR must sum to portfolio VaR exactly. This is
  the identity that makes the decomposition interpretable as a risk allocation,
  and it holds because VaR is homogeneous of degree one in the weights.
* `CopulaEngine` previously let `np.linalg.cholesky` raise a bare
  `LinAlgError` on a non-positive-definite input, and validated neither symmetry
  nor the unit diagonal — so a covariance matrix passed where a correlation
  matrix was expected would either fail cryptically or silently produce samples
  with the wrong dependence.
* `RiskEngine.confidence_levels` defaulted to the mutable literal
  `[0.95, 0.99]`, shared across all instances that did not pass one.
"""
import numpy as np
import pytest

from market_risk.estimators import RiskEngine
from portfolio_risk.decomposition import CopulaEngine, RiskDecomposer


def make_cov(n=4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n * 8, n))
    return np.cov(a, rowvar=False)


# --------------------------------------------------------------------------
# Euler additivity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_component_var_sums_to_portfolio_var(alpha, seed):
    n = 5
    cov = make_cov(n, seed)
    weights = np.full(n, 1 / n)
    dec = RiskDecomposer(weights, cov)

    assert dec.calculate_component_var(alpha).sum() == pytest.approx(
        dec.portfolio_var(alpha), rel=1e-12
    )


def test_component_var_sums_for_uneven_weights():
    cov = make_cov(4, 7)
    weights = np.array([0.5, 0.25, 0.15, 0.10])
    dec = RiskDecomposer(weights, cov)
    assert dec.calculate_component_var(0.99).sum() == pytest.approx(
        dec.portfolio_var(0.99), rel=1e-12
    )


def test_single_asset_component_var_is_the_whole_var():
    cov = np.array([[0.04]])
    dec = RiskDecomposer(np.array([1.0]), cov)
    # VaR = z * sigma = z * 0.2
    from scipy import stats

    assert dec.portfolio_var(0.99) == pytest.approx(stats.norm.ppf(0.99) * 0.2)
    assert dec.calculate_component_var(0.99)[0] == pytest.approx(dec.portfolio_var(0.99))


def test_marginal_var_is_the_gradient_of_portfolio_var():
    """MVaR must equal dVaR/dw, checked by finite differences."""
    cov = make_cov(4, 3)
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    dec = RiskDecomposer(weights, cov)
    analytic = dec.calculate_marginal_var(0.99)

    h = 1e-7
    fd = np.empty_like(weights)
    for i in range(len(weights)):
        up, down = weights.copy(), weights.copy()
        up[i] += h
        down[i] -= h
        fd[i] = (
            RiskDecomposer(up, cov).portfolio_var(0.99)
            - RiskDecomposer(down, cov).portfolio_var(0.99)
        ) / (2 * h)

    np.testing.assert_allclose(analytic, fd, rtol=1e-5)


def test_uncorrelated_equal_assets_share_risk_equally():
    cov = np.eye(4) * 0.04
    dec = RiskDecomposer(np.full(4, 0.25), cov)
    comp = dec.calculate_component_var(0.95)
    np.testing.assert_allclose(comp, comp[0])


def test_decomposer_validates_shapes_and_symmetry():
    with pytest.raises(ValueError, match="square"):
        RiskDecomposer(np.ones(3), np.ones((3, 2)))
    with pytest.raises(ValueError, match="weights"):
        RiskDecomposer(np.ones(3), np.eye(4))
    with pytest.raises(ValueError, match="symmetric"):
        RiskDecomposer(np.ones(2), np.array([[1.0, 0.5], [0.2, 1.0]]))
    with pytest.raises(ValueError, match="confidence_level"):
        RiskDecomposer(np.ones(2), np.eye(2)).calculate_marginal_var(1.5)


def test_zero_weights_are_rejected_rather_than_dividing_by_zero():
    dec = RiskDecomposer(np.zeros(3), np.eye(3))
    with pytest.raises(ValueError, match="not positive"):
        dec.calculate_marginal_var(0.95)


# --------------------------------------------------------------------------
# Copula
# --------------------------------------------------------------------------
def test_copula_samples_are_uniform_marginals():
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    u = CopulaEngine(corr).generate_gaussian_copula_samples(200_000, seed=1)
    assert u.shape == (200_000, 2)
    assert u.min() > 0.0 and u.max() < 1.0
    # Uniform marginals have mean 1/2 and variance 1/12.
    np.testing.assert_allclose(u.mean(axis=0), 0.5, atol=5e-3)
    np.testing.assert_allclose(u.var(axis=0), 1 / 12, atol=5e-3)


def test_copula_reproduces_the_requested_rank_correlation():
    """Gaussian copula: Spearman rho = (6/pi)*arcsin(r/2)."""
    from scipy import stats

    r = 0.6
    corr = np.array([[1.0, r], [r, 1.0]])
    u = CopulaEngine(corr).generate_gaussian_copula_samples(200_000, seed=2)

    expected = (6 / np.pi) * np.arcsin(r / 2)
    observed = stats.spearmanr(u[:, 0], u[:, 1]).statistic
    assert observed == pytest.approx(expected, abs=0.01)


def test_copula_is_reproducible_from_a_seed():
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    eng = CopulaEngine(corr)
    a = eng.generate_gaussian_copula_samples(1000, seed=5)
    b = eng.generate_gaussian_copula_samples(1000, seed=5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, eng.generate_gaussian_copula_samples(1000, seed=6))


def test_non_positive_definite_correlation_gives_a_diagnosis():
    """The regression test for the bare LinAlgError."""
    # Perfectly collinear third asset.
    corr = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="not positive definite"):
        CopulaEngine(corr)


def test_covariance_matrix_passed_as_correlation_is_caught():
    """A covariance matrix has a non-unit diagonal; that must be flagged."""
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    with pytest.raises(ValueError, match="unit diagonal"):
        CopulaEngine(cov)


def test_asymmetric_correlation_is_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        CopulaEngine(np.array([[1.0, 0.5], [0.2, 1.0]]))


def test_non_square_correlation_is_rejected():
    with pytest.raises(ValueError, match="square"):
        CopulaEngine(np.array([[1.0, 0.5, 0.2]]))


def test_copula_rejects_non_positive_sample_count():
    eng = CopulaEngine(np.eye(2))
    with pytest.raises(ValueError, match="n_samples"):
        eng.generate_gaussian_copula_samples(0)


# --------------------------------------------------------------------------
# RiskEngine defaults
# --------------------------------------------------------------------------
def test_default_confidence_levels_are_not_shared_between_instances():
    """The regression test for the mutable default argument."""
    first = RiskEngine()
    first.confidence_levels.append(0.999)

    second = RiskEngine()
    assert second.confidence_levels == [0.95, 0.99], (
        "mutating one engine's levels must not change the default for the next"
    )


def test_passed_confidence_levels_are_copied():
    levels = [0.99]
    engine = RiskEngine(confidence_levels=levels)
    levels.append(0.5)
    assert engine.confidence_levels == [0.99]


@pytest.mark.parametrize("bad", [[], [0.0], [1.0], [0.95, 1.5], [-0.1]])
def test_invalid_confidence_levels_are_rejected(bad):
    with pytest.raises(ValueError):
        RiskEngine(confidence_levels=bad)
