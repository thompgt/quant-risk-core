"""Validation tests for the EVT peaks-over-threshold engine.

Defects these were written to catch, in `estimate_risk`:

* ``VaR = u + (beta/xi) * (ratio**-xi - 1)`` divides by the shape parameter with
  no guard. A fitted ``xi`` near zero — the exponential-tail case, which is
  entirely plausible for equity losses — blows the expression up.
* ``ES = (VaR + beta - xi*u)/(1 - xi)`` is only the tail mean when ``xi < 1``.
  For ``xi >= 1`` the GPD has infinite mean; the old code returned a finite
  number, and for ``xi > 1`` a *negative* one, i.e. a negative expected loss.
* Nothing rejected ``alpha`` levels below the fitted threshold, where the tail
  model carries no information.

The VaR reference is independent of the closed form under test: rather than
re-deriving the quantile algebraically, each returned VaR is pushed back through
`scipy.stats.genpareto.sf` and the implied exceedance probability must equal
``1 - alpha``. The ES reference is numerical integration of the tail.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from market_risk.extreme_value import EVTEngine


def make_engine(xi, beta, u=0.02, pu=0.05, n_total=4000):
    """Construct a fitted engine with known parameters, bypassing estimation."""
    eng = EVTEngine(threshold_quantile=1 - pu)
    eng.xi, eng.beta, eng.u = xi, beta, u
    eng.n_total = n_total
    eng.n_excess = int(round(pu * n_total))
    eng._fitted = True
    return eng


def implied_tail_prob(eng, var):
    """P(loss > var) implied by the fitted GPD — the independent inversion."""
    return eng.exceedance_rate * stats.genpareto.sf(
        var - eng.u, eng.xi, loc=0, scale=eng.beta
    )


def tail_mean_by_quadrature(eng, var):
    """E[L | L > var] via numerical integration of the GPD tail."""
    sf_at_var = stats.genpareto.sf(var - eng.u, eng.xi, loc=0, scale=eng.beta)

    def integrand(x):
        return stats.genpareto.pdf(x - eng.u, eng.xi, loc=0, scale=eng.beta) * x

    mass = integrate.quad(integrand, var, np.inf, limit=400)[0]
    return mass / sf_at_var


@pytest.mark.parametrize("xi", [-0.15, 0.0, 1e-9, 0.05, 0.2, 0.45])
@pytest.mark.parametrize("alpha", [0.96, 0.99, 0.999])
def test_var_inverts_to_the_requested_tail_probability(xi, alpha):
    eng = make_engine(xi=xi, beta=0.01)
    var, _ = eng.estimate_risk(alpha)
    assert implied_tail_prob(eng, var) == pytest.approx(1 - alpha, rel=1e-8)


@pytest.mark.parametrize("xi", [-0.15, 0.05, 0.2, 0.45])
@pytest.mark.parametrize("alpha", [0.99, 0.999])
def test_es_matches_quadrature_tail_mean(xi, alpha):
    eng = make_engine(xi=xi, beta=0.01)
    var, es = eng.estimate_risk(alpha)
    assert es == pytest.approx(tail_mean_by_quadrature(eng, var), rel=1e-6)


def test_xi_zero_uses_the_exponential_limit_and_stays_finite():
    """The regression test for the division-by-xi defect."""
    eng = make_engine(xi=0.0, beta=0.01)
    var, es = eng.estimate_risk(0.99)
    assert np.isfinite(var) and np.isfinite(es)
    # Closed form for the exponential tail, stated independently.
    expected = eng.u + eng.beta * np.log(eng.exceedance_rate / 0.01)
    assert var == pytest.approx(expected, rel=1e-12)
    # For an exponential excess the mean excess above any level is beta.
    assert es == pytest.approx(var + eng.beta, rel=1e-12)


def test_xi_approaching_zero_is_continuous():
    """No discontinuity at the tolerance boundary."""
    below = make_engine(xi=1e-9, beta=0.01).estimate_risk(0.99)[0]
    above = make_engine(xi=1e-4, beta=0.01).estimate_risk(0.99)[0]
    assert below == pytest.approx(above, rel=1e-3)


@pytest.mark.parametrize("xi", [1.0, 1.4])
def test_infinite_mean_tail_reports_infinite_es(xi):
    """The regression test for the xi >= 1 defect.

    The GPD has infinite mean once xi >= 1, so ES does not exist as a finite
    number. VaR is still well defined.
    """
    eng = make_engine(xi=xi, beta=0.01)
    var, es = eng.estimate_risk(0.99)
    assert np.isfinite(var), "VaR is still well defined for xi >= 1"
    assert es == float("inf")


def test_old_es_formula_was_wrong_beyond_xi_one():
    """Document precisely how the unguarded ES expression failed.

    At xi == 1 it divides by zero; for xi > 1 it returns a *negative* ES, i.e. a
    negative expected loss, which no risk report can carry. Both are silent in
    the sense that neither is a plausible-looking number a reviewer would catch.
    """
    def old_es(eng, var, xi):
        return (var + eng.beta - xi * eng.u) / (1.0 - xi)

    eng_at_one = make_engine(xi=1.0, beta=0.01)
    var_at_one, _ = eng_at_one.estimate_risk(0.99)
    with pytest.raises(ZeroDivisionError):
        old_es(eng_at_one, var_at_one, 1.0)

    eng_above = make_engine(xi=1.4, beta=0.01)
    var_above, _ = eng_above.estimate_risk(0.99)
    assert old_es(eng_above, var_above, 1.4) < 0


def test_rejects_alpha_inside_the_distribution_body():
    eng = make_engine(xi=0.2, beta=0.01, pu=0.05)
    # 1 - 0.90 = 0.10 > 0.05, so this level is below the fitted threshold.
    with pytest.raises(ValueError, match="exceeds the fitted exceedance rate"):
        eng.estimate_risk(0.90)


def test_alpha_exactly_at_the_threshold_is_allowed():
    """A request at the declared threshold level must not be rejected.

    The empirical exceedance count can fall marginally below the nominal
    1 - threshold_quantile, because the threshold is an interpolated sample
    quantile and exceedances are counted strictly. Comparing against the
    empirical rate alone would reject alpha=0.95 on a 95% threshold, where the
    answer is simply VaR = u.
    """
    eng = make_engine(xi=0.2, beta=0.01, pu=0.05)
    eng.n_excess = eng.n_excess - 3  # nudge empirical pu just under nominal
    assert eng.exceedance_rate < 0.05

    var, es = eng.estimate_risk(0.95)
    assert np.isfinite(var) and np.isfinite(es)
    assert es > var
    # Because empirical pu sits fractionally below the requested tail
    # probability, the estimator interpolates a hair below the threshold rather
    # than extrapolating above it. That is the accepted cost of admitting the
    # boundary case; it must stay negligible relative to u.
    assert var == pytest.approx(eng.u, rel=0.02)


def test_var_is_monotone_in_alpha():
    eng = make_engine(xi=0.2, beta=0.01)
    levels = [0.96, 0.98, 0.99, 0.995, 0.999]
    vars_ = [eng.estimate_risk(a)[0] for a in levels]
    assert all(b > a for a, b in zip(vars_, vars_[1:]))


def test_es_dominates_var():
    eng = make_engine(xi=0.2, beta=0.01)
    for alpha in (0.96, 0.99, 0.999):
        var, es = eng.estimate_risk(alpha)
        assert es > var


def test_unfitted_engine_raises():
    with pytest.raises(ValueError, match="must be fitted"):
        EVTEngine().estimate_risk(0.99)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5, 1.5])
def test_rejects_invalid_alpha(alpha):
    eng = make_engine(xi=0.2, beta=0.01)
    with pytest.raises(ValueError):
        eng.estimate_risk(alpha)


@pytest.mark.parametrize("q", [0.0, 1.0, 1.5])
def test_rejects_invalid_threshold_quantile(q):
    with pytest.raises(ValueError, match="threshold_quantile"):
        EVTEngine(threshold_quantile=q)


def test_fit_recovers_known_gpd_parameters():
    """End-to-end fit on synthetic GPD-tailed losses."""
    rng = np.random.default_rng(20240726)
    true_xi, true_beta, u = 0.25, 0.01, 0.02

    body = rng.uniform(0.0, u, size=19_000)
    tail = u + stats.genpareto.rvs(
        true_xi, loc=0, scale=true_beta, size=1_000, random_state=rng
    )
    losses = np.concatenate([body, tail])
    returns = pd.Series(-losses)

    eng = EVTEngine(threshold_quantile=0.95)
    eng.fit(returns)

    assert eng.xi == pytest.approx(true_xi, abs=0.08)
    assert eng.beta == pytest.approx(true_beta, rel=0.25)
    assert eng.n_excess == pytest.approx(1_000, rel=0.10)


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError, match="No observations"):
        EVTEngine().fit(pd.Series([], dtype=float))
