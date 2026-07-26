"""Golden-value tests for parametric VaR and Expected Shortfall.

These lock down the sign convention of the mean term, which is the defect this
file was written to catch: ES was implemented as ``mu + sigma * k`` while VaR
was ``-(mu + sigma * q)``, so a non-zero mean pushed ES the wrong way by
``2 * mu``. Every pre-existing test either used ``mu = 0`` or only asserted
``ES > VaR``, so the bug was invisible.

Reference values are produced by numerical quadrature of the tail expectation

    ES_a = -mu - sigma * s * (1 / (1 - a)) * INT_{-inf}^{q} t f(t) dt

which shares no algebra with the closed forms under test (`scipy.integrate.quad`
against `scipy.stats.norm.pdf` / `t.pdf`). The generating script is recorded in
the docstring of `reference_es` below so the numbers can be regenerated.

Convention: both VaR and ES are positive loss quantities. A positive expected
return *reduces* the expected loss, so both must decrease as `mu` increases.
"""
import numpy as np
import pytest

from market_risk.estimators import RiskEngine

SIGMA = 0.02

# (dist, df, mu, alpha): (VaR, ES) from quadrature — see module docstring.
GOLDEN = {
    ("normal", None, 0.0, 0.95): (0.032897072539, 0.041254256150),
    ("normal", None, 0.0, 0.99): (0.046526957481, 0.053304284407),
    ("normal", None, 0.001, 0.95): (0.031897072539, 0.040254256150),
    ("normal", None, 0.001, 0.99): (0.045526957481, 0.052304284407),
    ("normal", None, -0.002, 0.95): (0.034897072539, 0.043254256150),
    ("normal", None, -0.002, 0.99): (0.048526957481, 0.055304284407),
    ("t", 5.0, 0.0, 0.95): (0.031216995167, 0.044773685109),
    ("t", 5.0, 0.0, 0.99): (0.052129271388, 0.068976735201),
    ("t", 5.0, 0.001, 0.95): (0.030216995167, 0.043773685109),
    ("t", 5.0, 0.001, 0.99): (0.051129271388, 0.067976735201),
    ("t", 5.0, -0.002, 0.95): (0.033216995167, 0.046773685109),
    ("t", 5.0, -0.002, 0.99): (0.054129271388, 0.070976735201),
}


def reference_es(mu, sigma, alpha, dist, df=None):
    """Independent quadrature reference — the generator for the GOLDEN table.

    Kept in the test file (rather than only in a scratch script) so the golden
    numbers remain auditable and regenerable.
    """
    from scipy import integrate, stats

    if dist == "normal":
        rv, s = stats.norm(), 1.0
    else:
        rv, s = stats.t(df), np.sqrt((df - 2) / df)
    q = rv.ppf(1 - alpha)
    var = -(mu + sigma * q * s)
    num = integrate.quad(lambda t: t * rv.pdf(t), -np.inf, q)[0]
    es = -mu - sigma * s * num / (1 - alpha)
    return var, es


@pytest.mark.parametrize("key", list(GOLDEN))
def test_parametric_var_es_matches_golden(key):
    dist, df, mu, alpha = key
    expected_var, expected_es = GOLDEN[key]

    engine = RiskEngine(confidence_levels=[alpha])
    res = engine.parametric_var_es(mu, SIGMA, dist=dist, df=df)

    assert res[f"VaR_{alpha}"] == pytest.approx(expected_var, abs=1e-10)
    assert res[f"ES_{alpha}"] == pytest.approx(expected_es, abs=1e-10)


@pytest.mark.parametrize("key", list(GOLDEN))
def test_golden_table_matches_its_generator(key):
    """Guard against the GOLDEN table drifting from the quadrature reference."""
    dist, df, mu, alpha = key
    var, es = reference_es(mu, SIGMA, alpha, dist, df)
    assert (var, es) == pytest.approx(GOLDEN[key], abs=1e-10)


@pytest.mark.parametrize("dist,df", [("normal", None), ("t", 5.0)])
def test_mean_shifts_var_and_es_in_the_same_direction(dist, df):
    """The regression test for the sign bug.

    Raising the expected return by d must lower BOTH VaR and ES by exactly d.
    The buggy implementation *raised* ES by d instead.
    """
    d = 0.001
    engine = RiskEngine(confidence_levels=[0.99])
    lo = engine.parametric_var_es(0.0, SIGMA, dist=dist, df=df)
    hi = engine.parametric_var_es(d, SIGMA, dist=dist, df=df)

    assert hi["VaR_0.99"] == pytest.approx(lo["VaR_0.99"] - d, abs=1e-12)
    assert hi["ES_0.99"] == pytest.approx(lo["ES_0.99"] - d, abs=1e-12)


@pytest.mark.parametrize("dist,df", [("normal", None), ("t", 5.0)])
@pytest.mark.parametrize("mu", [0.0, 0.001, -0.002, 0.05])
def test_es_dominates_var(dist, df, mu):
    """ES is an average over the tail beyond VaR, so ES >= VaR for any mean."""
    engine = RiskEngine(confidence_levels=[0.95, 0.99])
    res = engine.parametric_var_es(mu, SIGMA, dist=dist, df=df)
    for alpha in (0.95, 0.99):
        assert res[f"ES_{alpha}"] > res[f"VaR_{alpha}"]


def test_student_t_requires_df_above_two():
    engine = RiskEngine(confidence_levels=[0.99])
    for bad_df in (None, 1.0, 2.0):
        with pytest.raises(ValueError, match="Degrees of freedom"):
            engine.parametric_var_es(0.0, SIGMA, dist="t", df=bad_df)


def test_unsupported_distribution_rejected():
    engine = RiskEngine(confidence_levels=[0.99])
    with pytest.raises(ValueError, match="Unsupported distribution"):
        engine.parametric_var_es(0.0, SIGMA, dist="laplace")
