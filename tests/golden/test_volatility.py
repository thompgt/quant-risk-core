"""Validation tests for the GARCH and regime-switching engines.

Defect these were written to catch: `GARCHEngine.fit` wrapped the optimiser call
in ``warnings.simplefilter('ignore')``, discarding `arch`'s ConvergenceWarning,
and then read the fitted parameters regardless. A failed optimisation produced a
fully populated model whose conditional volatility and VaR looked entirely
well-formed.

This is not hypothetical. On 1000 draws from the legacy global RNG at seed 42 —
the exact data the pre-existing test suite used — the optimiser returns code 4,
"Inequality constraints incompatible", and fits alpha[1] = 6.3e-08. The old
test asserted only ``all(forecast > 0)``, which passed.

A second silent truncation: for p or q above 1 the scalar `alpha` and `beta`
attributes held only lag 1, so `GARCHEngine(p=2, q=2)` quietly discarded
alpha[2] and beta[2].
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from quant_risk_core.market_risk.volatility import (
    GARCHEngine,
    GARCHNotConvergedError,
    RegimeSwitchingEngine,
)


def non_converging_returns():
    """The exact series that makes the optimiser return status 4."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 0.01, 1000))


def garch_like_returns(n=2000, seed=7):
    """Simulate a genuine GARCH(1,1) process, which fits cleanly."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.08, 0.90
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        eps = rng.standard_normal()
        r[t] = np.sqrt(var) * eps
        var = omega + alpha * r[t] ** 2 + beta * var
    return pd.Series(r)


# --------------------------------------------------------------------------
# Convergence reporting
# --------------------------------------------------------------------------
def test_non_convergence_is_surfaced_not_swallowed():
    """The regression test for the blanket warning suppression."""
    engine = GARCHEngine(p=1, q=1)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        engine.fit(non_converging_returns())

    assert engine.converged is False
    assert engine.convergence_flag == 4
    assert "Inequality constraints" in engine.convergence_message


def test_non_convergence_still_populates_parameters():
    """Document the trap: the numbers look fine, which is why it must warn."""
    engine = GARCHEngine(p=1, q=1)
    with pytest.warns(RuntimeWarning):
        engine.fit(non_converging_returns())

    vol = engine.conditional_volatility()
    forecast = engine.forecast_volatility(5)

    assert np.all(vol > 0), "a failed fit still yields plausible-looking output"
    assert np.all(forecast > 0)
    assert engine.alpha < 1e-6, "the fit is degenerate despite looking healthy"


def test_strict_mode_raises_on_non_convergence():
    engine = GARCHEngine(p=1, q=1, strict=True)
    with pytest.raises(GARCHNotConvergedError, match="did not converge"):
        engine.fit(non_converging_returns())


def test_successful_fit_reports_convergence_and_does_not_warn():
    engine = GARCHEngine(p=1, q=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        engine.fit(garch_like_returns())

    assert engine.converged is True
    assert engine.convergence_flag == 0


def test_strict_mode_accepts_a_converging_fit():
    engine = GARCHEngine(p=1, q=1, strict=True)
    engine.fit(garch_like_returns())
    assert engine.converged is True


# --------------------------------------------------------------------------
# Parameter recovery and higher-order models
# --------------------------------------------------------------------------
def test_recovers_known_garch_parameters():
    """End-to-end: fit a simulated GARCH(1,1) and recover its coefficients."""
    engine = GARCHEngine(p=1, q=1)
    engine.fit(garch_like_returns(n=6000, seed=3))

    assert engine.converged is True
    # True alpha = 0.08, beta = 0.90.
    assert engine.alpha == pytest.approx(0.08, abs=0.05)
    assert engine.beta == pytest.approx(0.90, abs=0.08)
    assert engine.persistence == pytest.approx(0.98, abs=0.03)


def test_higher_order_parameters_are_not_silently_dropped():
    """The regression test for the truncated scalar attributes."""
    engine = GARCHEngine(p=2, q=2)
    with pytest.warns(RuntimeWarning, match="only the first lag"):
        engine.fit(garch_like_returns(n=3000))

    assert engine.alpha_params.size == 2
    assert engine.beta_params.size == 2
    assert engine.alpha == pytest.approx(engine.alpha_params[0])
    assert set(engine.params.index) >= {"alpha[1]", "alpha[2]", "beta[1]", "beta[2]"}


def test_persistence_sums_all_lags():
    engine = GARCHEngine(p=2, q=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit(garch_like_returns(n=3000))
    assert engine.persistence == pytest.approx(
        engine.alpha_params.sum() + engine.beta_params.sum()
    )
    assert engine.persistence < 1.0, "a stationary fit must have persistence below 1"


def test_student_t_fit_reports_degrees_of_freedom():
    engine = GARCHEngine(p=1, q=1, dist="t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit(garch_like_returns(n=3000))
    assert engine.nu is not None and engine.nu > 2


def asymmetric_returns(lev, n=4000, seed=11):
    """Simulate a GJR-type variance process with a controllable asymmetry.

    lev > 0: negative shocks amplify next-period variance (classic leverage).
    lev < 0: positive shocks amplify instead.
    lev = 0: symmetric.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.05, 0.90
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        eps = rng.standard_normal()
        r[t] = np.sqrt(var) * eps
        shock = alpha * r[t] ** 2
        if (r[t] < 0 and lev > 0) or (r[t] > 0 and lev < 0):
            shock += abs(lev) * r[t] ** 2
        var = omega + shock + beta * var
    return pd.Series(r)


def fit_quietly(engine, returns, model_type):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit(returns, model_type=model_type)
    return engine


def test_egarch_actually_estimates_the_leverage_term():
    """The regression test for the permanently-zero gamma.

    `arch` puts EGARCH's asymmetry under the `o` order, which defaults to 0. The
    old code never passed it, so the `gamma[1]` it read was never estimated and
    gamma was *always* exactly 0.0 — the effect EGARCH exists to capture was
    silently absent.

    Only the presence and non-zeroness of the term is asserted here. The
    simulated process is GJR-type rather than EGARCH, so EGARCH is misspecified
    against it and the fitted gamma's sign is not interpretable; the sign
    behaviour is tested on GJR below, where it is identified.
    """
    engine = fit_quietly(GARCHEngine(p=1, q=1), asymmetric_returns(0.15), "EGARCH")

    assert engine.params is not None
    assert "gamma[1]" in engine.params.index, "the leverage term must be fitted"
    assert engine.gamma == pytest.approx(float(engine.params["gamma[1]"]))
    assert engine.gamma != 0.0


def test_plain_garch_fits_no_leverage_term_by_default():
    engine = fit_quietly(GARCHEngine(p=1, q=1), garch_like_returns(n=2000), "Garch")
    assert engine.gamma_params.size == 0
    assert engine.gamma == 0.0


def test_gjr_gamma_tracks_the_direction_of_the_asymmetry():
    """GJR-GARCH: gamma is identified, so its sign must follow the DGP.

    sigma^2_t = omega + alpha*e^2 + gamma*e^2*I[e<0] + beta*sigma^2, so gamma is
    positive when downside shocks amplify variance and negative when upside
    shocks do.
    """
    downside = fit_quietly(
        GARCHEngine(p=1, q=1, o=1), asymmetric_returns(0.15), "Garch"
    )
    upside = fit_quietly(
        GARCHEngine(p=1, q=1, o=1), asymmetric_returns(-0.15), "Garch"
    )
    symmetric = fit_quietly(
        GARCHEngine(p=1, q=1, o=1), asymmetric_returns(0.0), "Garch"
    )

    assert downside.gamma_params.size == 1
    assert downside.gamma > 0.02, "downside amplification must give positive gamma"
    assert upside.gamma < -0.02, "upside amplification must give negative gamma"
    assert abs(symmetric.gamma) < 0.05, "a symmetric process needs no asymmetry term"
    assert downside.gamma > symmetric.gamma > upside.gamma


# --------------------------------------------------------------------------
# Forecasting and validation
# --------------------------------------------------------------------------
def test_forecast_shape_and_positivity():
    engine = GARCHEngine(p=1, q=1)
    engine.fit(garch_like_returns())
    forecast = engine.forecast_volatility(horizon=10)
    assert forecast.shape == (10,)
    assert np.all(forecast > 0)


def test_forecast_converges_towards_the_unconditional_level():
    """A stationary GARCH forecast must mean-revert as the horizon grows."""
    engine = GARCHEngine(p=1, q=1)
    engine.fit(garch_like_returns(n=6000, seed=3))
    long_horizon = engine.forecast_volatility(horizon=400)

    unconditional = np.sqrt(engine.omega / (1 - engine.persistence))
    assert abs(long_horizon[-1] - unconditional) < abs(long_horizon[0] - unconditional)


def test_unfitted_engine_raises():
    engine = GARCHEngine()
    with pytest.raises(ValueError, match="fitted"):
        engine.forecast_volatility(5)
    with pytest.raises(ValueError, match="fitted"):
        engine.conditional_volatility()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(p=0), "at least 1"),
        (dict(q=0), "at least 1"),
        (dict(dist="laplace"), "normal"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GARCHEngine(**kwargs)


def test_fit_validates_inputs():
    engine = GARCHEngine()
    with pytest.raises(ValueError, match="model_type"):
        engine.fit(garch_like_returns(n=200), model_type="TGARCH")
    with pytest.raises(ValueError, match="no observations"):
        engine.fit(pd.Series([], dtype=float))
    with pytest.raises(ValueError, match="zero variance"):
        engine.fit(pd.Series(np.zeros(100)))


# --------------------------------------------------------------------------
# Regime switching
# --------------------------------------------------------------------------
def test_regime_probabilities_sum_to_one():
    rng = np.random.default_rng(4)
    calm = rng.normal(0, 0.005, 400)
    stressed = rng.normal(0, 0.03, 200)
    returns = pd.Series(np.concatenate([calm, stressed, calm]))

    engine = RegimeSwitchingEngine(k_regimes=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit(returns)

    probs = engine.get_regime_probabilities()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)
    assert probs.shape[1] == 2


def test_regime_engine_validation():
    with pytest.raises(ValueError, match="at least 2"):
        RegimeSwitchingEngine(k_regimes=1)
    with pytest.raises(ValueError, match="fitted"):
        RegimeSwitchingEngine().get_regime_probabilities()
    with pytest.raises(ValueError, match="no observations"):
        RegimeSwitchingEngine().fit(pd.Series([], dtype=float))
