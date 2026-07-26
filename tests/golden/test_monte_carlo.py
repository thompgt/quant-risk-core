"""Reproducibility and convergence tests for the Monte Carlo VaR engine.

The defect these were written to catch: `simulate_paths` was
`@njit(parallel=True)` and drew normals via `np.random.standard_normal` inside a
`prange` loop. Numba gives each worker thread its own RNG state that no
Python-level `np.random.seed` can reach, so two identical calls produced
different VaR numbers. A risk figure that cannot be regenerated cannot be
validated or signed off, which makes this an audit blocker rather than a
nuisance.

Convergence is checked against the closed-form parametric result, which is an
independent reference: for a single-step GBM the simple return is
lognormal, so MC VaR must approach the analytic quantile as paths increase.
"""
import numpy as np
import pytest

from market_risk.estimators import RiskEngine, resolve_rng, simulate_paths

S0, MU, SIGMA = 100.0, 0.0005, 0.02


def test_same_seed_is_bit_identical():
    a = simulate_paths(S0, MU, SIGMA, horizon=10, paths=5000, seed=1337)
    b = simulate_paths(S0, MU, SIGMA, horizon=10, paths=5000, seed=1337)
    assert np.array_equal(a, b), "identical seeds must give bit-identical paths"


def test_different_seeds_differ():
    a = simulate_paths(S0, MU, SIGMA, horizon=10, paths=5000, seed=1)
    b = simulate_paths(S0, MU, SIGMA, horizon=10, paths=5000, seed=2)
    assert not np.array_equal(a, b)


def test_var_es_reproducible_from_seed():
    engine = RiskEngine(confidence_levels=[0.95, 0.99])
    kw = dict(initial_value=1.0, mu=MU, sigma=SIGMA, horizon=1, paths=20_000)
    a = engine.monte_carlo_var_es(seed=42, **kw)
    b = engine.monte_carlo_var_es(seed=42, **kw)
    assert a == b


def test_result_reports_the_seed_it_used():
    """An unseeded call must still be replayable from what it returns."""
    engine = RiskEngine(confidence_levels=[0.99])
    kw = dict(initial_value=1.0, mu=MU, sigma=SIGMA, horizon=1, paths=10_000)

    first = engine.monte_carlo_var_es(**kw)
    assert first["seed"] >= 0
    assert first["paths"] == 10_000

    replay = engine.monte_carlo_var_es(seed=int(first["seed"]), **kw)
    assert replay["VaR_0.99"] == first["VaR_0.99"]
    assert replay["ES_0.99"] == first["ES_0.99"]


def test_generator_passed_in_is_marked_unreplayable():
    rng = np.random.default_rng(7)
    _, seed = resolve_rng(rng)
    assert seed == -1, "a live Generator has no reproducing integer seed"


def test_converges_to_parametric_var():
    """MC VaR must approach the analytic lognormal quantile as paths grow.

    For one step, S_T/S0 - 1 = exp((mu - sigma^2/2) + sigma*Z) - 1, so the
    alpha-level VaR has the closed form below — computed here directly from
    scipy, independent of RiskEngine.
    """
    from scipy import stats

    alpha = 0.99
    q = stats.norm.ppf(1 - alpha)
    analytic = -(np.exp((MU - 0.5 * SIGMA**2) + SIGMA * q) - 1.0)

    engine = RiskEngine(confidence_levels=[alpha])
    errors = []
    for paths in (5_000, 200_000):
        res = engine.monte_carlo_var_es(
            initial_value=1.0, mu=MU, sigma=SIGMA, horizon=1, paths=paths, seed=99
        )
        errors.append(abs(res[f"VaR_{alpha}"] - analytic))

    assert errors[-1] < 1e-3, f"MC VaR {errors[-1]:.2e} from analytic {analytic:.6f}"
    assert errors[-1] < errors[0], "error should shrink with more paths"


def test_horizon_scales_dispersion():
    """Terminal dispersion must grow with sqrt(horizon)."""
    short = simulate_paths(S0, 0.0, SIGMA, horizon=1, paths=200_000, seed=5)
    long = simulate_paths(S0, 0.0, SIGMA, horizon=16, paths=200_000, seed=5)
    ratio = np.std(np.log(long / S0)) / np.std(np.log(short / S0))
    assert ratio == pytest.approx(4.0, rel=0.05)


@pytest.mark.parametrize("bad", [dict(horizon=0), dict(horizon=-1), dict(paths=0)])
def test_rejects_degenerate_sizes(bad):
    kw = dict(S0=S0, mu=MU, sigma=SIGMA, horizon=5, paths=100)
    kw.update(bad)
    with pytest.raises(ValueError):
        simulate_paths(kw["S0"], kw["mu"], kw["sigma"], kw["horizon"], kw["paths"])


def test_zero_volatility_is_deterministic_drift():
    out = simulate_paths(S0, MU, 0.0, horizon=4, paths=10, seed=3)
    assert out == pytest.approx(S0 * np.exp(MU * 4))
