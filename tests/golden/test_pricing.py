"""Golden-value tests for Black-Scholes prices and Greeks.

Defect these were written to catch: `calculate_prices` supported both calls and
puts, but `calculate_greeks` took no `option_type` and always returned *call*
Greeks. Anyone risking a put position got the wrong sign on delta, theta and rho.
`calculate_prices` also treated any unrecognised `option_type` as a put, so a
typo silently priced the wrong instrument.

References:
* Hull, *Options, Futures and Other Derivatives*, worked example: S=42, K=40,
  r=10%, sigma=20%, T=0.5 gives a call of 4.76 and a put of 0.81.
* Greeks are validated against central finite differences of the price function,
  which is independent of the closed-form Greek expressions under test.
* Put-call parity, C - P = S - K*exp(-rT), is an identity that must hold exactly
  and is checked directly.
"""
import numpy as np
import pytest

from market_risk.pricing import BlackScholesEngine as BS

# Hull's worked example.
HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def test_call_price_matches_hull():
    assert BS.calculate_prices(option_type="call", **HULL) == pytest.approx(4.76, abs=5e-3)


def test_put_price_matches_hull():
    assert BS.calculate_prices(option_type="put", **HULL) == pytest.approx(0.81, abs=5e-3)


def test_put_call_parity_holds_exactly():
    call = BS.calculate_prices(option_type="call", **HULL)
    put = BS.calculate_prices(option_type="put", **HULL)
    lhs = call - put
    rhs = HULL["S"] - HULL["K"] * np.exp(-HULL["r"] * HULL["T"])
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_prices_are_vectorised_over_spot():
    spots = np.array([30.0, 42.0, 60.0])
    prices = BS.calculate_prices(spots, HULL["K"], HULL["T"], HULL["r"], HULL["sigma"])
    assert prices.shape == spots.shape
    assert np.all(np.diff(prices) > 0), "a call must increase in spot"


def test_unknown_option_type_is_rejected_not_treated_as_a_put():
    """The regression test for the silent fallthrough."""
    put = BS.calculate_prices(option_type="put", **HULL)
    with pytest.raises(ValueError, match="option_type"):
        BS.calculate_prices(option_type="putt", **HULL)
    with pytest.raises(ValueError, match="option_type"):
        BS.calculate_greeks(option_type="Call ", **HULL)
    # Guard the premise: the typo would previously have returned the put price.
    assert put > 0


# --------------------------------------------------------------------------
# Greeks vs finite differences
# --------------------------------------------------------------------------
def price(option_type, **kw):
    args = {**HULL, **kw}
    return float(BS.calculate_prices(option_type=option_type, **args))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type):
    h = 1e-5
    fd = (price(option_type, S=HULL["S"] + h) - price(option_type, S=HULL["S"] - h)) / (2 * h)
    assert BS.calculate_greeks(option_type=option_type, **HULL)["Delta"] == pytest.approx(
        fd, rel=1e-6
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_second_finite_difference(option_type):
    h = 1e-3
    fd = (
        price(option_type, S=HULL["S"] + h)
        - 2 * price(option_type, S=HULL["S"])
        + price(option_type, S=HULL["S"] - h)
    ) / h**2
    assert BS.calculate_greeks(option_type=option_type, **HULL)["Gamma"] == pytest.approx(
        fd, rel=1e-4
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(option_type):
    h = 1e-6
    fd = (
        price(option_type, sigma=HULL["sigma"] + h)
        - price(option_type, sigma=HULL["sigma"] - h)
    ) / (2 * h)
    assert BS.calculate_greeks(option_type=option_type, **HULL)["Vega"] == pytest.approx(
        fd, rel=1e-6
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type):
    h = 1e-7
    fd = (price(option_type, r=HULL["r"] + h) - price(option_type, r=HULL["r"] - h)) / (2 * h)
    assert BS.calculate_greeks(option_type=option_type, **HULL)["Rho"] == pytest.approx(
        fd, rel=1e-5
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_negative_time_derivative(option_type):
    """Theta is d/dt, i.e. minus d/dT."""
    h = 1e-6
    fd = -(price(option_type, T=HULL["T"] + h) - price(option_type, T=HULL["T"] - h)) / (2 * h)
    assert BS.calculate_greeks(option_type=option_type, **HULL)["Theta"] == pytest.approx(
        fd, rel=1e-5
    )


# --------------------------------------------------------------------------
# Call vs put relationships
# --------------------------------------------------------------------------
def test_put_greeks_differ_from_call_greeks():
    """The regression test: the two must not be the same dictionary."""
    call = BS.calculate_greeks(option_type="call", **HULL)
    put = BS.calculate_greeks(option_type="put", **HULL)

    assert call["Delta"] != pytest.approx(put["Delta"])
    assert call["Rho"] != pytest.approx(put["Rho"])
    assert call["Theta"] != pytest.approx(put["Theta"])


def test_delta_parity_and_signs():
    call = BS.calculate_greeks(option_type="call", **HULL)
    put = BS.calculate_greeks(option_type="put", **HULL)

    # d/dS of (C - P) = d/dS of (S - K e^{-rT}) = 1
    assert call["Delta"] - put["Delta"] == pytest.approx(1.0, rel=1e-12)
    assert 0.0 < call["Delta"] < 1.0
    assert -1.0 < put["Delta"] < 0.0


def test_rho_signs_are_opposite():
    call = BS.calculate_greeks(option_type="call", **HULL)
    put = BS.calculate_greeks(option_type="put", **HULL)
    assert call["Rho"] > 0, "a call gains value as rates rise"
    assert put["Rho"] < 0, "a put loses value as rates rise"


def test_gamma_and_vega_are_shared():
    call = BS.calculate_greeks(option_type="call", **HULL)
    put = BS.calculate_greeks(option_type="put", **HULL)
    assert call["Gamma"] == pytest.approx(put["Gamma"], rel=1e-12)
    assert call["Vega"] == pytest.approx(put["Vega"], rel=1e-12)
    assert call["Gamma"] > 0 and call["Vega"] > 0


def test_both_thetas_are_negative_for_long_options_here():
    """At these parameters both long positions decay."""
    assert BS.calculate_greeks(option_type="call", **HULL)["Theta"] < 0
    assert BS.calculate_greeks(option_type="put", **HULL)["Theta"] < 0


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad", [dict(S=0.0), dict(S=-1.0), dict(K=0.0), dict(T=0.0), dict(T=-1.0), dict(sigma=0.0), dict(sigma=-0.2)]
)
@pytest.mark.parametrize("fn", [BS.calculate_prices, BS.calculate_greeks])
def test_degenerate_inputs_are_rejected(bad, fn):
    """T=0 or sigma=0 would divide by zero in d1 and return nan."""
    with pytest.raises(ValueError):
        fn(**{**HULL, **bad})
