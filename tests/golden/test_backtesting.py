"""Validation tests for VaR backtesting: Kupiec, Christoffersen, Basel zones.

Defects these were written to catch:

* `kupiec_pof_test` returned a hard-coded p-value of 1.0 for zero exceptions.
  That is not a pass — observing no breaches over a long window is evidence the
  model is too conservative, and the test rejects it. At n=250, p=0.01 the true
  result is LR = 5.0252, p = 0.024982: a rejection at the 5% level that the old
  code reported as perfect.
* `basel_traffic_light` correctly defaults to n_obs = 250 because the Basel zone
  boundaries are calibrated for a 250-day window, but `evaluate` passed the full
  sample length (1000 in run_validation.py). The returned string was then not a
  Basel zone at all.

References:
* Basel Committee on Banking Supervision (1996), *Supervisory Framework for the
  Use of Backtesting*: at 250 observations and 99% VaR the zones are Green 0-4,
  Yellow 5-9, Red 10 or more. Reproduced exactly in BASEL_TABLE below.
* Kupiec (1995), *Techniques for Verifying the Accuracy of Risk Measurement
  Models*. LR values here are computed independently in `kupiec_lr_by_hand`.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from market_risk.backtesting import BASEL_WINDOW, RiskBacktester

# Basel Committee (1996) published zones: 250 observations, 99% VaR.
BASEL_TABLE = {
    **{x: "Green" for x in range(0, 5)},
    **{x: "Yellow" for x in range(5, 10)},
    **{x: "Red" for x in range(10, 16)},
}


def kupiec_lr_by_hand(exceptions, n_obs, p):
    """Kupiec LR written out directly from the definition, for cross-checking.

    LR = -2 ln[ L(p) / L(x/n) ] where L is the binomial likelihood. Uses
    math.log on explicit products rather than the implementation's structure.
    """
    import math

    x, n = exceptions, n_obs
    phat = x / n

    def loglik(rate):
        total = 0.0
        if n - x > 0:
            total += (n - x) * math.log(1 - rate)
        if x > 0:
            total += x * math.log(rate)
        return total

    return -2 * (loglik(p) - loglik(phat))


# --------------------------------------------------------------------------
# Basel traffic light
# --------------------------------------------------------------------------
@pytest.mark.parametrize("exceptions", sorted(BASEL_TABLE))
def test_basel_zones_match_published_table(exceptions):
    """The cumulative-probability rule must reproduce the 1996 published table."""
    bt = RiskBacktester(confidence_level=0.99)
    assert bt.basel_traffic_light(exceptions, BASEL_WINDOW) == BASEL_TABLE[exceptions]


def test_basel_zone_boundaries_are_where_the_table_says():
    bt = RiskBacktester(confidence_level=0.99)
    assert bt.basel_traffic_light(4, 250) == "Green"
    assert bt.basel_traffic_light(5, 250) == "Yellow"
    assert bt.basel_traffic_light(9, 250) == "Yellow"
    assert bt.basel_traffic_light(10, 250) == "Red"


def test_evaluate_scores_basel_on_a_250_day_window():
    """The regression test for the wrong-window defect.

    Build 1000 observations whose breaches sit entirely in the first 750, so the
    trailing 250-day window is clean. Scoring the full sample would count all of
    them; scoring the Basel window must count none.
    """
    n = 1000
    returns = pd.Series(np.full(n, -0.001))
    var = pd.Series(np.full(n, 0.02))
    # 12 breaches, all in the first 750 observations.
    breach_idx = list(range(0, 750, 62))[:12]
    returns.iloc[breach_idx] = -0.05

    bt = RiskBacktester(confidence_level=0.99)
    res = bt.evaluate(returns, var)

    assert res["Observations"] == n
    assert res["Exceptions"] == 12
    assert res["Basel_Window"] == 250
    assert res["Basel_Exceptions"] == 0
    assert res["Basel_Zone"] == "Green"


def test_evaluate_basel_window_reports_recent_deterioration():
    """Breaches concentrated in the recent window must show up as Red."""
    n = 1000
    returns = pd.Series(np.full(n, -0.001))
    var = pd.Series(np.full(n, 0.02))
    returns.iloc[-12:] = -0.05  # 12 breaches, all recent

    bt = RiskBacktester(confidence_level=0.99)
    res = bt.evaluate(returns, var)

    assert res["Basel_Window"] == 250
    assert res["Basel_Exceptions"] == 12
    assert res["Basel_Zone"] == "Red"


def test_full_sample_zone_differs_from_windowed_zone():
    """Demonstrates the two are genuinely different numbers, not a cosmetic change."""
    n = 1000
    returns = pd.Series(np.full(n, -0.001))
    var = pd.Series(np.full(n, 0.02))
    returns.iloc[list(range(0, 750, 62))[:12]] = -0.05

    bt = RiskBacktester(confidence_level=0.99)
    windowed = bt.evaluate(returns, var)
    whole = bt.evaluate(returns, var, basel_window=None)

    assert windowed["Basel_Window"] == 250
    assert whole["Basel_Window"] == 1000
    assert windowed["Basel_Exceptions"] != whole["Basel_Exceptions"]


def test_short_sample_reports_the_window_it_actually_used():
    n = 100
    returns = pd.Series(np.full(n, -0.001))
    var = pd.Series(np.full(n, 0.02))

    bt = RiskBacktester(confidence_level=0.99)
    res = bt.evaluate(returns, var)
    assert res["Basel_Window"] == n, "must not claim 250 when only 100 exist"


# --------------------------------------------------------------------------
# Kupiec POF
# --------------------------------------------------------------------------
def test_zero_exceptions_is_rejected_not_waved_through():
    """The regression test for the hard-coded 1.0.

    Zero breaches in 250 days against a 99% VaR means the model is materially
    too conservative. Published-style hand computation: LR = -2*n*ln(1-p) =
    -2*250*ln(0.99) = 5.0252, giving p = 0.024982.
    """
    bt = RiskBacktester(confidence_level=0.99)
    p_value = bt.kupiec_pof_test(0, 250)

    expected_lr = -2 * 250 * np.log(0.99)
    expected_p = 1 - stats.chi2.cdf(expected_lr, df=1)

    assert expected_lr == pytest.approx(5.0252, abs=1e-4)
    assert p_value == pytest.approx(expected_p, rel=1e-12)
    assert p_value == pytest.approx(0.024982, abs=1e-6)
    assert p_value < 0.05, "zero exceptions over 250 days must reject the model"


def test_zero_exceptions_rejects_harder_on_a_longer_window():
    bt = RiskBacktester(confidence_level=0.99)
    assert bt.kupiec_pof_test(0, 1000) < bt.kupiec_pof_test(0, 250)


def test_all_exceptions_endpoint_does_not_blow_up():
    """x == n makes log(1 - x/n) undefined; the 0*log(0) convention handles it."""
    bt = RiskBacktester(confidence_level=0.99)
    p_value = bt.kupiec_pof_test(250, 250)
    assert np.isfinite(p_value)
    assert p_value < 1e-6, "every day breaching must reject decisively"


@pytest.mark.parametrize("exceptions", [0, 1, 3, 5, 12, 25, 100, 249, 250])
def test_kupiec_matches_independent_hand_computation(exceptions):
    bt = RiskBacktester(confidence_level=0.99)
    expected_lr = kupiec_lr_by_hand(exceptions, 250, 0.01)
    expected_p = 1 - stats.chi2.cdf(max(expected_lr, 0.0), df=1)
    assert bt.kupiec_pof_test(exceptions, 250) == pytest.approx(expected_p, rel=1e-10)


def test_kupiec_is_least_significant_at_the_nominal_rate():
    """The p-value peaks when the observed rate equals the model's rate."""
    bt = RiskBacktester(confidence_level=0.99)
    at_nominal = bt.kupiec_pof_test(10, 1000)  # exactly 1%
    assert at_nominal == pytest.approx(1.0, abs=1e-9)
    for other in (0, 3, 5, 20, 40):
        assert bt.kupiec_pof_test(other, 1000) < at_nominal


@pytest.mark.parametrize("bad", [(-1, 250), (251, 250), (0, 0), (0, -5)])
def test_kupiec_rejects_impossible_counts(bad):
    bt = RiskBacktester(confidence_level=0.99)
    with pytest.raises(ValueError):
        bt.kupiec_pof_test(*bad)


# --------------------------------------------------------------------------
# Christoffersen independence
# --------------------------------------------------------------------------
def test_clustered_breaches_are_flagged_as_dependent():
    """All breaches consecutive — maximal clustering."""
    hits = np.zeros(250, dtype=int)
    hits[100:112] = 1
    bt = RiskBacktester(confidence_level=0.99)
    assert bt.christoffersen_independence_test(hits) < 0.01


def test_evenly_spaced_breaches_are_not_flagged():
    hits = np.zeros(250, dtype=int)
    hits[::25] = 1
    bt = RiskBacktester(confidence_level=0.99)
    assert bt.christoffersen_independence_test(hits) > 0.10


def test_christoffersen_degenerate_inputs_return_one():
    bt = RiskBacktester(confidence_level=0.99)
    assert bt.christoffersen_independence_test(np.zeros(250, dtype=int)) == 1.0
    assert bt.christoffersen_independence_test(np.array([1])) == 1.0
    assert bt.christoffersen_independence_test(np.array([], dtype=int)) == 1.0


def test_christoffersen_matches_hand_computed_transition_counts():
    """Small explicit sequence with countable transitions.

    hits = [0,1,1,0,0,1,0,0,0,0] gives, over the 9 transitions:
      n00 = 4, n01 = 2, n10 = 2, n11 = 1
    """
    hits = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    pi0, pi1, pi = 2 / 6, 1 / 3, 3 / 9

    log_l_null = (4 + 2) * np.log(1 - pi) + (2 + 1) * np.log(pi)
    log_l_alt = (
        4 * np.log(1 - pi0) + 2 * np.log(pi0) + 2 * np.log(1 - pi1) + 1 * np.log(pi1)
    )
    expected = 1 - stats.chi2.cdf(max(-2 * (log_l_null - log_l_alt), 0.0), df=1)

    bt = RiskBacktester(confidence_level=0.99)
    assert bt.christoffersen_independence_test(hits) == pytest.approx(
        expected, rel=1e-10
    )


# --------------------------------------------------------------------------
# evaluate() plumbing
# --------------------------------------------------------------------------
def test_evaluate_rejects_non_overlapping_series():
    bt = RiskBacktester(confidence_level=0.99)
    returns = pd.Series([0.01, 0.02], index=[0, 1])
    var = pd.Series([0.03, 0.03], index=[10, 11])
    with pytest.raises(ValueError, match="No overlapping observations"):
        bt.evaluate(returns, var)


def test_evaluate_counts_breaches_at_the_right_sign():
    """A breach is return < -VaR, with VaR a positive loss."""
    returns = pd.Series([-0.03, -0.01, 0.05, -0.021])
    var = pd.Series([0.02, 0.02, 0.02, 0.02])
    bt = RiskBacktester(confidence_level=0.99)
    res = bt.evaluate(returns, var, basel_window=None)
    assert res["Exceptions"] == 2  # -0.03 and -0.021
    assert res["Observations"] == 4


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_rejects_invalid_confidence_level(bad_alpha):
    with pytest.raises(ValueError, match="confidence_level"):
        RiskBacktester(confidence_level=bad_alpha)
