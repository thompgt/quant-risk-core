"""Validation tests for counterparty exposure, CVA, netting and collateral.

Defects these were written to catch:

* **CVA was undiscounted.** `calculate_cva` summed EE * dPD with no discount
  factors, overstating CVA — materially so on a long netting set.
* **EPE dropped its first grid point.** The time weights came from
  ``np.diff(time_grid, prepend=time_grid[0])``, whose first element is
  identically zero, and the result was divided by ``time_grid[-1]`` rather than
  the grid's actual span.
* **The first marginal PD was forced to zero.** ``np.diff(pd_curve,
  prepend=pd_curve[0])`` discarded default probability in the first interval.
* **`threshold`, `mta` and `mpor_days` were dead parameters.** They were stored
  and never used, so a CollateralManager configured with a 10-day MPOR, a
  threshold and an MTA behaved exactly like one with none of them.
  `scripts/generate_readme_figures.py` had to reimplement the lag and threshold
  inline to get the documented behaviour, which is the clearest evidence the
  logic belonged in the class.

References: Gregory, *Counterparty Credit Risk and Credit Value Adjustment*,
for the EPE definition as the time average of EE and the standard discrete CVA
sum. Numeric expectations below are computed by hand in the test bodies.
"""
import numpy as np
import pytest

from quant_risk_core.credit_risk.counterparty import CounterpartyRiskEngine, RatingMigrationEngine
from quant_risk_core.credit_risk.mitigation import CollateralManager, NettingEngine


# --------------------------------------------------------------------------
# EPE time weighting
# --------------------------------------------------------------------------
def test_epe_is_the_trapezoidal_time_average():
    """EPE must equal the trapezoidal average, including the first point."""
    grid = np.array([1.0, 2.0, 3.0])
    eng = CounterpartyRiskEngine(grid)
    # One path, so EE equals the path itself.
    eng.set_portfolio_paths(np.array([[10.0, 20.0, 30.0]]))

    prof = eng.calculate_exposure_profiles()
    np.testing.assert_allclose(prof["EE"], [10.0, 20.0, 30.0])

    # Trapezoid over [1,3]: (10+20)/2*1 + (20+30)/2*1 = 15 + 25 = 40; span 2 -> 20.
    assert prof["EPE"][0] == pytest.approx(20.0)


def test_epe_does_not_drop_the_first_grid_point():
    """The regression test for the zero first weight.

    A profile that is large at the first point and zero afterwards must produce a
    positive EPE. The old weighting gave the first point zero weight, so this
    returned 0.
    """
    grid = np.array([1.0, 2.0, 3.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[90.0, 0.0, 0.0]]))

    epe = eng.calculate_exposure_profiles()["EPE"][0]

    old_weights = np.diff(grid, prepend=grid[0])
    old_epe = np.sum(np.array([90.0, 0.0, 0.0]) * old_weights) / grid[-1]
    assert old_epe == 0.0, "confirms the old weighting ignored the first point"
    assert epe > 0.0


def test_epe_uses_the_grid_span_not_the_final_time():
    """A grid that does not start at zero must be averaged over its own span."""
    grid = np.array([10.0, 11.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[100.0, 100.0]]))

    # Constant exposure of 100 must average to exactly 100 regardless of where
    # the grid sits on the timeline.
    assert eng.calculate_exposure_profiles()["EPE"][0] == pytest.approx(100.0)


def test_epe_of_constant_exposure_equals_that_constant():
    grid = np.linspace(0.0, 5.0, 21)
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.full((3, 21), 42.0))
    assert eng.calculate_exposure_profiles()["EPE"][0] == pytest.approx(42.0)


def test_single_point_grid_epe_is_the_point_itself():
    eng = CounterpartyRiskEngine(np.array([1.0]))
    eng.set_portfolio_paths(np.array([[7.0], [3.0]]))
    prof = eng.calculate_exposure_profiles()
    assert prof["EPE"][0] == pytest.approx(5.0)


def test_ee_floors_negative_mtm_at_zero():
    grid = np.array([0.5, 1.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[-10.0, 10.0], [10.0, -20.0]]))
    np.testing.assert_allclose(eng.calculate_exposure_profiles()["EE"], [5.0, 5.0])


def test_pfe_is_the_requested_quantile_of_exposure():
    grid = np.array([1.0, 2.0])
    eng = CounterpartyRiskEngine(grid)
    paths = np.array([[0.0, 1.0], [10.0, 2.0], [20.0, 3.0], [30.0, 4.0]])
    eng.set_portfolio_paths(paths)
    prof = eng.calculate_exposure_profiles(quantile=0.5)
    np.testing.assert_allclose(prof["PFE"], np.quantile(paths, 0.5, axis=0))


# --------------------------------------------------------------------------
# CVA
# --------------------------------------------------------------------------
def test_cva_matches_hand_computed_discounted_sum():
    grid = np.array([1.0, 2.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[100.0, 200.0]]))

    pd_curve = np.array([0.01, 0.03])
    df = np.array([0.95, 0.90])
    recovery = 0.4

    # marginal PD = [0.01 - 0, 0.03 - 0.01] = [0.01, 0.02]
    # CVA = 0.6 * (0.95*100*0.01 + 0.90*200*0.02) = 0.6 * (0.95 + 3.60) = 2.73
    expected = 0.6 * (0.95 * 100 * 0.01 + 0.90 * 200 * 0.02)
    assert expected == pytest.approx(2.73)
    assert eng.calculate_cva(recovery, pd_curve, df) == pytest.approx(expected)


def test_discounting_reduces_cva():
    """The regression test for the missing discount factors."""
    grid = np.array([1.0, 5.0, 10.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[100.0, 100.0, 100.0]]))
    pd_curve = np.array([0.01, 0.05, 0.10])

    df = np.exp(-0.05 * grid)
    undiscounted = eng.calculate_cva(0.4, pd_curve)
    discounted = eng.calculate_cva(0.4, pd_curve, df)

    assert discounted < undiscounted
    # On a 10y grid at 5% the overstatement is not a rounding difference.
    assert discounted < 0.85 * undiscounted


def test_unit_discount_factors_reproduce_the_undiscounted_result():
    grid = np.array([1.0, 2.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[100.0, 200.0]]))
    pd_curve = np.array([0.01, 0.03])

    assert eng.calculate_cva(0.4, pd_curve, np.ones(2)) == pytest.approx(
        eng.calculate_cva(0.4, pd_curve)
    )


def test_first_interval_default_probability_is_counted():
    """The regression test for the zeroed first marginal PD.

    All default probability sits in the first interval, so a CVA that ignores it
    is exactly zero.
    """
    grid = np.array([1.0, 2.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[100.0, 100.0]]))
    pd_curve = np.array([0.05, 0.05])  # flat after the first interval

    cva = eng.calculate_cva(0.4, pd_curve)

    old_marginal = np.diff(pd_curve, prepend=pd_curve[0])
    assert old_marginal.sum() == 0.0, "confirms the old code saw no default risk"
    assert cva == pytest.approx(0.6 * 100 * 0.05)


def test_cva_scales_linearly_with_loss_given_default():
    grid = np.array([1.0, 2.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[100.0, 100.0]]))
    pd_curve = np.array([0.01, 0.02])

    assert eng.calculate_cva(0.0, pd_curve) == pytest.approx(
        2 * eng.calculate_cva(0.5, pd_curve)
    )
    assert eng.calculate_cva(1.0, pd_curve) == 0.0


def test_wwr_multiplier_scales_the_discounted_base():
    grid = np.array([1.0, 2.0])
    eng = CounterpartyRiskEngine(grid)
    eng.set_portfolio_paths(np.array([[100.0, 200.0]]))
    pd_curve = np.array([0.01, 0.03])
    df = np.array([0.95, 0.90])

    base = eng.calculate_cva(0.4, pd_curve, df)
    assert eng.calculate_cva_wwr(0.4, pd_curve, 1.4, df) == pytest.approx(base * 1.4)


@pytest.mark.parametrize(
    "pd_curve,match",
    [
        (np.array([0.05, 0.02]), "non-decreasing"),
        (np.array([-0.1, 0.2]), r"\[0, 1\]"),
        (np.array([0.1, 1.5]), r"\[0, 1\]"),
        (np.array([0.1]), "one entry per grid point"),
    ],
)
def test_cva_rejects_invalid_pd_curves(pd_curve, match):
    eng = CounterpartyRiskEngine(np.array([1.0, 2.0]))
    eng.set_portfolio_paths(np.array([[100.0, 100.0]]))
    with pytest.raises(ValueError, match=match):
        eng.calculate_cva(0.4, pd_curve)


def test_cva_rejects_invalid_discount_and_recovery():
    eng = CounterpartyRiskEngine(np.array([1.0, 2.0]))
    eng.set_portfolio_paths(np.array([[100.0, 100.0]]))
    pd_curve = np.array([0.01, 0.02])

    with pytest.raises(ValueError, match="recovery_rate"):
        eng.calculate_cva(1.5, pd_curve)
    with pytest.raises(ValueError, match="discount_factors must be positive"):
        eng.calculate_cva(0.4, pd_curve, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="one entry per grid point"):
        eng.calculate_cva(0.4, pd_curve, np.ones(3))


def test_engine_rejects_bad_grids_and_paths():
    with pytest.raises(ValueError, match="strictly increasing"):
        CounterpartyRiskEngine(np.array([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="non-empty"):
        CounterpartyRiskEngine(np.array([]))

    eng = CounterpartyRiskEngine(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Portfolio paths not set"):
        eng.calculate_exposure_profiles()
    with pytest.raises(ValueError, match="must be 2-D"):
        eng.set_portfolio_paths(np.array([1.0, 2.0]))


# --------------------------------------------------------------------------
# Netting
# --------------------------------------------------------------------------
def test_netted_exposure_never_exceeds_gross():
    rng = np.random.default_rng(11)
    contracts = rng.normal(0, 10, size=(4, 200, 12))
    netted = NettingEngine(True).aggregate_mtm(contracts)
    gross = NettingEngine(False).aggregate_mtm(contracts)
    assert np.all(netted <= gross + 1e-12)


def test_netting_rejects_wrong_dimensionality():
    with pytest.raises(ValueError, match="num_contracts"):
        NettingEngine().aggregate_mtm(np.array([[1.0, 2.0]]))


# --------------------------------------------------------------------------
# Collateral: threshold, MTA, MPOR
# --------------------------------------------------------------------------
def test_threshold_leaves_exposure_below_it_unsecured():
    cm = CollateralManager(threshold=25.0, mpor_days=0, mta=0.0)
    exposure = np.array([10.0, 25.0, 40.0])
    np.testing.assert_allclose(cm.required_margin(exposure), [0.0, 0.0, 15.0])


def test_mpor_lag_delays_collateral_by_the_margin_period():
    """The regression test for the dead mpor_days parameter."""
    grid = np.arange(10) / 252.0  # daily grid in years
    cm = CollateralManager(threshold=0.0, mpor_days=3, mta=0.0)
    assert cm.mpor_steps(grid) == 3

    exposure = np.array([100.0] * 10)
    held = cm.compute_variation_margin(exposure, grid)

    # First three points carry no collateral; the rest are fully margined.
    np.testing.assert_allclose(held[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(held[3:], [100.0] * 7)


def test_mpor_makes_collateralised_exposure_strictly_worse():
    """With a lag, collateral cannot fully extinguish a rising exposure."""
    grid = np.arange(10) / 252.0
    exposure = np.linspace(10.0, 100.0, 10)

    lagged = CollateralManager(threshold=0.0, mpor_days=3, mta=0.0)
    instant = CollateralManager(threshold=0.0, mpor_days=0, mta=0.0)

    with_lag = lagged.apply_collateral(exposure, time_grid=grid)
    without_lag = instant.apply_collateral(exposure, time_grid=grid)

    assert np.all(without_lag == pytest.approx(0.0))
    assert with_lag.sum() > 0.0, "an MPOR lag must leave residual exposure"


def test_mta_makes_the_collateral_balance_sticky():
    """Balance only moves when the required amount shifts by at least the MTA."""
    grid = np.arange(4) / 252.0
    cm = CollateralManager(threshold=0.0, mpor_days=0, mta=10.0)

    exposure = np.array([100.0, 105.0, 130.0, 132.0])
    held = cm.compute_variation_margin(exposure, grid)

    # 100 -> called. +5 < MTA so held stays. 130 is +30 >= MTA so it moves.
    # 132 is +2 < MTA so it stays at 130.
    np.testing.assert_allclose(held, [100.0, 100.0, 130.0, 130.0])


def test_zero_mta_tracks_exposure_exactly():
    grid = np.arange(4) / 252.0
    cm = CollateralManager(threshold=0.0, mpor_days=0, mta=0.0)
    exposure = np.array([100.0, 105.0, 130.0, 132.0])
    np.testing.assert_allclose(cm.compute_variation_margin(exposure, grid), exposure)


def test_collateral_reduces_exposure_and_never_goes_negative():
    rng = np.random.default_rng(5)
    grid = np.arange(20) / 252.0
    exposure = np.abs(rng.normal(50, 20, size=(50, 20)))

    cm = CollateralManager(threshold=25.0, mpor_days=10, mta=5.0)
    adjusted = cm.apply_collateral(exposure, time_grid=grid)

    assert np.all(adjusted >= 0.0)
    assert np.all(adjusted <= exposure + 1e-12)


def test_collateral_works_on_2d_path_arrays():
    grid = np.arange(6) / 252.0
    cm = CollateralManager(threshold=0.0, mpor_days=2, mta=0.0)
    exposure = np.tile(np.array([100.0] * 6), (4, 1))
    held = cm.compute_variation_margin(exposure, grid)
    assert held.shape == exposure.shape
    np.testing.assert_allclose(held[:, :2], 0.0)
    np.testing.assert_allclose(held[:, 2:], 100.0)


def test_missing_time_grid_warns_that_mpor_is_skipped():
    cm = CollateralManager(threshold=0.0, mpor_days=10, mta=0.0)
    with pytest.warns(RuntimeWarning, match="margin period of risk"):
        cm.compute_variation_margin(np.array([100.0, 100.0]))


def test_non_uniform_grid_warns():
    """MPOR chosen so it is representable on the first step, isolating the
    non-uniformity warning from the sub-grid one."""
    cm = CollateralManager(threshold=0.0, mpor_days=30, mta=0.0)
    grid = np.array([0.0, 0.1, 0.5, 2.0])
    with pytest.warns(RuntimeWarning, match="not uniformly spaced"):
        assert cm.mpor_steps(grid) == 1


def test_mpor_shorter_than_the_grid_step_warns():
    """A sub-grid MPOR silently becomes no lag at all; that must be loud.

    The figures script previously papered over this with max(1, ...), which
    promoted an unrepresentable 10-day MPOR to a full grid step of ~31 business
    days without saying so.
    """
    cm = CollateralManager(threshold=0.0, mpor_days=10, mta=0.0)
    uniform_coarse = np.linspace(0.0, 5.0, 41)  # 0.125y steps
    with pytest.warns(RuntimeWarning, match="rounds to zero steps"):
        assert cm.mpor_steps(uniform_coarse) == 0


def test_explicit_collateral_shape_is_validated():
    cm = CollateralManager()
    with pytest.raises(ValueError, match="does not"):
        cm.apply_collateral(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        cm.apply_collateral(np.array([1.0, 2.0]), np.array([-1.0, 0.0]))


@pytest.mark.parametrize(
    "kwargs", [dict(threshold=-1), dict(mta=-1), dict(mpor_days=-1)]
)
def test_collateral_manager_rejects_negative_parameters(kwargs):
    with pytest.raises(ValueError):
        CollateralManager(**kwargs)


def test_collateral_lowers_cva():
    """End-to-end: mitigation must reduce the charge."""
    grid = np.linspace(1 / 252, 1.0, 60)
    rng = np.random.default_rng(3)
    exposure = np.abs(rng.normal(100, 30, size=(200, 60)))

    cm = CollateralManager(threshold=25.0, mpor_days=10, mta=5.0)
    collateralised = cm.apply_collateral(exposure, time_grid=grid)

    pd_curve = 1 - np.exp(-0.02 * grid)
    df = np.exp(-0.03 * grid)

    uncollat = CounterpartyRiskEngine(grid)
    uncollat.set_portfolio_paths(exposure)
    collat = CounterpartyRiskEngine(grid)
    collat.set_portfolio_paths(collateralised)

    assert collat.calculate_cva(0.4, pd_curve, df) < uncollat.calculate_cva(
        0.4, pd_curve, df
    )


# --------------------------------------------------------------------------
# Rating migration
# --------------------------------------------------------------------------
def test_migration_is_reproducible_from_a_seed():
    tm = np.array([[0.9, 0.1], [0.2, 0.8]])
    eng = RatingMigrationEngine(tm)
    a = eng.simulate_migration(0, 50, seed=7)
    b = eng.simulate_migration(0, 50, seed=7)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, eng.simulate_migration(0, 50, seed=8))


def test_absorbing_default_state_is_never_left():
    # State 2 is absorbing (default).
    tm = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.0, 0.0, 1.0]])
    eng = RatingMigrationEngine(tm)
    path = eng.simulate_migration(0, 200, seed=1)
    if 2 in path:
        first = int(np.argmax(path == 2))
        assert np.all(path[first:] == 2)


def test_migration_validates_matrix_and_arguments():
    with pytest.raises(ValueError, match="sum to 1.0"):
        RatingMigrationEngine(np.array([[0.5, 0.4], [0.2, 0.8]]))
    with pytest.raises(ValueError, match="square"):
        RatingMigrationEngine(np.array([[0.5, 0.5, 0.0]]))

    eng = RatingMigrationEngine(np.array([[0.9, 0.1], [0.2, 0.8]]))
    with pytest.raises(ValueError, match="initial_rating_idx"):
        eng.simulate_migration(5, 10)
