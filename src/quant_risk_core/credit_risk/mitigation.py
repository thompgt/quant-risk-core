import warnings

import numpy as np
from typing import List, Optional, Tuple

# Business days per year, used to convert a margin period of risk quoted in
# business days into the year fractions the exposure time grid is measured in.
BUSINESS_DAYS_PER_YEAR = 252


class NettingEngine:
    def __init__(self, netting_enabled: bool = True):
        self.netting_enabled = netting_enabled

    def aggregate_mtm(self, contract_values: np.ndarray) -> np.ndarray:
        """
        Aggregate mark-to-market values across contracts.
        contract_values: shape (num_contracts, num_paths, num_points)
        Returns: exposure shape (num_paths, num_points)

        With netting the set is collapsed before flooring at zero, so in-the-money
        and out-of-the-money trades offset. Without it each trade is floored
        first, which is why gross exposure always dominates netted exposure.
        """
        contract_values = np.asarray(contract_values, dtype=float)
        if contract_values.ndim != 3:
            raise ValueError(
                "contract_values must have shape (num_contracts, num_paths, "
                f"num_points); got {contract_values.shape}."
            )

        if self.netting_enabled:
            # sum V_i, then max(., 0)
            net_value = np.sum(contract_values, axis=0)
            exposure = np.maximum(net_value, 0)
        else:
            # sum max(V_i, 0)
            exposures = np.maximum(contract_values, 0)
            exposure = np.sum(exposures, axis=0)

        return exposure


class CollateralManager:
    def __init__(
        self,
        threshold: float = 0.0,
        mpor_days: int = 10,
        mta: float = 0.0,
        business_days_per_year: int = BUSINESS_DAYS_PER_YEAR,
    ):
        """
        threshold: unsecured exposure the counterparty may carry before any
            collateral is called.
        mpor_days: Margin Period of Risk, in business days. The delay between
            observing an exposure and actually holding collateral against it.
        mta: Minimum Transfer Amount. Margin is only moved when the required
            amount differs from the amount held by at least this much.
        business_days_per_year: conversion used to express `mpor_days` in the
            year fractions of the exposure time grid.

        All three parameters were previously stored and never used —
        `apply_collateral` simply subtracted whatever collateral the caller
        passed in, so a CollateralManager configured with a threshold, an MTA and
        a 10-day MPOR behaved identically to one with none of them. Callers that
        wanted the documented behaviour had to reimplement it themselves.
        """
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")
        if mta < 0:
            raise ValueError("mta must be non-negative.")
        if mpor_days < 0:
            raise ValueError("mpor_days must be non-negative.")
        if business_days_per_year <= 0:
            raise ValueError("business_days_per_year must be positive.")

        self.threshold = threshold
        self.mpor_days = mpor_days
        self.mta = mta
        self.business_days_per_year = business_days_per_year

    @property
    def mpor_years(self) -> float:
        """Margin period of risk expressed as a year fraction."""
        return self.mpor_days / self.business_days_per_year

    def mpor_steps(self, time_grid: np.ndarray) -> int:
        """
        Number of grid steps the margin period of risk spans.

        Assumes a uniformly spaced grid and warns otherwise, since a single
        integer lag cannot represent the MPOR on an irregular grid.
        """
        time_grid = np.asarray(time_grid, dtype=float)
        if time_grid.size < 2:
            return 0

        steps = np.diff(time_grid)
        dt = float(steps[0])
        if not np.allclose(steps, dt, rtol=1e-6):
            warnings.warn(
                "time_grid is not uniformly spaced; the margin period of risk "
                "is applied as a constant number of steps derived from the "
                "first interval, which understates or overstates the lag "
                "elsewhere on the grid.",
                RuntimeWarning,
                stacklevel=2,
            )
        if dt <= 0:
            raise ValueError("time_grid must be strictly increasing.")

        steps_exact = self.mpor_years / dt
        lag = int(round(steps_exact))

        if self.mpor_days > 0 and lag == 0:
            warnings.warn(
                f"A margin period of risk of {self.mpor_days} business days "
                f"({self.mpor_years:.4f}y) is shorter than half the grid step "
                f"({dt:.4f}y), so it rounds to zero steps and no lag is applied. "
                f"Collateralised exposure will be understated. Use a grid step of "
                f"at most {self.mpor_years:.4f}y to represent this MPOR.",
                RuntimeWarning,
                stacklevel=2,
            )

        return lag

    def required_margin(self, exposure: np.ndarray) -> np.ndarray:
        """Collateral called for at each point, before MTA and MPOR: max(E - threshold, 0)."""
        exposure = np.asarray(exposure, dtype=float)
        return np.maximum(exposure - self.threshold, 0.0)

    def compute_variation_margin(
        self, exposure: np.ndarray, time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Variation margin actually held at each time point.

        Applies the three contractual features in the order they bite:

        1. **Threshold** — margin is only called on exposure above it.
        2. **MTA** — the balance is left alone unless the required amount moves
           away from the amount held by at least the minimum transfer amount, so
           the collateral balance is sticky rather than tracking exposure exactly.
        3. **MPOR** — the balance is lagged, because collateral held today
           reflects an exposure observed one margin period ago. The first
           `mpor_steps` points therefore carry no collateral at all.

        `exposure` may be 1-D (a single profile) or 2-D (num_paths, num_points);
        the time axis is always the last one. If `time_grid` is omitted the MPOR
        lag is skipped and a warning is issued, since the lag cannot be expressed
        in steps without knowing the spacing.
        """
        exposure = np.asarray(exposure, dtype=float)
        if exposure.ndim == 0:
            raise ValueError("exposure must be at least 1-D over time.")

        required = self.required_margin(exposure)

        # MTA: sticky balance, only re-margined on a large enough move.
        held = np.empty_like(required)
        current = np.zeros(required.shape[:-1], dtype=float)
        for t in range(required.shape[-1]):
            target = required[..., t]
            if self.mta > 0:
                move = np.abs(target - current) >= self.mta
                current = np.where(move, target, current)
            else:
                current = target
            held[..., t] = current

        # MPOR: collateral is stale by the margin period of risk.
        if time_grid is None:
            if self.mpor_days > 0:
                warnings.warn(
                    "time_grid not supplied, so the margin period of risk cannot "
                    "be converted to grid steps and no lag is applied. "
                    "Collateralised exposure will be understated.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            lag = self.mpor_steps(time_grid)
            if lag > 0:
                n = held.shape[-1]
                lagged = np.zeros_like(held)
                if lag < n:
                    lagged[..., lag:] = held[..., : n - lag]
                held = lagged

        return held

    def apply_collateral(
        self,
        exposure: np.ndarray,
        collateral_held: Optional[np.ndarray] = None,
        time_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Adjust exposure by Variation Margin collateral held.

        If `collateral_held` is None it is derived from `exposure` via
        `compute_variation_margin`, applying the configured threshold, MTA and
        MPOR. Pass `collateral_held` explicitly to override with an externally
        modelled balance.
        """
        exposure = np.asarray(exposure, dtype=float)

        if collateral_held is None:
            collateral_held = self.compute_variation_margin(exposure, time_grid)
        else:
            collateral_held = np.asarray(collateral_held, dtype=float)
            if collateral_held.shape != exposure.shape:
                raise ValueError(
                    f"collateral_held shape {collateral_held.shape} does not "
                    f"match exposure shape {exposure.shape}."
                )
            if np.any(collateral_held < 0):
                raise ValueError("collateral_held must be non-negative.")

        # Uncollateralized exposure is max(exposure - collateral, 0)
        return np.maximum(exposure - collateral_held, 0)
