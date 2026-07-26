import numpy as np
from typing import Dict, Optional

# np.trapz was renamed to np.trapezoid in NumPy 2.0. Alias so the package works
# on either major version until the dependency floor is pinned.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class CounterpartyRiskEngine:
    def __init__(self, time_grid: np.ndarray):
        """
        time_grid: array of forward time points (in years)
        """
        time_grid = np.asarray(time_grid, dtype=float)
        if time_grid.ndim != 1 or time_grid.size == 0:
            raise ValueError("time_grid must be a non-empty 1-D array.")
        if time_grid.size > 1 and np.any(np.diff(time_grid) <= 0):
            raise ValueError("time_grid must be strictly increasing.")
        if np.any(time_grid < 0):
            raise ValueError("time_grid must be non-negative.")

        self.time_grid = time_grid
        self.num_points = len(time_grid)
        self.portfolio_paths = None # shape: (num_paths, num_points)

    def set_portfolio_paths(self, paths: np.ndarray) -> None:
        """
        paths: array of simulated portfolio values of shape (num_paths, num_points)
        """
        paths = np.asarray(paths, dtype=float)
        if paths.ndim != 2:
            raise ValueError(
                f"paths must be 2-D (num_paths, num_points); got shape {paths.shape}."
            )
        if paths.shape[1] != self.num_points:
            raise ValueError("Paths must match the time grid length.")
        self.portfolio_paths = paths

    def _require_paths(self) -> np.ndarray:
        if self.portfolio_paths is None:
            raise ValueError("Portfolio paths not set.")
        return self.portfolio_paths

    def calculate_exposure_profiles(self, quantile: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calculate EE, EPE, and PFE.

        EPE is the time-weighted average of the expected exposure over the grid,

            EPE = (1/(T - t_0)) * INT_{t_0}^{T} EE(t) dt

        evaluated with the trapezoidal rule. The previous implementation weighted
        each point by ``np.diff(time_grid, prepend=time_grid[0])``, which sets the
        first weight to zero and so dropped the leading point from the average
        entirely, then divided by ``time_grid[-1]`` rather than the actual span
        of the grid — inconsistent whenever the grid did not start at zero.
        """
        exposure_paths = np.maximum(self._require_paths(), 0)

        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must lie in (0, 1).")

        ee = np.mean(exposure_paths, axis=0)

        span = float(self.time_grid[-1] - self.time_grid[0])
        if self.num_points > 1 and span > 0:
            epe = float(_trapezoid(ee, self.time_grid) / span)
        else:
            epe = float(ee[0])

        pfe = np.quantile(exposure_paths, quantile, axis=0)

        return {
            'EE': ee,
            'EPE': np.array([epe]),
            'PFE': pfe
        }

    def calculate_cva(
        self,
        recovery_rate: float,
        pd_curve: np.ndarray,
        discount_factors: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate Credit Value Adjustment.

        pd_curve: array of cumulative default probabilities corresponding to
            time_grid.
        discount_factors: risk-free discount factors at each grid point. If
            omitted, all are treated as 1.0 — i.e. the CVA is undiscounted, which
            overstates it, materially so on a long netting set. Passing a curve
            is strongly preferred; the default exists only for backward
            compatibility.

            CVA = (1 - R) * SUM_i DF(t_i) * EE(t_i) * dPD(t_i)

        The marginal default probability in the first interval is taken as
        ``pd_curve[0] - 0``, i.e. default between time zero and the first grid
        point. The previous implementation used
        ``np.diff(pd_curve, prepend=pd_curve[0])``, which forced that increment
        to zero and silently discarded the first interval's contribution.
        """
        ee = self.calculate_exposure_profiles()['EE']

        if not 0.0 <= recovery_rate <= 1.0:
            raise ValueError("recovery_rate must lie in [0, 1].")

        pd_curve = np.asarray(pd_curve, dtype=float)
        if pd_curve.shape != (self.num_points,):
            raise ValueError(
                f"pd_curve must have one entry per grid point "
                f"({self.num_points}); got shape {pd_curve.shape}."
            )
        if np.any(pd_curve < 0) or np.any(pd_curve > 1):
            raise ValueError("pd_curve holds cumulative probabilities; values must lie in [0, 1].")
        if np.any(np.diff(pd_curve) < 0):
            raise ValueError("pd_curve is cumulative and must be non-decreasing.")

        if discount_factors is None:
            df = np.ones(self.num_points, dtype=float)
        else:
            df = np.asarray(discount_factors, dtype=float)
            if df.shape != (self.num_points,):
                raise ValueError(
                    f"discount_factors must have one entry per grid point "
                    f"({self.num_points}); got shape {df.shape}."
                )
            if np.any(df <= 0):
                raise ValueError("discount_factors must be positive.")

        # Marginal PD: probability of default between t_{i-1} and t_i, with the
        # first interval running from time zero.
        marginal_pd = np.diff(pd_curve, prepend=0.0)

        cva = (1 - recovery_rate) * np.sum(df * ee * marginal_pd)

        return float(cva)

    def calculate_cva_wwr(
        self,
        recovery_rate: float,
        pd_curve: np.ndarray,
        alpha_wwr: float = 1.1,
        discount_factors: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate CVA with Wrong-Way Risk (WWR) multiplier.
        alpha_wwr: Basel multiplier (usually 1.0 to 1.4) to account for correlation between EE and PD.

        This is a flat scaling of the base CVA, not a modelled dependence between
        exposure and default probability. It cannot capture *when* the
        correlation bites, only that it does on average.
        """
        if alpha_wwr <= 0:
            raise ValueError("alpha_wwr must be positive.")
        base_cva = self.calculate_cva(recovery_rate, pd_curve, discount_factors)
        return base_cva * alpha_wwr


class RatingMigrationEngine:
    def __init__(self, transition_matrix: np.ndarray):
        """
        transition_matrix: (N, N) matrix where cell (i, j) is prob of moving from rating i to j.
        """
        transition_matrix = np.asarray(transition_matrix, dtype=float)
        if transition_matrix.ndim != 2 or (
            transition_matrix.shape[0] != transition_matrix.shape[1]
        ):
            raise ValueError("transition_matrix must be square.")
        if np.any(transition_matrix < 0):
            raise ValueError("transition_matrix must be non-negative.")
        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Rows of transition matrix must sum to 1.0")

        self.tm = transition_matrix

    def simulate_migration(
        self,
        initial_rating_idx: int,
        horizons: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate a rating path over H discrete time steps.

        `seed` draws from an explicit Generator so a simulated path can be
        reproduced; without it the global NumPy RNG state is used and the path
        cannot be regenerated.
        """
        n_states = self.tm.shape[0]
        if not 0 <= initial_rating_idx < n_states:
            raise ValueError(
                f"initial_rating_idx must lie in [0, {n_states - 1}]."
            )
        if horizons < 0:
            raise ValueError("horizons must be non-negative.")

        rng = np.random.default_rng(seed)
        current_rating = initial_rating_idx
        path = [current_rating]
        for _ in range(horizons):
            probs = self.tm[current_rating]
            current_rating = int(rng.choice(n_states, p=probs))
            path.append(current_rating)
        return np.array(path)
