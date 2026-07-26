import warnings

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from typing import Tuple, Dict

# |xi| below this is treated as the xi -> 0 (exponential) limiting case, where
# the (beta/xi) * (y^-xi - 1) form is numerically unstable.
XI_ZERO_TOL = 1e-6


class EVTEngine:
    def __init__(self, threshold_quantile: float = 0.95):
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must lie in (0, 1).")
        self.threshold_quantile = threshold_quantile
        self.xi = 0.0 # shape
        self.beta = 0.0 # scale
        self.u = 0.0 # threshold
        self.n_total = 0
        self.n_excess = 0
        self._fitted = False

    def fit(self, returns: pd.Series) -> None:
        """
        Fit EVT POT using Generalized Pareto Distribution (GPD).
        We model losses, so we take negative returns.
        """
        losses = -returns.dropna()
        self.n_total = len(losses)
        if self.n_total == 0:
            raise ValueError("No observations to fit.")

        self.u = losses.quantile(self.threshold_quantile)

        excesses = losses[losses > self.u] - self.u
        self.n_excess = len(excesses)

        if self.n_excess == 0:
            raise ValueError("No excesses found above threshold.")

        c, loc, scale = genpareto.fit(excesses, floc=0)
        self.xi = c
        self.beta = scale
        self._fitted = True

        if self.beta <= 0:
            raise ValueError(f"Fitted GPD scale must be positive, got {self.beta}.")
        if self.xi >= 1.0:
            warnings.warn(
                f"Fitted GPD shape xi={self.xi:.4f} >= 1: the tail has infinite "
                "mean, so Expected Shortfall is not finite. VaR remains defined.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def exceedance_rate(self) -> float:
        """Empirical P(loss > threshold), the `pu` of the POT formulae."""
        return self.n_excess / self.n_total

    def estimate_risk(self, alpha: float) -> Tuple[float, float]:
        """
        Derive EVT-adjusted VaR and ES from the fitted shape and scale.

        Uses the standard peaks-over-threshold estimators (McNeil, Frey &
        Embrechts, *Quantitative Risk Management*, Sec. 7.2):

            VaR_a = u + (beta/xi) * [ ((1-a)/pu)^-xi - 1 ]
            ES_a  = (VaR_a + beta - xi*u) / (1 - xi)

        Two degenerate cases the naive form gets wrong, both guarded here:

        * ``xi -> 0``. The VaR expression divides by xi. The limit as xi -> 0 is
          the exponential-tail form ``u + beta * log(pu/(1-a))``, which is used
          whenever ``|xi| < XI_ZERO_TOL``.
        * ``xi >= 1``. The GPD then has infinite mean and the ES expression
          returns a finite-looking but meaningless number (it goes negative for
          xi > 1). ES is reported as ``+inf`` instead.

        Extrapolation below the fitted threshold is also rejected: if the
        requested tail probability exceeds the exceedance rate, the quantile
        lies in the body of the distribution, where the GPD tail fit says
        nothing and historical simulation is the right tool.

        The boundary is taken as ``max(pu, 1 - threshold_quantile)``. The
        empirical ``pu`` can fall a little below its nominal value because the
        threshold comes from an interpolated sample quantile and exceedances are
        counted strictly, so comparing against ``pu`` alone would reject a
        request at exactly the declared threshold level — e.g. alpha=0.95 with a
        95% threshold, where the answer is simply VaR = u.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1).")

        pu = self.exceedance_rate
        tail_prob = 1.0 - alpha
        max_tail_prob = max(pu, 1.0 - self.threshold_quantile)

        if tail_prob > max_tail_prob:
            raise ValueError(
                f"alpha={alpha} implies a tail probability of {tail_prob:.4g}, "
                f"which exceeds the fitted exceedance rate of {max_tail_prob:.4g}. "
                f"The POT fit only describes losses above the "
                f"{self.threshold_quantile:.0%} threshold; use historical "
                f"simulation for this level."
            )

        ratio = tail_prob / pu

        if abs(self.xi) < XI_ZERO_TOL:
            # Exponential limit: lim_{xi->0} (beta/xi)(ratio^-xi - 1) = -beta*ln(ratio)
            var = self.u + self.beta * np.log(1.0 / ratio)
        else:
            var = self.u + (self.beta / self.xi) * (ratio ** (-self.xi) - 1.0)

        if self.xi >= 1.0:
            es = float(np.inf)
        else:
            es = (var + self.beta - self.xi * self.u) / (1.0 - self.xi)

        return float(var), float(es)
