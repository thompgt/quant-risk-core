import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, Optional, Union

# The Basel traffic-light zone boundaries are defined for a 250-day
# (approximately one trading year) backtest window. Scoring a different number
# of observations against them does not yield a Basel zone.
BASEL_WINDOW = 250


class RiskBacktester:
    def __init__(self, confidence_level: float):
        """
        Backtesting VaR estimates.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")
        self.alpha = confidence_level

    def kupiec_pof_test(self, exceptions: int, n_obs: int) -> float:
        """
        Kupiec Proportion of Failure (POF) Test.

        Likelihood ratio against the null that the true exception rate equals
        the model's nominal rate p = 1 - alpha:

            LR = -2*[(n-x)*ln(1-p) + x*ln(p)]
                 +2*[(n-x)*ln(1-x/n) + x*ln(x/n)]

        asymptotically chi-square with 1 degree of freedom, using the convention
        0*ln(0) = 0 so the x = 0 and x = n endpoints are handled by the same
        expression.

        The zero-exception case is *not* an automatic pass. An earlier version
        returned a hard-coded p-value of 1.0 there, which is wrong in a way that
        matters: observing no exceptions over a long window is evidence the model
        is too conservative, and the test rejects it. At n = 250 and p = 0.01,
        zero exceptions gives LR = 5.03 and p = 0.025 — a rejection at the 5%
        level that the old code reported as a perfect pass.
        """
        if n_obs <= 0:
            raise ValueError("n_obs must be positive.")
        if not 0 <= exceptions <= n_obs:
            raise ValueError(f"exceptions must lie in [0, {n_obs}], got {exceptions}.")

        p = 1 - self.alpha
        x, n = int(exceptions), int(n_obs)

        def xlogy(count: int, prob: float) -> float:
            """count * log(prob) with the 0 * log(0) = 0 convention."""
            if count == 0:
                return 0.0
            return count * np.log(prob)

        # Log-likelihood under the null (rate fixed at p) and under the
        # unrestricted alternative (rate estimated as x/n).
        log_l_null = xlogy(n - x, 1 - p) + xlogy(x, p)
        log_l_alt = xlogy(n - x, 1 - x / n) + xlogy(x, x / n)

        lr = -2 * (log_l_null - log_l_alt)
        # Guard against a tiny negative LR from floating-point cancellation.
        lr = max(lr, 0.0)

        return float(1 - stats.chi2.cdf(lr, df=1))

    def christoffersen_independence_test(self, hits: np.ndarray) -> float:
        """
        Christoffersen Independence Test.
        hits: 0/1 array.

        Tests whether exceptions cluster, i.e. whether P(hit | previous hit)
        differs from P(hit | no previous hit). Returns 1.0 when the transition
        counts cannot identify both conditional rates (fewer than two
        observations, or no exceptions at all), since there is then no evidence
        of clustering to weigh.
        """
        hits = np.asarray(hits)
        if len(hits) < 2 or hits.sum() == 0:
            return 1.0

        prev, curr = hits[:-1], hits[1:]
        n00 = int(np.sum((prev == 0) & (curr == 0)))
        n01 = int(np.sum((prev == 0) & (curr == 1)))
        n10 = int(np.sum((prev == 1) & (curr == 0)))
        n11 = int(np.sum((prev == 1) & (curr == 1)))

        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)

        def safe_log(x: float) -> float:
            return float(np.log(x)) if x > 0 else 0.0

        log_l_null = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
        log_l_alt = (
            n00 * safe_log(1 - pi0)
            + n01 * safe_log(pi0)
            + n10 * safe_log(1 - pi1)
            + n11 * safe_log(pi1)
        )

        lr_ind = max(-2 * (log_l_null - log_l_alt), 0.0)
        return float(1 - stats.chi2.cdf(lr_ind, df=1))

    def basel_traffic_light(self, exceptions: int, n_obs: int = BASEL_WINDOW) -> str:
        """
        Basel Traffic Light System.

        Zones are set by the cumulative binomial probability of observing at most
        `exceptions` breaches: Green below 95%, Yellow below 99.99%, Red above.
        At the framework's n_obs = 250 and a 99% VaR this reproduces the
        published table exactly (Green 0-4, Yellow 5-9, Red 10+); see
        tests/golden/test_backtesting.py.

        `n_obs` defaults to 250 because that is the window the Basel boundaries
        are calibrated for. Passing a different length rescales the binomial and
        no longer yields a Basel zone — `evaluate` therefore scores the zone on
        the trailing 250 observations rather than on the whole sample.
        """
        if n_obs <= 0:
            raise ValueError("n_obs must be positive.")
        if not 0 <= exceptions <= n_obs:
            raise ValueError(f"exceptions must lie in [0, {n_obs}], got {exceptions}.")

        p = 1 - self.alpha
        cum_prob = stats.binom.cdf(exceptions, n_obs, p)

        if cum_prob < 0.95:
            return "Green"
        elif cum_prob < 0.9999:
            return "Yellow"
        else:
            return "Red"

    def evaluate(
        self,
        returns: pd.Series,
        var_estimates: pd.Series,
        basel_window: Optional[int] = BASEL_WINDOW,
    ) -> Dict[str, Union[float, str, int]]:
        """
        Comprehensive evaluation.

        The Kupiec and Christoffersen tests use the full aligned sample, since
        both gain power with more observations. The Basel zone is scored on the
        trailing `basel_window` observations, because its boundaries are only
        defined for a 250-day window. The window length and the exception count
        within it are both reported so the zone is interpretable.

        Set `basel_window=None` to score the zone on the whole sample; the
        result is then not a Basel zone and `Basel_Window` reflects that.
        """
        data = pd.DataFrame({'returns': returns, 'var': var_estimates}).dropna()
        if data.empty:
            raise ValueError("No overlapping observations between returns and VaR.")

        hits = (data['returns'] < -data['var']).astype(int).values
        exceptions = int(hits.sum())
        n_obs = int(len(hits))

        if basel_window is None:
            window_hits = hits
        else:
            window_hits = hits[-basel_window:]
        window_n = int(len(window_hits))
        window_exceptions = int(window_hits.sum())

        return {
            'Exceptions': exceptions,
            'Observations': n_obs,
            'Kupiec_p_value': float(self.kupiec_pof_test(exceptions, n_obs)),
            'Christoffersen_p_value': float(
                self.christoffersen_independence_test(hits)
            ),
            'Basel_Zone': self.basel_traffic_light(window_exceptions, window_n),
            'Basel_Window': window_n,
            'Basel_Exceptions': window_exceptions,
        }
