import pandas as pd
import numpy as np
from typing import Optional


class DataPreprocessor:
    def __init__(
        self,
        forward_fill: bool = True,
        drop_zero_variance: bool = True,
        align: str = "intersection",
    ):
        """
        forward_fill: carry the last observation forward over gaps before
            differencing, so an untraded day does not create a spurious return.
        drop_zero_variance: drop columns whose returns never move. These break
            downstream covariance work — a zero-variance column makes a
            correlation matrix singular, so `CopulaEngine`'s Cholesky fails.
        align: how to handle dates where some assets are missing.
            'intersection' (default, and the historical behaviour) keeps only
            dates where *every* asset has a return. 'union' keeps every date and
            leaves NaN in place for the assets that are missing.

        The alignment choice matters more than it looks. Under 'intersection' a
        single asset with a short history, a different trading calendar, or a
        mid-sample listing truncates the sample for all the others: adding one
        illiquid name to a basket can silently discard most of the panel. The
        default is retained for compatibility, but `align='union'` is the honest
        choice when assets genuinely trade on different calendars, with the
        caveat that downstream covariance estimation must then handle NaN.
        """
        if align not in ("intersection", "union"):
            raise ValueError("align must be 'intersection' or 'union'.")
        self.forward_fill = forward_fill
        self.drop_zero_variance = drop_zero_variance
        self.align = align

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Computes continuous log-returns from asset price series.
        r_t = ln(P_t / P_{t-1})
        """
        if prices is None or len(prices) == 0:
            raise ValueError("prices contains no observations.")

        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        if self.forward_fill:
            prices = prices.ffill()

        # Drop rows that are entirely NaN after ffill if any
        prices = prices.dropna(how='all')

        non_positive = (prices <= 0).to_numpy().sum()
        if non_positive:
            raise ValueError(
                f"prices contains {non_positive} non-positive value(s); log "
                "returns are undefined. Check the data source for zero-filled "
                "or placeholder rows."
            )

        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))

        if self.align == "intersection":
            # Keep only dates where every asset has a return, so the resulting
            # panel is rectangular and directly usable for covariance work.
            log_returns = log_returns.dropna()
        else:
            # Keep every date, dropping only the leading row that differencing
            # always makes entirely NaN.
            log_returns = log_returns.dropna(how='all')

        if self.drop_zero_variance:
            # Check for columns with zero variance
            variances = log_returns.var()
            zero_var_cols = variances[variances == 0].index
            if len(zero_var_cols) > 0:
                log_returns = log_returns.drop(columns=zero_var_cols)

        if log_returns.empty:
            raise ValueError(
                "No returns survived preprocessing. Under align='intersection' "
                "this usually means the assets have non-overlapping histories."
            )

        return log_returns
