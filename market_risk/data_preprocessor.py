import pandas as pd
import numpy as np
from typing import Optional

class DataPreprocessor:
    def __init__(self, forward_fill: bool = True, drop_zero_variance: bool = True):
        self.forward_fill = forward_fill
        self.drop_zero_variance = drop_zero_variance

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Computes continuous log-returns from asset price series.
        r_t = ln(P_t / P_{t-1})
        """
        if self.forward_fill:
            prices = prices.ffill()
            
        # Drop rows that are entirely NaN after ffill if any
        prices = prices.dropna(how='all')
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Drop the first row which will be NaN
        log_returns = log_returns.dropna(how='all')
        
        if self.drop_zero_variance:
            # Check for columns with zero variance
            variances = log_returns.var()
            zero_var_cols = variances[variances == 0].index
            if len(zero_var_cols) > 0:
                log_returns = log_returns.drop(columns=zero_var_cols)
                
        return log_returns
