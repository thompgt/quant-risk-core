import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, Union

class RiskBacktester:
    def __init__(self, confidence_level: float):
        """
        Backtesting VaR estimates.
        """
        self.alpha = confidence_level

    def kupiec_pof_test(self, exceptions: int, n_obs: int) -> float:
        """
        Kupiec Proportion of Failure (POF) Test.
        """
        p = 1 - self.alpha
        
        if exceptions == 0:
            # Check if 0 exceptions is reasonable
            return 1.0
            
        # Likelihood ratio statistic
        lr = -2 * (exceptions * np.log(p) + (n_obs - exceptions) * np.log(1 - p)) \
             + 2 * (exceptions * np.log(exceptions / n_obs) + (n_obs - exceptions) * np.log(1 - exceptions / n_obs))
             
        p_value = 1 - stats.chi2.cdf(lr, df=1)
        return p_value

    def christoffersen_independence_test(self, hits: np.ndarray) -> float:
        """
        Christoffersen Independence Test.
        hits: 0/1 array.
        """
        if len(hits) < 2 or hits.sum() == 0:
            return 1.0
            
        n00, n01, n10, n11 = 0, 0, 0, 0
        
        for i in range(1, len(hits)):
            if hits[i-1] == 0 and hits[i] == 0: n00 += 1
            elif hits[i-1] == 0 and hits[i] == 1: n01 += 1
            elif hits[i-1] == 1 and hits[i] == 0: n10 += 1
            elif hits[i-1] == 1 and hits[i] == 1: n11 += 1
            
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        def safe_log(x): return np.log(x) if x > 0 else 0
        
        log_L_null = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
        log_L_alt = n00 * safe_log(1 - pi0) + n01 * safe_log(pi0) + n10 * safe_log(1 - pi1) + n11 * safe_log(pi1)
        
        lr_ind = -2 * (log_L_null - log_L_alt)
        p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
        
        return p_value

    def basel_traffic_light(self, exceptions: int, n_obs: int = 250) -> str:
        """
        Basel Traffic Light System.
        """
        p = 1 - self.alpha
        cum_prob = stats.binom.cdf(exceptions, n_obs, p)
        
        if cum_prob < 0.95:
            return "Green"
        elif cum_prob < 0.9999:
            return "Yellow"
        else:
            return "Red"
            
    def evaluate(self, returns: pd.Series, var_estimates: pd.Series) -> Dict[str, Union[float, str, int]]:
        """
        Comprehensive evaluation.
        """
        data = pd.DataFrame({'returns': returns, 'var': var_estimates}).dropna()
        
        hits = (data['returns'] < -data['var']).astype(int).values
        exceptions = hits.sum()
        n_obs = len(hits)
        
        return {
            'Exceptions': int(exceptions),
            'Observations': int(n_obs),
            'Kupiec_p_value': float(self.kupiec_pof_test(exceptions, n_obs)),
            'Christoffersen_p_value': float(self.christoffersen_independence_test(hits)),
            'Basel_Zone': self.basel_traffic_light(exceptions, n_obs)
        }
