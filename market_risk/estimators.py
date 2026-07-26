import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
import scipy.stats as stats
from numba import njit, prange

@njit(parallel=True)
def simulate_paths(S0: float, mu: float, sigma: float, horizon: int, paths: int, dt: float = 1.0) -> np.ndarray:
    """
    Simulate geometric Brownian motion using Numba for Monte Carlo VaR.
    """
    results = np.zeros(paths)
    for i in prange(paths):
        Z = np.random.standard_normal(horizon)
        S = S0
        for t in range(horizon):
            S = S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])
        results[i] = S
    return results

class RiskEngine:
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels

    def parametric_var_es(self, mu: float, sigma: float, dist: str = 'normal', df: Optional[float] = None) -> Dict[str, float]:
        """
        Parametric VaR and ES using Normal or Student-t distribution.

        Both VaR and ES are returned as *positive loss* quantities, i.e. for a
        return R with mean `mu` and standard deviation `sigma`, the loss is
        L = -R and

            VaR_a = -(mu + sigma * q_{1-a})
            ES_a  = -mu + sigma * k_a

        where q is the (1-a) quantile of the standardised distribution and k_a
        is the corresponding standardised tail expectation. Note the mean term
        enters ES with a *negative* sign, exactly as it does in VaR: a positive
        expected return reduces the expected loss.
        """
        results = {}
        for alpha in self.confidence_levels:
            if dist == 'normal':
                z = stats.norm.ppf(1 - alpha)
                var = -(mu + sigma * z)
                # Standardised normal tail expectation: phi(z) / (1 - alpha)
                es = -mu + sigma * (stats.norm.pdf(z) / (1 - alpha))
            elif dist == 't':
                if df is None or df <= 2:
                    raise ValueError("Degrees of freedom must be > 2 for ES.")
                t_val = stats.t.ppf(1 - alpha, df)
                var = -(mu + sigma * t_val * np.sqrt((df - 2) / df))

                # Standardised Student-t tail expectation. The sqrt((df-2)/df)
                # factor rescales the raw t to unit variance so that `sigma` is
                # interpreted as the standard deviation of the return.
                scale_t = stats.t.pdf(t_val, df) / (1 - alpha)
                adj = (df + t_val**2) / (df - 1)
                es = -mu + sigma * scale_t * adj * np.sqrt((df - 2) / df)
            else:
                raise ValueError("Unsupported distribution.")
                
            results[f'VaR_{alpha}'] = var
            results[f'ES_{alpha}'] = es
            
        return results

    def historical_var_es(self, returns: pd.Series) -> Dict[str, float]:
        """
        Historical Simulation for VaR and ES.
        """
        results = {}
        for alpha in self.confidence_levels:
            var = -np.quantile(returns, 1 - alpha)
            tail_returns = returns[returns < -var]
            if len(tail_returns) > 0:
                es = -tail_returns.mean()
            else:
                es = var
                
            results[f'VaR_{alpha}'] = var
            results[f'ES_{alpha}'] = es
            
        return results

    def monte_carlo_var_es(self, initial_value: float, mu: float, sigma: float, horizon: int, paths: int = 50000) -> Dict[str, float]:
        """
        Monte Carlo Simulation for VaR and ES.
        """
        terminal_values = simulate_paths(initial_value, mu, sigma, horizon, paths)
        returns = (terminal_values - initial_value) / initial_value
        
        results = {}
        for alpha in self.confidence_levels:
            var = -np.quantile(returns, 1 - alpha)
            tail_returns = returns[returns < -var]
            if len(tail_returns) > 0:
                es = -tail_returns.mean()
            else:
                es = var
                
            results[f'VaR_{alpha}'] = var
            results[f'ES_{alpha}'] = es
            
        return results
