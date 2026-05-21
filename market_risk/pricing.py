import numpy as np
import scipy.stats as stats
from typing import Tuple, Dict

class BlackScholesEngine:
    @staticmethod
    def calculate_prices(S: np.ndarray, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> np.ndarray:
        """
        Vectorized Black-Scholes pricing.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        delta = stats.norm.cdf(d1)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)
        theta = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Vega': vega,
            'Theta': theta,
            'Rho': rho
        }
