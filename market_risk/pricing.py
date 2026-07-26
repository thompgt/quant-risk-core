import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple

VALID_OPTION_TYPES = ("call", "put")


def _normalise_option_type(option_type: str) -> str:
    """
    Validate and canonicalise the option type.

    `calculate_prices` previously treated *any* value other than 'call' as a
    put, so a typo silently returned the wrong instrument's price.
    """
    kind = str(option_type).lower()
    if kind not in VALID_OPTION_TYPES:
        raise ValueError(
            f"option_type must be one of {VALID_OPTION_TYPES}, got {option_type!r}."
        )
    return kind


def _validate_inputs(S, K: float, T: float, r: float, sigma: float) -> None:
    if np.any(np.asarray(S) <= 0):
        raise ValueError("Spot price S must be positive.")
    if K <= 0:
        raise ValueError("Strike K must be positive.")
    if T <= 0:
        raise ValueError("Time to expiry T must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")


def _d1_d2(S, K: float, T: float, r: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


class BlackScholesEngine:
    @staticmethod
    def calculate_prices(S: np.ndarray, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> np.ndarray:
        """
        Vectorized Black-Scholes pricing.
        """
        kind = _normalise_option_type(option_type)
        _validate_inputs(S, K, T, r, sigma)
        d1, d2 = _d1_d2(S, K, T, r, sigma)

        if kind == 'call':
            return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    @staticmethod
    def calculate_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
    ) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks for a call or a put.

        `option_type` was previously absent: `calculate_prices` supported puts but
        `calculate_greeks` always returned *call* greeks, so anyone risking a put
        book got the wrong sign on delta, theta and rho. Gamma and vega are
        genuinely identical across the two by put-call parity, since the parity
        relation is linear in S and independent of sigma.

        Theta is the derivative with respect to calendar time (so it is normally
        negative for a long option) and is quoted per year, as are vega and rho.
        """
        kind = _normalise_option_type(option_type)
        _validate_inputs(S, K, T, r, sigma)
        d1, d2 = _d1_d2(S, K, T, r, sigma)

        sqrt_t = np.sqrt(T)
        pdf_d1 = stats.norm.pdf(d1)
        discount = np.exp(-r * T)

        # Identical for calls and puts: the parity relation C - P = S - K*e^{-rT}
        # is linear in S and free of sigma, so its second S-derivative and its
        # sigma-derivative both vanish.
        gamma = pdf_d1 / (S * sigma * sqrt_t)
        vega = S * pdf_d1 * sqrt_t

        common_theta = -(S * pdf_d1 * sigma) / (2 * sqrt_t)

        if kind == 'call':
            delta = stats.norm.cdf(d1)
            theta = common_theta - r * K * discount * stats.norm.cdf(d2)
            rho = K * T * discount * stats.norm.cdf(d2)
        else:
            delta = stats.norm.cdf(d1) - 1.0
            theta = common_theta + r * K * discount * stats.norm.cdf(-d2)
            rho = -K * T * discount * stats.norm.cdf(-d2)

        return {
            'Delta': float(delta),
            'Gamma': float(gamma),
            'Vega': float(vega),
            'Theta': float(theta),
            'Rho': float(rho),
        }
