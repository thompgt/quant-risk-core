import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Tuple
import scipy.stats as stats

SeedLike = Union[int, np.random.Generator, None]


def resolve_rng(seed: SeedLike = None) -> Tuple[np.random.Generator, int]:
    """
    Build a Generator and report the integer seed that reproduces it.

    Risk numbers must be regenerable to be auditable, so callers that pass no
    seed still get one drawn from OS entropy and returned to them, rather than
    an unrecoverable global RNG state.
    """
    if isinstance(seed, np.random.Generator):
        # Caller owns the stream; there is no integer that reproduces its
        # current position, so report -1 to mark the result as not replayable
        # from a seed alone.
        return seed, -1
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**63))
    return np.random.default_rng(seed), int(seed)


def simulate_paths(
    S0: float,
    mu: float,
    sigma: float,
    horizon: int,
    paths: int,
    dt: float = 1.0,
    seed: SeedLike = None,
) -> np.ndarray:
    """
    Simulate terminal values of a geometric Brownian motion.

    Returns an array of `paths` terminal values S_T.

    The terminal value of a GBM depends only on the *sum* of the horizon's
    normal draws,

        S_T = S0 * exp((mu - 0.5*sigma^2)*dt*H + sigma*sqrt(dt)*sum_t Z_t)

    so a per-step loop is unnecessary for a terminal-only payoff and the whole
    simulation reduces to one vectorised expression.

    Reproducibility
    ---------------
    Draws come from an explicit `numpy.random.Generator` seeded by `seed`. An
    earlier implementation of this function was `@njit(parallel=True)` and
    called `np.random.standard_normal` inside a `prange` loop; Numba gives each
    thread an independent RNG state that no Python-level seed can control, so
    results were not reproducible run to run. Benchmarked on 50k paths at
    H=10, that kernel was ~1.75x faster warm (8.4ms vs 14.6ms) but cost ~5.7s
    of import plus ~6.5s of JIT compilation. Saving 6ms per call is not worth
    forfeiting reproducibility, so the JIT was removed.
    """
    if horizon < 1:
        raise ValueError("horizon must be a positive number of steps.")
    if paths < 1:
        raise ValueError("paths must be positive.")

    rng, _ = resolve_rng(seed)
    z_sum = rng.standard_normal((paths, horizon)).sum(axis=1)

    drift = (mu - 0.5 * sigma**2) * dt * horizon
    diffusion = sigma * np.sqrt(dt) * z_sum
    return S0 * np.exp(drift + diffusion)

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

    def monte_carlo_var_es(
        self,
        initial_value: float,
        mu: float,
        sigma: float,
        horizon: int,
        paths: int = 50000,
        seed: SeedLike = None,
    ) -> Dict[str, Union[int, float]]:
        """
        Monte Carlo Simulation for VaR and ES.

        The returned dict carries the `seed` and `paths` actually used, so any
        figure produced from it can be regenerated exactly. If `seed` is None a
        seed is drawn from OS entropy and reported back; it is never left
        implicit.
        """
        rng, resolved_seed = resolve_rng(seed)
        terminal_values = simulate_paths(
            initial_value, mu, sigma, horizon, paths, seed=rng
        )
        returns = (terminal_values - initial_value) / initial_value

        results: Dict[str, Union[int, float]] = {
            'seed': resolved_seed,
            'paths': int(paths),
        }
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
