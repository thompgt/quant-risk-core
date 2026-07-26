import numpy as np
import scipy.stats as stats
from typing import Optional, Tuple


class CopulaEngine:
    def __init__(self, correlation_matrix: np.ndarray):
        """
        correlation_matrix: (N, N) symmetric positive definite matrix

        Limitation: this is a *Gaussian* copula. It reproduces a given
        correlation structure but has zero tail dependence — the probability of
        joint extreme moves vanishes relative to a Student-t or Clayton copula.
        That is precisely the assumption implicated in the 2008 structured-credit
        losses, and it means multi-asset tail risk built on this engine is
        understated in exactly the scenarios that matter most.
        """
        corr = np.asarray(correlation_matrix, dtype=float)

        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError(
                f"correlation_matrix must be square; got shape {corr.shape}."
            )
        if not np.allclose(corr, corr.T, rtol=1e-8, atol=1e-10):
            raise ValueError("correlation_matrix must be symmetric.")
        if not np.allclose(np.diag(corr), 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError(
                "correlation_matrix must have unit diagonal; pass a correlation "
                "matrix, not a covariance matrix."
            )

        self.corr = corr
        self.dim = corr.shape[0]

        try:
            self.cholesky = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            # Report the actual diagnosis rather than a bare LinAlgError.
            eigenvalues = np.linalg.eigvalsh(corr)
            raise ValueError(
                "correlation_matrix is not positive definite (smallest "
                f"eigenvalue {eigenvalues.min():.3e}), so it cannot be "
                "factorised. This usually means a perfectly collinear pair of "
                "assets, a zero-variance column, or a matrix estimated from "
                "fewer observations than assets."
            ) from exc

    def generate_gaussian_copula_samples(
        self, n_samples: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate N-dimensional uniform samples with Gaussian Copula dependency.

        `seed` draws from an explicit Generator so the sample can be reproduced;
        without it the global NumPy RNG state is used and the draw cannot be
        regenerated.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")

        rng = np.random.default_rng(seed)
        # Generate correlated standard normal samples
        z = rng.standard_normal((n_samples, self.dim))
        correlated_z = z @ self.cholesky.T

        # Transform to uniform using standard normal CDF
        return stats.norm.cdf(correlated_z)


class RiskDecomposer:
    def __init__(self, weights: np.ndarray, cov_matrix: np.ndarray):
        """
        weights: portfolio weights, one per asset.
        cov_matrix: (N, N) covariance matrix of asset returns.

        Assumptions, both material and neither previously stated:

        * **Zero mean.** VaR is taken as ``z_alpha * portfolio_vol`` with no
          drift term, so the decomposition describes a zero-mean return
          distribution. Over a one-day horizon this is a standard and largely
          harmless simplification; over longer horizons the omitted drift is not
          negligible.
        * **Normality.** ``z_alpha`` is a Gaussian quantile, so the component
          contributions inherit thin tails and understate the concentration of
          risk in fat-tailed assets.

        What the method does deliver exactly is Euler additivity: component VaR
        sums to portfolio VaR, because VaR is homogeneous of degree one in the
        weights. That identity holds regardless of the two assumptions above and
        is what makes the decomposition interpretable as an allocation.
        """
        weights = np.asarray(weights, dtype=float)
        cov = np.asarray(cov_matrix, dtype=float)

        if weights.ndim != 1:
            raise ValueError(f"weights must be 1-D; got shape {weights.shape}.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov_matrix must be square; got shape {cov.shape}.")
        if cov.shape[0] != weights.size:
            raise ValueError(
                f"cov_matrix is {cov.shape[0]}x{cov.shape[1]} but there are "
                f"{weights.size} weights."
            )
        if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-12):
            raise ValueError("cov_matrix must be symmetric.")

        self.w = weights
        self.cov = cov

    def portfolio_volatility(self) -> float:
        """Standard deviation of the portfolio return."""
        variance = float(self.w.T @ self.cov @ self.w)
        if variance <= 0:
            raise ValueError(
                "Portfolio variance is not positive; the weights may be zero or "
                "the covariance matrix degenerate, so risk cannot be allocated."
            )
        return float(np.sqrt(variance))

    def calculate_marginal_var(self, confidence_level: float = 0.95) -> np.ndarray:
        """
        MVaR = dVaR / dw = z_alpha * (Cov * w) / sqrt(w' * Cov * w)
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")

        z_alpha = stats.norm.ppf(confidence_level)
        portfolio_vol = self.portfolio_volatility()
        return z_alpha * (self.cov @ self.w) / portfolio_vol

    def calculate_component_var(self, confidence_level: float = 0.95) -> np.ndarray:
        """
        CVaR = w_i * MVaR_i

        These sum to portfolio VaR by Euler's theorem, since VaR is homogeneous
        of degree one in the weights.
        """
        mvar = self.calculate_marginal_var(confidence_level)
        return self.w * mvar

    def portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Portfolio VaR, equal to the sum of the component VaRs by construction."""
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")
        return float(
            stats.norm.ppf(confidence_level) * self.portfolio_volatility()
        )
