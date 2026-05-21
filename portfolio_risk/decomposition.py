import numpy as np
import scipy.stats as stats
from typing import Tuple

class CopulaEngine:
    def __init__(self, correlation_matrix: np.ndarray):
        """
        correlation_matrix: (N, N) symmetric positive semi-definite matrix
        """
        self.corr = correlation_matrix
        self.dim = correlation_matrix.shape[0]
        self.cholesky = np.linalg.cholesky(correlation_matrix)

    def generate_gaussian_copula_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate N-dimensional uniform samples with Gaussian Copula dependency.
        """
        # Generate correlated standard normal samples
        z = np.random.standard_normal((n_samples, self.dim))
        correlated_z = z @ self.cholesky.T
        
        # Transform to uniform using standard normal CDF
        u = stats.norm.cdf(correlated_z)
        return u

class RiskDecomposer:
    def __init__(self, weights: np.ndarray, cov_matrix: np.ndarray):
        self.w = weights
        self.cov = cov_matrix

    def calculate_marginal_var(self, confidence_level: float = 0.95) -> np.ndarray:
        """
        MVaR = dVaR / dw = z_alpha * (Cov * w) / sqrt(w' * Cov * w)
        """
        z_alpha = stats.norm.ppf(confidence_level)
        portfolio_vol = np.sqrt(self.w.T @ self.cov @ self.w)
        mvar = z_alpha * (self.cov @ self.w) / portfolio_vol
        return mvar

    def calculate_component_var(self, confidence_level: float = 0.95) -> np.ndarray:
        """
        CVaR = w_i * MVaR_i
        """
        mvar = self.calculate_marginal_var(confidence_level)
        cvar = self.w * mvar
        return cvar
