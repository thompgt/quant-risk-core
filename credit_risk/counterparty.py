import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from numba import njit, prange

class CounterpartyRiskEngine:
    def __init__(self, time_grid: np.ndarray):
        """
        time_grid: array of forward time points (in years)
        """
        self.time_grid = time_grid
        self.num_points = len(time_grid)
        self.portfolio_paths = None # shape: (num_paths, num_points)

    def set_portfolio_paths(self, paths: np.ndarray) -> None:
        """
        paths: array of simulated portfolio values of shape (num_paths, num_points)
        """
        if paths.shape[1] != self.num_points:
            raise ValueError("Paths must match the time grid length.")
        self.portfolio_paths = paths

    def calculate_exposure_profiles(self, quantile: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calculate EE, EPE, and PFE.
        """
        if self.portfolio_paths is None:
            raise ValueError("Portfolio paths not set.")
            
        exposure_paths = np.maximum(self.portfolio_paths, 0)
        
        ee = np.mean(exposure_paths, axis=0)
        
        # EPE is the time-weighted average of EE
        if len(self.time_grid) > 1:
            dt = np.diff(self.time_grid, prepend=self.time_grid[0])
            epe = np.sum(ee * dt) / self.time_grid[-1] if self.time_grid[-1] > 0 else ee[0]
        else:
            epe = ee[0]
            
        pfe = np.quantile(exposure_paths, quantile, axis=0)
        
        return {
            'EE': ee,
            'EPE': np.array([epe]),
            'PFE': pfe
        }

    def calculate_cva(self, recovery_rate: float, pd_curve: np.ndarray) -> float:
        """
        Calculate Credit Value Adjustment.
        pd_curve: array of cumulative default probabilities corresponding to time_grid
        """
        if self.portfolio_paths is None:
            raise ValueError("Portfolio paths not set.")
            
        profiles = self.calculate_exposure_profiles()
        ee = profiles['EE']
        
        # Marginal PD: probability of default between t_{i-1} and t_i
        marginal_pd = np.diff(pd_curve, prepend=pd_curve[0])
        
        # CVA ≈ (1 - R) * sum(EE(t_i) * dPD(t_i))
        cva = (1 - recovery_rate) * np.sum(ee * marginal_pd)
        
        return float(cva)

    def calculate_cva_wwr(self, recovery_rate: float, pd_curve: np.ndarray, alpha_wwr: float = 1.1) -> float:
        """
        Calculate CVA with Wrong-Way Risk (WWR) multiplier.
        alpha_wwr: Basel multiplier (usually 1.0 to 1.4) to account for correlation between EE and PD.
        """
        base_cva = self.calculate_cva(recovery_rate, pd_curve)
        return base_cva * alpha_wwr

class RatingMigrationEngine:
    def __init__(self, transition_matrix: np.ndarray):
        """
        transition_matrix: (N, N) matrix where cell (i, j) is prob of moving from rating i to j.
        """
        self.tm = transition_matrix
        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Rows of transition matrix must sum to 1.0")

    def simulate_migration(self, initial_rating_idx: int, horizons: int) -> np.ndarray:
        """
        Simulate a rating path over H discrete time steps.
        """
        current_rating = initial_rating_idx
        path = [current_rating]
        for _ in range(horizons):
            probs = self.tm[current_rating]
            current_rating = np.random.choice(len(probs), p=probs)
            path.append(current_rating)
        return np.array(path)
