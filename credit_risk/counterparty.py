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
