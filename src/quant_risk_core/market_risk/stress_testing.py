import numpy as np
import pandas as pd
from typing import Dict, List

class ScenarioEngine:
    def __init__(self, baseline_data: pd.DataFrame):
        self.data = baseline_data

    def apply_historical_shock(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Extract a specific historical window (e.g., 2008) to use as a stress period.
        """
        stress_window = self.data.loc[start_date:end_date]
        return stress_window

    def apply_hypothetical_scenario(self, shocks: Dict[str, float]) -> pd.DataFrame:
        """
        Apply manual percentage shocks to specific assets.
        shocks: {'Asset_A': -0.10, 'Asset_B': 0.05}
        """
        shocked_data = self.data.copy()
        for asset, shock in shocks.items():
            if asset in shocked_data.columns:
                shocked_data[asset] = shocked_data[asset] * (1 + shock)
        return shocked_data

class FactorStresser:
    @staticmethod
    def shift_volatility(vols: np.ndarray, shift: float) -> np.ndarray:
        """
        Parallel shift of the volatility surface.
        """
        return vols + shift

    @staticmethod
    def tilt_correlation(corr_matrix: np.ndarray, factor: float) -> np.ndarray:
        """
        Increase all off-diagonal correlations by a factor (e.g., during market crash).
        """
        stressed_corr = corr_matrix * factor
        np.fill_diagonal(stressed_corr, 1.0)
        return np.clip(stressed_corr, -1.0, 1.0)
