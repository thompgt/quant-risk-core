import numpy as np
import pandas as pd
from scipy.stats import genpareto
from typing import Tuple, Dict

class EVTEngine:
    def __init__(self, threshold_quantile: float = 0.95):
        self.threshold_quantile = threshold_quantile
        self.xi = 0.0 # shape
        self.beta = 0.0 # scale
        self.u = 0.0 # threshold
        self.n_total = 0
        self.n_excess = 0

    def fit(self, returns: pd.Series) -> None:
        """
        Fit EVT POT using Generalized Pareto Distribution (GPD).
        We model losses, so we take negative returns.
        """
        losses = -returns.dropna()
        self.n_total = len(losses)
        
        self.u = losses.quantile(self.threshold_quantile)
        
        excesses = losses[losses > self.u] - self.u
        self.n_excess = len(excesses)
        
        if self.n_excess == 0:
            raise ValueError("No excesses found above threshold.")
            
        c, loc, scale = genpareto.fit(excesses, floc=0)
        self.xi = c
        self.beta = scale

    def estimate_risk(self, alpha: float) -> Tuple[float, float]:
        """
        Derive EVT-adjusted VaR and ES formulas using fitted shape and scale.
        """
        if self.n_total == 0:
            raise ValueError("Model must be fitted first.")
            
        pu = self.n_excess / self.n_total
        
        var = self.u + (self.beta / self.xi) * (((1 - alpha) / pu)**(-self.xi) - 1)
        es = (var + self.beta - self.xi * self.u) / (1 - self.xi)
        
        return var, es
