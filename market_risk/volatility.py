import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from arch import arch_model
import warnings

class GARCHEngine:
    def __init__(self, p: int = 1, q: int = 1, dist: str = 'normal'):
        """
        GARCH(p, q) conditional variance engine.
        dist: 'normal' or 't' (Student-t)
        """
        self.p = p
        self.q = q
        if dist not in ['normal', 't']:
            raise ValueError("dist must be 'normal' or 't'")
        self.dist = dist
        self.model_fit = None
        self.omega = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.nu = None # Degrees of freedom for t dist

    def fit(self, returns: pd.Series, model_type: str = 'Garch') -> None:
        """
        Fits the specified volatility model.
        model_type: 'Garch' or 'EGARCH'
        """
        # arch_model expects returns to have zero mean if mean='Zero'
        am = arch_model(returns, vol=model_type, p=self.p, q=self.q, dist=self.dist, mean='Zero', rescale=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model_fit = am.fit(disp='off', update_freq=0)
            
        params = self.model_fit.params
        self.omega = params.get('omega', 0.0)
        
        if model_type == 'Garch':
            self.alpha = params.get('alpha[1]', 0.0)
            self.beta = params.get('beta[1]', 0.0)
        elif model_type == 'EGARCH':
            self.alpha = params.get('alpha[1]', 0.0) # size effect
            self.gamma = params.get('gamma[1]', 0.0) # leverage effect
            self.beta = params.get('beta[1]', 0.0) # persistence
        
        if self.dist == 't':
            self.nu = params.get('nu', 5.0)
            
    def forecast_volatility(self, horizon: int) -> np.ndarray:
        """
        Out-of-sample forecasting function across forward horizon H.
        Returns array of length H of forecasted volatilities.
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting.")
            
        forecasts = self.model_fit.forecast(horizon=horizon)
        var_forecasts = forecasts.variance.iloc[-1].values
        
        return np.sqrt(var_forecasts)
        
    def conditional_volatility(self) -> pd.Series:
        """
        Returns the in-sample conditional volatility.
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted first.")
        return self.model_fit.conditional_volatility

import statsmodels.api as sm

class RegimeSwitchingEngine:
    def __init__(self, k_regimes: int = 2):
        self.k = k_regimes
        self.model = None
        self.results = None

    def fit(self, returns: pd.Series) -> None:
        """
        Fit a Markov Switching Dynamic Regression model.
        """
        self.model = sm.tsa.MarkovRegression(returns, k_regimes=self.k, trend='c', switching_variance=True)
        self.results = self.model.fit()

    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Return the smoothed probabilities of being in each regime.
        """
        if self.results is None:
            raise ValueError("Model must be fitted first.")
        return self.results.smoothed_marginal_probabilities
