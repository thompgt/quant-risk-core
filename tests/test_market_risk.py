import numpy as np
import pandas as pd
from market_risk.estimators import RiskEngine
from market_risk.backtesting import RiskBacktester
from market_risk.volatility import GARCHEngine

def test_var_estimators():
    # Setup dummy data
    np.random.seed(42)
    mu, sigma = 0.001, 0.02
    
    engine = RiskEngine(confidence_levels=[0.95, 0.99])
    res = engine.parametric_var_es(mu, sigma, dist='normal')
    
    assert res['VaR_0.95'] > 0
    assert res['ES_0.95'] > res['VaR_0.95']
    assert res['VaR_0.99'] > res['VaR_0.95']

def test_backtesting():
    bt = RiskBacktester(confidence_level=0.95)
    
    n_obs = 250
    exceptions = int(n_obs * 0.05) # 12
    p_val_kupiec = bt.kupiec_pof_test(exceptions, n_obs)
    
    assert p_val_kupiec > 0.05
    
    zone = bt.basel_traffic_light(exceptions, n_obs)
    assert zone == "Green"

def test_garch_engine():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    garch = GARCHEngine(p=1, q=1)
    garch.fit(returns)
    forecast = garch.forecast_volatility(horizon=5)
    
    assert len(forecast) == 5
    assert all(forecast > 0)
