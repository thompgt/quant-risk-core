import numpy as np
import pandas as pd
from quant_risk_core.market_risk.volatility import GARCHEngine, RegimeSwitchingEngine

def test_egarch():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    garch = GARCHEngine()
    garch.fit(returns, model_type='EGARCH')
    
    assert hasattr(garch, 'gamma')

def test_regime_switching():
    np.random.seed(42)
    # 500 days of low vol, 500 days of high vol
    low_vol = np.random.normal(0, 0.01, 500)
    high_vol = np.random.normal(0, 0.05, 500)
    returns = pd.Series(np.concatenate([low_vol, high_vol]))
    
    engine = RegimeSwitchingEngine(k_regimes=2)
    engine.fit(returns)
    probs = engine.get_regime_probabilities()
    
    assert probs.shape == (1000, 2)
    # Check if it identifies a shift
    assert probs.iloc[100, 0] > 0.5
    assert probs.iloc[900, 1] > 0.5
