import numpy as np
import pandas as pd
from quant_risk_core.market_risk.stress_testing import ScenarioEngine, FactorStresser

def test_hypothetical_scenario():
    df = pd.DataFrame({'AAPL': [150, 155], 'GOOG': [2800, 2850]})
    engine = ScenarioEngine(df)
    
    shocks = {'AAPL': -0.1}
    shocked = engine.apply_hypothetical_scenario(shocks)
    
    assert shocked['AAPL'].iloc[0] == 135.0
    assert shocked['GOOG'].iloc[0] == 2800.0 # Unchanged

def test_correlation_tilt():
    corr = np.array([[1.0, 0.2], [0.2, 1.0]])
    stressed = FactorStresser.tilt_correlation(corr, 2.0)
    
    assert stressed[0, 1] == 0.4
    assert stressed[0, 0] == 1.0
