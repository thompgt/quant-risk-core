import numpy as np
from quant_risk_core.market_risk.liquidity import LiquidityRiskEngine

def test_liquidity_adjustment():
    engine = LiquidityRiskEngine(position_size=1000, mid_price=100)
    var_base = 2000 # Baseline VaR
    
    # s = 0.01 (1%), sigma_s = 0.1
    l_var = engine.calculate_l_var(var_base, bid_ask_spread=0.01, spread_vol=0.1)
    
    assert l_var > var_base
    
    impact = engine.price_impact_adjustment(daily_volume=10000)
    assert impact > 0
