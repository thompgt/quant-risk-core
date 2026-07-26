import numpy as np
from quant_risk_core.credit_risk.counterparty import CounterpartyRiskEngine
from quant_risk_core.credit_risk.mitigation import NettingEngine

def test_counterparty_profiles():
    # Grid of 5 points
    time_grid = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    engine = CounterpartyRiskEngine(time_grid)
    
    # 2 paths, 5 points
    paths = np.array([
        [-10, 10, 20, 30, -5],
        [ 10, 20, -5, -10, 10]
    ])
    engine.set_portfolio_paths(paths)
    
    profiles = engine.calculate_exposure_profiles()
    
    # EE: mean of max(V, 0)
    # Path1: [0, 10, 20, 30, 0]
    # Path2: [10, 20, 0, 0, 10]
    # EE: [5, 15, 10, 15, 5]
    np.testing.assert_array_equal(profiles['EE'], np.array([5, 15, 10, 15, 5]))
    
    pd_curve = np.array([0.01, 0.02, 0.05, 0.10, 0.15])
    cva = engine.calculate_cva(recovery_rate=0.4, pd_curve=pd_curve)
    assert cva > 0

def test_netting():
    # 2 contracts, 1 path, 2 points
    c_vals = np.array([
        [[10, -20]],  # Contract 1
        [[-5, 30]]    # Contract 2
    ])
    
    # With netting
    net_engine = NettingEngine(netting_enabled=True)
    exposure_net = net_engine.aggregate_mtm(c_vals)
    # net_value = [[5, 10]] -> exposure = [[5, 10]]
    np.testing.assert_array_equal(exposure_net, np.array([[5, 10]]))
    
    # Without netting
    no_net_engine = NettingEngine(netting_enabled=False)
    exposure_no_net = no_net_engine.aggregate_mtm(c_vals)
    # max(C1) = [[10, 0]], max(C2) = [[0, 30]]
    # sum = [[10, 30]]
    np.testing.assert_array_equal(exposure_no_net, np.array([[10, 30]]))
