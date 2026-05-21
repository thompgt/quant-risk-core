import numpy as np
from credit_risk.counterparty import CounterpartyRiskEngine, RatingMigrationEngine

def test_wwr_cva():
    time_grid = np.array([0.5, 1.0])
    engine = CounterpartyRiskEngine(time_grid)
    paths = np.array([[10, 20], [5, 10]])
    engine.set_portfolio_paths(paths)
    
    pd_curve = np.array([0.01, 0.02])
    cva_base = engine.calculate_cva(0.4, pd_curve)
    cva_wwr = engine.calculate_cva_wwr(0.4, pd_curve, alpha_wwr=1.2)
    
    assert cva_wwr == cva_base * 1.2

def test_migration():
    # 2x2: State 0 (Investment Grade), State 1 (High Yield)
    tm = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ])
    engine = RatingMigrationEngine(tm)
    path = engine.simulate_migration(initial_rating_idx=0, horizons=10)
    
    assert len(path) == 11
    assert np.all((path == 0) | (path == 1))
