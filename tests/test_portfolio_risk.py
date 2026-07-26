import numpy as np
from quant_risk_core.portfolio_risk.decomposition import RiskDecomposer

def test_risk_decomposition():
    weights = np.array([0.6, 0.4])
    cov = np.array([
        [0.0004, 0.0001],
        [0.0001, 0.0009]
    ])
    
    decomposer = RiskDecomposer(weights, cov)
    mvar = decomposer.calculate_marginal_var(0.95)
    cvar = decomposer.calculate_component_var(0.95)
    
    # Portfolio VaR = sum(CVaR)
    port_vol = np.sqrt(weights.T @ cov @ weights)
    port_var = 1.64485 * port_vol
    
    np.testing.assert_approx_equal(np.sum(cvar), port_var, significant=4)

def test_copula_samples():
    from quant_risk_core.portfolio_risk.decomposition import CopulaEngine
    # Reusing the file for brevity in this step, normally separate
    corr = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    engine = CopulaEngine(corr)
    samples = engine.generate_gaussian_copula_samples(1000)
    
    assert samples.shape == (1000, 2)
    assert np.all(samples >= 0) and np.all(samples <= 1)
