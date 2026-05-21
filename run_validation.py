import numpy as np
import pandas as pd
import argparse
from market_risk.volatility import GARCHEngine
from market_risk.estimators import RiskEngine
from market_risk.backtesting import RiskBacktester

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='latest')
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    
    np.random.seed(1337)
    # Generate 1000 days of synthetic returns with slight volatility clustering
    returns = np.random.normal(0, 0.01, 1000)
    # Introduce shocks
    returns[200:250] *= 3
    returns[700:750] *= 2
    returns_series = pd.Series(returns)

    print("Fitting GARCH(1,1) Model...")
    garch = GARCHEngine(p=1, q=1, dist='normal')
    garch.fit(returns_series)
    
    vol = garch.conditional_volatility()
    
    engine = RiskEngine(confidence_levels=[0.99])
    
    # Compute 99% VaR daily
    var_99 = np.zeros(len(returns))
    for i in range(len(returns)):
        res = engine.parametric_var_es(mu=0, sigma=vol.iloc[i], dist='normal')
        var_99[i] = res['VaR_0.99']
        
    bt = RiskBacktester(confidence_level=0.99)
    metrics = bt.evaluate(returns_series, pd.Series(var_99))
    
    # Convert Exceptions/Observations into an accuracy metric
    accuracy = 1.0 - (metrics['Exceptions'] / metrics['Observations'])
    
    print("\n=== Validation Metrics ===")
    print(f"Total Observations : {metrics['Observations']}")
    print(f"Exceptions         : {metrics['Exceptions']}")
    print(f"Kupiec p-value     : {metrics['Kupiec_p_value']:.4f}")
    print(f"Christoffersen p-val: {metrics['Christoffersen_p_value']:.4f}")
    print(f"Basel Zone         : {metrics['Basel_Zone']}")
    print(f"Accuracy           : {accuracy:.4f}")
    print("==========================\n")
    
if __name__ == "__main__":
    main()
