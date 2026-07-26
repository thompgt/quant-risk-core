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
        
    confidence = 0.99
    bt = RiskBacktester(confidence_level=confidence)
    metrics = bt.evaluate(returns_series, pd.Series(var_99))

    # Observed vs expected breach rate. Deliberately NOT reported as an
    # "accuracy" of 1 - exceptions/observations: that figure rewards *fewer*
    # breaches, so a model with zero exceptions would score a perfect 1.0 while
    # the Kupiec test correctly rejects it for being too conservative. For a VaR
    # model the target is to breach at the nominal rate, not to breach rarely.
    expected_rate = 1.0 - confidence
    observed_rate = metrics['Exceptions'] / metrics['Observations']

    print("\n=== Validation Metrics ===")
    print(f"Total Observations : {metrics['Observations']}")
    print(f"Exceptions         : {metrics['Exceptions']}")
    print(f"Observed breach rate: {observed_rate:.4%}  (expected {expected_rate:.4%})")
    print(f"Kupiec p-value     : {metrics['Kupiec_p_value']:.4f}")
    print(f"Christoffersen p-val: {metrics['Christoffersen_p_value']:.4f}")
    print(
        f"Basel Zone         : {metrics['Basel_Zone']} "
        f"({metrics['Basel_Exceptions']} exceptions in the trailing "
        f"{metrics['Basel_Window']} days)"
    )
    print("==========================\n")
    
if __name__ == "__main__":
    main()
