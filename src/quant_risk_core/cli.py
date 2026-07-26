"""``quant-risk-validate`` — CLI VaR backtest report.

Fits a GARCH(1,1) to a synthetic return series, rolls a 99% parametric VaR, and
scores it with the Kupiec POF test, the Christoffersen independence test and the
Basel traffic-light zone.
"""
import argparse

import numpy as np
import pandas as pd

from quant_risk_core import __version__
from quant_risk_core.market_risk.volatility import GARCHEngine
from quant_risk_core.market_risk.estimators import RiskEngine
from quant_risk_core.market_risk.backtesting import RiskBacktester

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="quant-risk-validate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--seed', type=int, default=1337,
                        help='RNG seed for the synthetic return series (default: 1337).')
    parser.add_argument('--observations', type=int, default=1000,
                        help='Length of the synthetic return series (default: 1000).')
    parser.add_argument('--confidence', type=float, default=0.99,
                        help='VaR confidence level (default: 0.99).')
    parser.add_argument('--version', action='version', version=f'quant-risk-core {__version__}')
    args = parser.parse_args(argv)

    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must lie strictly between 0 and 1.")
    if args.observations < 2:
        parser.error("--observations must be at least 2.")

    # Explicitly seeded so the printed report is reproducible; the seed is echoed
    # below so any number in it can be regenerated.
    rng = np.random.default_rng(args.seed)
    returns = rng.normal(0, 0.01, args.observations)
    # Two volatility clusters, placed proportionally so a shorter series still
    # exercises the clustering the GARCH fit is meant to pick up.
    n = args.observations
    returns[int(0.20 * n):int(0.25 * n)] *= 3
    returns[int(0.70 * n):int(0.75 * n)] *= 2
    returns_series = pd.Series(returns)

    print("Fitting GARCH(1,1) Model...")
    # strict=True: a non-converged fit still produces plausible-looking
    # parameters, and every VaR below would be built on them. A validation
    # report is exactly the place that must not paper over that.
    garch = GARCHEngine(p=1, q=1, dist='normal', strict=True)
    garch.fit(returns_series)

    vol = garch.conditional_volatility()

    confidence = args.confidence
    engine = RiskEngine(confidence_levels=[confidence])
    var_key = f'VaR_{confidence}'

    var_series = np.array([
        engine.parametric_var_es(mu=0, sigma=s, dist='normal')[var_key] for s in vol
    ])

    bt = RiskBacktester(confidence_level=confidence)
    metrics = bt.evaluate(returns_series, pd.Series(var_series, index=returns_series.index))

    # Observed vs expected breach rate. Deliberately NOT reported as an
    # "accuracy" of 1 - exceptions/observations: that figure rewards *fewer*
    # breaches, so a model with zero exceptions would score a perfect 1.0 while
    # the Kupiec test correctly rejects it for being too conservative. For a VaR
    # model the target is to breach at the nominal rate, not to breach rarely.
    expected_rate = 1.0 - confidence
    observed_rate = metrics['Exceptions'] / metrics['Observations']

    print("\n=== Validation Metrics ===")
    print(f"Library version    : {__version__}")
    print(f"Seed               : {args.seed}")
    print(f"Confidence level   : {confidence:.2%}")
    print(f"Total Observations : {metrics['Observations']}")
    print(f"Exceptions         : {metrics['Exceptions']}")
    print(f"Observed breach rate: {observed_rate:.4%}  (expected {expected_rate:.4%})")
    print(f"Kupiec p-value     : {metrics['Kupiec_p_value']:.4f}")
    print(f"Christoffersen p-val: {metrics['Christoffersen_p_value']:.4f}")
    basel_n = metrics['Basel_Exceptions']
    print(
        f"Basel Zone         : {metrics['Basel_Zone']} "
        f"({basel_n} exception{'' if basel_n == 1 else 's'} in the trailing "
        f"{metrics['Basel_Window']} days)"
    )
    print("==========================\n")
    
if __name__ == "__main__":
    main()
