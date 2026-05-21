import numpy as np

class LiquidityRiskEngine:
    def __init__(self, position_size: float, mid_price: float):
        self.size = position_size
        self.mid = mid_price

    def calculate_l_var(self, var_base: float, bid_ask_spread: float, spread_vol: float = 0.0, confidence_level: float = 0.95) -> float:
        """
        L-VaR = VaR_base + Liquidity_Adjustment
        Adjustment = 0.5 * P * s * (1 + z_alpha * sigma_s)
        """
        from scipy.stats import norm
        z_alpha = norm.ppf(confidence_level)
        
        # Spread adjustment (Exogenous)
        spread_adj = 0.5 * self.mid * self.size * bid_ask_spread * (1 + z_alpha * spread_vol)
        
        return var_base + spread_adj

    def price_impact_adjustment(self, daily_volume: float, lambda_impact: float = 0.1) -> float:
        """
        Endogenous Liquidity: Kyly-style impact model.
        Adjustment = lambda * (size / daily_volume)^k
        """
        if daily_volume <= 0:
            return 0.0
        
        impact = lambda_impact * (self.size / daily_volume)**0.5
        return impact * self.mid * self.size
