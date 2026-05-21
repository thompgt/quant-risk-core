import numpy as np
from typing import List, Tuple

class NettingEngine:
    def __init__(self, netting_enabled: bool = True):
        self.netting_enabled = netting_enabled

    def aggregate_mtm(self, contract_values: np.ndarray) -> np.ndarray:
        """
        Aggregate mark-to-market values across contracts.
        contract_values: shape (num_contracts, num_paths, num_points)
        Returns: exposure shape (num_paths, num_points)
        """
        if self.netting_enabled:
            # ∑ V_i, then Max(..., 0)
            net_value = np.sum(contract_values, axis=0)
            exposure = np.maximum(net_value, 0)
        else:
            # ∑ Max(V_i, 0)
            exposures = np.maximum(contract_values, 0)
            exposure = np.sum(exposures, axis=0)
            
        return exposure

class CollateralManager:
    def __init__(self, threshold: float = 0.0, mpor_days: int = 10, mta: float = 0.0):
        """
        threshold: collateral threshold
        mpor_days: Margin Period of Risk
        mta: Minimum Transfer Amount
        """
        self.threshold = threshold
        self.mpor_days = mpor_days
        self.mta = mta
        
    def apply_collateral(self, exposure: np.ndarray, collateral_held: np.ndarray) -> np.ndarray:
        """
        Adjust exposure by Variation Margin collateral held.
        """
        # Uncollateralized exposure is max(exposure - collateral, 0)
        adjusted_exposure = np.maximum(exposure - collateral_held, 0)
        return adjusted_exposure
