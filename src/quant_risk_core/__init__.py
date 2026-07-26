"""Quant Risk Core — market, credit and portfolio risk engines.

The subpackages are deliberately *not* imported here. ``arch`` and
``statsmodels`` each cost hundreds of milliseconds to import, and a caller that
only wants ``quant_risk_core.__version__`` (the service's ``/version`` endpoint,
for instance) should not pay for the whole econometrics stack. Import the engine
you need directly::

    from quant_risk_core.market_risk.estimators import RiskEngine
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("quant-risk-core")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
