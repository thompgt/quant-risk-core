import warnings

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from arch import arch_model
import statsmodels.api as sm

VALID_MODEL_TYPES = ('Garch', 'EGARCH')


class GARCHNotConvergedError(RuntimeError):
    """Raised when a volatility fit fails to converge and strict=True."""


class GARCHEngine:
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = 'normal',
        strict: bool = False,
        o: Optional[int] = None,
    ):
        """
        GARCH(p, q) conditional variance engine.
        dist: 'normal' or 't' (Student-t)
        strict: raise GARCHNotConvergedError instead of warning when the
            optimiser fails to converge. Production callers should prefer True:
            a non-converged fit still yields parameters, and every downstream VaR
            built on them looks perfectly well-formed.
        o: asymmetric (leverage) order. `arch` puts the EGARCH leverage term
            under `o`, not under `p`, and defaults it to 0. The previous
            implementation never passed it, so `gamma` — documented as the
            leverage effect — read a `gamma[1]` parameter that the fit had not
            estimated and was therefore *always* 0.0: the asymmetry EGARCH exists
            to capture was silently never modelled. Defaults to 1 for EGARCH and
            0 for GARCH; pass explicitly to override.
        """
        if p < 1 or q < 1:
            raise ValueError("p and q must be at least 1.")
        if o is not None and o < 0:
            raise ValueError("o must be non-negative.")
        self.p = p
        self.q = q
        self.o = o
        if dist not in ['normal', 't']:
            raise ValueError("dist must be 'normal' or 't'")
        self.dist = dist
        self.strict = strict

        self.model_fit = None
        self.params: Optional[pd.Series] = None
        self.omega = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0 # Leverage effect (EGARCH only)
        self.nu = None # Degrees of freedom for t dist

        # Full parameter vectors, so p or q above 1 is not silently truncated.
        self.alpha_params: np.ndarray = np.array([])
        self.beta_params: np.ndarray = np.array([])
        self.gamma_params: np.ndarray = np.array([])
        self._fitted_o: int = 0

        # Convergence diagnostics.
        self.converged: Optional[bool] = None
        self.convergence_flag: Optional[int] = None
        self.convergence_message: Optional[str] = None

    def fit(self, returns: pd.Series, model_type: str = 'Garch') -> None:
        """
        Fits the specified volatility model.
        model_type: 'Garch' or 'EGARCH'

        Convergence is checked and reported rather than discarded. The previous
        implementation wrapped the fit in ``warnings.simplefilter('ignore')``,
        which swallowed `arch`'s own ConvergenceWarning, and then read the
        parameters regardless. That is not a cosmetic issue: on 1000 draws from
        the legacy global RNG at seed 42 — the data the existing test suite uses
        — the optimiser returns code 4 ("Inequality constraints incompatible")
        and fits alpha[1] = 6.3e-08, an essentially degenerate model. Every
        conditional volatility and VaR derived from it is well-formed and wrong,
        and the old test's ``all(forecast > 0)`` assertion passed happily.

        After fitting, inspect `converged`, `convergence_flag` and
        `convergence_message`, or construct with `strict=True` to make failure
        fatal.
        """
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {VALID_MODEL_TYPES}, got {model_type!r}."
            )

        returns = pd.Series(returns).dropna()
        if returns.empty:
            raise ValueError("returns contains no observations.")
        if returns.var() == 0:
            raise ValueError(
                "returns has zero variance; there is no volatility to model."
            )

        # EGARCH's leverage term lives under `o`; without it the asymmetry is
        # simply not estimated and gamma stays at zero.
        o = self.o if self.o is not None else (1 if model_type == 'EGARCH' else 0)

        # arch_model expects returns to have zero mean if mean='Zero'
        am = arch_model(
            returns,
            vol=model_type,
            p=self.p,
            o=o,
            q=self.q,
            dist=self.dist,
            mean='Zero',
            rescale=False,
        )
        self._fitted_o = o

        # arch's ConvergenceWarning is captured rather than ignored, so it can be
        # re-raised with the parameters that came out of the failed fit attached.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.model_fit = am.fit(disp='off', update_freq=0)

        self.params = params = self.model_fit.params

        self.omega = float(params.get('omega', 0.0))
        self.alpha_params = self._collect(params, 'alpha')
        self.beta_params = self._collect(params, 'beta')

        # Scalar attributes remain the first lag, for backward compatibility.
        self.alpha = float(self.alpha_params[0]) if self.alpha_params.size else 0.0
        self.beta = float(self.beta_params[0]) if self.beta_params.size else 0.0

        self.gamma_params = self._collect(params, 'gamma')
        self.gamma = float(self.gamma_params[0]) if self.gamma_params.size else 0.0

        # Reported after the parameters are populated, so a non-convergence
        # message can quote the values the failed fit actually produced.
        self._record_convergence(caught)

        if self._fitted_o > 0 and self.gamma_params.size == 0:
            warnings.warn(
                f"Requested asymmetric order o={self._fitted_o} but the fit "
                "produced no gamma term; the leverage effect is not being "
                "estimated.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.alpha_params.size > 1 or self.beta_params.size > 1:
            warnings.warn(
                f"Fitted a {model_type}({self.p}, {self.q}) model, but the "
                "scalar `alpha` and `beta` attributes hold only the first lag. "
                "Use `alpha_params`, `beta_params` or `params` for the full "
                "parameter vector.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.dist == 't':
            self.nu = float(params.get('nu', 5.0))

    @staticmethod
    def _collect(params: pd.Series, prefix: str) -> np.ndarray:
        """Gather prefix[1], prefix[2], ... in lag order."""
        names = [n for n in params.index if n.startswith(f'{prefix}[')]
        names.sort(key=lambda n: int(n.split('[')[1].rstrip(']')))
        return np.array([float(params[n]) for n in names])

    def _record_convergence(self, caught) -> None:
        """Store optimiser status and surface a failure."""
        opt = getattr(self.model_fit, 'optimization_result', None)
        flag = getattr(self.model_fit, 'convergence_flag', None)

        self.convergence_flag = int(flag) if flag is not None else None
        self.convergence_message = (
            str(getattr(opt, 'message', '')) if opt is not None else None
        )

        if opt is not None and getattr(opt, 'success', None) is not None:
            self.converged = bool(opt.success)
        elif self.convergence_flag is not None:
            self.converged = self.convergence_flag == 0
        else:
            self.converged = None

        if self.converged is False:
            detail = (
                f"optimiser status {self.convergence_flag}: "
                f"{self.convergence_message}"
            )
            message = (
                f"Volatility fit did not converge ({detail}). The fitted "
                f"parameters are still populated (omega={self.omega:.4g}, "
                f"alpha={self.alpha:.4g}, beta={self.beta:.4g}) and any "
                "volatility or VaR derived from them will look well-formed but "
                "should not be trusted. Construct with strict=True to make this "
                "an error."
            )
            if self.strict:
                raise GARCHNotConvergedError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        else:
            # Re-emit anything else arch reported, which the old blanket
            # suppression discarded (data-scaling advice, for instance).
            for w in caught:
                warnings.warn(w.message, w.category, stacklevel=3)

    @property
    def persistence(self) -> float:
        """
        Sum of the ARCH and GARCH coefficients.

        For a GARCH model this must be below 1 for the variance process to be
        stationary; at or above 1 the unconditional variance does not exist and
        long-horizon forecasts diverge.
        """
        return float(self.alpha_params.sum() + self.beta_params.sum())

    def _require_fit(self) -> None:
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting.")

    def forecast_volatility(self, horizon: int) -> np.ndarray:
        """
        Out-of-sample forecasting function across forward horizon H.
        Returns array of length H of forecasted volatilities.
        """
        self._require_fit()
        if horizon < 1:
            raise ValueError("horizon must be a positive number of steps.")

        forecasts = self.model_fit.forecast(horizon=horizon, reindex=False)
        var_forecasts = forecasts.variance.iloc[-1].values

        return np.sqrt(var_forecasts)

    def conditional_volatility(self) -> pd.Series:
        """
        Returns the in-sample conditional volatility.
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted first.")
        return self.model_fit.conditional_volatility


class RegimeSwitchingEngine:
    def __init__(self, k_regimes: int = 2):
        if k_regimes < 2:
            raise ValueError("k_regimes must be at least 2.")
        self.k = k_regimes
        self.model = None
        self.results = None

    def fit(self, returns: pd.Series) -> None:
        """
        Fit a Markov Switching Dynamic Regression model.

        Note that regime *labels* are not identified: which fitted regime is
        "high volatility" can swap between runs or between datasets. Order the
        regimes by fitted variance before interpreting them.
        """
        returns = pd.Series(returns).dropna()
        if returns.empty:
            raise ValueError("returns contains no observations.")

        self.model = sm.tsa.MarkovRegression(
            returns, k_regimes=self.k, trend='c', switching_variance=True
        )
        self.results = self.model.fit()

    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Return the smoothed probabilities of being in each regime.
        """
        if self.results is None:
            raise ValueError("Model must be fitted first.")
        return self.results.smoothed_marginal_probabilities
