"""Generate the figures embedded in README.md.

Every chart is produced by the repository's own engines running on real market
data pulled through ``data.data_connectors.YahooFinanceConnector``.

Usage:
    python scripts/generate_readme_figures.py

Output: docs/images/*.png (150 dpi)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from credit_risk.counterparty import CounterpartyRiskEngine  # noqa: E402
from credit_risk.mitigation import CollateralManager, NettingEngine  # noqa: E402
from data.data_connectors import YahooFinanceConnector  # noqa: E402
from market_risk.backtesting import RiskBacktester  # noqa: E402
from market_risk.data_preprocessor import DataPreprocessor  # noqa: E402
from market_risk.estimators import RiskEngine  # noqa: E402
from market_risk.extreme_value import EVTEngine  # noqa: E402
from market_risk.volatility import GARCHEngine  # noqa: E402
from portfolio_risk.decomposition import RiskDecomposer  # noqa: E402

OUT_DIR = ROOT / "docs" / "images"

TICKERS = ["SPY", "AAPL", "MSFT", "XOM", "JPM"]
START, END = "2018-01-01", "2024-12-31"

# Palette (validated categorical order, light surface)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRIT = "#d03b3b"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


def load_returns() -> pd.DataFrame:
    print(f"Fetching {', '.join(TICKERS)} {START} -> {END} via YahooFinanceConnector ...")
    prices = YahooFinanceConnector.fetch_historical_prices(TICKERS, START, END)
    returns = DataPreprocessor().compute_log_returns(prices)
    print(f"  {returns.shape[0]} observations x {returns.shape[1]} assets")
    return returns


# --------------------------------------------------------------------------
# 1. GARCH volatility + rolling VaR backtest
# --------------------------------------------------------------------------
def fig_garch_backtest(spy: pd.Series) -> None:
    garch = GARCHEngine(p=1, q=1, dist="t")
    garch.fit(spy)
    vol = garch.conditional_volatility()

    engine = RiskEngine(confidence_levels=[0.99])
    var = pd.Series(
        [
            engine.parametric_var_es(mu=0.0, sigma=s, dist="t", df=float(garch.nu))["VaR_0.99"]
            for s in vol
        ],
        index=spy.index,
    )
    metrics = RiskBacktester(confidence_level=0.99).evaluate(spy, var)
    breaches = spy[spy < -var]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(spy.index, spy.values, lw=0.7, color="#b8b7b0", label="SPY log return")
    ax1.plot(var.index, -var.values, lw=2, color=C1, label="99% GARCH-t VaR")
    ax1.scatter(
        breaches.index,
        breaches.values,
        s=22,
        color=CRIT,
        zorder=5,
        edgecolor="#fcfcfb",
        linewidth=0.6,
        label=f"VaR exceptions ({len(breaches)})",
    )
    ax1.set_ylabel("Daily log return")
    ax1.set_title("GARCH(1,1)-t conditional VaR vs realised SPY returns (2018-2024)")
    ax1.legend(loc="lower left", ncol=3, fontsize=9)

    ax2.fill_between(vol.index, vol.values * np.sqrt(252) * 100, color=C1, alpha=0.25)
    ax2.plot(vol.index, vol.values * np.sqrt(252) * 100, lw=1.6, color=C1)
    ax2.set_ylabel("Annualised vol (%)")
    ax2.set_title("Conditional volatility")

    caption = (
        f"omega={garch.omega:.2e}  alpha={garch.alpha:.3f}  beta={garch.beta:.3f}  nu={garch.nu:.1f}\n"
        f"Exceptions {metrics['Exceptions']}/{metrics['Observations']}  |  "
        f"Kupiec p={metrics['Kupiec_p_value']:.3f}  |  "
        f"Christoffersen p={metrics['Christoffersen_p_value']:.3f}  |  "
        f"Basel zone: {metrics['Basel_Zone']}"
    )
    ax2.text(
        0.0, -0.42, caption, transform=ax2.transAxes, fontsize=8.5, color="#52514e", va="top"
    )
    save(fig, "garch_var_backtest.png")


# --------------------------------------------------------------------------
# 2. VaR / ES estimator comparison
# --------------------------------------------------------------------------
def fig_var_methods(spy: pd.Series) -> None:
    engine = RiskEngine(confidence_levels=[0.95, 0.99])
    mu, sigma = float(spy.mean()), float(spy.std())

    methods = {
        "Parametric\n(normal)": engine.parametric_var_es(mu, sigma, dist="normal"),
        "Parametric\n(Student-t)": engine.parametric_var_es(mu, sigma, dist="t", df=5.0),
        "Historical\nsimulation": engine.historical_var_es(spy),
        "Monte Carlo\n(50k GBM paths)": engine.monte_carlo_var_es(
            initial_value=1.0, mu=mu, sigma=sigma, horizon=1, paths=50_000
        ),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    hist = engine.historical_var_es(spy)
    ax1.hist(spy.values * 100, bins=120, color="#c9dcf5", edgecolor="#fcfcfb", linewidth=0.3)
    ax1.axvline(-hist["VaR_0.99"] * 100, color=C1, lw=2, label=f"99% VaR {hist['VaR_0.99']*100:.2f}%")
    ax1.axvline(-hist["ES_0.99"] * 100, color=CRIT, lw=2, ls="--", label=f"99% ES {hist['ES_0.99']*100:.2f}%")
    ax1.set_xlabel("Daily return (%)")
    ax1.set_ylabel("Days")
    ax1.set_title("SPY return distribution with historical VaR / ES")
    ax1.legend(fontsize=9)

    labels = list(methods)
    x = np.arange(len(labels))
    w = 0.2
    series = [
        ("VaR 95%", [methods[m]["VaR_0.95"] * 100 for m in labels], C1),
        ("ES 95%", [methods[m]["ES_0.95"] * 100 for m in labels], C3),
        ("VaR 99%", [methods[m]["VaR_0.99"] * 100 for m in labels], C2),
        ("ES 99%", [methods[m]["ES_0.99"] * 100 for m in labels], C4),
    ]
    for i, (label, vals, color) in enumerate(series):
        bars = ax2.bar(x + (i - 1.5) * w, vals, w * 0.9, label=label, color=color)
        ax2.bar_label(bars, fmt="%.2f", fontsize=7, color="#52514e", padding=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Loss (% of position)")
    ax2.set_title("1-day VaR / ES by estimation method")
    ax2.legend(fontsize=9, ncol=4, loc="upper left")
    ax2.set_ylim(0, max(v for _, vals, _ in series for v in vals) * 1.35)

    save(fig, "var_es_methods.png")


# --------------------------------------------------------------------------
# 3. Extreme value theory (peaks over threshold)
# --------------------------------------------------------------------------
def fig_evt(spy: pd.Series) -> None:
    evt = EVTEngine(threshold_quantile=0.95)
    evt.fit(spy)

    losses = -spy.dropna()
    excesses = np.sort((losses[losses > evt.u] - evt.u).values)
    emp_cdf = np.arange(1, len(excesses) + 1) / (len(excesses) + 1)
    grid = np.linspace(0, excesses.max() * 1.05, 300)
    fitted = genpareto.cdf(grid, evt.xi, loc=0, scale=evt.beta)

    alphas = np.array([0.95, 0.975, 0.99, 0.995, 0.999])
    engine = RiskEngine(confidence_levels=list(alphas))
    hist = engine.historical_var_es(spy)
    evt_var, evt_es = zip(*(evt.estimate_risk(a) for a in alphas))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.step(excesses * 100, emp_cdf, where="post", color=MUTED, lw=1.4, label="Empirical excesses")
    ax1.plot(grid * 100, fitted, color=C1, lw=2.2, label=f"Fitted GPD (xi={evt.xi:.3f}, beta={evt.beta:.4f})")
    ax1.set_xlabel(f"Loss in excess of threshold u = {evt.u*100:.2f}% (%)")
    ax1.set_ylabel("CDF")
    ax1.set_title(f"Peaks-over-threshold fit ({evt.n_excess} exceedances of {evt.n_total} days)")
    ax1.legend(fontsize=9, loc="lower right")

    a = alphas * 100
    ax2.plot(a, np.array(evt_var) * 100, "o-", color=C1, lw=2, ms=7, label="EVT VaR")
    ax2.plot(a, np.array(evt_es) * 100, "o--", color=CRIT, lw=2, ms=7, label="EVT ES")
    ax2.plot(
        a,
        [hist[f"VaR_{al}"] * 100 for al in alphas],
        "s-",
        color=C2,
        lw=2,
        ms=6,
        label="Historical VaR",
    )
    ax2.set_xlabel("Confidence level (%)")
    ax2.set_ylabel("Loss (% of position)")
    ax2.set_title("EVT tail extrapolation vs historical simulation")
    ax2.legend(fontsize=9)

    save(fig, "evt_tail_risk.png")


# --------------------------------------------------------------------------
# 4. Counterparty credit exposure profiles
# --------------------------------------------------------------------------
def fig_credit_exposure(spy: pd.Series) -> None:
    rng = np.random.default_rng(1337)
    sigma_ann = float(spy.std()) * np.sqrt(252)  # vol calibrated on real SPY data

    # The grid step is chosen so a 10-business-day margin period of risk is
    # representable as exactly one step (10/252 = 0.0397y). A coarser grid cannot
    # express the MPOR at all: on the previous 41-point grid the step was 0.125y,
    # so 10 days rounded to zero steps and the old inline `max(1, ...)` silently
    # promoted it to one step, i.e. ~31 business days rather than 10.
    mpor_days = 10
    years, dt_years = 5.0, mpor_days / 252
    n_steps = int(round(years / dt_years)) + 1
    n_paths, n_contracts = 10_000, 3
    grid = np.linspace(0.0, years, n_steps)
    dt = np.diff(grid, prepend=0.0)

    # Three swap-like contracts, driftless MtM diffusions calibrated to SPY vol
    shocks = rng.standard_normal((n_contracts, n_paths, n_steps)) * np.sqrt(dt)
    notionals = np.array([1.0, -0.6, 0.4]).reshape(-1, 1, 1)
    contract_values = notionals * sigma_ann * 100.0 * np.cumsum(shocks, axis=2)

    netted = NettingEngine(netting_enabled=True).aggregate_mtm(contract_values)
    gross = NettingEngine(netting_enabled=False).aggregate_mtm(contract_values)

    # CollateralManager now applies the threshold, the minimum transfer amount
    # and the MPOR lag itself; this used to be reimplemented inline here because
    # those parameters were stored but unused.
    cm = CollateralManager(threshold=25.0, mpor_days=mpor_days, mta=5.0)
    collateralised = cm.apply_collateral(netted, time_grid=grid)

    eng_net = CounterpartyRiskEngine(grid)
    eng_net.set_portfolio_paths(netted)
    prof_net = eng_net.calculate_exposure_profiles(quantile=0.95)

    eng_gross = CounterpartyRiskEngine(grid)
    eng_gross.set_portfolio_paths(gross)
    prof_gross = eng_gross.calculate_exposure_profiles(quantile=0.95)

    eng_coll = CounterpartyRiskEngine(grid)
    eng_coll.set_portfolio_paths(collateralised)
    prof_coll = eng_coll.calculate_exposure_profiles(quantile=0.95)

    hazard, disc_rate = 0.02, 0.03
    pd_curve = 1 - np.exp(-hazard * grid)
    # Discount the exposure. Omitting the curve leaves CVA undiscounted, which
    # overstates it by roughly 8% on this five-year grid.
    df = np.exp(-disc_rate * grid)
    cva_gross = eng_gross.calculate_cva(0.4, pd_curve, df)
    cva_net = eng_net.calculate_cva(0.4, pd_curve, df)
    cva_coll = eng_coll.calculate_cva(0.4, pd_curve, df)
    cva_wwr = eng_net.calculate_cva_wwr(0.4, pd_curve, alpha_wwr=1.4, discount_factors=df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.plot(grid, prof_gross["PFE"], lw=2, ls="--", color=MUTED, label="PFE 95% (no netting)")
    ax1.plot(grid, prof_net["PFE"], lw=2.2, color=C2, label="PFE 95% (netted)")
    ax1.fill_between(grid, prof_net["EE"], color=C1, alpha=0.2)
    ax1.plot(grid, prof_net["EE"], lw=2.2, color=C1, label="EE (netted)")
    ax1.plot(grid, prof_coll["PFE"], lw=2, color=C3, label="PFE 95% (collateralised, MPoR 10d)")
    ax1.axhline(
        prof_net["EPE"][0], color=C4, lw=1.8, ls=":", label=f"EPE = {prof_net['EPE'][0]:.1f}"
    )
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Exposure")
    ax1.set_title("Counterparty exposure profiles under a 3-trade netting set")
    ax1.legend(fontsize=8.5)

    names = ["Gross\n(no netting)", "Netted", "Netted +\ncollateral", "Netted\n+ WWR 1.4x"]
    vals = [cva_gross, cva_net, cva_coll, cva_wwr]
    bars = ax2.bar(names, vals, color=[MUTED, C1, C3, CRIT], width=0.62)
    ax2.bar_label(bars, fmt="%.2f", fontsize=9, color="#52514e", padding=2)
    ax2.set_ylabel("CVA")
    ax2.set_title(
        "CVA impact of risk mitigation (LGD 60%, 2% hazard, 3% discount)"
    )
    ax2.set_ylim(0, max(vals) * 1.2)
    ax2.grid(axis="x", visible=False)

    save(fig, "credit_exposure_cva.png")


# --------------------------------------------------------------------------
# 5. Portfolio risk decomposition
# --------------------------------------------------------------------------
def fig_portfolio(returns: pd.DataFrame) -> None:
    assets = list(returns.columns)
    weights = np.full(len(assets), 1.0 / len(assets))
    cov = returns.cov().values * 252
    corr = returns.corr().values

    dec = RiskDecomposer(weights, cov)
    comp = dec.calculate_component_var(0.99)
    marg = dec.calculate_marginal_var(0.99)
    port_var = float(norm.ppf(0.99) * np.sqrt(weights @ cov @ weights))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    order = np.argsort(comp)[::-1]
    bars = ax1.barh(
        [assets[i] for i in order][::-1], comp[order][::-1] * 100, color=C1, height=0.62
    )
    ax1.bar_label(bars, fmt="%.2f%%", fontsize=9, color="#52514e", padding=3)
    ax1.set_xlabel("Component VaR (annualised, % of portfolio value)")
    ax1.set_title(
        f"99% component VaR decomposition - equal weights\n"
        f"portfolio VaR {port_var*100:.2f}% = sum of components"
    )
    ax1.grid(axis="y", visible=False)
    ax1.set_xlim(0, comp.max() * 100 * 1.25)

    im = ax2.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(assets)), assets)
    ax2.set_yticks(range(len(assets)), assets)
    ax2.grid(visible=False)
    for i in range(len(assets)):
        for j in range(len(assets)):
            ax2.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#fcfcfb" if corr[i, j] > 0.6 else INK,
            )
    ax2.set_title("Return correlation matrix (Gaussian copula input)")
    fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.04)

    print(
        "  marginal VaR: "
        + ", ".join(f"{a}={m*100:.2f}%" for a, m in zip(assets, marg))
    )
    save(fig, "portfolio_decomposition.png")


def main() -> None:
    returns = load_returns()
    spy = returns["SPY"].dropna()

    print("Figure 1/5: GARCH VaR backtest")
    fig_garch_backtest(spy)
    print("Figure 2/5: VaR / ES estimator comparison")
    fig_var_methods(spy)
    print("Figure 3/5: EVT tail risk")
    fig_evt(spy)
    print("Figure 4/5: Counterparty exposure & CVA")
    fig_credit_exposure(spy)
    print("Figure 5/5: Portfolio decomposition")
    fig_portfolio(returns)
    print("Done.")


if __name__ == "__main__":
    main()
