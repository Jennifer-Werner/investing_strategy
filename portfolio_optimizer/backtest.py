# backtest.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from returns import to_log_returns, bayes_stein_shrinkage
from risk import ledoit_wolf_cov
from bl import bl_posterior
from optimizer import optimize_weights

@dataclass
class RunConfig:
    tickers: List[str]
    benchmark: str
    initial_nav: float
    min_weight: float
    max_weight: float
    sector_caps: Dict[str, float]
    default_sector_cap: float
    turnover_cap: float
    te_annual_target: float
    tau: float
    eta_bayes_stein: float
    risk_aversion: float | None
    div_floor_abs: float
    div_slack: float

def semiannual_dates(prices_index: pd.DatetimeIndex, start: str, end: str) -> List[pd.Timestamp]:
    """Last trading day of June and December within [start, end]."""
    idx = prices_index[(prices_index >= start) & (prices_index <= end)]
    if len(idx) == 0:
        return []
    months = [6, 12]
    dates: List[pd.Timestamp] = []
    for y in sorted(set(idx.year)):
        for m in months:
            d = idx[(idx.year == y) & (idx.month == m)]
            if len(d) > 0:
                dates.append(d[-1])
    return [d for d in dates if (d >= idx[0]) and (d <= idx[-1])]

def synth_bonds(prices: pd.DataFrame, yield_annual: float = 0.0425) -> pd.DataFrame:
    """Append synthetic bond price series compounding at yield_annual if 'BONDS' is in universe."""
    if "BONDS" in prices.columns:
        return prices
    idx = prices.index
    daily = 1.0 + yield_annual / 252.0
    bonds_px = pd.Series(100.0 * (daily ** np.arange(len(idx))), index=idx, name="BONDS")
    return prices.join(bonds_px)

def run_backtest(
    prices: pd.DataFrame,
    sectors: pd.Series,
    div_yields: pd.Series,
    market_weights: pd.Series,
    cfg: RunConfig,
    start: str,
    end: str,
    lookback_years: int,
    views_cfg: list[dict],
    green_energy: dict | None,
    holdings0: Optional[pd.Series] = None
) -> dict:
    tickers = cfg.tickers
    bench = cfg.benchmark

    # Ensure prices include benchmark and synth BONDS if needed
    cols_needed = list(set([c for c in tickers if c != bench] + [bench]))
    prices = prices.copy()[[c for c in cols_needed if c in prices.columns]].dropna(how="all")
    prices = synth_bonds(prices)

    # Rebalance dates (semiannual)
    rebal_dates = semiannual_dates(prices.index, start, end)

    weights_hist: List[tuple[pd.Timestamp, pd.Series]] = []
    trades_hist: List[tuple[pd.Timestamp, pd.Series]] = []

    prev_w = holdings0.reindex(tickers).fillna(0.0) if holdings0 is not None else None

    for dt in rebal_dates:
        est_start = dt - pd.offsets.DateOffset(years=lookback_years)
        est_px = prices.loc[(prices.index > est_start) & (prices.index <= dt), [c for c in tickers if c in prices.columns]]
        est_px = est_px.dropna(how="all", axis=1)
        if est_px.shape[0] < 252:
            continue

        # ----- Estimation window for this rebalance date -----
        rets_cols = [c for c in tickers if c in prices.columns]  # defensive
        est_px = prices.loc[(prices.index > est_start) & (prices.index <= dt), rets_cols]
        est_px = est_px.dropna(how="all", axis=1)

        # Require at least 252 obs and at least 2 assets to build Σ
        if (est_px.shape[0] < 252) or (est_px.shape[1] < 2):
            continue

        rets = to_log_returns(est_px).dropna()
        cov = ledoit_wolf_cov(rets)
        mu_hist = rets.mean() * 252.0
        mu_bs = bayes_stein_shrinkage(mu_hist, cov, T=rets.shape[0])

        # Universe available *this* period
        active_all = list(est_px.columns)                     # includes sleeves if present
        active_bl  = [t for t in active_all if t != "BONDS"]  # BL on non-bonds only

        # If still too narrow, skip
        if len(active_bl) < 1:
            continue

        # Per-period sectors & benchmark weights (equal-weight for TE unless you prefer sleeves-aware TE here too)
        sectors_t = sectors.reindex(active_all)
        mkt_w_t   = pd.Series(1.0 / len(active_all), index=active_all)

        # ----- Black–Litterman on the *active* set only -----
        # Use only the submatrix that exists now
        cov_sub = cov.loc[active_bl, active_bl]
        px_sub  = prices.loc[:dt, active_bl + [bench]].dropna(how="all")
        mu_bl, cov_bl = bl_posterior(
            cov_sub,
            px_sub,
            active_bl,
            sectors_t,
            mkt_w_t.reindex(active_bl),
            cfg.risk_aversion,
            bench,
            cfg.tau,
            mu_bs.reindex(active_bl),
            views_cfg,
            green_energy,
        )

        # Build μ for *all* active names; set BONDS proxy if present
        mu_all = pd.Series(index=active_all, dtype=float)
        mu_all.loc[active_bl] = mu_bl.reindex(active_bl).fillna(0.0).values
        if "BONDS" in active_all:
            mu_all.loc["BONDS"] = 0.0425  # 4.25% annual proxy

        # ----- Fixed sleeves and per-asset bounds (only for active names) -----
        fixed_weights = {"BONDS": 0.10, "VOO": 0.15, "SMH": 0.05, "IWF": 0.025, "QQQ": 0.025}
        per_asset_bounds = {}
        for tkr in active_all:
            if tkr in fixed_weights:
                per_asset_bounds[tkr] = (fixed_weights[tkr], fixed_weights[tkr])  # exact if present this period
            else:
                per_asset_bounds[tkr] = (0.0, min(cfg.max_weight, 0.05))          # ≤5% cap for single stock

        # Sector soft targets (“light”)
        sector_targets = {
            "Information Technology": 0.20,
            "Healthcare": 0.13,
            "Renewable Energy": 0.07,
            "Industrials": 0.10,
            "Utilities": 0.03,
            "Consumer Discretionary": 0.10,
            "Communication Services": 0.03,
        }
        sector_target_penalty = 0.5

        # Dividend yields aligned to active set
        y = div_yields.reindex(active_all).fillna(0.0)

        # TE benchmark for this period (equal-weight across active set)
        # (If you prefer sleeves-aware TE, mirror your optimize() bench_w logic here)
        bench_w_t = pd.Series(1.0 / len(active_all), index=active_all)

        # ----- Optimize on the active set -----
        w_star_active = optimize_weights(
            mu=mu_all,
            cov=cov.reindex(index=active_all, columns=active_all),
            benchmark_w=bench_w_t,
            sectors_series=sectors_t,
            prev_w=(prev_w.reindex(active_all).fillna(0.0) if prev_w is not None else None),
            min_w=cfg.min_weight,
            max_w=cfg.max_weight,
            sector_caps=cfg.sector_caps,
            default_sector_cap=cfg.default_sector_cap,
            turnover_cap=cfg.turnover_cap,
            te_annual_target=cfg.te_annual_target,
            div_yields=y,
            initial_nav=cfg.initial_nav,
            div_floor_abs=cfg.div_floor_abs,
            div_slack=cfg.div_slack,
            risk_aversion=cfg.risk_aversion,
            per_asset_bounds=per_asset_bounds,
            fixed_weights={k: v for k, v in fixed_weights.items() if k in active_all},
            sector_targets=sector_targets,
            sector_target_penalty=sector_target_penalty,
        )

        # Expand to full cfg.tickers index with zeros for missing names
        w_star = pd.Series(0.0, index=tickers, dtype=float)
        w_star.loc[w_star_active.index] = w_star_active.values

        weights_hist.append((dt, w_star))
        if prev_w is not None:
            trades_hist.append((dt, (w_star - prev_w).reindex(w_star.index).fillna(0.0)))
        prev_w = w_star

    weights_df = pd.DataFrame({d: w for d, w in weights_hist}).T if weights_hist else pd.DataFrame()
    trades_df = pd.DataFrame({d: t for d, t in trades_hist}).T if trades_hist else pd.DataFrame()
    return {"weights": weights_df, "trades": trades_df}