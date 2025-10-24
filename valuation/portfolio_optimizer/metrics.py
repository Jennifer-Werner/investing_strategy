# metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

BOND_YIELD_ANNUAL = 0.0425  # 4.25% synthetic bond sleeve

def _ensure_bonds_in_prices(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """If weights contain BONDS but prices don't, synthesize a bond price path."""
    if "BONDS" in getattr(weights, "columns", []) and "BONDS" not in prices.columns:
        idx = prices.index
        daily = 1.0 + BOND_YIELD_ANNUAL / 252.0
        bonds_px = pd.Series(100.0 * (daily ** np.arange(len(idx))), index=idx, name="BONDS")
        prices = prices.join(bonds_px)
    return prices

def portfolio_returns(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    Reconstruct daily portfolio returns:
      - Forward-fill semiannual weights to daily
      - Use daily percent changes on prices
      - Supports synthetic 'BONDS' sleeve
    """
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    px = prices.copy().sort_index()
    px = _ensure_bonds_in_prices(weights, px)
    # Align columns present in both weights and prices
    cols = [c for c in weights.columns if c in px.columns]
    if not cols:
        return pd.Series(dtype=float)
    w_daily = weights[cols].reindex(px.index).ffill().dropna(how="all")
    rets = px[cols].pct_change().fillna(0.0)
    return (w_daily * rets).sum(axis=1)

def ann_return(r: pd.Series, periods: int = 252) -> float:
    return float((1 + r).prod() ** (periods / max(len(r), 1)) - 1) if len(r) > 0 else float("nan")

def ann_vol(r: pd.Series, periods: int = 252) -> float:
    return float(r.std() * np.sqrt(periods)) if len(r) > 1 else float("nan")

def max_drawdown(r: pd.Series) -> float:
    if len(r) == 0:
        return float("nan")
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return float((cum / peak - 1.0).min())

def sharpe(r: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    ex = r - rf / periods
    return float(ex.mean() / ex.std() * np.sqrt(periods))

def sortino(r: pd.Series, target: float = 0.0, periods: int = 252) -> float:
    if len(r) < 2:
        return float("nan")
    ex = r - target / periods
    downside = ex[ex < 0]
    if downside.std() == 0 or np.isnan(downside.std()):
        return float("nan")
    return float((ex.mean() * periods) / (downside.std() * np.sqrt(periods)))

def omega(r: pd.Series, theta: float = 0.0, periods: int = 252) -> float:
    if len(r) == 0:
        return float("nan")
    thr = theta / periods
    gains = (r - thr).clip(lower=0).sum()
    losses = (thr - r).clip(lower=0).sum()
    return float(gains / losses) if losses > 0 else float("inf")

def deflated_sharpe_ratio(r: pd.Series, trials: int = 1) -> float:
    """
    Approximate (probabilistic) Sharpe deflation. This is a compact proxy suitable for reporting.
    For full DSR as in Bailey et al., pass a reasonable 'trials' (>1) to reflect model selection.

    Returns a value in [0,1] that can be read as a (rough) probability the Sharpe is not spurious.
    """
    if len(r) < 3 or r.std() == 0:
        return float("nan")
    s = sharpe(r)  # annualized
    # Convert to daily Sharpe to scale by sqrt(T)
    s_daily = s / np.sqrt(252.0)
    T = len(r)
    # Simple deflation by sqrt(log(trials)) in z-space (rough, conservative)
    from math import erf, sqrt, log
    z = s_daily * np.sqrt(T - 1.0)
    z_deflated = max(0.0, z - np.sqrt(2.0 * max(1.0, log(max(1, trials)))))
    # Map to [0,1]
    return float(0.5 * (1.0 + erf(z_deflated / np.sqrt(2.0))))