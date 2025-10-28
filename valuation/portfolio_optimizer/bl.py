# bl.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from pypfopt import black_litterman

# --- small helper: sector-name canonicalization (aligns with run_pipeline) ---
def _canon_sector(name: Optional[str]) -> str:
    if name is None:
        return "Unknown"
    m = {
        "Information Technology": "Information Technology",
        "Technology": "Information Technology",
        "Info Tech": "Information Technology",
        "Health Care": "Healthcare",
        "Healthcare": "Healthcare",
        "Communication": "Communication Services",
        "Comm Services": "Communication Services",
        "Communication Services": "Communication Services",
        "Consumer Discretionary": "Consumer Discretionary",
        "Consumer Cyclical": "Consumer Discretionary",
        "Industrials": "Industrials",
        "Utilities": "Utilities",
        "Renewable Energy": "Renewable Energy",
        "ETF": "ETF",
        "Bonds": "Bonds",
    }
    # exact hit
    if name in m:
        return m[name]
    # try a few fuzzy startswith
    low = name.lower()
    if low.startswith("health"):
        return "Healthcare"
    if low.startswith("tech"):
        return "Information Technology"
    if low.startswith("comm"):
        return "Communication Services"
    if "cyc" in low:  # “Consumer Cyclical”
        return "Consumer Discretionary"
    return name

def market_implied_prior(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: Optional[float],
    benchmark_prices: pd.Series,
) -> pd.Series:
    """Build BL market-implied prior Pi."""
    if risk_aversion is None:
        bm = benchmark_prices.dropna()
        if len(bm) >= 252:
            ret = bm.pct_change().dropna()
            mu = ret.mean() * 252.0
            var = ret.var() * 252.0
            delta = float(mu / var) if var > 0 else 2.5
        else:
            delta = 2.5
    else:
        delta = float(risk_aversion)

    # Use market weights as pseudo "caps"
    caps = market_weights.copy()
    if (caps <= 0).any():
        caps = (caps - caps.min()) + 1e-6
    caps = caps / caps.min()

    Pi = black_litterman.market_implied_prior_returns(
        market_caps=caps.loc[cov.index],
        cov_matrix=cov,
        risk_aversion=delta,
    )
    return Pi

def expand_sector_views(
    views_cfg: List[Dict],
    sectors: pd.Series,
    tickers: List[str],
    mu_prior: pd.Series,               # correct position: mu_prior
    green_energy: Optional[Dict] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Convert views to per-asset absolute-return views and confidences.
    Returns:
      abs_views: dict[ticker] -> absolute mu view (annual, decimal)
      conf_by:  dict[ticker] -> confidence in [0,1]
    """
    abs_views: Dict[str, float] = {}
    conf_by: Dict[str, float] = {}

    # normalize sector labels once
    sec_norm = sectors.copy()
    sec_norm = sec_norm.apply(_canon_sector)

    # Pre-tag green tickers if provided
    green_tickers = set()
    if green_energy:
        green_tickers |= set([t.upper() for t in green_energy.get("tickers", [])])

    for v in (views_cfg or []):
        vtype = (v.get("type") or "").lower()

        # Absolute asset view
        if vtype == "absolute":
            t = (v.get("asset") or v.get("ticker") or "").upper()
            if t in tickers and pd.notna(mu_prior.get(t)):
                mu_abs = float(v["mu"])
                conf   = float(v.get("conf", v.get("confidence", 0.5)))
                if (t not in abs_views) or (conf > conf_by.get(t, 0.0)):
                    abs_views[t] = mu_abs
                    conf_by[t]   = conf
            continue

        # Per-asset tilt relative to prior
        if vtype == "asset":
            t = (v.get("ticker") or "").upper()
            if t in tickers and pd.notna(mu_prior.get(t)):
                tilt  = float(v.get("tilt", 0.0))
                mu_abs = float(mu_prior[t]) + tilt
                conf   = float(v.get("confidence", 0.5))
                if (t not in abs_views) or (conf > conf_by.get(t, 0.0)):
                    abs_views[t] = mu_abs
                    conf_by[t]   = conf
            continue

        # Sector tilt: add tilt to every name in that sector
        if vtype == "sector":
            sector_raw = str(v["sector"])
            sector = _canon_sector(sector_raw)
            tilt   = float(v.get("tilt", 0.0))
            conf   = float(v.get("confidence", 0.5))

            if sector.lower().startswith("green"):
                names = [t for t in tickers if (t in green_tickers) or ("NEE" in t) or ("ENPH" in t)]
            else:
                names = [t for t in tickers if _canon_sector(sec_norm.get(t)) == sector]

            for t in names:
                if t not in tickers or pd.isna(mu_prior.get(t)):
                    continue
                mu_abs = float(mu_prior[t]) + tilt
                if (t not in abs_views) or (conf > conf_by.get(t, 0.0)):
                    abs_views[t] = mu_abs
                    conf_by[t]   = conf

    return abs_views, conf_by

# bl.py (patch)

def bl_posterior(
    cov: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list[str],
    sectors: pd.Series,
    market_weights: pd.Series,
    risk_aversion: float | None,
    benchmark: str,
    tau: float,
    mu_bs: pd.Series,
    views_cfg: list[dict],
    green_energy: dict | None,
    eta_bayes_stein: float = 0.25,       # <-- NEW: pass from cfg
    omega_scale: float = 1.0,            # <-- NEW: tunable scale for Ω
):
    # 1) Prior (market implied Pi) then blend toward Bayes–Stein by eta
    Pi = market_implied_prior(
        cov.loc[tickers, tickers],
        market_weights.loc[tickers],
        risk_aversion,
        prices[benchmark],
    )

    eta = float(eta_bayes_stein)
    mu_prior = (1.0 - eta) * Pi.loc[tickers] + eta * mu_bs.loc[tickers]

    # 2) Expand sector/asset views into absolute μ views per asset
    abs_views, conf_by = expand_sector_views(
        views_cfg, sectors, tickers, mu_prior, green_energy  # NOTE: mu_prior here
    )

    # 3) Build P, q, Ω from absolute views (ignore relative for now)
    idx_map = {t: i for i, t in enumerate(tickers)}
    P_rows, q_list, conf_list = [], [], []
    for t, mu_v in abs_views.items():
        if t in idx_map:
            row = np.zeros(len(tickers))
            row[idx_map[t]] = 1.0
            P_rows.append(row)
            q_list.append(float(mu_v))
            conf_list.append(float(conf_by.get(t, 0.5)))

    if P_rows:
        P = np.vstack(P_rows)
        q = np.array(q_list)

        # Confidence → Ω mapping (smaller Ω = stronger view)
        # Use the BL natural scale S_view = P (tau Σ) P^T
        Sigma_prior = cov.loc[tickers, tickers].values
        S_view = P @ (tau * Sigma_prior) @ P.T
        eps = 1e-9
        diag = []
        for i, c in enumerate(conf_list):
            c = min(max(c, 0.01), 0.99)
            base = float(S_view[i, i]) if S_view[i, i] > eps else 1.0
            strength = (1.0 - c) / c
            diag.append(omega_scale * strength * base)
        Omega = np.diag(diag)

        bl = black_litterman.BlackLittermanModel(
            cov_matrix=cov.loc[tickers, tickers],
            pi=mu_prior.loc[tickers],
            tau=tau,
            P=P,
            Q=q,
            omega=Omega,
        )
    else:
        # no views found → just return the prior
        bl = black_litterman.BlackLittermanModel(
            cov_matrix=cov.loc[tickers, tickers],
            pi=mu_prior.loc[tickers],
            tau=tau,
        )

    mu_bl = bl.bl_returns()
    cov_bl = bl.bl_cov()
    return mu_bl, cov_bl