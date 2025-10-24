# bl.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from pypfopt import black_litterman

def market_implied_prior(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: Optional[float],
    benchmark_prices: pd.Series,
) -> pd.Series:
    """
    Build BL market-implied prior Pi using either a provided risk aversion (delta)
    or infer delta from the benchmark's simple historical mean/variance.
    """
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

    # Use market weights as pseudo "caps" (OK for BL prior construction)
    caps = market_weights.copy()
    # Avoid zeros/negatives; scale so smallest becomes 1
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
    green_energy: Optional[Dict] = None
) -> Tuple[Dict[str, float], List[float]]:
    """
    Convert sector-level tilts into per-asset absolute views and confidences.
    - Supports v = {"type":"sector","sector":..., "tilt":..., "confidence":...}
      and v = {"type":"asset","ticker":..., "tilt":..., "confidence":...}
    """
    abs_views: Dict[str, float] = {}
    confs: List[float] = []

    # Pre-tag green energy tickers if provided
    green_tickers = set()
    if green_energy:
        green_tickers |= set([t.upper() for t in green_energy.get("tickers", [])])

    for v in (views_cfg or []):
        vtype = v.get("type")
        if vtype == "sector":
            sector = v["sector"]
            tilt = float(v["tilt"])
            conf = float(v.get("confidence", 0.5))
            if sector.lower().startswith("green"):
                names = [t for t in tickers if (t in green_tickers) or ("NEE" in t) or ("ENPH" in t)]
            else:
                names = [t for t in tickers if (sectors.get(t) == sector)]
            if not names:
                continue
            for t in names:
                abs_views[t] = abs_views.get(t, 0.0) + tilt
                confs.append(conf)
        elif vtype == "asset":
            t = v["ticker"].upper()
            tilt = float(v["tilt"])
            conf = float(v.get("confidence", 0.5))
            if t in tickers:
                abs_views[t] = abs_views.get(t, 0.0) + tilt
                confs.append(conf)

    return abs_views, confs

def bl_posterior(
    cov: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: List[str],
    sectors: pd.Series,
    market_weights: pd.Series,
    risk_aversion: Optional[float],
    benchmark: str,
    tau: float,
    mu_bs: pd.Series,
    views_cfg: List[Dict],
    green_energy: Optional[Dict]
):
    """
    Build BL posterior mean/covariance for the provided tickers.
    Notes:
      - 'tickers' should exclude any synthetic assets (e.g., BONDS).
      - 'prices' must contain the benchmark column.
    """
    Pi = market_implied_prior(
        cov.loc[tickers, tickers],
        market_weights.loc[tickers],
        risk_aversion,
        prices[benchmark],
    )

    # Blend market-implied prior with Bayes–Stein shrinkage mean
    eta = 0.25  # blend weight toward Bayes–Stein
    mu_prior = (1.0 - eta) * Pi.loc[tickers] + eta * mu_bs.loc[tickers]

    # Expand sector/asset views to absolute views
    abs_views, confs = expand_sector_views(views_cfg, sectors, tickers, green_energy)

    # Build BL model
    bl = black_litterman.BlackLittermanModel(
        cov_matrix=cov.loc[tickers, tickers],
        pi=mu_prior.loc[tickers],
        tau=tau,
        absolute_views=abs_views if abs_views else None,
        view_confidences=confs if confs else None,
        omega=None,
    )
    mu_bl = bl.bl_returns()
    cov_bl = bl.bl_cov()
    return mu_bl, cov_bl