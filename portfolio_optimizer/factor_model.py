
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def load_factor_exposures(path: str) -> pd.DataFrame:
    """
    Expected CSV format: columns = ['ticker','value','quality','profitability','momentum']
    Index by ticker, lowercase factor names.
    """
    df = pd.read_csv(path)
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df = df.set_index('ticker')
    df.columns = [c.lower() for c in df.columns]
    return df

def load_factor_cov_from_csv(path: str) -> pd.DataFrame:
    """CSV containing a covariance matrix with header and index of factor names."""
    df = pd.read_csv(path, index_col=0)
    df.columns = [c.lower() for c in df.columns]
    df.index = [i.lower() for i in df.index]
    return df

def load_ff5_cov_from_excel(path: str, annualize: bool = True) -> pd.DataFrame:
    """
    Loads Fama-French 5 factor + RF monthly data and returns a covariance matrix
    for selected factors (value=HML, profitability=RMW, quality≈(RMW+CMA)/2, momentum not present).
    The Excel is expected in the canonical Kenneth French format.
    """
    xls = pd.read_excel(path, sheet_name=0, skiprows=3)
    xls.columns = [str(c).strip().lower() for c in xls.columns]
    cols = [c for c in xls.columns if c in ['mkt-rf','smb','hml','rmw','cma','rf']]
    df = xls[cols].dropna()
    # convert to decimal returns if percentages
    if df.abs().mean().mean() > 0.3:  # likely in percent
        df = df / 100.0
    cov_m = df[['hml','rmw','cma']].cov()  # monthly cov
    factors = ['value','quality','profitability','momentum']
    cov_full = pd.DataFrame(np.nan, index=factors, columns=factors, dtype=float)
    # Linear transform from [hml, rmw, cma] to our factors
    A = np.array([
        [1.0, 0.0, 0.0],   # value
        [0.0, 0.5, 0.5],   # quality proxy
        [0.0, 1.0, 0.0],   # profitability
        [0.0, 0.0, 0.0],   # momentum (not present)
    ])
    cov3 = cov_m.values
    cov4 = A @ cov3 @ A.T
    cov_full.loc[:, :] = cov4
    if annualize:
        cov_full *= 12.0
    return cov_full

def active_exposures(B: pd.DataFrame, w: pd.Series, w_b: pd.Series) -> pd.Series:
    """Compute active factor exposures: B^T (w - w_b)."""
    tickers = [t for t in B.index if t in w.index]
    b = B.loc[tickers]
    a = (w.reindex(tickers).fillna(0.0) - w_b.reindex(tickers).fillna(0.0)).values
    e = b.T.values @ a
    return pd.Series(e, index=b.columns)

def factor_risk_contributions(e: pd.Series, F: pd.DataFrame) -> pd.DataFrame:
    """Return exposures, marginal contributions, absolute contributions and share per factor."""
    F = F.reindex(index=e.index, columns=e.index)
    Fe = F.values @ e.values
    rc = e.values * Fe
    var = float(e.values.T @ Fe)
    shares = rc / var if var > 0 else np.zeros_like(rc)
    df = pd.DataFrame({
        'exposure': e.values,
        'marginal': Fe,
        'contribution': rc,
        'share': shares
    }, index=e.index)
    df.attrs['total_variance'] = var
    return df

def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s*0.0
    return (s - mu) / sd

def load_scores_to_exposures(path: str, prices: pd.DataFrame|None=None, asof: str|None=None) -> pd.DataFrame:
    """
    Convert scores to factor exposures by cross-sectional z-scoring.
    Expected columns: ticker, quality_score, profitability_score, value_score, composite_score (optional).
    Momentum exposure will be derived from prices (12-1) if provided; else 0.
    """
    df = pd.read_csv(path)
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df = df.set_index('ticker')
    cols = {c.lower():c for c in df.columns}
    out = pd.DataFrame(index=df.index, columns=['value','quality','profitability','momentum'], dtype=float)
    # Map scores -> exposures via z-score
    if 'value_score' in cols:
        out['value'] = _zscore(df[cols['value_score']])
    elif 'composite_score' in cols:
        # fallback: use composite for value
        out['value'] = _zscore(df[cols['composite_score']])
    else:
        out['value'] = 0.0
    if 'quality_score' in cols:
        out['quality'] = _zscore(df[cols['quality_score']])
    else:
        out['quality'] = 0.0
    if 'profitability_score' in cols:
        out['profitability'] = _zscore(df[cols['profitability_score']])
    else:
        out['profitability'] = 0.0
    # Momentum from prices: 12-1 total return z-score
    mom = pd.Series(0.0, index=out.index)
    if prices is not None and len(prices.index)>260:
        asof_dt = pd.to_datetime(asof) if asof else prices.index[-1]
        px = prices.reindex(columns=[t for t in out.index if t in prices.columns]).loc[:asof_dt].dropna(how='all')
        if px.shape[0] >= 260:
            r12 = px.iloc[-21:].mean()  # placeholder if insufficient history
            try:
                ret_12m = px.iloc[-21] / px.iloc[-252] - 1.0
                ret_skip1 = px.iloc[-1] / px.iloc[-21] - 1.0
                # classical 12-1 momentum approximated by price_{t-1m} / price_{t-12m} - 1
                mom_raw = (px.iloc[-21] / px.iloc[-252] - 1.0)
            except Exception:
                mom_raw = px.pct_change(252).iloc[-1]
            mom = _zscore(mom_raw).reindex(out.index).fillna(0.0)
    out['momentum'] = mom
    return out.fillna(0.0)
