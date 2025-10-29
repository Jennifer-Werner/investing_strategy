# returns.py
from __future__ import annotations
import numpy as np
import pandas as pd

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from adjusted prices.
    Drops rows where all assets are NaN. Guards against non-positive prices.
    """
    px = prices.replace([np.inf, -np.inf], np.nan).copy()
    # Avoid log of non-positive values
    px[px <= 0] = np.nan
    lr = np.log(px).diff()
    return lr.dropna(how="all")

def annualize_mean(mu_per_period: pd.Series, periods: int = 252) -> pd.Series:
    """
    Scale a per-period mean vector to annualized units via simple linear scaling.
    Use when mu_per_period is already a mean of returns per period (not compounding log-mean).
    """
    return mu_per_period.astype(float) * float(periods)

def annualize_cov(cov_per_period: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """
    Scale a per-period covariance matrix to annualized units.
    """
    return cov_per_period.astype(float) * float(periods)

def bayes_stein_shrinkage(
    mu: pd.Series,
    cov: pd.DataFrame,
    T: int,
    floor: float = 0.0,
    ceil: float = 1.0
) -> pd.Series:
    """
    Empirical Bayes (James–Stein) shrinkage toward the cross-sectional mean (Jorion-style).

    Parameters
    ----------
    mu : pd.Series
        Sample mean vector (units must match cov/T usage; e.g., if mu is annualized,
        cov should be annualized too, or both in daily units with consistent T).
    cov : pd.DataFrame
        Covariance matrix in the same units as mu (see note above).
    T : int
        Sample size (number of return observations used to estimate mu/cov).
    floor, ceil : float
        Clamp the shrinkage intensity phi into [floor, ceil].

    Returns
    -------
    pd.Series
        Shrunk mean vector in the same units as mu.
    """
    mu = mu.astype(float)
    cov = cov.astype(float)
    n = len(mu)
    if T <= 1 or n <= 2:
        return mu

    m = float(mu.mean())
    diff = mu - m

    # Average variance of the mean estimator across assets
    var_mean = float(np.trace(cov.values)) / float(n) / float(T)

    denom = float(np.dot(diff.values, diff.values))
    if denom <= 1e-16:
        phi = 1.0
    else:
        phi = ((n - 2.0) * var_mean) / denom

    phi = float(np.clip(phi, floor, ceil))
    mu_bs = m + (1.0 - phi) * diff
    return mu_bs