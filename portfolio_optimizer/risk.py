# risk.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit–Wolf shrinkage covariance on (log) returns.
    - Drops rows with any NaNs.
    - Returns a symmetric DataFrame aligned to input columns.
    """
    R = returns.dropna(how="any").astype(float)
    if R.shape[0] < 2 or R.shape[1] < 1:
        # empty / degenerate -> zero matrix
        cols = list(returns.columns)
        return pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    lw = LedoitWolf().fit(R.values)
    cov = pd.DataFrame(lw.covariance_, index=R.columns, columns=R.columns)
    # ensure symmetry (numerical)
    cov = (cov + cov.T) / 2.0
    return cov.reindex(index=returns.columns, columns=returns.columns)

def chol_psd(mat: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """
    Robust Cholesky for PSD matrices:
    - Attempts Cholesky with escalating jitter.
    - Falls back to eigenvalue clamp/sqrt if needed.
    Returns lower-triangular-like factor L such that L @ L.T ≈ mat (PSD).
    """
    A = np.array(mat, dtype=float)
    n = A.shape[0]
    I = np.eye(n)
    jit = float(jitter)
    for _ in range(10):
        try:
            return np.linalg.cholesky((A + A.T) / 2.0 + jit * I)
        except np.linalg.LinAlgError:
            jit *= 10.0
    # Eigen fallback
    vals, vecs = np.linalg.eigh((A + A.T) / 2.0)
    vals[vals < 0] = 0.0
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T