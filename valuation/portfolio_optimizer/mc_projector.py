# mc_projector.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- IO ----------
def _read_mu(mu_csv: str) -> pd.Series:
    df = pd.read_csv(mu_csv)
    if "asset" in df.columns and "mu" in df.columns:
        s = df.set_index("asset")["mu"]
    else:
        s = df.set_index(df.columns[0]).iloc[:, 0]
        s.name = "mu"
    return s.astype(float)

def _read_cov(cov_csv: str) -> pd.DataFrame:
    cov = pd.read_csv(cov_csv, index_col=0).astype(float)
    return (cov + cov.T) / 2.0

def _read_weights(weights_csv: str | None) -> pd.Series | None:
    if not weights_csv:
        return None
    df = pd.read_csv(weights_csv)
    if {"asset","weight"} <= set(df.columns):
        w = df.set_index("asset")["weight"].astype(float)
    else:
        w = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
        w.name = "weight"
    w = w.clip(lower=0)
    return w / w.sum() if w.sum() > 0 else w

def _read_returns_cache(path: str | None) -> pd.Series | None:
    if not path:
        return None
    s = pd.read_parquet(path)
    if isinstance(s, pd.DataFrame):
        if "Strategy" in s.columns:
            s = s["Strategy"]
        else:
            s = s.iloc[:, 0]
    s = pd.Series(s).astype(float).sort_index()
    s.name = "Strategy"
    return s

# ---------- align/build ----------
def _align_assets(mu: pd.Series, cov: pd.DataFrame, weights: pd.Series | None):
    common = mu.index.intersection(cov.index).intersection(cov.columns)
    if weights is not None:
        common = common.intersection(weights.index)
    mu = mu.loc[common]
    cov = cov.loc[common, common]
    if weights is not None:
        weights = weights.loc[common]
    return mu, cov, weights

def _build_weights(mu: pd.Series, weights: pd.Series | None) -> pd.Series:
    if weights is not None and weights.sum() > 0:
        return weights / weights.sum()
    fixed = {"BONDS": 0.10, "VOO": 0.15, "SMH": 0.05, "IWF": 0.025, "QQQ": 0.025}
    w = pd.Series(0.0, index=mu.index)
    fixed_sum = 0.0
    for t, v in fixed.items():
        if t in w.index:
            w.loc[t] = float(v); fixed_sum += float(v)
    residual = max(0.0, 1.0 - fixed_sum)
    others = [t for t in w.index if t not in fixed]
    if residual > 0 and others:
        w.loc[others] = residual / len(others)
    return w / w.sum() if w.sum() > 0 else w

# ---------- PMPT (match report.py) ----------
def _downside_std_negative_only(r_daily: np.ndarray) -> np.ndarray:
    """
    r_daily: (batch, days), returns std over r<0 only, ddof=1, NaN if no negatives.
    """
    mask = r_daily < 0.0
    # Use NaN-masked std to avoid counting zeros/positives
    neg = np.where(mask, r_daily, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanstd(neg, axis=1, ddof=1)

def _pmpt_from_daily_paths(r_daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sortino (annualized) and Omega computed from daily returns.
    Sortino matches report.py: (mean_daily*252) / (std(daily<0)*sqrt(252))
    Omega threshold = 0 on daily returns.
    """
    mean_d = r_daily.mean(axis=1)
    down_std = _downside_std_negative_only(r_daily)

    with np.errstate(divide='ignore', invalid='ignore'):
        sortino = (mean_d * 252.0) / (down_std * np.sqrt(252.0))
    sortino[~np.isfinite(sortino)] = np.nan

    gains = np.clip(r_daily, 0.0, None).sum(axis=1)
    losses = np.clip(-r_daily, 0.0, None).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        omega = gains / losses
    omega[~np.isfinite(omega)] = np.nan

    return sortino, omega

# ---------- calibration ----------
def _calibrate_from_returns(
    r: pd.Series, simulate: str, default_nu: int | None
) -> tuple[float, float, int | None]:
    """
    Returns (mu_p_daily, sigma_p_daily, nu_for_t or None)
    """
    r = r.dropna().astype(float)
    mu_d = float(r.mean())
    sigma_d = float(r.std(ddof=1))

    # Calibrate nu from excess kurtosis if using t-daily
    nu = None
    if simulate == "t-daily":
        # sample excess kurtosis (Fisher)
        if r.shape[0] > 10 and sigma_d > 0:
            x = r.values - mu_d
            m2 = float(np.mean(x**2))
            m4 = float(np.mean(x**4))
            if m2 > 0:
                excess = m4 / (m2**2) - 3.0
                if excess > 0:
                    # For t, excess kurtosis = 6/(nu-4) -> nu = 6/excess + 4
                    nu_est = 6.0 / excess + 4.0
                    if np.isfinite(nu_est) and nu_est >= 4.5:
                        nu = int(round(nu_est))
        if nu is None:
            nu = default_nu if default_nu is not None else 8  # milder tails than 6
    return mu_d, sigma_d, nu

# ---------- core MC ----------
def mc_projection(
    mu: pd.Series,
    cov: pd.DataFrame,
    weights: pd.Series | None,
    total_nav: float = 500_000.0,
    years: int = 10,
    sims: int = 10_000,
    seed: int = 42,
    cov_is_daily: bool = True,
    out_png: str = "outputs/monte_carlo_projection.png",
    out_json: str = "outputs/monte_carlo_summary.json",
    simulate: str = "gaussian-daily",
    sims_batch: int = 2000,
    returns_cache: str | None = None,
    nu_override: int | None = None,
):
    """
    Memory-safe MC at the *portfolio* level. You can calibrate μ/σ (and ν for t) from
    realized daily strategy returns via --returns_cache to align MC PMPT with the report.
    """
    np.random.seed(seed)

    # If a returns cache is provided, calibrate from realized daily returns (preferred)
    mu_p_daily = sigma_p_daily = None
    nu_used: int | None = None
    r_hist = _read_returns_cache(returns_cache) if returns_cache else None

    if r_hist is not None and r_hist.shape[0] > 30:
        mu_p_daily, sigma_p_daily, nu_used = _calibrate_from_returns(r_hist, simulate, nu_override)
    else:
        # Use posterior mu & cov
        mu, cov, weights = _align_assets(mu, cov, weights)
        w = _build_weights(mu, weights).values.reshape(1, -1)

        if cov_is_daily:
            Sigma_daily = cov.values
            mu_daily_vec = (mu.values.reshape(-1, 1)) / 252.0
        else:
            Sigma_daily = cov.values / 252.0
            mu_daily_vec = (mu.values.reshape(-1, 1)) / 252.0

        mu_p_daily = float((w @ mu_daily_vec).item())
        sigma_p_daily = float(np.sqrt((w @ Sigma_daily @ w.T).item()))
        nu_used = nu_override if (simulate == "t-daily") else None

    # Derived annual scalars (for info)
    mu_p_annual    = mu_p_daily * 252.0
    sigma_p_annual = sigma_p_daily * np.sqrt(252.0)

    # Simulation
    T_years = int(years)
    days = 252 * T_years
    idxs = np.array([252 * y - 1 for y in range(1, T_years + 1)], dtype=int)
    nav0 = float(total_nav)
    dtype = np.float32

    year_end_navs = []
    finals = []
    all_sortino = []
    all_omega = []
    done = 0

    while done < sims:
        b = min(sims_batch, sims - done)
        if simulate == "gaussian-annual":
            # annual step simulation
            rets_ann = (mu_p_annual + sigma_p_annual * np.random.randn(b, T_years)).astype(dtype)
            # convert to daily-equivalent paths for PMPT (stretch each annual into 252 equal daily steps)
            # Just to compute daily PMPT consistently with report:
            r_daily = np.repeat((rets_ann / 252.0), 252, axis=1).astype(dtype)

            srt_b, omg_b = _pmpt_from_daily_paths(r_daily)
            all_sortino.append(srt_b); all_omega.append(omg_b)

            navs = nav0 * np.cumprod(1.0 + rets_ann, axis=1, dtype=dtype)
            year_end_navs.append(navs)
            finals.append(navs[:, -1])
        else:
            if simulate == "t-daily":
                nu = nu_used if nu_used is not None else 8
                z = np.random.standard_t(df=nu, size=(b, days)).astype(dtype) / np.sqrt(nu / (nu - 2))
            else:
                z = np.random.randn(b, days).astype(dtype)

            r_p_daily = (mu_p_daily + sigma_p_daily * z).astype(dtype)
            srt_b, omg_b = _pmpt_from_daily_paths(r_p_daily)
            all_sortino.append(srt_b); all_omega.append(omg_b)

            navs = nav0 * np.cumprod(1.0 + r_p_daily, axis=1, dtype=dtype)
            year_end_navs.append(navs[:, idxs])
            finals.append(navs[:, -1])

        done += b

    # Aggregate
    if simulate == "gaussian-annual":
        navs_all = np.vstack(year_end_navs).astype(np.float64)
        years_axis = np.arange(1, T_years + 1, dtype=int)
    else:
        navs_all = np.vstack(year_end_navs).astype(np.float64)
        years_axis = np.arange(1, T_years + 1, dtype=int)

    pct = np.percentile(navs_all, [5, 25, 50, 75, 95], axis=0)
    all_sortino = np.concatenate(all_sortino)
    all_omega   = np.concatenate(all_omega)
    sortino = float(np.nanmedian(all_sortino))
    omega   = float(np.nanmedian(all_omega))

    # Chart
    plt.figure(figsize=(9, 5))
    plt.fill_between(years_axis, pct[0], pct[4], alpha=0.20, label="5–95%")
    plt.fill_between(years_axis, pct[1], pct[3], alpha=0.40, label="25–75%")
    plt.plot(years_axis, pct[2], label="Median")
    plt.plot(years_axis, [nav0] * len(years_axis), linestyle="--", label="Initial NAV")
    plt.title(f"{T_years}-Year Monte Carlo Projection")
    plt.xlabel("Years"); plt.ylabel("Projected Portfolio Value (USD)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

    summary = {
        "assumptions": {
            "simulate": simulate,
            "mu_annual": mu_p_annual,
            "sigma_annual": sigma_p_annual,
            "nu": nu_used,
            "calibrated_from_cache": bool(r_hist is not None and r_hist.shape[0] > 30),
        },
        "sortino": sortino,
        "omega": omega,
        "nav_percentiles": {
            "y": years_axis.tolist(),
            "p05": pct[0].tolist(),
            "p25": pct[1].tolist(),
            "p50": pct[2].tolist(),
            "p75": pct[3].tolist(),
            "p95": pct[4].tolist(),
        },
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    return out_png, out_json

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu_csv", required=True)
    ap.add_argument("--cov_csv", required=True)
    ap.add_argument("--weights_csv", default=None)
    ap.add_argument("--total_nav", type=float, default=500_000.0)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--sims_batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cov_is_daily", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True)
    ap.add_argument("--simulate", choices=["gaussian-annual","gaussian-daily","t-daily"], default="gaussian-daily")
    ap.add_argument("--out_png", default="outputs/monte_carlo_projection.png")
    ap.add_argument("--out_json", default="outputs/monte_carlo_summary.json")
    # NEW: empirical calibration & t df override
    ap.add_argument("--returns_cache", default=None, help="Parquet with strategy daily returns to calibrate μ/σ/(ν).")
    ap.add_argument("--nu", type=int, default=None, help="Override Student-t degrees of freedom for --simulate t-daily.")
    args = ap.parse_args()

    mu = _read_mu(args.mu_csv)
    cov = _read_cov(args.cov_csv)
    weights = _read_weights(args.weights_csv)

    mc_projection(
        mu=mu, cov=cov, weights=weights,
        total_nav=args.total_nav, years=args.years, sims=args.sims,
        sims_batch=args.sims_batch, seed=args.seed,
        cov_is_daily=bool(args.cov_is_daily),
        out_png=args.out_png, out_json=args.out_json,
        simulate=args.simulate,
        returns_cache=args.returns_cache,
        nu_override=args.nu,
    )