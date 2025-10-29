#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BOND_YIELD_ANNUAL = 0.0425

def _ensure_bonds_in_prices(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if "BONDS" in getattr(weights, "columns", []) and "BONDS" not in prices.columns:
        idx = prices.index
        daily = 1.0 + BOND_YIELD_ANNUAL / 252.0
        bonds_px = pd.Series(100.0 * (daily ** np.arange(len(idx))), index=idx, name="BONDS")
        prices = prices.join(bonds_px)
    return prices

def portfolio_returns(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    px = _ensure_bonds_in_prices(weights, prices.copy().sort_index())
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
    d = ex[ex < 0]
    if d.std() == 0 or np.isnan(d.std()):
        return float("nan")
    return float((ex.mean() * periods) / (d.std() * np.sqrt(periods)))

def omega(r: pd.Series, theta: float = 0.0, periods: int = 252) -> float:
    if len(r) == 0:
        return float("nan")
    thr = theta / periods
    gains = (r - thr).clip(lower=0).sum()
    losses = (thr - r).clip(lower=0).sum()
    return float(gains / losses) if losses > 0 else float("inf")

def deflated_sharpe_ratio(r: pd.Series, trials: int = 1) -> float:
    if len(r) < 3 or r.std() == 0:
        return float("nan")
    s = sharpe(r)  # annualized
    s_daily = s / np.sqrt(252.0)
    T = len(r)
    from math import erf, sqrt, log
    z = s_daily * np.sqrt(T - 1.0)
    z_deflated = max(0.0, z - np.sqrt(2.0 * max(1.0, log(max(1, trials)))))
    return float(0.5 * (1.0 + erf(z_deflated / np.sqrt(2.0))))

def newey_west_ttest_mean(x: pd.Series, lags: int | None = None) -> dict:
    """HAC t-test for mean(x) using Bartlett kernel with Newey-West bandwidth."""
    x = pd.Series(x).dropna().astype(float)
    T = x.shape[0]
    if T < 30:
        return {"t": float("nan"), "p": float("nan"), "lags": 0, "n": T, "mean": float(x.mean())}
    if lags is None:
        # Andrews (1991)-style automatic bandwidth, rounded
        lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
        lags = max(1, lags)
    x_centered = x - x.mean()
    # gamma_0
    gamma0 = float(np.dot(x_centered, x_centered) / T)
    # HAC long-run variance with Bartlett weights
    lrv = gamma0
    for L in range(1, lags + 1):
        cov = float(np.dot(x_centered[L:], x_centered[:-L]) / T)
        w = 1.0 - L / (lags + 1.0)
        lrv += 2.0 * w * cov
    # variance of the sample mean
    var_mean = lrv / T
    if var_mean <= 0 or not np.isfinite(var_mean):
        return {"t": float("nan"), "p": float("nan"), "lags": lags, "n": T, "mean": float(x.mean())}
    tstat = float(x.mean() / np.sqrt(var_mean))
    # two-sided p-value under N(0,1)
    from math import erf, sqrt
    # Φ(|t|) = 0.5 * (1 + erf(|t| / sqrt(2)))
    Phi = 0.5 * (1.0 + erf(abs(tstat) / np.sqrt(2.0)))
    pval = 2.0 * (1.0 - Phi)
    return {"t": tstat, "p": float(pval), "lags": lags, "n": T, "mean": float(x.mean())}

def fetch_prices_yf(tickers: list[str], start: str, end: str,
                    retries: int = 3, timeout: int = 20) -> pd.DataFrame:
    import time
    import yfinance as yf
    series_list = []
    for t in tickers:
        last_err = None
        for _ in range(retries):
            try:
                df = yf.download(t, start=start, end=end,
                                 progress=False, auto_adjust=True, threads=False)
                if df is None or df.empty:
                    raise ValueError("empty download")
                if "Adj Close" in df.columns:
                    s = df["Adj Close"].copy()
                elif "Close" in df.columns:
                    s = df["Close"].copy()
                else:
                    s = df.iloc[:, -1].copy()
                s.name = t
                series_list.append(s)
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0)
        if (not series_list) or (series_list[-1].name != t):
            print(f"[warn] failed to fetch {t}: {last_err}")
    if not series_list:
        return pd.DataFrame()
    px = pd.concat(series_list, axis=1).sort_index().ffill()
    return px

def plot_equity_curve(curves: dict[str,pd.Series], out_png: str):
    plt.figure(figsize=(9, 5))
    for name, r in curves.items():
        nav = (1 + r).cumprod()
        plt.plot(nav.index, nav.values, label=name)
    plt.title("Equity Curve"); plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_drawdown(curves: dict[str,pd.Series], out_png: str):
    plt.figure(figsize=(9, 4))
    for name, r in curves.items():
        nav = (1 + r).cumprod()
        dd = nav / nav.cummax() - 1.0
        plt.plot(dd.index, dd.values, label=name)
    plt.title("Drawdowns"); plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_return_hist(r: pd.Series, out_png: str, bins: int = 50):
    plt.figure(figsize=(8, 4))
    plt.hist(r.dropna().values, bins=bins)
    plt.title("Return Distribution (Daily)"); plt.xlabel("Return"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# --- safe reader for cached returns ---
def _read_returns_cache(path: str) -> pd.Series:
    obj = pd.read_parquet(path)
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0]
        elif "strategy" in obj.columns:
            s = obj["strategy"]
        else:
            s = obj.iloc[:, 0]
    else:
        s = obj
    s = pd.Series(s).astype(float).sort_index()
    s.name = "Strategy"
    return s

def white_reality_check_spa(benchmark: pd.Series, alts: dict[str, pd.Series],
                            reps: int = 1000, seed: int = 42) -> dict:
    """Try arch.SPA; if not available or outputs missing, fall back to a stationary bootstrap surrogate."""
    # Build aligned DataFrame: benchmark is renamed to 'bench'
    df = pd.concat([benchmark.rename("bench")] + [v.rename(k) for k, v in alts.items()], axis=1).dropna()
    if df.shape[0] < 30:
        return {"error": "insufficient data for SPA"}

    # HAC (Newey–West) t-test on Strategy - Benchmark (use 'bench' here)
    if "Strategy" in df.columns and "bench" in df.columns:
        excess = (df["Strategy"] - df["bench"]).dropna()
        nw = newey_west_ttest_mean(excess)
        print(json.dumps({"NW_ttest_excess_mean": nw}, indent=2))
    else:
        print("[warn] SPA NW test skipped: columns missing")

    # SPA inputs: loss = -return
    L_b = (-df["bench"]).to_numpy()
    L_m = (-df.drop(columns=["bench"]).to_numpy())

    def _as_list(x):
        if x is None:
            return None
        if hasattr(x, "to_numpy"):
            x = x.to_numpy()
        x = np.asarray(x)
        return x.astype(float).ravel().tolist()

    # Fallback bootstrap helper
    def _fallback_bs(L_b_arr, L_m_arr, B: int = reps, block: int = 5):
        try:
            from arch.bootstrap import StationaryBootstrap
        except Exception as e:
            return {"error": f"SPA and fallback bootstrap unavailable: {e}"}
        T, M = L_m_arr.shape
        pvals = []
        for j in range(M):
            d = (L_m_arr[:, j] - L_b_arr)
            obs = float(np.mean(d))
            bs = StationaryBootstrap(block, d)
            draws = []
            for arr in bs.bootstrap(B):
                x = arr
                while isinstance(x, (tuple, list)):
                    x = x[0]
                x = np.asarray(x)
                if x.size:
                    draws.append(float(np.mean(x)))
            pvals.append(float(np.mean(np.asarray(draws) <= obs)) if draws else float("nan"))
        return {"method": "fallback_block_bootstrap", "models": list(alts.keys()),
                "spa_statistics": None, "spa_pvalues": pvals}

    # Try arch.SPA
    try:
        from arch.bootstrap import SPA
        try:
            spa = SPA(L_b, L_m, reps=reps, seed=seed)
        except TypeError:
            spa = SPA(L_b, L_m, reps=reps)
        res = spa.compute()

        def _get_any(obj, names):
            for n in names:
                if hasattr(obj, n):
                    v = getattr(obj, n)
                    try:
                        v = v() if callable(v) else v
                    except Exception:
                        pass
                    if v is not None:
                        return v
            return None

        stat = _get_any(spa, ("test_statistics", "statistics", "tstats", "stat")) \
               or _get_any(res, ("test_statistics", "statistics", "tstats", "stat"))
        pval = _get_any(spa, ("pvalues", "pvalues_spa", "pval", "pvals")) \
               or _get_any(res, ("pvalues", "pvalues_spa", "pval", "pvals"))

        if (stat is not None) and (pval is not None):
            return {"method": "arch.SPA", "models": list(alts.keys()),
                    "spa_statistics": _as_list(stat), "spa_pvalues": _as_list(pval)}
        return _fallback_bs(L_b, L_m, B=reps)

    except Exception:
        return _fallback_bs(L_b, L_m, B=reps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--start", required=False)
    ap.add_argument("--end", required=False)
    ap.add_argument("--spa_reps", type=int, default=1000)
    # optional caches (CLI)
    ap.add_argument("--returns_cache", default=None)
    ap.add_argument("--prices_cache", default=None)
    ap.add_argument("--ignore_caches", action="store_true")
    ap.add_argument("--debug_compare", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Read weights (for fallbacks and EW baseline)
    w_path = os.path.join(outdir, "weights.parquet")
    if not os.path.exists(w_path):
        raise SystemExit(f"Missing {w_path}. Run backtest first.")
    weights = pd.read_parquet(w_path).sort_index()  # index=dates, columns=tickers
    tickers = list(weights.columns)

    # Determine window
    start = args.start or (weights.index.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end   = args.end   or (weights.index.max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    start_ts = pd.to_datetime(args.start) if args.start else None
    end_ts = pd.to_datetime(args.end) if args.end else None

    # Auto-detect pipeline caches
    default_prices = os.path.join(outdir, "prices_used.parquet")
    default_strat  = os.path.join(outdir, "port_daily_returns.parquet")
    default_eqw    = os.path.join(outdir, "eqw_daily_returns.parquet")

    # Load prices
    px = None
    if not args.ignore_caches:
        path_prices = args.prices_cache or (default_prices if os.path.exists(default_prices) else None)
        if path_prices:
            try:
                px = pd.read_parquet(path_prices).sort_index()
            except Exception as e:
                print(f"[warn] failed to read prices cache: {e}")
                px = None
    if px is None:
        px = fetch_prices_yf([t for t in tickers if t != "BONDS"] + [args.benchmark], start, end)

    # Strategy returns
    strat_r = None
    if not args.ignore_caches:
        path_returns = args.returns_cache or (default_strat if os.path.exists(default_strat) else None)
        if path_returns and os.path.exists(path_returns):
            strat_r = _read_returns_cache(path_returns)
            if len(px.index) > 0:
                strat_r = strat_r.reindex(px.index).dropna()
    if strat_r is None:
        cols_for_port = [c for c in px.columns if c != args.benchmark]
        strat_r = portfolio_returns(weights, px[cols_for_port])

    # Equal-weight (prefer cached sleeves-aware baseline)
    eq_r = None
    if not args.ignore_caches and os.path.exists(default_eqw):
        try:
            eq_r = pd.read_parquet(default_eqw).iloc[:, 0].rename("EqualWeight")
        except Exception:
            eq_r = None
    if eq_r is None:
        # simple equal weight across all tickers (fallback)
        eq_w = pd.Series(1.0 / len(tickers), index=tickers)
        eq_weights = weights.copy(); eq_weights.iloc[:] = eq_w.values
        cols_for_port = [c for c in px.columns if c != args.benchmark]
        eq_r = portfolio_returns(eq_weights, px[cols_for_port]).rename("EqualWeight")

    # Apply date slices
    if start_ts is not None:
        px = px.loc[px.index >= start_ts]
        strat_r = strat_r.loc[strat_r.index >= start_ts]
        eq_r = eq_r.loc[eq_r.index >= start_ts]
    if end_ts is not None:
        px = px.loc[px.index <= end_ts]
        strat_r = strat_r.loc[strat_r.index <= end_ts]
        eq_r = eq_r.loc[eq_r.index <= end_ts]

    # Benchmark series aligned
    bench_r = px[args.benchmark].pct_change().reindex(strat_r.index).fillna(0.0).rename("Benchmark")

    df = pd.concat({"Strategy": strat_r, "EqualWeight": eq_r.reindex(strat_r.index), "Benchmark": bench_r}, axis=1).dropna()

    if args.debug_compare and args.returns_cache:
        # Compare cached vs recomputed series on same grid
        def _metrics(series):
            return {
                "ann_return": ann_return(series),
                "ann_vol": ann_vol(series),
                "max_drawdown": max_drawdown(series),
                "sharpe": sharpe(series),
                "sortino": sortino(series),
                "omega": omega(series),
                "len": int(series.shape[0]),
                "start": str(series.index.min()) if len(series) > 0 else None,
                "end": str(series.index.max()) if len(series) > 0 else None,
                "mean_daily": float(series.mean()),
                "std_daily": float(series.std())
            }
        mA = _metrics(df["Strategy"])
        cols_for_port = [c for c in px.columns if c != args.benchmark]
        strat_r_B = portfolio_returns(pd.read_parquet(os.path.join(outdir, "weights.parquet")), px[cols_for_port])
        if start_ts is not None: strat_r_B = strat_r_B.loc[strat_r_B.index >= start_ts]
        if end_ts is not None: strat_r_B = strat_r_B.loc[strat_r_B.index <= end_ts]
        strat_r_B = strat_r_B.reindex(df.index).dropna()
        mB = _metrics(strat_r_B)
        common = df.index.intersection(strat_r_B.index)
        max_abs_diff = float((df["Strategy"].reindex(common) - strat_r_B.reindex(common)).abs().max())
        print(json.dumps({"COMPARE": {"cache_series": mA, "recomputed_series": mB, "max_abs_diff": max_abs_diff}}, indent=2))

    # Metrics on Strategy
    r = df["Strategy"]
    metrics = {
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "max_drawdown": max_drawdown(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "omega": omega(r),
        "deflated_sharpe_ratio": deflated_sharpe_ratio(r)
    }

    # SPA (Strategy & 1/N vs market)
    spa = white_reality_check_spa(df["Benchmark"], {"Strategy": df["Strategy"], "EqualWeight": df["EqualWeight"]}, reps=args.spa_reps, seed=42)
    if isinstance(spa, dict) and "error" in spa:
        print(f"[warn] SPA skipped: {spa['error']}")

    # Visuals
    plot_equity_curve({"Strategy": df["Strategy"], "EqualWeight": df["EqualWeight"], "Benchmark": df["Benchmark"]},
                      os.path.join(outdir, "equity_curve.png"))
    plot_drawdown({"Strategy": df["Strategy"], "EqualWeight": df["EqualWeight"], "Benchmark": df["Benchmark"]},
                  os.path.join(outdir, "drawdowns.png"))
    plot_return_hist(df["Strategy"], os.path.join(outdir, "return_hist.png"))

    # Monte Carlo stitched if available
    mc = {}
    mc_json = os.path.join(outdir, "monte_carlo_summary.json")
    if os.path.exists(mc_json):
        with open(mc_json, "r") as f:
            mc = json.load(f)

    # Persist report
    report = {"metrics": metrics, "spa": spa, "monte_carlo": mc}
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print(json.dumps({
        "metrics": metrics,
        "spa": {"models": spa.get("models"), "pvalues": spa.get("spa_pvalues")},
        "mc_sortino": mc.get("sortino"),
        "mc_omega": mc.get("omega")
    }, indent=2))

if __name__ == "__main__":
    main()