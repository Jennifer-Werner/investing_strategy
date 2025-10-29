#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------- small utils --------------
def _ann_return(r: pd.Series, periods: int = 252) -> float:
    r = pd.Series(r).dropna()
    if r.empty:
        return float("nan")
    return float((1 + r).prod() ** (periods / max(len(r), 1)) - 1)

def _clean_index(idx: pd.Index) -> pd.Index:
    return pd.Index(str(x).strip().upper() for x in idx)

# -------------- IO helpers --------------
def _read_weights(path: str) -> pd.Series:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if df.ndim == 2:
            s = df.iloc[-1].copy()
            s.name = "weight"
            s.index = _clean_index(s.index)
            return s.astype(float)
    else:
        df = pd.read_csv(path)
        if {"asset","weight"} <= set(df.columns):
            s = df.set_index("asset")["weight"].astype(float)
        else:
            s = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
            s.name = "weight"
        s.index = _clean_index(s.index)
        return s
    raise SystemExit(f"Could not parse weights from {path}")

def _read_mu(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if {"asset","mu"} <= set(df.columns):
        s = df.set_index("asset")["mu"].astype(float)
    else:
        s = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
        s.name = "mu"
    s.index = _clean_index(s.index)
    return s

def _read_cov(path: str) -> pd.DataFrame:
    cov = pd.read_csv(path, index_col=0).astype(float)
    cov.index = _clean_index(cov.index)
    cov.columns = _clean_index(cov.columns)
    return (cov + cov.T) / 2.0

def _read_sectors(path: str | None) -> pd.Series:
    """Accept CSV with columns (ticker,sector) or (asset,sector)."""
    if not path or not os.path.exists(path):
        return pd.Series(dtype=object)
    df = pd.read_csv(path)
    if "ticker" in df.columns: kcol = "ticker"
    elif "asset" in df.columns: kcol = "asset"
    else: kcol = df.columns[0]
    if "sector" not in df.columns:
        scol = [c for c in df.columns if c != kcol][0]
    else:
        scol = "sector"
    s = df.set_index(kcol)[scol].astype(str).str.strip()
    s.index = _clean_index(s.index)
    return s

def _read_returns_cache(path: str | None) -> pd.Series | None:
    if not path or not os.path.exists(path):
        return None
    obj = pd.read_parquet(path)
    if isinstance(obj, pd.DataFrame):
        if "Strategy" in obj.columns:
            s = obj["Strategy"]
        else:
            s = obj.iloc[:, 0]
    else:
        s = obj
    return pd.Series(s).astype(float).sort_index()

def _read_benchmark_w(path: str | None) -> pd.Series | None:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if {"asset","weight"} <= set(df.columns):
        s = df.set_index("asset")["weight"].astype(float)
    else:
        s = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
        s.name = "weight"
    s.index = _clean_index(s.index)
    return s

# -------------- core helpers --------------
def _align(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, verbose: bool = True):
    common = w.index.intersection(mu.index).intersection(cov.index).intersection(cov.columns)
    dropped = sorted(set(w.index) - set(common))
    if verbose:
        print(json.dumps({
            "ALIGN": {
                "weights_n": int(len(w)),
                "mu_n": int(len(mu)),
                "cov_n": int(len(cov)),
                "common_n": int(len(common)),
                "dropped_from_weights": dropped[:10] + (["..."] if len(dropped) > 10 else [])
            }
        }, indent=2))

    if len(common) == 0:
        raise SystemExit("No overlapping tickers among weights, mu, and cov.")
    w = w.loc[common].clip(lower=0)
    if w.sum() == 0:
        raise SystemExit("All weights are zero after clipping.")
    w = w / w.sum()
    mu = mu.loc[common]
    cov = cov.loc[common, common]
    return w, mu, cov

def _risk_contributions(w: pd.Series, cov_annual: pd.DataFrame):
    Sigma = cov_annual.values
    wv = w.values.reshape(-1, 1)
    port_var = float((wv.T @ Sigma @ wv).item())
    port_vol = np.sqrt(max(port_var, 0.0))
    if port_vol > 0:
        mcr = (Sigma @ wv)[:, 0] / port_vol
        rc = w.values * mcr
    else:
        mcr = np.zeros_like(w.values)
        rc = np.zeros_like(w.values)
    asset_vol = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    return port_vol, pd.Series(mcr, index=w.index, name="marg_risk"), pd.Series(rc, index=w.index, name="risk_contrib"), pd.Series(asset_vol, index=w.index, name="asset_vol")

def _labels(contrib_ret: pd.Series, risk_contrib: pd.Series) -> pd.Series:
    r_med = contrib_ret.median()
    rc_hi = risk_contrib.quantile(0.80)
    rc_lo = risk_contrib.quantile(0.20)
    lab = []
    top_ret_cut = contrib_ret.quantile(0.70)
    for t in contrib_ret.index:
        cr = contrib_ret.loc[t]; rc = risk_contrib.loc[t]
        if cr >= top_ret_cut:
            lab.append("Strong")
        elif (rc >= rc_hi) and (cr < r_med):
            lab.append("Risky")
        elif (rc <= rc_lo) and (cr > 0):
            lab.append("Safe")
        else:
            lab.append("Weak")
    return pd.Series(lab, index=contrib_ret.index, name="label")

def _percentiles(s: pd.Series) -> pd.Series:
    qs = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
    out = s.quantile(qs)
    out.index = [f"p{int(q*100):02d}" for q in qs]
    return out

# -------------- plots (matplotlib only) --------------
def _plot_weights_pie(w: pd.Series, out_png: str):
    plt.figure(figsize=(8, 8))
    w_sorted = w.sort_values(ascending=False)
    big = w_sorted[w_sorted >= 0.005]
    small = w_sorted[w_sorted < 0.005]
    if small.sum() > 0:
        big = pd.concat([big, pd.Series({"Other": small.sum()})])
    plt.pie(big.values, labels=big.index, autopct="%1.1f%%", startangle=90)
    plt.title("Portfolio Weights")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _plot_bar(series: pd.Series, title: str, ylabel: str, out_png: str):
    plt.figure(figsize=(12, 5))
    s = series.sort_values(ascending=False)
    plt.bar(range(len(s)), s.values)
    plt.xticks(range(len(s)), s.index, rotation=90)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _plot_scatter(x: pd.Series, y: pd.Series, out_png: str, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(7, 6))
    plt.scatter(x.values, y.values)
    plt.axvline(x.median(), linestyle="--")
    plt.axhline(y.median(), linestyle="--")
    for t in x.index:
        plt.annotate(t, (x.loc[t], y.loc[t]), fontsize=7)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/weights_latest_full.csv")
    ap.add_argument("--mu", default="outputs/mu_bl.csv")
    ap.add_argument("--cov", default="outputs/cov_for_mc.csv")
    ap.add_argument("--sectors", default="data/universe.csv", help="CSV with columns ticker,sector (or asset,sector)")
    ap.add_argument("--returns_cache", default="outputs/port_daily_returns.parquet")
    ap.add_argument("--benchmark_weights", default=None, help="Optional CSV with benchmark weights (asset, weight)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--excel", default="outputs/portfolio_summary.xlsx")
    ap.add_argument("--cov_is_daily", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True)
    ap.add_argument("--mu_is_annual", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                    help="Set false if mu in CSV is DAILY; we will multiply by 252.")
    ap.add_argument("--min_display_weight", type=float, default=1e-4, help="Hide rows below this weight from invested-only tables")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load & align
    w = _read_weights(args.weights)
    mu = _read_mu(args.mu)
    cov = _read_cov(args.cov)
    w, mu, cov = _align(w, mu, cov, verbose=True)

    # Annualize mu if needed
    if not args.mu_is_annual:
        mu = mu * 252.0

    # Sectors
    sectors_map = _read_sectors(args.sectors)
    sectors = sectors_map.reindex(w.index).fillna("Unknown")

    # Annualize covariance if needed
    cov_annual = cov * (252.0 if args.cov_is_daily else 1.0)

    # Expected return (annual μ) and contributions
    contrib_ret = w * mu
    port_mu = float(contrib_ret.sum())

    # Risk pieces
    port_vol, mcr, rc, asset_vol = _risk_contributions(w, cov_annual)

    # Classify
    labels = _labels(contrib_ret, rc)

    # Realized (optional)
    realized_ann = None
    r_cache = _read_returns_cache(args.returns_cache)
    if r_cache is not None and not r_cache.empty:
        realized_ann = _ann_return(r_cache)

    # Optional tracking error vs benchmark_weights (approx, using cov)
    te_annual = None
    if args.benchmark_weights:
        wb = _read_benchmark_w(args.benchmark_weights)
        if wb is not None:
            common = w.index.intersection(wb.index).intersection(cov.index).intersection(cov.columns)
            if len(common) > 0:
                ww = w.reindex(common).fillna(0.0)
                bb = wb.reindex(common).fillna(0.0)
                ww = ww / ww.sum() if ww.sum() > 0 else ww
                bb = bb / bb.sum() if bb.sum() > 0 else bb
                d = (ww - bb).values.reshape(-1, 1)
                Sigma = cov.loc[common, common].values  # daily by default
                te_daily = float(np.sqrt(max(0.0, (d.T @ Sigma @ d).item())))
                te_annual = float(te_daily * np.sqrt(252.0 if args.cov_is_daily else 1.0))

    # Holdings table (full universe after align)
    df = pd.concat([
        sectors.rename("sector"),
        w.rename("weight"),
        mu.rename("mu_annual"),
        contrib_ret.rename("contrib_return"),
        asset_vol.rename("asset_vol_annual"),
        mcr,
        rc
    ], axis=1)
    rc_sum = df["risk_contrib"].sum()
    df["risk_contrib_pct"] = df["risk_contrib"] / (rc_sum if abs(rc_sum) > 0 else 1.0)
    df["label"] = labels

    # Invested-only view (hide tiny/zero)
    EPS = float(args.min_display_weight)
    invested = df[df["weight"] >= EPS].copy()
    invested_n = int((df["weight"] >= EPS).sum())

    # Sector rollup
    by_sector = df.groupby("sector").agg(
        weight=("weight","sum"),
        contrib_return=("contrib_return","sum"),
        risk_contrib=("risk_contrib","sum"),
        n_names=("weight","size")
    ).sort_values("weight", ascending=False)
    rc_sec_sum = by_sector["risk_contrib"].sum()
    by_sector["risk_contrib_pct"] = by_sector["risk_contrib"] / (rc_sec_sum if abs(rc_sec_sum) > 0 else 1.0)

    # Distributions (percentiles)
    dist = pd.concat({
        "weight": _percentiles(df["weight"]),
        "contrib_return": _percentiles(df["contrib_return"]),
        "risk_contrib": _percentiles(df["risk_contrib"])
    }, axis=1)

    # Per-asset expected returns (pretty tables)
    expected_returns = df[["weight","mu_annual","contrib_return"]].sort_values("contrib_return", ascending=False)
    invested_expected_returns = invested[["weight","mu_annual","contrib_return"]].sort_values("contrib_return", ascending=False)

    # Negative-μ guardrail
    neg = invested[invested["mu_annual"] <= 0].copy()
    neg_weight_sum = float(neg["weight"].sum()) if not neg.empty else 0.0

    # Save Excel
    with pd.ExcelWriter(args.excel, engine="xlsxwriter") as xw:
        # Summary row
        summary_cols = {
            "portfolio_expected_return_annual": port_mu,
            "portfolio_vol_annual": port_vol,
            "n_assets_total_aligned": len(df),
            "n_assets_invested": invested_n,
            "min_display_weight": EPS
        }
        if realized_ann is not None and np.isfinite(realized_ann):
            summary_cols["realized_return_annual"] = realized_ann
        if te_annual is not None:
            summary_cols["te_annual_vs_benchmark"] = te_annual

        summary = pd.DataFrame([summary_cols])
        summary.to_excel(xw, sheet_name="Summary", index=False)

        # Core tables
        df.sort_values("weight", ascending=False).to_excel(xw, sheet_name="Holdings")
        invested.sort_values("weight", ascending=False).to_excel(xw, sheet_name="InvestedOnly")
        by_sector.to_excel(xw, sheet_name="BySector")
        dist.to_excel(xw, sheet_name="Distributions")
        expected_returns.to_excel(xw, sheet_name="ExpectedReturns")
        invested_expected_returns.to_excel(xw, sheet_name="ExpectedReturns_Invested")

        # Negative μ names (if any)
        if not neg.empty:
            neg.to_excel(xw, sheet_name="NEGATIVE_MU")
        info = pd.DataFrame({
            "key": ["cov_is_daily","mu_is_annual","has_returns_cache","excel_path"],
            "value": [bool(args.cov_is_daily), bool(args.mu_is_annual), bool(r_cache is not None and not r_cache.empty), args.excel]
        })
        info.to_excel(xw, sheet_name="INFO", index=False)

        # widen columns
        for sh in ["Holdings","InvestedOnly","BySector","Top","Distributions","Summary","ExpectedReturns","ExpectedReturns_Invested","INFO","NEGATIVE_MU"]:
            if sh in xw.sheets:
                ws = xw.sheets[sh]
                ws.set_column(0, 0, 20)
                ws.set_column(1, 20, 14)

    # Plots to PNGs (use invested-only for pie so counts match)
    _plot_weights_pie(invested["weight"], os.path.join(args.outdir, "weights_pie.png"))
    _plot_bar(invested["contrib_return"], "Expected Return Contribution by Asset (Invested)", "w * μ (annual)", os.path.join(args.outdir, "return_contribs.png"))
    _plot_bar((df["risk_contrib"]), "Risk Contribution by Asset", "Risk Contribution", os.path.join(args.outdir, "risk_contribs.png"))
    _plot_scatter(invested["contrib_return"], invested["risk_contrib"], os.path.join(args.outdir, "contrib_return_vs_risk_contrib.png"),
                  "Return vs Risk Contribution (Invested)", "Return Contribution (w*μ)", "Risk Contribution")

    # Console summary
    print(f"Portfolio expected return (annual, model): {port_mu:.2%}")
    if realized_ann is not None and np.isfinite(realized_ann):
        print(f"Realized return (annual, backtest):   {realized_ann:.2%}")
    print(f"Portfolio volatility (annual):          {port_vol:.2%}")
    if te_annual is not None:
        print(f"Tracking error (annual, approx):       {te_annual:.2%}")
    if neg_weight_sum > 0:
        print(f"[warn] NEGATIVE μ within invested names: total weight ~ {neg_weight_sum:.2%} (see NEGATIVE_MU sheet)")
    print(f"Saved Excel: {args.excel}")
    print("Images saved to outputs/: weights_pie.png, return_contribs.png, risk_contribs.png, contrib_return_vs_risk_contrib.png")

if __name__ == "__main__":
    main()