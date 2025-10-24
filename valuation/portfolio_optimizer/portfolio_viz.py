#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- IO helpers ----------
def _read_weights(path: str) -> pd.Series:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if df.ndim == 2:
            s = df.iloc[-1].copy()
            s.name = "weight"
            return s.astype(float)
    else:
        df = pd.read_csv(path)
        if {"asset","weight"} <= set(df.columns):
            s = df.set_index("asset")["weight"].astype(float)
            return s
        s = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
        s.name = "weight"
        return s
    raise SystemExit(f"Could not parse weights from {path}")

def _read_mu(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if {"asset","mu"} <= set(df.columns):
        s = df.set_index("asset")["mu"].astype(float)
    else:
        s = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
        s.name = "mu"
    return s

def _read_cov(path: str) -> pd.DataFrame:
    cov = pd.read_csv(path, index_col=0).astype(float)
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
        # best effort: take second column
        scol = [c for c in df.columns if c != kcol][0]
    else:
        scol = "sector"
    s = df.set_index(kcol)[scol].astype(str).str.strip()
    s.index = s.index.str.upper()
    return s

# ---------- core helpers ----------
def _align(w: pd.Series, mu: pd.Series, cov: pd.DataFrame):
    common = w.index.intersection(mu.index).intersection(cov.index).intersection(cov.columns)
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

# ---------- plots (matplotlib only; default colors) ----------
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/weights_latest_full.csv")
    ap.add_argument("--mu", default="outputs/mu_bl.csv")
    ap.add_argument("--cov", default="outputs/cov_for_mc.csv")
    ap.add_argument("--sectors", default="data/universe.csv", help="CSV with columns ticker,sector (or asset,sector)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--excel", default="outputs/portfolio_summary.xlsx")
    ap.add_argument("--cov_is_daily", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load & align
    w = _read_weights(args.weights); w.index = w.index.str.upper()
    mu = _read_mu(args.mu);          mu.index = mu.index.str.upper()
    cov = _read_cov(args.cov);       cov.index = cov.index.str.upper(); cov.columns = cov.columns.str.upper()
    w, mu, cov = _align(w, mu, cov)

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

    # Holdings sheet
    df = pd.concat([
        sectors.rename("sector"),
        w.rename("weight"),
        mu.rename("mu_annual"),
        contrib_ret.rename("contrib_return"),
        asset_vol.rename("asset_vol_annual"),
        mcr,
        rc
    ], axis=1)
    df["risk_contrib_pct"] = df["risk_contrib"] / df["risk_contrib"].sum()
    df["label"] = labels

    # Sector rollup
    by_sector = df.groupby("sector").agg(
        weight=("weight","sum"),
        contrib_return=("contrib_return","sum"),
        risk_contrib=("risk_contrib","sum"),
        n_names=("weight","size")
    ).sort_values("weight", ascending=False)
    by_sector["risk_contrib_pct"] = by_sector["risk_contrib"] / by_sector["risk_contrib"].sum()

    # Distributions (percentiles)
    dist = pd.concat({
        "weight": _percentiles(df["weight"]),
        "contrib_return": _percentiles(df["contrib_return"]),
        "risk_contrib": _percentiles(df["risk_contrib"])
    }, axis=1)

    # Top/bottom tables
    top_ret = df.sort_values("contrib_return", ascending=False).head(20)
    bot_ret = df.sort_values("contrib_return", ascending=True).head(20)
    top_risk = df.sort_values("risk_contrib", ascending=False).head(20)
    bot_risk = df.sort_values("risk_contrib", ascending=True).head(20)

    # Save Excel
    with pd.ExcelWriter(args.excel, engine="xlsxwriter") as xw:
        # Summary (one row)
        summary = pd.DataFrame({
            "portfolio_expected_return_annual":[port_mu],
            "portfolio_vol_annual":[port_vol],
            "n_assets":[len(df)],
            "n_sectors":[df["sector"].nunique()]
        })
        summary.to_excel(xw, sheet_name="Summary", index=False)
        df.sort_values("weight", ascending=False).to_excel(xw, sheet_name="Holdings")
        by_sector.to_excel(xw, sheet_name="BySector")
        dist.to_excel(xw, sheet_name="Distributions")

        # Top tables
        top_ret.to_excel(xw, sheet_name="Top", startrow=0, startcol=0)
        bot_ret.to_excel(xw, sheet_name="Top", startrow=len(top_ret)+3, startcol=0)
        top_risk.to_excel(xw, sheet_name="Top", startrow=0, startcol=9)
        bot_risk.to_excel(xw, sheet_name="Top", startrow=len(top_risk)+3, startcol=9)

        # optional: widen columns a bit
        for sh in ["Holdings","BySector","Top","Distributions","Summary"]:
            ws = xw.sheets[sh]
            ws.set_column(0, 0, 18)   # index col (asset/sector)
            ws.set_column(1, 20, 14)  # data cols

    # Plots to PNGs
    _plot_weights_pie(w, os.path.join(args.outdir, "weights_pie.png"))
    _plot_bar(contrib_ret, "Expected Return Contribution by Asset", "w * mu (annual)", os.path.join(args.outdir, "return_contribs.png"))
    _plot_bar(rc, "Risk Contribution by Asset", "Risk Contribution", os.path.join(args.outdir, "risk_contribs.png"))
    _plot_scatter(contrib_ret, rc, os.path.join(args.outdir, "contrib_return_vs_risk_contrib.png"),
                  "Return vs Risk Contribution", "Return Contribution (w*mu)", "Risk Contribution")

    # Console summary
    print(f"Portfolio expected return (annual): {port_mu:.4f}")
    print(f"Portfolio volatility (annual): {port_vol:.4f}")
    print(f"Saved Excel: {args.excel}")
    print("Images saved to outputs/: weights_pie.png, return_contribs.png, risk_contribs.png, contrib_return_vs_risk_contrib.png")

if __name__ == "__main__":
    main()