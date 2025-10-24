#!/usr/bin/env python3
from __future__ import annotations
import argparse, pandas as pd

def _read_w(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if {"asset","weight"} <= set(df.columns):
        s = df.set_index("asset")["weight"].astype(float)
    else:
        s = df.set_index(df.columns[0]).iloc[:,0].astype(float); s.name="weight"
    s.index = s.index.str.upper()
    s = s.clip(lower=0)
    if s.sum() > 0: s = s/s.sum()
    return s

def _read_mu(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if {"asset","mu"} <= set(df.columns):
        s = df.set_index("asset")["mu"].astype(float)
    else:
        s = df.set_index(df.columns[0]).iloc[:,0].astype(float); s.name="mu"
    s.index = s.index.str.upper()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/weights_latest_full.csv")
    ap.add_argument("--mu",      default="outputs/mu_bl.csv")
    ap.add_argument("--assume_mu_daily", action="store_true",
                    help="Treat mu as daily; convert to annual by *252 for diagnostics.")
    args = ap.parse_args()

    w  = _read_w(args.weights)
    mu = _read_mu(args.mu)

    common = w.index.intersection(mu.index)
    missing_in_mu = sorted(set(w.index) - set(mu.index))
    if missing_in_mu:
        print(f"[warn] {len(missing_in_mu)} tickers missing in mu (ignored in dot): "
              f"{missing_in_mu[:15]}{' ...' if len(missing_in_mu)>15 else ''}")
    w  = w.loc[common]
    mu = mu.loc[common]

    if args.assume_mu_daily:
        mu = mu * 252.0

    port_mu = float((w * mu).sum())
    sleeves = {"VOO","SMH","IWF","QQQ","BONDS"}
    sleeves_w = float(w.reindex(sleeves).fillna(0.0).sum())
    active_w  = 1.0 - sleeves_w

    sleeves_contrib = float((w.reindex(sleeves).fillna(0.0) * mu.reindex(sleeves).fillna(0.0)).sum())
    active_contrib  = float((w.drop(list(sleeves.intersection(w.index)), errors="ignore") *
                             mu.drop(list(sleeves.intersection(mu.index)), errors="ignore")).sum())

    print("\n=== Expected Return Debug ===")
    print(f"Portfolio expected return (Σ w·μ): {port_mu:.4%}")
    print(f"Sleeves weight: {sleeves_w:.2%}, Active weight: {active_w:.2%}")
    print(f"Sleeves contribution: {sleeves_contrib:.4%}, Active contribution: {active_contrib:.4%}")
    print("\nμ summary (annual decimals assumed unless --assume_mu_daily):")
    print(f"min={mu.min():.4%}  p25={mu.quantile(0.25):.4%}  median={mu.median():.4%}  "
          f"mean={mu.mean():.4%}  p75={mu.quantile(0.75):.4%}  max={mu.max():.4%}")

    contrib = (w * mu).sort_values(ascending=False)
    print("\nTop 12 contribution names:")
    print((contrib.head(12) * 100).map(lambda x: f"{x:.2f}%").to_string())
    print("\nBottom 12 contribution names:")
    print((contrib.tail(12) * 100).map(lambda x: f"{x:.2f}%").to_string())

    abs_mu = mu.abs()
    if abs_mu.median() < 0.002 and abs_mu.max() < 0.01:
        print("\n[hint] μ looks small; if μ is daily, re-run with --assume_mu_daily to see the annualized KPI.")

if __name__ == "__main__":
    main()