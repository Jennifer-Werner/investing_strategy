#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List

FIXED_SLEEVES = {
    "BONDS": 0.10,
    "VOO": 0.15,
    "SMH": 0.05,
    "IWF": 0.025,
    "QQQ": 0.025,
}
ACTIVE_TARGET = 1.0 - sum(FIXED_SLEEVES.values())  # 0.65 by default

def project_to_capped_simplex(v: pd.Series, total: float, cap: float) -> pd.Series:
    """
    Project nonnegative vector v onto the simplex {w: w>=0, sum w = total} with per-component cap (w_i<=cap).
    Water-filling: iteratively cap the largest and rescale the rest until all <= cap.
    """
    w = v.clip(lower=0).astype(float)
    if w.sum() <= 0:
        # fallback: equal weights
        w[:] = 1.0
    # initial scale to target
    w *= (total / w.sum())

    # iteratively cap and rescale remainder
    free = w.index.tolist()
    capped = set()
    while True:
        over = w[w > cap].index.tolist()
        over = [i for i in over if i in free]
        if not over:
            break
        # fix the overs at cap
        for i in over:
            w.loc[i] = cap
            capped.add(i)
        free = [i for i in w.index if i not in capped]
        remaining = total - cap * len(capped)
        if remaining < 0 or len(free) == 0:
            # all capped; just renormalize (tiny numerical safety)
            break
        # rescale only the free set proportionally to their original (pre-cap) values
        v_free = v.loc[free].clip(lower=0).astype(float)
        if v_free.sum() <= 0:
            # if no signal, use equal among free names
            w.loc[free] = remaining / len(free)
        else:
            w.loc[free] = v_free * (remaining / v_free.sum())
    # final tiny renorm to hit total
    if w.sum() > 0:
        w *= (total / w.sum())
    return w.clip(lower=0)

def normalize_holdings(df: pd.DataFrame, max_cap: float = 0.05) -> pd.DataFrame:
    """
    Input df columns: ['ticker','weight'] (any case). Returns normalized DataFrame with:
      - sleeves set exactly to FIXED_SLEEVES
      - other tickers projected to ACTIVE_TARGET with per-name cap = max_cap
      - total sums to 1.0
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    assert "ticker" in df.columns and "weight" in df.columns, "holdings.csv must have 'ticker,weight' cols"

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    # collapse duplicates if any
    df = df.groupby("ticker", as_index=False)["weight"].sum()

    tickers = df["ticker"].tolist()
    weights = pd.Series(df["weight"].astype(float).values, index=tickers)

    # Ensure sleeves are present; if missing, add them with 0 to begin with
    for t in FIXED_SLEEVES:
        if t not in weights.index:
            weights.loc[t] = 0.0

    # Non-sleeve universe
    non_sleeves = [t for t in weights.index if t not in FIXED_SLEEVES]

    # Base vector for projection among non-sleeves: use provided weights (>=0)
    v = weights.loc[non_sleeves].clip(lower=0)

    # Project non-sleeves to ACTIVE_TARGET with per-name cap
    w_active = project_to_capped_simplex(v, total=ACTIVE_TARGET, cap=max_cap)

    # Combine with fixed sleeves
    w = pd.Series(0.0, index=weights.index)
    w.loc[non_sleeves] = w_active
    for t, val in FIXED_SLEEVES.items():
        w.loc[t] = val

    # Drop any stray tickers with exact zeros (optional)
    w = w[w > 0].sort_values(ascending=False)

    # Final normalize to 1.0
    w = w / w.sum()

    out = pd.DataFrame({"ticker": w.index, "weight": w.values})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/holdings.csv", help="Path to current holdings.csv")
    ap.add_argument("--outfile", default="data/holdings.csv", help="Where to write normalized holdings.csv")
    ap.add_argument("--max_cap", type=float, default=0.05, help="Per-name cap for non-sleeves (<= 0.05)")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    out = normalize_holdings(df, max_cap=args.max_cap)
    out.to_csv(args.outfile, index=False)
    # Print a brief diff
    try:
        before = df.copy()
        before.columns = [c.strip().lower() for c in before.columns]
        before["ticker"] = before["ticker"].astype(str).str.upper().str.strip()
        print("Before sum:", float(before["weight"].sum()))
        print("After  sum:", float(out["weight"].sum()))
        print("Top 10 after:\n", out.head(10).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()