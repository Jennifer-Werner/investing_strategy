#!/usr/bin/env python3
import argparse, yaml, math
import pandas as pd
import numpy as np
from pathlib import Path

def _read_series(path, key_name, val_name):
    df = pd.read_csv(path)
    if key_name in df.columns and val_name in df.columns:
        s = df.set_index(key_name)[val_name]
    else:
        s = df.set_index(df.columns[0]).iloc[:, 0]
        s.name = val_name
    return s.astype(float)

def _read_cfg(cfg_path="config.yml"):
    if Path(cfg_path).exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def _sector_map(universe_csv="data/universe.csv"):
    if Path(universe_csv).exists():
        u = pd.read_csv(universe_csv)
        if {"ticker","sector"} <= set(u.columns):
            return u.set_index("ticker")["sector"]
    return pd.Series(dtype=object)

def _build_bench_w(tickers, fixed):
    sum_sleeves = sum(v for k, v in fixed.items() if k in tickers)
    active = [t for t in tickers if t not in fixed]
    w = pd.Series(0.0, index=tickers, dtype=float)
    if active:
        w.loc[active] = (1.0 - sum_sleeves) / len(active)
    for k, v in fixed.items():
        if k in w.index:
            w.loc[k] = v
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--weights_csv", default="outputs/weights_latest_full.csv")
    ap.add_argument("--mu_csv",      default="outputs/mu_bl.csv")
    ap.add_argument("--cov_csv",     default="outputs/cov_for_mc.csv")
    ap.add_argument("--cfg",         default="config.yml")
    args = ap.parse_args()

    # Load outputs
    w   = _read_series(args.weights_csv, "asset", "weight")
    mu  = _read_series(args.mu_csv,      "asset", "mu")
    cov = pd.read_csv(args.cov_csv, index_col=0).astype(float)

    # Basic presence checks
    print(f"\n=== Presence checks for {args.ticker} ===")
    print("In weights.csv? ", args.ticker in w.index)
    print("In mu_bl.csv?   ", args.ticker in mu.index)
    print("In cov matrix?  ", (args.ticker in cov.index) and (args.ticker in cov.columns))
    if args.ticker in w.index:
        print(f"Weight[{args.ticker}] = {float(w.get(args.ticker)):.6f}")
    if args.ticker in mu.index:
        print(f"mu_bl[{args.ticker}] = {float(mu.get(args.ticker)):.6f}")

    # Sector loads vs caps
    cfg = _read_cfg(args.cfg)
    sector_caps = (cfg.get("sector_caps") or {})
    default_cap = (cfg.get("sector_caps") or {}).get("default", 0.30)
    sector_map = _sector_map()
    sectors = sector_map.reindex(w.index).fillna("Unknown")
    sector_loads = w.groupby(sectors).sum().sort_values(ascending=False)
    print("\n=== Sector weights ===")
    print(sector_loads.to_string())

    print("\n=== Sector caps (if any) ===")
    for sec, load in sector_loads.items():
        cap = sector_caps.get(sec, default_cap)
        print(f"{sec:30s} load={load:6.3f}  cap={cap:6.3f}  {'(binding)' if load >= cap-1e-6 else ''}")

    # Dividend floor check (approx)
    init_nav = cfg.get("initial_nav", 500_000.0)
    div_floor_abs = cfg.get("dividend_income_target_abs", 0.0)
    div_slack = cfg.get("dividend_income_slack", 0.0)
    req_income = div_floor_abs * (1.0 - div_slack)

    # Try to read dividend yields if you saved them; else set to 0 (or add your own pull here)
    div_path = Path("outputs/dividend_yields.csv")
    if div_path.exists():
        div = _read_series(div_path, "asset", "div_yield")
    else:
        div = pd.Series(0.0, index=w.index)  # neutral if not saved

    income = init_nav * float((div.reindex(w.index).fillna(0.0) * w).sum())
    print(f"\n=== Dividend floor check ===")
    print(f"Required income: {req_income:,.0f}   Achieved: {income:,.0f}   {'(binding)' if income <= req_income+1 else ''}")

    # TE check vs sleeve-aware benchmark using daily cov (your cov_for_mc.csv is daily)
    fixed = {"BONDS":0.10,"VOO":0.15,"SMH":0.05,"IWF":0.025,"QQQ":0.025}
    wb = _build_bench_w(list(w.index), fixed)
    common = w.index.intersection(cov.index).intersection(cov.columns)
    d = (w.reindex(common).fillna(0.0) - wb.reindex(common).fillna(0.0)).values.reshape(-1,1)
    Sigma = cov.loc[common, common].values
    te_daily = float(np.sqrt(d.T @ Sigma @ d))
    te_annual = te_daily * math.sqrt(252.0)
    te_target = cfg.get("te_annual_target", 0.04)
    print(f"\n=== Tracking error (approx) ===")
    print(f"TE_annual ~ {te_annual:.4f}   target={te_target:.4f}   {'(binding)' if te_annual >= te_target-1e-6 else ''}")

    print("\nTip: If your ticker weight is 0, look for a (binding) label above.")

if __name__ == "__main__":
    main()