#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, json, yaml
import pandas as pd
import numpy as np
from pathlib import Path

SLEEVES = {"BONDS","VOO","SMH","IWF","QQQ"}

def _read_series(path, key_name, val_name):
    df = pd.read_csv(path)
    # Try explicit cols first (asset/ticker + value)
    key_candidates = [c for c in ["asset","ticker",key_name] if c in df.columns]
    key = key_candidates[0] if key_candidates else df.columns[0]
    if val_name in df.columns:
        s = df.set_index(key)[val_name]
    else:
        # fall back to first non-key numeric column
        non_key_cols = [c for c in df.columns if c != key]
        s = df.set_index(key)[non_key_cols[0]]
        s.name = val_name
    # Make upper-case index for ticker consistency
    s.index = s.index.astype(str).str.upper()
    # Cast to float when possible
    try:
        s = s.astype(float)
    except Exception:
        pass
    return s

def _read_cfg(cfg_path="config.yml"):
    if Path(cfg_path).exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def _sector_map(universe_csv="data/universe.csv"):
    if Path(universe_csv).exists():
        u = pd.read_csv(universe_csv)
        key = "ticker" if "ticker" in u.columns else u.columns[0]
        sec = "sector" if "sector" in u.columns else None
        if sec:
            s = u.set_index(key)[sec].astype(str)
            s.index = s.index.astype(str).str.upper()
            return s
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
    ap.add_argument("--ticker", default="HCA")
    ap.add_argument("--weights_csv", default="outputs/weights_latest_full.csv")
    ap.add_argument("--mu_csv",      default="outputs/mu_bl.csv")
    ap.add_argument("--cov_csv",     default="outputs/cov_for_mc.csv")
    ap.add_argument("--cfg",         default="config.yml")
    ap.add_argument("--universe_csv",default="data/universe.csv")
    ap.add_argument("--dividends_csv", default="outputs/dividend_yields.csv")
    args = ap.parse_args()

    # Load outputs
    w   = _read_series(args.weights_csv, "asset", "weight")
    mu  = _read_series(args.mu_csv,      "asset", "mu")
    cov = pd.read_csv(args.cov_csv, index_col=0).astype(float)
    cov.index = cov.index.astype(str).str.upper()
    cov.columns = cov.columns.astype(str).str.upper()

    # Universe / sectors (best effort)
    sectors = _sector_map(args.universe_csv).reindex(w.index).fillna("Unknown")

    # Config & knobs
    cfg = _read_cfg(args.cfg)
    sector_caps = (cfg.get("sector_caps") or {})
    default_cap = (cfg.get("sector_caps") or {}).get("default", 0.30)
    te_target = float(cfg.get("te_annual_target", 0.04))
    min_pos = float(cfg.get("min_position_active", 0.0))
    enforce_min = bool(cfg.get("enforce_min_position", False))
    max_weight = float(cfg.get("max_weight", 0.05))
    initial_nav = float(cfg.get("initial_nav", 500_000.0))
    div_floor_abs = float(cfg.get("dividend_income_target_abs", 0.0))
    div_slack = float(cfg.get("dividend_income_slack", 0.0))
    req_income = div_floor_abs * (1.0 - div_slack)

    # Dividends (if saved)
    if Path(args.dividends_csv).exists():
        div = _read_series(args.dividends_csv, "asset", "div_yield").reindex(w.index).fillna(0.0)
    else:
        div = pd.Series(0.0, index=w.index)

    name = args.ticker.strip().upper()

    # ---------------- Presence & basics ----------------
    presence = {
        "in_weights_csv": bool(name in w.index),
        "in_mu_csv": bool(name in mu.index),
        "in_cov": bool((name in cov.index) and (name in cov.columns))
    }
    w_name = float(w.get(name, 0.0))
    mu_name = float(mu.get(name, np.nan)) if name in mu.index else float("nan")
    div_name = float(div.get(name, np.nan)) if name in div.index else float("nan")
    sec_name = str(sectors.get(name, "Unknown"))

    # ---------------- Sector loads vs caps ----------------
    sector_loads = w.groupby(sectors).sum().sort_values(ascending=False)
    sector_cap_report = []
    for sec, load in sector_loads.items():
        cap = float(sector_caps.get(sec, default_cap))
        binding = bool(load >= cap - 1e-9)
        sector_cap_report.append({"sector": sec, "load": float(load), "cap": cap, "binding": binding})
    sector_cap_binding_here = next((r["binding"] for r in sector_cap_report if r["sector"] == sec_name), False)

    # ---------------- Dividend floor ----------------
    achieved_income = float(initial_nav * (div * w).sum())
    div_binding = bool(achieved_income <= req_income + 1.0)

    # ---------------- TE vs sleeve-aware benchmark ----------------
    fixed = {"BONDS":0.10,"VOO":0.15,"SMH":0.05,"IWF":0.025,"QQQ":0.025}
    wb = _build_bench_w(list(w.index), fixed)
    common = w.index.intersection(cov.index).intersection(cov.columns)
    d = (w.reindex(common).fillna(0.0) - wb.reindex(common).fillna(0.0)).values.reshape(-1,1)
    Sigma = cov.loc[common, common].values
    te_daily = float(np.sqrt(max(0.0, (d.T @ Sigma @ d).item())))
    te_annual = float(te_daily * math.sqrt(252.0))
    te_binding = bool(te_annual >= te_target - 1e-9)

    # ---------------- Min-position pass diagnostics ----------------
    sleeves_w = float(w.reindex(SLEEVES).fillna(0.0).sum())
    active_budget = max(0.0, 1.0 - sleeves_w)
    active_names = [t for t in w.index if t not in SLEEVES]
    n_ge_min = sum(1 for t in active_names if w[t] >= min_pos - 1e-12)
    K_feasible = int(np.floor(active_budget / min_pos)) if (enforce_min and min_pos > 0) else None
    min_pass_likely = bool(enforce_min and min_pos > 0 and any(w[t] == 0 for t in active_names) and n_ge_min > 0)
    # Top K by current weights (proxy for which names “won” if second pass used)
    winners = []
    if enforce_min and min_pos > 0:
        winners = list(pd.Series({t: w[t] for t in active_names}).sort_values(ascending=False).index[:max(0, K_feasible or 0)])

    # ---------------- Compose report ----------------
    report = {
        "ticker": name,
        "presence": presence,
        "final": {
            "weight": w_name,
            "weight_pct": f"{w_name:.2%}",
            "sector": sec_name,
            "mu_annual": mu_name,
            "div_yield": div_name
        },
        "portfolio_shape": {
            "sleeves_weight": sleeves_w,
            "active_budget": active_budget,
            "n_active": int(len(active_names)),
            "n_active_ge_min": int(n_ge_min),
            "min_position": min_pos,
            "enforce_min_position": bool(enforce_min),
            "K_feasible_from_budget": K_feasible,
            "min_position_pass_likely": bool(min_pass_likely)
        },
        "binding_signals": {
            "sector_cap_binding_for_ticker_sector": bool(sector_cap_binding_here),
            "dividend_floor_binding": bool(div_binding),
            "te_binding": bool(te_binding)
        },
        "tracking_error": {
            "te_annual": te_annual,
            "te_target": te_target
        },
        "dividends": {
            "required_after_slack": req_income,
            "achieved": achieved_income
        },
        "sector_loads_vs_caps": sector_cap_report[:20],   # trim for readability
        "top_active_by_weight": pd.Series({t: w[t] for t in active_names}).sort_values(ascending=False).head(15).to_dict(),
        "top_active_by_mu": pd.Series({t: float(mu.get(t, np.nan)) for t in active_names}).dropna().sort_values(ascending=False).head(15).to_dict(),
        "is_ticker_among_topK_by_weight": (name in winners) if winners else None
    }

    # Simple natural-language hint
    hints = []
    if not presence["in_mu_csv"] or not presence["in_cov"]:
        hints.append("Ticker missing from mu/cov -> check universe/prices mapping.")
    if presence["in_mu_csv"] and w_name == 0.0 and enforce_min and min_pos > 0:
        hints.append("Likely dropped in second-pass min-position (not in top-K active names).")
    if sector_cap_binding_here:
        hints.append(f"Sector cap looks binding for {sec_name} -> crowd out possible.")
    if div_binding:
        hints.append("Dividend floor looks binding -> low-yield names can be excluded.")
    if te_binding:
        hints.append("TE looks binding -> portfolio may hug benchmark; some names zeroed.")
    if (np.isfinite(mu_name) and mu_name <= 0):
        hints.append("Ticker μ is non-positive -> optimizer avoids it.")
    report["hints"] = hints

    print(json.dumps(report, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x))

if __name__ == "__main__":
    main()