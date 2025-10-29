#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

def ann_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty: return {}
    ar = float((1+r).prod()**(252/len(r))-1)
    vol = float(r.std()*np.sqrt(252))
    dd = float(((1+r).cumprod()/((1+r).cumprod()).cummax()-1).min())
    return {"ann_return":ar, "ann_vol":vol, "max_dd":dd}

def load_weights(path: str) -> pd.DataFrame:
    # weights.parquet from backtest: index=date, columns=tickers
    return pd.read_parquet(path).sort_index()

def load_prices(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).sort_index()

def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index().pct_change().fillna(0.0)

def group_weights(w: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    # sum weights by sector bucket per day
    sec = sectors.reindex(w.columns).fillna("Unknown")
    g = []
    for c in sorted(set(sec)):
        cols = sec.index[sec==c].tolist()
        if cols:
            g.append(w[cols].sum(axis=1).rename(c))
    return pd.concat(g, axis=1) if g else pd.DataFrame(index=w.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--sectors_csv", default="data/universe.csv")  # ticker,sector
    args = ap.parse_args()

    w = load_weights(os.path.join(args.outdir, "weights.parquet"))
    px = load_prices(os.path.join(args.outdir, "prices.parquet"))
    bench = args.benchmark

    # Daily returns for portfolio and benchmark
    r_px = pct_change(px)
    if bench not in r_px.columns:
        raise SystemExit(f"Benchmark {bench} not in cached prices.")
    # portfolio daily return = sum_{i} w_{t-1,i} * r_{t,i}
    w_aligned = w.reindex(r_px.index).ffill().dropna(how="all")
    r_port = (w_aligned * r_px[w_aligned.columns]).sum(axis=1).rename("port")
    r_bench = r_px[bench].reindex(r_port.index).fillna(0.0).rename("bench")
    r_active = (r_port - r_bench).rename("active")

    print("=== HEADLINE METRICS (CACHED BACKTEST STREAM) ===")
    print(ann_metrics(r_port))
    print()

    # Sleeves vs Active decomposition
    sleeves = {"VOO","SMH","IWF","QQQ","BONDS"}
    sleeves_in = [c for c in w.columns if c in sleeves]
    active_cols = [c for c in w.columns if c not in sleeves]
    w_sleeves = w_aligned[sleeves_in].sum(axis=1) if sleeves_in else pd.Series(0, index=w_aligned.index)
    w_active  = w_aligned[active_cols].sum(axis=1) if active_cols else pd.Series(0, index=w_aligned.index)

    # Compute sleeves return stream and active stream
    r_sleeves = (w_aligned[sleeves_in] * r_px[sleeves_in]).sum(axis=1) if sleeves_in else pd.Series(0, index=r_port.index)
    r_active_stream = (w_aligned[active_cols] * r_px[active_cols]).sum(axis=1) if active_cols else pd.Series(0, index=r_port.index)

    print("=== AVERAGE WEIGHTS ===")
    print({"sleeves_avg_w": float(w_sleeves.mean()), "active_avg_w": float(w_active.mean())})
    print()

    print("=== ANN METRICS: sleeves vs active vs bench ===")
    print({"sleeves": ann_metrics(r_sleeves), "active": ann_metrics(r_active_stream), "bench": ann_metrics(r_bench)})
    print()

    # Did constraints suppress active risk? Check realized TE
    # realized TE ~ std(port - bench)*sqrt(252)
    te_realized = float(r_active.std() * np.sqrt(252))
    print(f"=== REALIZED TE (annualized) ===\n{te_realized:.4%}\n")

    # Sector contributions and weights (using universe.csv)
    uni = pd.read_csv(args.sectors_csv)
    if "ticker" in uni.columns:
        s_map = uni.set_index("ticker")["sector"].astype(str)
    else:
        s_map = uni.set_index(uni.columns[0])[uni.columns[1]].astype(str)
    s_map.index = s_map.index.str.upper()

    # Canonicalize ETF/Bonds buckets
    for k,v in {"VOO":"ETF","SMH":"ETF","IWF":"ETF","QQQ":"ETF","BONDS":"Bonds"}.items():
        if k in s_map.index:
            s_map.loc[k]=v

    g_w = group_weights(w_aligned, s_map)
    # sector return contribution per day = sum_{i in sector} w_{t-1,i} * r_{t,i}
    sec_names = g_w.columns.tolist()
    sec_contrib = {}
    for sec in sec_names:
        cols = [c for c in w_aligned.columns if s_map.get(c,"Unknown")==sec]
        sec_contrib[sec] = (w_aligned[cols]*r_px[cols]).sum(axis=1)
    sec_contrib = pd.DataFrame(sec_contrib).reindex(r_port.index).fillna(0.0)

    # Aggregate over the backtest
    total_contrib = sec_contrib.sum(axis=0).sort_values(ascending=False)
    avg_w_by_sec  = g_w.mean().reindex(total_contrib.index)
    print("=== SECTOR AGGREGATES (avg weight, total return contribution) ===")
    print(pd.DataFrame({"avg_weight":avg_w_by_sec, "tot_contrib":total_contrib}))
    print()

    # Top/bottom assets by contribution
    asset_contrib = (w_aligned * r_px[w_aligned.columns]).sum(axis=0).sort_values(ascending=False)
    print("=== TOP 15 CONTRIBUTORS ===")
    print(asset_contrib.head(15)); print()
    print("=== BOTTOM 15 CONTRIBUTORS ===")
    print(asset_contrib.tail(15)); print()

    # Quick flags that often explain low returns
    # 1) Low realized TE vs target -> too tight TE or soft-penalty too strong
    # 2) Sleeves weight higher than planned
    # 3) Sector target penalty pulling away from winners
    print("=== QUICK FLAGS ===")
    flags = []
    # read config to get te target if available
    te_target = None
    cfg_path = os.path.join(args.outdir, "..", "config.yml")
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path,"r") as f:
                te_target = (yaml.safe_load(f) or {}).get("te_annual_target")
        except Exception:
            pass
    if te_target is not None and te_realized < 0.8*te_target:
        flags.append("Realized TE << target: active bets likely suppressed (raise te_annual_target or soften penalty).")
    if float(w_sleeves.mean()) > 0.36:
        flags.append("Sleeves+Bonds averaging above 35%: double-check per-asset bounds and fixed sleeves.")
    if not flags: flags.append("No obvious red flags from quick checks.")
    print("\n".join(flags))

if __name__ == "__main__":
    main()