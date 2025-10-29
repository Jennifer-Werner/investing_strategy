
# run_pipeline.py
from __future__ import annotations
import argparse, json, yaml
import numpy as np
import pandas as pd
from pathlib import Path

from data import load_universe, load_holdings, fetch_prices, fetch_dividend_yields
from data import load_trade_filter_from_excel
from returns import to_log_returns, bayes_stein_shrinkage
from risk import ledoit_wolf_cov
from bl import bl_posterior
from optimizer import optimize_weights, build_sector_matrix   # <— import build_sector_matrix
from backtest import run_backtest, RunConfig
from metrics import (
    portfolio_returns, ann_return, ann_vol, max_drawdown,
    sharpe, sortino, omega, deflated_sharpe_ratio
)

def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def synthesize_bonds(prices: pd.DataFrame, tickers: list[str], yield_annual: float = 0.0425) -> pd.DataFrame:
    if "BONDS" in tickers and "BONDS" not in prices.columns:
        idx = prices.index
        daily = 1.0 + yield_annual / 252.0
        bonds_px = pd.Series(100.0 * (daily ** np.arange(len(idx))), index=idx, name="BONDS")
        prices = prices.join(bonds_px)
    return prices

def _pin_sleeves_exact(w: pd.Series, fixed: dict[str, float]) -> pd.Series:
    """Set sleeves exactly to required weights and rescale the remaining active book."""
    w = w.copy()
    active = [t for t in w.index if t not in fixed]
    fixed_total = sum(fixed.get(t, 0.0) for t in fixed if t in w.index)
    active_target = max(0.0, 1.0 - fixed_total)
    for t, v in fixed.items():
        if t in w.index:
            w.loc[t] = float(v)
    s = w.loc[active].sum()
    if s > 0 and active_target > 0:
        w.loc[active] = w.loc[active] * (active_target / s)
    return w

# def _normalize_div_yields(div: pd.Series) -> pd.Series:
#     """Ensure dividend yields are decimals (0.042 not 4.2)."""
#     s = pd.to_numeric(div, errors="coerce").fillna(0.0).astype(float)
#     if (s > 1.0).mean() > 0.25 and (s <= 100.0).all():
#         s = s / 100.0
#     return s.clip(lower=0.0, upper=0.30)
def _normalize_div_yields(div: pd.Series,
                          hard_cap: float = 0.12,      # 12% global cap
                          min_valid: float = 0.0) -> pd.Series:
    """
    Normalize dividend yields to decimals (e.g., 0.025 = 2.5%).
    - Convert percents [1..100] -> divide by 100
    - Treat anything > hard_cap as invalid (set NaN; we'll impute later)
    - Keep zeros as zeros (some growth names truly have 0)
    """
    s = pd.to_numeric(div, errors="coerce").astype(float)

    # If many are in percent form (1..100), convert to decimals
    percmask = (s > 1.0) & (s <= 100.0)
    if percmask.mean() > 0.1:  # if ≥10% look like percents, convert those entries
        s = s.where(~percmask, s / 100.0)

    # Knock out absurd values; we'll impute them later
    s = s.where((s >= min_valid) & (s <= hard_cap))

    return s

def _achieved_div_dollars(weights: pd.Series, div_yields: pd.Series, nav: float) -> float:
    y = div_yields.reindex(weights.index).fillna(0.0)
    return float(nav * (weights.fillna(0.0) * y).sum())

# ---------- NEW: quick LP to upper-bound achievable dividend floor ----------
def _max_dividend_income_upper_bound(
    tickers: list[str],
    sectors: pd.Series,
    div_yields: pd.Series,
    initial_nav: float,
    per_asset_bounds: dict[str, tuple[float, float]],
    fixed_weights: dict[str, float],
    sector_caps: dict[str, float],
    default_sector_cap: float,
) -> float:
    """
    Maximize initial_nav * (y @ w) subject to:
      sum w = 1, w >= 0, per-asset bounds (incl. sleeves fixed), sector hard caps.
    No TE / turnover here. This is a generous *upper bound* on dividend income.
    """
    import cvxpy as cp

    idx = list(tickers)
    n = len(idx)
    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1.0]

    # Per-asset bounds & sleeves exact
    for j, t in enumerate(idx):
        if t in fixed_weights:
            cons += [w[j] == float(fixed_weights[t])]
        else:
            lb, ub = per_asset_bounds.get(t, (0.0, 1.0))
            if lb is not None: cons += [w[j] >= float(lb)]
            if ub is not None: cons += [w[j] <= float(ub)]

    # Sector caps (include sleeves on LHS for a conservative bound)
    if sectors is not None and len(sectors) > 0:
        S, caps, _ = build_sector_matrix(idx, sectors, sector_caps or {}, float(default_sector_cap))
        for i in range(S.shape[0]):
            cons += [S[i, :] @ w <= caps[i]]

    y = div_yields.reindex(idx).fillna(0.0).values
    obj = cp.Maximize(y @ w)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except Exception:
        prob.solve(solver="ECOS", verbose=False)
    if w.value is None:
        return 0.0
    y_opt = float(y @ w.value)
    return float(initial_nav * y_opt)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['optimize', 'backtest'], required=True)
    ap.add_argument('--asof', type=str)
    ap.add_argument('--start', type=str)
    ap.add_argument('--end', type=str)
    ap.add_argument('--config', type=str, default='config.yml')
    ap.add_argument('--views', type=str, default='views.yaml')
    ap.add_argument('--sectors', type=str, default='sectors.yml')
    ap.add_argument('--universe', type=str, default='data/universe.csv')
    ap.add_argument('--holdings', type=str, default='data/holdings.csv')
    ap.add_argument('--outdir', type=str, default='outputs')
    # debug helpers
    ap.add_argument('--ignore_holdings', action='store_true',
                    help='Ignore holdings when optimizing (disables turnover constraint).')
    ap.add_argument('--debug_relax', action='store_true',
                    help='If infeasible, progressively relax constraints and report where it became feasible.')
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    def _normalize_views(vs):
        out = []
        for v in (vs or []):
            v = dict(v)  # copy
            # unify confidence key
            if "conf" not in v and "confidence" in v:
                v["conf"] = v["confidence"]
            # absolute: accept 'mu' or 'value'
            if v.get("type") == "absolute":
                if "value" not in v and "mu" in v:
                    v["value"] = v["mu"]
            # sector name canonicalization for views (match your runtime sectors)
            if v.get("type") == "sector":
                sec = str(v.get("sector", "")).strip()
                canon = {
                    "Information Technology": "Information Technology",
                    "Technology": "Information Technology",
                    "Info Tech": "Information Technology",
                    "Health Care": "Healthcare",
                    "Healthcare": "Healthcare",
                    "Communication": "Communication Services",
                    "Comm Services": "Communication Services",
                    "Communication Services": "Communication Services",
                    "Consumer Discretionary": "Consumer Discretionary",
                    "Consumer Cyclical": "Consumer Discretionary",  # <-- important
                    "Industrials": "Industrials",
                    "Utilities": "Utilities",
                    "Renewable Energy": "Renewable Energy",
                    "Energy": "Energy",
                }
                v["sector"] = canon.get(sec, sec)
            out.append(v)
        return out

    views_cfg_raw = load_yaml(args.views).get('views', [])
    views_cfg = _normalize_views(views_cfg_raw)

    # Canonicalize sector names used in views so BL actually picks them up
    _sector_canon_views = {
        "Health Care": "Healthcare",
        "Consumer Cyclical": "Consumer Discretionary",
        "Consumer Defensive": "Consumer Discretionary",
        "Comm Services": "Communication Services",
        "Communication": "Communication Services",
    }
    for v in views_cfg:
        if isinstance(v, dict) and v.get("type") == "sector":
            sec = v.get("sector")
            if isinstance(sec, str):
                v["sector"] = _sector_canon_views.get(sec, sec)

    sectors_cfg = load_yaml(args.sectors) if Path(args.sectors).exists() else {}
    green_energy = (sectors_cfg.get('green_energy') or {})
    sector_map = (sectors_cfg.get('map') or {})

    # Universe + sectors
    uni = load_universe(args.universe)
    tickers = uni['ticker'].astype(str).str.upper().tolist()
    sectors = uni.set_index('ticker')['sector'].to_dict()
    sectors.update({k.upper(): v for k, v in sector_map.items()})
    sectors = pd.Series(sectors)

    # Optional pre-filter (tradability)
    tradable = load_trade_filter_from_excel(
        "valuation/comprehensive-analysis/output/sheets/Comprehensive_Valuation_Report_Normalized.xlsx",
        sheet="Complete_Data",
        min_dollar_vol_m=20.0,
        require_can_trade=True
    )
    if len(tradable) > 0:
        keep = set(["BONDS", "VOO", "SMH", "IWF", "QQQ"]) | set(tradable)
        tickers = [t for t in tickers if t in keep]
        sectors = sectors.reindex(tickers)

    # Holdings (optional)
    if (not args.ignore_holdings) and Path(args.holdings).exists() and Path(args.holdings).stat().st_size > 0:
        holdings = load_holdings(args.holdings).reindex(tickers).fillna(0.0)
        holdings = (holdings / holdings.sum()) if holdings.sum() > 0 else None
    else:
        holdings = None

    # Data window
    price_years = cfg.get('price_history_years', 10)
    end_date = args.end or args.asof or pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=price_years)).strftime('%Y-%m-%d')

    # Fetch prices
    bench = cfg['benchmark']
    real_tickers = [t for t in tickers if t != "BONDS"]
    prices = fetch_prices(real_tickers + [bench], start=start_date, end=end_date)

    # Ensure sleeve ETFs present if missing
    sleeves = ["VOO", "SMH", "IWF", "QQQ"]
    sleeves_needed = [s for s in sleeves if (s in tickers) and (s not in prices.columns or prices[s].dropna().empty)]
    if sleeves_needed:
        sleeve_px = fetch_prices(sleeves_needed, start=start_date, end=end_date, chunk_size=1, retries=5, timeout=60)
        missing_cols = [c for c in sleeves_needed if c not in prices.columns or prices[c].dropna().empty]
        sleeve_px = sleeve_px.reindex(columns=missing_cols)
        prices = prices.join(sleeve_px, how="outer")

    # Benchmark retry if needed
    if bench not in prices.columns or prices[bench].dropna().empty:
        print("[warn] Benchmark missing; retrying benchmark alone …")
        bench_px = fetch_prices([bench], start=start_date, end=end_date, chunk_size=1, retries=3, timeout=30)
        prices = prices.join(bench_px, how="outer")
        if bench not in prices.columns or prices[bench].dropna().empty:
            raise SystemExit(f"Benchmark {bench} has no data. Please re-run or check network.")

    # Drop assets with no usable history and warn
    available = [c for c in prices.columns if c != bench and prices[c].notna().sum() > 0]
    missing = sorted(set(real_tickers) - set(available))
    if missing:
        print(f"[warn] Dropping {len(missing)} tickers with no data: {missing[:15]}{' ...' if len(missing) > 15 else ''}")
    tickers = [t for t in tickers if (t == 'BONDS') or (t in available)]
    sectors = sectors.reindex(tickers)

    prices = synthesize_bonds(prices, tickers, yield_annual=0.0425)

    # === Canonicalize sectors; map sleeves ===
    _sectors_dict = sectors.to_dict()
    _sector_canon = {
        "Information Technology":"Information Technology","Technology":"Information Technology","Info Tech":"Information Technology",
        "Health Care":"Healthcare","Healthcare":"Healthcare",
        "Communication":"Communication Services","Comm Services":"Communication Services","Communication Services":"Communication Services",
        "Consumer Discretionary":"Consumer Discretionary","Industrials":"Industrials","Utilities":"Utilities",
        "Renewable Energy":"Renewable Energy","ETF":"ETF","Bonds":"Bonds", None:"Unknown"
    }
    _sector_force = {"VOO":"ETF","SMH":"ETF","IWF":"ETF","QQQ":"ETF","BONDS":"Bonds"}
    sectors = pd.Series({k: _sector_canon.get(_sector_force.get(k, _sectors_dict.get(k)), _sectors_dict.get(k, "Unknown")) for k in tickers}).fillna("Unknown")

    # Dividend yields snapshot (+ normalization)
    div_raw = fetch_dividend_yields(tickers)
    y_over = {"BONDS": 0.0425}
    y_over.update(cfg.get("dividend_yield_overrides", {}))
    # was:
    # div = _normalize_div_yields(div_raw).reindex(tickers).fillna(0.0)

    # fix:
    div = _normalize_div_yields(div_raw).reindex(tickers)
    # for k, v in y_over.items():
    #     if k in div.index:
    #         div.loc[k] = float(v)

    # 1) Sector-median imputation for invalid/missing values
    sec_for_impute = sectors.reindex(tickers).fillna("Unknown")
    # compute medians on valid values only
    sec_meds = div.groupby(sec_for_impute).transform(lambda x: x.median(skipna=True))
    div = div.fillna(sec_meds)

    # 2) Fallback for anything still NaN (e.g., sectors with all NaN)
    fallback_yield = float(cfg.get("dividend_yield_fallback", 0.015))  # 1.5% default
    div = div.fillna(fallback_yield)

    # 3) Reasonable ETF defaults unless explicitly overridden in config
    etf_defaults = {"VOO": 0.012, "QQQ": 0.005, "IWF": 0.007, "SMH": 0.012}
    for k, v in etf_defaults.items():
        if k in div.index and pd.isna(y_over.get(k, np.nan)):  # only set if not overridden in cfg
            div.loc[k] = v

    # 4) Apply explicit overrides last (always wins)
    for k, v in y_over.items():
        if k in div.index:
            div.loc[k] = float(v)

    # 5) Final safety cap (keep numbers sane)
    final_cap = float(cfg.get("dividend_yield_cap", 0.12))
    div = div.clip(lower=0.0, upper=final_cap)

    # 6) Diagnostics (so you can catch it immediately next time)
    print("[diag] Dividend yields (decimals) after cleaning:",
          f"min={div.min():.4f}, p50={div.median():.4f}, p95={div.quantile(0.95):.4f}, max={div.max():.4f}")
    blend = float((div.reindex(tickers).fillna(0.0) * (pd.Series(1.0, index=tickers) / len(tickers))).sum())
    print(f"[diag] Example blended yield if equal-weight: ~{blend:.2%}")

    # Estimation window
    est_cols = [c for c in tickers if c in prices.columns]
    rets = to_log_returns(prices[est_cols]).dropna()
    if rets.shape[0] < 252:
        raise SystemExit("Not enough history for estimation (need >= 252 trading days)")

    # Risk & mean estimates
    cov = ledoit_wolf_cov(rets)
    mu_hist_ann = rets.mean() * 252.0
    mu_bs = bayes_stein_shrinkage(mu_hist_ann, cov, T=rets.shape[0])

    # Market weights proxy
    mkt_w = pd.Series(1.0 / len(tickers), index=tickers)

    # Black–Litterman posterior (exclude synthetic BONDS)
    tickers_ex_bonds = [t for t in tickers if t != "BONDS"]

    ra_bl = cfg.get('risk_aversion_bl', 2.5)
    ra_opt = cfg.get('risk_aversion_optim', 0.05)

    mu_bl, cov_bl = bl_posterior(
        cov, prices, tickers_ex_bonds, sectors, mkt_w,
        ra_bl, bench, cfg['tau'], mu_bs,
        views_cfg, green_energy,
        eta_bayes_stein=cfg.get('eta_bayes_stein', 0.25),
        omega_scale=cfg.get('omega_scale', 1.0),
    )

    mu_over = (cfg.get("mu_overrides") or {})
    if mu_over:
        mu_bl = mu_bl.copy()
        for k, v in mu_over.items():
            kk = str(k).upper()
            if kk in mu_bl.index:
                mu_bl.loc[kk] = float(v)

    # Optimization config
    rc = RunConfig(
        tickers=tickers, benchmark=bench, initial_nav=cfg['initial_nav'],
        min_weight=cfg['min_weight'], max_weight=cfg['max_weight'],
        sector_caps=cfg.get('sector_caps', {}), default_sector_cap=cfg.get('sector_caps', {}).get('default', 0.30),
        turnover_cap=cfg['turnover_cap'], te_annual_target=cfg['te_annual_target'],
        tau=cfg['tau'], eta_bayes_stein=cfg['eta_bayes_stein'], risk_aversion=ra_opt,
        div_floor_abs=cfg['dividend_income_target_abs'], div_slack=cfg['dividend_income_slack'],
    )

    # After rc:
    min_pos = float(cfg.get('min_position_active', 0.0))
    enforce_min_pos = bool(cfg.get('enforce_min_position', False))
    sector_target_penalty_cfg = float(cfg.get('sector_target_penalty', 0.3))

    # Fixed sleeves (present in universe)
    fixed_weights = {"BONDS": 0.10, "VOO": 0.15, "SMH": 0.05, "IWF": 0.025, "QQQ": 0.025}
    fixed_in_universe = {k: v for k, v in fixed_weights.items() if k in tickers}

    per_asset_bounds: dict[str, tuple[float, float]] = {}
    for t in tickers:
        if t in fixed_in_universe:
            v = fixed_in_universe[t]
            per_asset_bounds[t] = (v, v)  # exact
        else:
            per_asset_bounds[t] = (0.0, rc.max_weight)



    # NVDA >= 3.25% if not a sleeve
    if "NVDA" in tickers and "NVDA" not in fixed_in_universe:
        lb = 0.0325
        ub = max(rc.max_weight, lb)
        per_asset_bounds["NVDA"] = (lb, ub)

    # TE benchmark respecting sleeves — configurable modes
    sum_sleeves_present = sum(fixed_in_universe.values())
    active_budget = max(0.0, 1.0 - sum_sleeves_present)

    te_mode = (cfg.get("te_benchmark_mode") or "sleeves_only").lower()

    bench_w = pd.Series(0.0, index=tickers, dtype=float)
    # sleeves always pinned in the TE benchmark
    for k, v in fixed_in_universe.items():
        bench_w.loc[k] = v

    if te_mode == "sleeves_only":
        # current behavior: benchmark gives 0 weight to all active names
        pass

    elif te_mode == "equal_weight_active":
        # spread the active budget equally across all non-sleeve names
        act = [t for t in tickers if t not in fixed_in_universe]
        if act and active_budget > 0:
            bench_w.loc[act] = active_budget / len(act)

    elif te_mode == "holdings" and (holdings is not None) and holdings.sum() > 0:
        # use your current holdings as the TE benchmark
        bench_w = holdings.reindex(tickers).fillna(0.0)

    elif te_mode == "sector_targets":
        # distribute active budget by sector_targets in config, equal-weight within each sector
        st_cfg = (cfg.get("sector_targets") or {})
        tot = float(sum(st_cfg.values()))
        if tot > 0 and active_budget > 0:
            for sec, w_sec in st_cfg.items():
                sec_w = active_budget * (float(w_sec) / tot)
                names = [t for t in tickers if (t not in fixed_in_universe) and sectors.get(t) == sec]
                if names and sec_w > 0:
                    bench_w.loc[names] = sec_w / len(names)
    else:
        print(f"[info] te_benchmark_mode='{te_mode}' not recognized; using sleeves_only.")

    print(f"[diag] TE benchmark mode: {te_mode}; sleeves={sum_sleeves_present:.2%}, "
          f"active_bench_sum={(bench_w.sum()-sum_sleeves_present):.2%}")

    # Desired sector totals (absolute) → soft targets + hard caps on the active book
    desired_sector_targets = {
        "Information Technology": 0.1950,
        "Healthcare":             0.1300,
        "Renewable Energy":       0.0650,
        "Industrials":            0.0975,
        "Utilities":              0.0325,
        "Consumer Discretionary": 0.0975,
        "Communication Services": 0.0325,
    }
    present_sleeves_sum = sum_sleeves_present
    target_total = sum(desired_sector_targets.values())
    scale = (1.0 - present_sleeves_sum) / target_total if target_total > 0 else 1.0

    # soft targets (pull)
    sector_targets = {k: v * scale for k, v in desired_sector_targets.items()}
    # hard caps with slack
    cap_slack = 0.01  # 1%
    sector_caps_for_opt = {k: v * scale + cap_slack for k, v in desired_sector_targets.items()}
    default_sector_cap_for_opt = 1.0

    _total_after = present_sleeves_sum + sum(sector_targets.values())
    if abs(_total_after - 1.0) > 1e-6:
        print(f"[warn] After scaling, sleeves+sectors != 100% (got {_total_after:.4f}). "
              f"Sleeves={present_sleeves_sum:.4f}, Sectors={sum(sector_targets.values()):.4f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Save dividends for diagnostics
    (outdir / 'dividend_yields.csv').write_text(div.rename_axis("asset").to_frame("div_yield").to_csv(index=True))

    # Heads-up on dividend floor feasibility (sleeves only)
    req_income = rc.div_floor_abs * (1.0 - rc.div_slack)
    sleeve_income = _achieved_div_dollars(pd.Series(fixed_in_universe), div, rc.initial_nav)
    if sleeve_income < req_income:
        print(f"[warn] Dividend floor may bind: sleeves income ${sleeve_income:,.0f} vs floor ${req_income:,.0f}.")

    # ---------- NEW: tighten dividend floor to what’s actually achievable ----------
    div_upper = _max_dividend_income_upper_bound(
        tickers=tickers,
        sectors=sectors,
        div_yields=div,
        initial_nav=rc.initial_nav,
        per_asset_bounds=per_asset_bounds,
        fixed_weights=fixed_in_universe,
        sector_caps=sector_caps_for_opt,
        default_sector_cap=default_sector_cap_for_opt,
    )
    if div_upper <= 0:
        print("[warn] Dividend feasibility probe failed; continuing with configured floor.")
        effective_div_floor = rc.div_floor_abs
    else:
        if div_upper < req_income:
            # If floor is not reachable, clamp it to 99% of the theoretical max to keep feasibility.
            print(f"[warn] Your dividend floor (${req_income:,.0f}) exceeds the maximum achievable "
                  f"under current bounds (${div_upper:,.0f}). Lowering floor to maintain feasibility.")
            effective_div_floor = div_upper / (1.0 - rc.div_slack) * 0.99
        else:
            effective_div_floor = rc.div_floor_abs
    # ------------------------------------------------------------------------------

    def _fmt_series_pct(s: pd.Series, ndp: int = 2) -> pd.Series:
        """Return a string-formatted % series with ndp decimals (e.g., 0.1234 -> '12.34%')."""
        return s.apply(lambda v: f"{v * 100:,.{ndp}f}%")

    # Second-pass helper (min position)
    def _second_pass_min_position(
            first_w: pd.Series,
            min_pos: float,
            tickers: list[str],
            fixed_in_universe: dict[str, float],
            rc,
            mu_all: pd.Series,
            cov: pd.DataFrame,
            bench_w: pd.Series,
            sectors: pd.Series,
            div: pd.Series,
            sector_caps_for_opt: dict,
            default_sector_cap_for_opt: float,
            sector_targets: dict,
            sector_target_penalty: float
    ) -> pd.Series:
        """
        Enforce a minimum position on a sector-aware subset of names.
        - Greedy pick: walk first-pass weights high→low, but only add a name if
          (i) sector cap has room for +min_pos and (ii) active budget has room.
        - Recompute a *feasible* dividend-floor upper bound with these new bounds,
          and clamp the floor before calling the optimizer to avoid infeasibility.
        """
        if min_pos <= 0:
            return first_w

        from collections import defaultdict

        # Active budget (ex sleeves)
        sleeves_sum = sum(fixed_in_universe.values())
        active_budget = max(0.0, 1.0 - sleeves_sum)

        # Active universe (no sleeves), sorted by the first-pass weight
        # Active universe (no sleeves), rank by expected return μ (desc) and skip μ<=0
        active_names = [t for t in tickers if t not in fixed_in_universe]

        # Pull μ for actives, drop NaNs and non-positive μ, then sort high → low
        mu_active = mu_all.reindex(active_names)
        mu_active = mu_active[mu_active > 0].sort_values(ascending=False)

        # Sector capacities (hard caps on LHS for the active book)
        sec_cap = defaultdict(lambda: float(default_sector_cap_for_opt))
        for s_name, cap in (sector_caps_for_opt or {}).items():
            sec_cap[s_name] = float(cap)

        # Running usage of sector cap from the min_pos selections
        used_sec = defaultdict(float)

        # Explicit floors we must preserve (e.g., NVDA >= 3.25%)
        explicit_floors = {}
        if "NVDA" in active_names and min_pos < 0.0325:
            explicit_floors["NVDA"] = 0.0325

        # Seed sector usage with explicit floors
        for t, lb in explicit_floors.items():
            s = sectors.get(t, "Unknown")
            used_sec[s] += lb

        # Greedy sector-aware selection
        selected = set(explicit_floors.keys())
        used_budget = sum(explicit_floors.values())

        for t in mu_active.index:  # iterate in order of highest μ first
            if t in selected:
                continue
            if used_budget + min_pos > active_budget + 1e-12:
                break  # no more budget
            s = sectors.get(t, "Unknown")
            # Can this sector take one more min_pos?
            if used_sec[s] + min_pos <= sec_cap.get(s, default_sector_cap_for_opt) + 1e-12:
                selected.add(t)
                used_sec[s] += min_pos
                used_budget += min_pos
            # else: skip (sector is at cap for min_pos increments)

        if not selected:
            # Nothing to enforce safely; return original solution
            return first_w

        # Build second-pass per-asset bounds
        per2 = {}
        for t in tickers:
            if t in fixed_in_universe:
                v = fixed_in_universe[t]
                per2[t] = (v, v)
            elif t in selected:
                lb = max(min_pos, explicit_floors.get(t, 0.0))
                per2[t] = (lb, rc.max_weight)
            else:
                per2[t] = (0.0, 0.0)  # excluded from active book in second pass

        # === IMPORTANT: recompute a feasible dividend floor for these bounds ===
        # Upper-bound the achievable dividends with current per2 + sector caps.
        div_upper_2 = _max_dividend_income_upper_bound(
            tickers=tickers,
            sectors=sectors,
            div_yields=div,
            initial_nav=rc.initial_nav,
            per_asset_bounds=per2,
            fixed_weights=fixed_in_universe,
            sector_caps=sector_caps_for_opt,
            default_sector_cap=default_sector_cap_for_opt,
        )
        # effective_div_floor is the "abs floor" we used in first pass (outer scope).
        # Bring it down if it exceeds what’s achievable now that many names are zeroed.
        # (Note: rc.div_slack is already applied when the constraint is enforced.)
        div_floor_abs_2 = effective_div_floor
        if div_upper_2 > 0:
            # Convert the absolute *income* upper bound to an absolute floor
            # that will meet the floor after slack is applied inside the optimizer
            # (i.e., floor_abs * (1 - slack) <= div_upper_2).
            bound_as_abs = div_upper_2 / max(1e-12, (1.0 - rc.div_slack))
            if bound_as_abs < div_floor_abs_2:
                print(f"[warn] Second-pass dividend floor tightened from ${div_floor_abs_2:,.0f} "
                      f"to ${bound_as_abs * 0.99:,.0f} to remain feasible under min-position selections.")
                div_floor_abs_2 = bound_as_abs * 0.99  # 1% cushion

        # Final solve (sector targets kept; if it still fails we’ll try once without them)
        try:
            w2 = optimize_weights(
                mu=mu_all, cov=cov, benchmark_w=bench_w, sectors_series=sectors, prev_w=None,
                min_w=rc.min_weight, max_w=rc.max_weight,
                sector_caps=sector_caps_for_opt,
                default_sector_cap=default_sector_cap_for_opt,
                turnover_cap=rc.turnover_cap, te_annual_target=rc.te_annual_target,
                div_yields=div, initial_nav=rc.initial_nav, div_floor_abs=div_floor_abs_2,
                div_slack=rc.div_slack, risk_aversion=rc.risk_aversion,
                per_asset_bounds=per2,
                fixed_weights=fixed_in_universe,
                sector_targets=sector_targets,
                sector_target_penalty=sector_target_penalty,
                te_upper_mult=cfg.get("te_upper_mult"),
            )

            return w2
        except Exception:
            # Retry once with sector targets disabled (caps still enforced)
            w2 = optimize_weights(
                mu=mu_all, cov=cov, benchmark_w=bench_w, sectors_series=sectors, prev_w=None,
                min_w=rc.min_weight, max_w=rc.max_weight,
                sector_caps=sector_caps_for_opt,
                default_sector_cap=default_sector_cap_for_opt,
                turnover_cap=rc.turnover_cap, te_annual_target=rc.te_annual_target,
                div_yields=div, initial_nav=rc.initial_nav, div_floor_abs=div_floor_abs_2,
                div_slack=rc.div_slack, risk_aversion=rc.risk_aversion,
                per_asset_bounds=per2,
                fixed_weights=fixed_in_universe,
                sector_targets={},  # soft pulls off
                sector_target_penalty=0.0,
                te_upper_mult=cfg.get("te_upper_mult"),
            )
            return w2

    if args.mode == 'optimize':
        prev_w = None if args.ignore_holdings else (holdings if holdings is not None else None)

        # Build μ for all tickers: BL for non-bonds; bonds = 4.25%
        mu_all = pd.Series(index=tickers, dtype=float)
        mu_all.loc[tickers_ex_bonds] = mu_bl.reindex(tickers_ex_bonds).fillna(0.0).values
        if "BONDS" in tickers:
            mu_all.loc["BONDS"] = y_over.get("BONDS", 0.0425)

        # --- APPLY μ OVERRIDES FROM CONFIG (if any) ---
        mu_ovr = cfg.get("mu_overrides") or {}
        if mu_ovr:
            # normalize keys to UPPER and coerce to float
            changed = {}
            for k, v in mu_ovr.items():
                ku = str(k).upper()
                if ku in mu_all.index:
                    mu_all.loc[ku] = float(v)
                    changed[ku] = float(v)
            if changed:
                print("[info] Applied mu_overrides to:", ", ".join(f"{k}={v:.2%}" for k, v in changed.items()))
            else:
                print("[warn] mu_overrides provided but none matched tickers in the optimization universe.")

        # --- ban non-positive μ from getting any active weight (sleeves still exact) ---
        for t in mu_all.index:
            if t in fixed_in_universe:  # keep sleeves pinned
                continue
            if float(mu_all.get(t, 0.0)) <= 0.0:
                per_asset_bounds[t] = (0.0, 0.0)  # freeze out

        # Convenience wrapper to try with different relax settings
        def _try_opt(turnover_cap, div_floor_abs, te_target, per_name_cap, use_sector_caps=True, use_sector_targets=True):
            pab = per_asset_bounds.copy()
            if per_name_cap is not None:
                for t in pab:
                    lb, ub = pab[t]
                    if ub is None or ub > per_name_cap:
                        pab[t] = (lb, per_name_cap)
                for t, v in fixed_in_universe.items():
                    pab[t] = (v, v)
                if "NVDA" in pab and pab["NVDA"][0] is not None:
                    pab["NVDA"] = (pab["NVDA"][0], max(pab["NVDA"][1], pab["NVDA"][0]))


            return optimize_weights(
                mu=mu_all, cov=cov, benchmark_w=bench_w, sectors_series=sectors,
                prev_w=(None if args.ignore_holdings else prev_w),
                min_w=rc.min_weight, max_w=rc.max_weight,
                sector_caps=(sector_caps_for_opt if use_sector_caps else {}),
                default_sector_cap=(default_sector_cap_for_opt if use_sector_caps else 1.0),
                turnover_cap=turnover_cap, te_annual_target=te_target,
                div_yields=div, initial_nav=rc.initial_nav,
                div_floor_abs=div_floor_abs, div_slack=rc.div_slack,
                risk_aversion=rc.risk_aversion,
                per_asset_bounds=pab, fixed_weights=fixed_in_universe,
                sector_targets=(sector_targets if use_sector_targets else {}),
                sector_target_penalty=(sector_target_penalty_cfg if use_sector_targets else 0.0),
                te_upper_mult = cfg.get("te_upper_mult")
            )

        attempt_log = []
        w_star = None
        try:
            # A: Intended constraints with *feasible* dividend floor
            w_star = _try_opt(rc.turnover_cap if prev_w is not None else 1.0,
                              effective_div_floor, rc.te_annual_target, rc.max_weight, True, True)
            attempt_log.append("A: original (with feasible dividend floor)")
        except Exception as eA:
            # if not args.debug_relax:
            #     raise
            try:
                # B: relax turnover
                w_star = _try_opt(1.0, effective_div_floor, rc.te_annual_target, rc.max_weight, True, True)
                attempt_log.append("B: relax turnover")
            except Exception:
                try:
                    # C: relax sector targets (keep caps)
                    w_star = _try_opt(1.0, effective_div_floor, rc.te_annual_target, rc.max_weight, True, False)
                    attempt_log.append("C: drop sector targets")
                except Exception:
                    try:
                        # D: loosen TE & per-name cap
                        w_star = _try_opt(1.0, effective_div_floor, max(0.15, rc.te_annual_target), max(0.08, rc.max_weight), True, False)
                        attempt_log.append("D: loosen TE/per-name; no targets")
                    except Exception:
                        # E: drop sector caps & targets
                        w_star = _try_opt(1.0, effective_div_floor, max(0.15, rc.te_annual_target), max(0.10, rc.max_weight), False, False)
                        attempt_log.append("E: drop sector caps/targets")

        if enforce_min_pos and min_pos > 0:
            w_star = _second_pass_min_position(
                first_w=w_star, min_pos=min_pos, tickers=tickers,
                fixed_in_universe=fixed_in_universe, rc=rc,
                mu_all=mu_all, cov=cov, bench_w=bench_w, sectors=sectors, div=div,
                sector_caps_for_opt=sector_caps_for_opt,
                default_sector_cap_for_opt=default_sector_cap_for_opt,
                sector_targets=sector_targets,
                sector_target_penalty=sector_target_penalty_cfg
            )

        # Pin sleeves exactly and renormalize active
        w_star = _pin_sleeves_exact(w_star, fixed_in_universe)

        # --- NEW: clip numerical dust and re-scale active only ---
        EPS = 1e-6  # 0.0001% of NAV
        w_star = w_star.where(w_star >= EPS, 0.0)

        # Keep sleeves exact; renormalize only the active sleeve to its target
        active = [t for t in w_star.index if t not in fixed_in_universe]
        target_active = 1.0 - sum(fixed_in_universe.values())
        active_sum = float(w_star.loc[active].sum())
        if active_sum > 0 and target_active > 0:
            w_star.loc[active] *= (target_active / active_sum)

        # Write outputs
        outdir = Path(args.outdir)
        w_star.to_frame('weight').to_csv(outdir / 'weights_latest.csv')
        mu_all.rename_axis("asset").to_frame("mu").to_csv(outdir / 'mu_bl.csv')
        cov.to_csv(outdir / 'cov_for_mc.csv')
        w_star.rename("weight").rename_axis("asset").to_csv(outdir / 'weights_latest_full.csv')

        # Sector breakdown
        sec_df = (
            pd.DataFrame({"weight": w_star})
            .assign(sector=lambda d: d.index.map(sectors.to_dict()))
            .groupby("sector")["weight"].sum()
            .sort_values(ascending=False)
        )
        sec_df.to_csv(outdir / "sector_breakdown.csv")

        # --- DEBUG: check for negative-μ names that still carry material weight ---
        df_diag = pd.concat(
            [w_star.rename('weight'), mu_all.rename('mu')],
            axis=1
        ).dropna()
        df_diag['contrib'] = df_diag['weight'] * df_diag['mu']

        EPS = 1e-6  # ignore numerical dust
        neg = df_diag[(df_diag['mu'] <= 0) & (df_diag['weight'] > EPS)]

        print("\n[diag] Any negative-μ names with *material* weight? ->", not neg.empty)
        if not neg.empty:
            print(neg[['weight', 'mu', 'contrib']]
                  .sort_values('weight', ascending=False)
                  .to_string(float_format=lambda x: f"{x:.6f}"))
        # -------------------------------------------------------------------------

        port_mu = float((w_star * mu_all).sum())
        print(f"\n[diag] Portfolio expected return (Σ w·μ): {port_mu:.2%}")

        # Quick TE check
        Sigma = cov.loc[w_star.index, w_star.index].values
        d = (w_star - bench_w.reindex(w_star.index).fillna(0.0)).values.reshape(-1,1)
        te_daily = float(np.sqrt(max(0.0, (d.T @ Sigma @ d).item())))
        te_annual = float(te_daily * np.sqrt(252.0))

        print("=== Optimize summary ===")
        if attempt_log:
            print("Solve path:", " → ".join(attempt_log))

        print("Optimized weights (latest):")
        ws = w_star.sort_values(ascending=False)
        min_show = 0.0001  # 1 bp
        ws_disp = ws[ws >= min_show]
        print(_fmt_series_pct(ws_disp, 2).to_string())

        print("\nRealized sector weights (saved to sector_breakdown.csv):")
        print(_fmt_series_pct(sec_df, 2).to_string())

        print(f"\nApprox TE(annual) ~ {te_annual:.2%}  (target={cfg['te_annual_target']:.2%})")
        inc = _achieved_div_dollars(w_star, div, rc.initial_nav)

        print(
            "Dividend $ implied ~ "
            f"{inc:,.0f}  "
            f"(floor ≥ {(effective_div_floor * (1.0 - rc.div_slack)):,.0f}; "
            f"configured floor={rc.div_floor_abs:,.0f}, slack={rc.div_slack:.0%})"
        )

    else:
        # Backtest
        bt = run_backtest(
            prices, sectors, div, mkt_w, rc,
            start=args.start or (pd.Timestamp(end_date) - pd.DateOffset(years=5)).strftime('%Y-%m-%d'),
            end=end_date, lookback_years=cfg.get('lookback_years', 3),
            views_cfg=views_cfg, green_energy=green_energy, holdings0=(None if args.ignore_holdings else holdings)
        )
        weights = bt['weights']; trades = bt['trades']
        outdir = Path(args.outdir)
        weights.to_parquet(outdir / 'weights.parquet')
        trades.to_parquet(outdir / 'trades.parquet')

        pr = prices[tickers].copy()
        port_r = portfolio_returns(weights, pr)
        bench_r = prices[bench].pct_change().reindex(port_r.index).fillna(0.0)
        eq_w = pd.Series(1.0 / len(tickers), index=tickers)
        eq_weights = weights.copy(); eq_weights.iloc[:] = eq_w.values
        eq_r = portfolio_returns(eq_weights, pr)

        # --- add these caches so report.py can reuse the exact data ---
        used_prices = prices[[c for c in tickers if c in prices.columns] + [bench]].sort_index()
        outdir.mkdir(parents=True, exist_ok=True)
        used_prices.to_parquet(outdir / 'prices_used.parquet')  # full price grid the backtest used
        port_r.rename('Strategy').to_frame().to_parquet(outdir / 'port_daily_returns.parquet')
        eq_r.rename('EqualWeight').to_frame().to_parquet(outdir / 'eqw_daily_returns.parquet')
        # ----------------------------------------------------------------


        metrics = {
            "ann_return": ann_return(port_r),
            "ann_vol": ann_vol(port_r),
            "max_drawdown": max_drawdown(port_r),
            "sharpe": sharpe(port_r),
            "sortino": sortino(port_r),
            "omega": omega(port_r),
            "deflated_sharpe_ratio_approx": deflated_sharpe_ratio(port_r)
        }
        (outdir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()