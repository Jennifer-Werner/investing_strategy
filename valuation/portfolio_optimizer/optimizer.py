# optimizer.py
from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
from risk import chol_psd

def build_sector_matrix(
    tickers: List[str],
    sectors: pd.Series,
    sector_caps: Dict[str, float],
    default_cap: float
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    unique_sectors = sorted(set([(sectors.get(t, 'Unknown') or 'Unknown') for t in tickers]))
    S = np.zeros((len(unique_sectors), len(tickers)))
    caps = np.zeros(len(unique_sectors))
    for i, sec in enumerate(unique_sectors):
        caps[i] = float(sector_caps.get(sec, default_cap))
        for j, t in enumerate(tickers):
            if (sectors.get(t, 'Unknown') or 'Unknown') == sec:
                S[i, j] = 1.0
    return S, caps, unique_sectors

def optimize_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    benchmark_w: pd.Series,
    sectors_series: Optional[pd.Series] = None,
    prev_w: Optional[pd.Series] = None,
    min_w: float = 0.0,
    max_w: float = 0.05,
    sector_caps: Optional[Dict[str, float]] = None,
    default_sector_cap: float = 0.30,
    turnover_cap: float = 0.20,
    te_annual_target: float = 0.04,
    div_yields: Optional[pd.Series] = None,
    initial_nav: float = 1.0,
    div_floor_abs: float = 0.0,
    div_slack: float = 0.0,
    risk_aversion: Optional[float] = None,
    factor_B: Optional[pd.DataFrame] = None,
    factor_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    enforce_factor_limits: bool = False,
    per_asset_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    fixed_weights: Optional[Dict[str, float]] = None,
    sector_targets: Optional[Dict[str, float]] = None,
    sector_target_penalty: float = 0.0,
    # NEW: TE ceiling multiplier (None = no ceiling)
    te_upper_mult: Optional[float] = None,
) -> pd.Series:
    """Convex MVO with TE-centering (no hard lower bound), optional TE ceiling, turnover L1,
    dividend floor, sector caps, fixed sleeves, soft sector targets."""
    tickers = list(mu.index)
    n = len(tickers)

    w = cp.Variable(n)

    Sigma = cov.loc[tickers, tickers].values
    lam = float(risk_aversion) if risk_aversion is not None else 1.0

    # Soft sector targets (optional)
    soft_penalty = 0
    if (sector_targets is not None) and (sectors_series is not None) and (sector_target_penalty > 0):
        uniq_secs = sorted(set([(sectors_series.get(t, 'Unknown') or 'Unknown') for t in tickers]))
        S_soft = np.zeros((len(uniq_secs), n))
        target = np.zeros(len(uniq_secs))
        for i, sec in enumerate(uniq_secs):
            target[i] = float(sector_targets.get(sec, 0.0))
            for j, t in enumerate(tickers):
                if (sectors_series.get(t, 'Unknown') or 'Unknown') == sec:
                    S_soft[i, j] = 1.0
        sec_w = S_soft @ w
        soft_penalty = sector_target_penalty * cp.sum_squares(sec_w - target)

    obj_expr = lam * cp.quad_form(w, Sigma) - mu.values @ w + soft_penalty

    cons: List = []
    cons += [cp.sum(w) == 1.0, w >= 0]

    # Per-asset bounds & sleeves exact
    if per_asset_bounds:
        for j, t in enumerate(tickers):
            if fixed_weights and (t in fixed_weights):
                v = float(fixed_weights[t])
                cons += [w[j] == v]
            else:
                lb, ub = per_asset_bounds.get(t, (None, None))
                if lb is not None:
                    cons += [w[j] >= float(lb)]
                if ub is not None:
                    cons += [w[j] <= float(ub)]
    else:
        cons += [w >= float(min_w), w <= float(max_w)]
        if fixed_weights:
            for j, t in enumerate(tickers):
                if t in fixed_weights:
                    cons += [w[j] == float(fixed_weights[t])]

    # Sector hard caps — exclude sleeves from LHS so sleeves don't consume cap
    if sectors_series is not None:
        S, caps, _ = build_sector_matrix(tickers, sectors_series, sector_caps or {}, float(default_sector_cap))
        fixed_idx = {j for j, t in enumerate(tickers) if fixed_weights and t in fixed_weights}
        for i in range(S.shape[0]):
            row = S[i, :].copy()
            if fixed_idx:
                for j in fixed_idx:
                    row[j] = 0.0
            cons += [row @ w <= caps[i]]

    # Tracking error centering (convex)
    te_daily_target = float(te_annual_target) / np.sqrt(252.0)
    Sigma_sqrt = chol_psd(Sigma)
    wb = benchmark_w.reindex(tickers).fillna(0.0).values if benchmark_w is not None else np.zeros(n)

    z_te = cp.Variable(nonneg=True)
    cons += [z_te >= cp.norm(Sigma_sqrt @ (w - wb), 2)]
    if te_upper_mult is not None and te_upper_mult > 0:
        cons += [z_te <= te_upper_mult * te_daily_target]

    gamma_te_center = 500.0
    obj_expr += gamma_te_center * cp.square(z_te - te_daily_target)

    # Turnover L1
    if (prev_w is not None) and (set(prev_w.index) >= set(tickers)):
        cons += [cp.norm1(w - prev_w.reindex(tickers).fillna(0.0).values) <= float(turnover_cap)]

    # Dividend floor
    if div_yields is not None and div_floor_abs > 0:
        y = div_yields.reindex(tickers).fillna(0.0).values
        cons += [float(initial_nav) * (y @ w) >= float(div_floor_abs) * (1.0 - float(div_slack))]

    # Factor bounds (optional)
    if enforce_factor_limits and (factor_B is not None) and (factor_bounds is not None):
        if set(factor_B.index) >= set(tickers):
            B = factor_B.loc[tickers].values
            for k, fac in enumerate(factor_B.columns):
                bounds = factor_bounds.get(fac)
                if not bounds:
                    continue
                lb, ub = bounds if isinstance(bounds, (tuple, list)) else (None, None)
                expr = B[:, k] @ (w - wb)
                if lb is not None:
                    cons += [expr >= float(lb)]
                if ub is not None:
                    cons += [expr <= float(ub)]

    problem = cp.Problem(cp.Minimize(obj_expr), cons)
    solved = False
    last_status = None
    last_solver = None
    installed = cp.installed_solvers()

    solver_order = [s for s in ["CLARABEL", "ECOS", "OSQP", "SCS"] if s in installed]
    for solver in solver_order:
        try:
            if solver == "SCS":
                problem.solve(solver=solver, verbose=False, max_iters=20000)
            else:
                problem.solve(solver=solver, verbose=False)
            last_status = problem.status
            last_solver = solver
            if w.value is not None and last_status in ("optimal", "optimal_inaccurate"):
                solved = True
                break
        except Exception:
            continue

    if not solved or w.value is None:
        raise RuntimeError(
            f"Optimization failed (status={last_status}, solver_tried={solver_order}, last_solver={last_solver}). "
            f"Likely infeasible with current TE/dividend/sector caps/per-asset bounds. "
            f"Try relaxing dividend floor, sector caps, per-name bounds, or TE limits."
        )

    sol = pd.Series(np.clip(w.value, 0.0, 1.0), index=tickers)
    tot = float(sol.sum())
    return sol / tot if abs(tot - 1.0) > 1e-6 else sol