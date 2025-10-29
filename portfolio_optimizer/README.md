# Portfolio Optimizer (BL/MVO/PMPT) — Semiannual

This is a reference implementation of a robust portfolio construction pipeline:
- Ledoit–Wolf covariance shrinkage
- Bayes–Stein mean shrinkage
- Black–Litterman posterior for views (Idzorek confidence)
- Constrained MVO in CVXPY with: sector caps, per-name brackets, turnover, tracking error target, dividend-income floor
- Semiannual rebalance with walk-forward OOS backtest
- Baseline 1/N and benchmark SPY
- PMPT metrics (Sortino, Omega), Deflated Sharpe Ratio, and SPA/Reality Check

> Date: 2025-10-23

## Quick start

1) Install deps (Python 3.10+ recommended):

```bash
pip install numpy pandas yfinance scikit-learn cvxpy pypfopt arch PyYAML matplotlib
```

2) Place your universe and (optionally) previous holdings:

- `data/universe.csv`: one TICKER per line, or columns `ticker,sector`.
- `data/holdings.csv`: columns `ticker,weight` (weights sum to 1).

3) Adjust settings in `config.yml` and `views.yaml` (sector tilts preloaded).

4) Run a one-shot optimization for the latest date:

```bash
python run_pipeline.py --mode optimize --asof YYYY-MM-DD
```

5) Run the semiannual walk-forward backtest:

```bash
python run_pipeline.py --mode backtest --start 2015-01-01 --end 2025-01-01
```

Outputs saved to `outputs/weights.parquet`, `outputs/trades.parquet`, and `outputs/metrics.json`.

> This code is **reference** quality — tune further for production (logging, retries, data validation, etc.).


## Factor model (reporting/limits)

- Provide exposures `data/factor_exposures.csv` with columns: `ticker,value,quality,profitability,momentum`.
- Provide factor covariance either as `data/factor_cov.csv` (matrix) **or** set `ff5_excel` in `config.yml` to a Fama-French 5 Factors Excel (we proxy *value*=HML, *profitability*=RMW, *quality*≈(RMW+CMA)/2, *momentum* requires your own data).
- Set desired risk budget shares in `config.yml -> factor_model -> risk_share_targets` (e.g., 0.30/0.30/0.20/0.20). The pipeline writes `outputs/factor_report.json` with exposures and risk-contribution shares.
- To enforce exposure bounds while keeping Ledoit–Wolf Σ in the objective, set:
  ```yaml
  factor_model:
    report_only: false
    enforce_limits: true
    exposure_bounds:
      momentum: [-0.05, 0.30]
      value: [0.00, 0.50]
  ```


### Using scores instead of exposures
If you can't provide factor exposures **B**, place `data/scores.csv` with columns
`ticker,quality_score,profitability_score,value_score,composite_score`.
The pipeline converts scores to **exposures** by cross-sectional z-scoring each metric at the rebalance,
and derives **momentum exposure** from 12–1 price momentum when prices are available.
Set `config.yml -> factor_model.enabled: true` and run with `--scores data/scores.csv`.
