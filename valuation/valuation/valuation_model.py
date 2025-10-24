"""
COMPREHENSIVE VALUATION MODEL
DCF + Relative Valuation (Comps) + EVA

Location: valuation/valuation/valuation_model.py
Outputs: valuation/valuation/output/sheets/ and /images/
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ----------------------- DIRECTORY SETUP ----------------------- #
def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("output/sheets", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)


ensure_directories()

# ----------------------- SETTINGS ----------------------- #
EXPORT_EXCEL = "output/sheets/Valuation_Analysis.xlsx"
CAPM_INPUT_FILE = "../risk-models/output/sheets/Risk_Model_Output.xlsx"

# Market assumptions
TAX_RATE = 0.21  # Corporate tax rate

# --- New flags & guardrails ---
MID_YEAR = True                # mid-year convention for discounting
TV_WARN_THRESHOLD = 0.70       # warn if PV(TV) > 70% of EV
TV_FADE_TRIGGER = 0.75         # trigger a fade stage if PV(TV) share exceeds 75%
FADE_YEARS = 3                 # years to fade growth to terminal
SHRINK_TO_YAHOO = False        # keep but OFF by default
PER_TICKER_IMAGES = True       # write per-ticker PNGs under output/images/<tickerlower>

# --- NEW: robustness flags (requested fixes) ---
GROWTH_FLOOR = 0.02            # minimum realistic growth floor (≈ inflation)
PEERS_US_ONLY = True           # restrict comps to US / USD to reduce accounting noise
EVA_INCLUDE_GOODWILL = False   # optionally include goodwill in invested capital
EVA_INCLUDE_INTANG = False     # optionally include intangibles in invested capital

DEBUG_YF = True  # turn on to print Yahoo Finance reconciliation per ticker

print("=" * 70)
print("COMPREHENSIVE VALUATION MODEL")
print("DCF + Relative Valuation + EVA + Margin of Safety")
print("=" * 70)

# ----------------------- LOAD PRE-CALCULATED CAPM ----------------------- #

import math
def fmt_money(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "$-"
    absx = abs(x)
    if absx >= 1e12:  # trillion
        return f"${x/1e12:,.2f}T"
    if absx >= 1e9:   # billion
        return f"${x/1e9:,.2f}B"
    if absx >= 1e6:   # million
        return f"${x/1e6:,.2f}M"
    return f"${x:,.0f}"

def fmt_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-%"
    return f"{100*x:.1f}%"

def load_capm_data():
    """Load pre-calculated CAPM results from risk models"""
    try:
        capm_df = pd.read_excel(CAPM_INPUT_FILE, sheet_name='CAPM_Results')
        print(f"✅ Loaded CAPM data for {len(capm_df)} stocks from {CAPM_INPUT_FILE}")
        return capm_df
    except FileNotFoundError:
        print(f"⚠️  CAPM file not found at {CAPM_INPUT_FILE}")
        print(f"    Will calculate CAPM from scratch for this analysis")
        return None
    except Exception as e:
        print(f"⚠️  Error loading CAPM data: {e}")
        print(f"    Will calculate CAPM from scratch for this analysis")
        return None


# ----------------------- AUTO-CALCULATE VALUATION PARAMETERS ----------------------- #
def calculate_valuation_parameters():
    """
    Automatically calculate valuation parameters based on financial theory:

    1. Terminal Growth Rate: Conservative estimate between historical inflation (2-3%)
       and GDP growth (4-5%). Default: 2.5%

    2. WACC Sensitivity Range: ±1.5% (creating a 3% total spread, which is standard
       for mature companies)

    3. Terminal Growth Sensitivity Range: ±0.5% (allowing range of 2.0% to 3.0%)
    """
    # Terminal growth rate: Conservative estimate aligned with long-term inflation
    terminal_growth = 0.025  # 2.5%

    # WACC sensitivity: ±1.5% creates a 3% total range (e.g., 7%-10% if base is 8.5%)
    wacc_sensitivity = 0.015  # ±1.5%

    # Terminal growth sensitivity: ±0.5% creates reasonable range
    terminal_g_sensitivity = 0.005  # ±0.5%

    return terminal_growth, wacc_sensitivity, terminal_g_sensitivity


# ----------------------- USER INPUTS ----------------------- #
n = int(input("\nEnter the number of tickers to analyze: "))
tickers = [input(f"Enter ticker {i + 1}: ").upper() for i in range(n)]

print("\n--- Valuation Parameters ---")
DCF_HORIZON = int(input("DCF forecast horizon (years, e.g., 5-10): "))

# Auto-calculate parameters
TERMINAL_GROWTH, WACC_SENSITIVITY, TERMINAL_G_SENSITIVITY = calculate_valuation_parameters()

print(f"\n📊 Auto-Calculated Valuation Parameters:")
print(f"   Terminal Growth Rate: {TERMINAL_GROWTH*100:.2f}%")
print(f"   WACC Sensitivity Range: ±{WACC_SENSITIVITY*100:.2f}%")
print(f"   Terminal Growth Sensitivity Range: ±{TERMINAL_G_SENSITIVITY*100:.2f}%")

# Market assumptions
RISK_FREE_RATE = 0.045
MARKET_RISK_PREMIUM = 0.07

# Weights for blended fair value
DCF_WEIGHT = 0.50
COMPS_WEIGHT = 0.30
EVA_WEIGHT = 0.20

print(f"\nBlended Fair Value Weights: DCF={DCF_WEIGHT*100:.0f}%, Comps={COMPS_WEIGHT*100:.0f}%, EVA={EVA_WEIGHT*100:.0f}%")

# Load pre-calculated CAPM
print("\n--- Loading Risk Model Data ---")

# ----------------------- HELPERS ----------------------- #
def safe_get(dct, key, default=None):
    try:
        v = dct.get(key, default)
    except Exception:
        v = default
    return default if v is None else v

def _get_total_debt_from_bs(balance_sheet):
    try:
        if 'Total Debt' in balance_sheet.index:
            td = float(balance_sheet.loc['Total Debt'].iloc[0])
            if td and td > 0:
                return td
    except Exception:
        pass
    try:
        ltd = float(balance_sheet.loc['Long Term Debt'].iloc[0]) if 'Long Term Debt' in balance_sheet.index else 0.0
    except Exception:
        ltd = 0.0
    try:
        std = float(balance_sheet.loc['Current Debt'].iloc[0]) if 'Current Debt' in balance_sheet.index else 0.0
    except Exception:
        std = 0.0
    return (0.0 if pd.isna(ltd) else ltd) + (0.0 if pd.isna(std) else std)

def _discount_factor(wacc, t):
    return (1 + wacc) ** (t - 0.5) if MID_YEAR else (1 + wacc) ** t

def _pv_of_projections(fcf_list, wacc):
    return sum(f['fcf'] / _discount_factor(wacc, f['year']) for f in fcf_list)

def _compute_nwc_ratio(balance_sheet, income_stmt):
    # NWC ≈ (CA - Cash) - (CL - Current Debt), relative to revenue
    try:
        _ = income_stmt.columns  # ensure frame exists
    except Exception:
        return 0.12
    def _vec(idx):
        try:
            return balance_sheet.loc[idx].astype(float).values[:3]
        except Exception:
            return None
    def _vec_is(idx):
        try:
            return income_stmt.loc[idx].astype(float).values[:3]
        except Exception:
            return None
    ca = _vec('Total Current Assets')
    cl = _vec('Total Current Liabilities')
    cash = _vec('Cash And Cash Equivalents')
    cur_debt = _vec('Current Debt')
    rev = _vec_is('Total Revenue')
    if rev is None or (np.array(rev) <= 0).any():
        return 0.12
    ca = np.zeros(3) if ca is None else ca
    cl = np.zeros(3) if cl is None else cl
    cash = np.zeros(3) if cash is None else cash
    cur_debt = np.zeros(3) if cur_debt is None else cur_debt
    nwc = (ca - cash) - (cl - cur_debt)
    with np.errstate(all='ignore'):
        ratios = nwc / rev
        ratios = ratios[np.isfinite(ratios)]
    return float(np.median(ratios)) if ratios.size else 0.12

# --- NEW HELPERS ---
def _effective_cash_tax(income_stmt):
    """Multi-year effective tax rate; bounded to [10%, 35%]."""
    try:
        taxes, pretax = [], []
        for i in range(min(3, income_stmt.shape[1])):
            t = None; ptx = None
            for k in ['Tax Provision', 'Income Tax Expense']:
                if k in income_stmt.index:
                    t = income_stmt.loc[k].astype(float).values[i]; break
            for k in ['Pretax Income', 'Earnings Before Tax']:
                if k in income_stmt.index:
                    ptx = income_stmt.loc[k].astype(float).values[i]; break
            if t is not None and ptx is not None and ptx != 0:
                taxes.append(t); pretax.append(ptx)
        if taxes and pretax:
            ratios = np.array(taxes, dtype=float) / np.array(pretax, dtype=float)
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                return float(np.clip(np.median(ratios), 0.10, 0.35))
    except Exception:
        pass
    return TAX_RATE

def _historical_fcf_positive(cash_flow: pd.DataFrame) -> bool:
    """Median of last ~3 periods of FCF > 0?"""
    try:
        if 'Free Cash Flow' in cash_flow.index:
            fcf_vec = cash_flow.loc['Free Cash Flow'].astype(float).values[:3]
        else:
            cfo = cash_flow.loc['Total Cash From Operating Activities'].astype(float).values[:3] \
                  if 'Total Cash From Operating Activities' in cash_flow.index else None
            capex = cash_flow.loc['Capital Expenditure'].astype(float).values[:3] \
                    if 'Capital Expenditure' in cash_flow.index else None
            fcf_vec = cfo - np.abs(capex) if (cfo is not None and capex is not None) else None
        if fcf_vec is None: return False
        v = np.array(fcf_vec, dtype=float)
        v = v[np.isfinite(v)]
        return v.size > 0 and np.median(v) > 0
    except Exception:
        return False

def _first_number(*vals):
    for v in vals:
        try:
            if v is not None and np.isfinite(float(v)) and float(v) > 0:
                return float(v)
        except Exception:
            pass
    return None

def _reconcile_price_shares(info: dict, fast: dict):
    """
    Choose best price/shares/market_cap among Yahoo sources and reconcile inconsistencies.
    Returns (price, shares, market_cap, currency, exchange, issues[list of strings]).
    Strategy: prefer consistency — if market cap & price exist, derive shares = mcap/price.
    """
    issues = []
    # Price candidates
    price = _first_number(
        (fast or {}).get('last_price'),
        info.get('currentPrice'),
        info.get('regularMarketPrice'),
        info.get('regularMarketPreviousClose'),
    )
    # Market cap candidates
    market_cap = _first_number((fast or {}).get('market_cap'), info.get('marketCap'))
    # Shares candidates
    shares_basic   = _first_number((fast or {}).get('shares_outstanding'), info.get('sharesOutstanding'))
    shares_implied = _first_number(info.get('impliedSharesOutstanding'))

    # Consistency-first: if both price and mcap exist, compute shares from them
    if market_cap and price:
        shares = market_cap / price
        issues.append("shares set = market_cap / price (consistency first)")
    else:
        # Fall back to diluted/implied if available; else basic
        shares = shares_implied or shares_basic

    # If no market cap but price & shares exist, compute it
    if market_cap is None and price and shares:
        market_cap = price * shares
        issues.append("computed market_cap = price * shares")

    # Diagnostic if inconsistent
    if price and shares and market_cap:
        implied_mkt = price * shares
        diff = abs(implied_mkt - market_cap) / market_cap if market_cap else 0.0
        if diff > 0.05:
            issues.append(f"⚠️ price*shares vs market_cap diff {diff*100:.1f}%")

    currency = info.get('currency') or (fast or {}).get('currency')
    exchange = info.get('exchange') or info.get('fullExchangeName')
    return price, shares, market_cap, currency, exchange, issues

def _yahoo_diagnostics(ticker: str, stock, info: dict, fast: dict, price, shares, market_cap):
    """
    Print a compact diagnostic to explain why Yahoo values may be 'off'.
    """
    try:
        hist_close = None
        hist = stock.history(period="5d")
        if hasattr(hist, "close") or ("Close" in hist.columns or "close" in hist.columns):
            close_col = "Close" if "Close" in hist.columns else ("close" if "close" in hist.columns else None)
            if close_col and not hist.empty:
                hist_close = float(hist[close_col].dropna().iloc[-1])
    except Exception:
        hist_close = None

    target = info.get('targetMeanPrice') or info.get('targetMedianPrice')
    analysts = info.get('numberOfAnalystOpinions')
    currency = info.get('currency') or (fast.get('currency') if fast else None)

    print("     — Yahoo Diagnostics —")
    print(f"       currency={currency}  exchange={info.get('fullExchangeName') or info.get('exchange')}")
    print(f"       price: last={price}  rmkt={info.get('regularMarketPrice')}  prevClose={info.get('regularMarketPreviousClose')}  fast={fast.get('last_price') if fast else None}  histClose={hist_close}")
    print(f"       market_cap: info={info.get('marketCap')}  fast={fast.get('market_cap') if fast else None}")
    print(f"       shares: info={info.get('sharesOutstanding')}  implied={info.get('impliedSharesOutstanding')}  fast={fast.get('shares_outstanding') if fast else None}")
    print(f"       analysts_target_mean={target}  #opinions={analysts}")

    # Simple heuristics for "off" explanations
    reasons = []
    if target and price:
        dev = abs(float(target) - float(price)) / float(price)
        if dev > 0.5:
            reasons.append(f"Target deviates {dev*100:.1f}% from price (analyst horizon vs spot)")

    if hist_close and price:
        mkt_dev = abs(price - hist_close) / hist_close
        if mkt_dev > 0.05:
            reasons.append(f"Price deviates {mkt_dev*100:.1f}% from recent close (pre/post-market or delayed quotes)")

    if (info.get('marketCap') is None) and (fast and fast.get('market_cap') is None):
        reasons.append("Missing market cap; reconstructed from price*shares")

    # International/currency caveat
    if currency and str(currency).upper() != "USD":
        reasons.append(f"Non-USD currency={currency} (ensure everything else is same currency)")

    if reasons:
        for r in reasons:
            print(f"       ↳ {r}")

capm_data = load_capm_data()


# ----------------------- DATA COLLECTION ----------------------- #
def get_financial_data(ticker):
    """Fetch financials + robust Yahoo price/shares/market_cap with sanity checks."""
    try:
        stock = yf.Ticker(ticker)

        # Statements
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        # Yahoo info (beware: can be stale) + fast_info (usually fresher)
        try:
            info = stock.info or {}
        except Exception:
            info = {}
        try:
            fast = stock.fast_info or {}
        except Exception:
            fast = {}

        # Reconcile key market fields
        price, shares, market_cap, currency, exchange, issues = _reconcile_price_shares(info, fast)

        if not price:
            # As a last resort, try recent close
            try:
                hist = stock.history(period="5d")
                if not hist.empty:
                    price = float(hist['Close'].dropna().iloc[-1])
                    issues.append("price from history Close")
            except Exception:
                pass

        # Ensure statements exist
        if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty or cash_flow is None or cash_flow.empty:
            print(f"❌ {ticker}: Insufficient financial data")
            return None

        # Yahoo analyst target
        yahoo_target = info.get('targetMeanPrice', None) or info.get('targetMedianPrice', None)

        # Optionally print diagnostics
        if DEBUG_YF:
            _yahoo_diagnostics(ticker, stock, info, fast, price, shares, market_cap)

        if not price or not shares:
            print(f"❌ {ticker}: Missing price or shares after reconciliation")
            return None

        print(f"✅ {ticker}: Financial data loaded (currency={currency}, exchange={exchange})")
        for note in issues:
            print(f"     ℹ️  {note}")

        return {
            'ticker': ticker,
            'info': info,
            'fast_info': fast,
            'income_stmt': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'current_price': float(price),
            'shares_outstanding': float(shares),
            'market_cap': float(market_cap) if market_cap else None,
            'yahoo_target': yahoo_target,
            'currency': currency,
            'exchange': exchange
        }

    except Exception as e:
        print(f"❌ {ticker}: Error fetching data - {e}")
        return None


def get_cost_of_equity(ticker, capm_df, data):
    """Get cost of equity from pre-calculated CAPM or calculate it"""
    if capm_df is not None and ticker in capm_df['Ticker'].values:
        # Use pre-calculated CAPM
        row = capm_df[capm_df['Ticker'] == ticker].iloc[0]
        cost_of_equity = row['Expected_Return']
        beta = row['Beta']
        print(f"     ✅ Using pre-calculated CAPM: Re = {cost_of_equity:.2%}, β = {beta:.2f}")
        return cost_of_equity, beta
    else:
        # Calculate CAPM from scratch
        info = data['info']
        beta = info.get('beta', 1.0)
        if beta is None or beta == 0:
            beta = 1.0
        cost_of_equity = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM
        print(f"     ⚠️  Calculated CAPM (no pre-calc): Re = {cost_of_equity:.2%}, β = {beta:.2f}")
        return cost_of_equity, beta



def calculate_wacc(data, ticker, capm_df=None):
    """Calculate WACC using CAPM inputs and robust debt"""
    balance_sheet = data['balance_sheet']

    cost_of_equity, beta = get_cost_of_equity(ticker, capm_df, data)

    current_price = data.get('current_price', 0)
    if current_price is None or np.isnan(current_price) or current_price <= 0:
        print(f"     ⚠️  Invalid price data, cannot calculate WACC")
        return None

    shares_outstanding = data.get('shares_outstanding', 0)
    if shares_outstanding is None or np.isnan(shares_outstanding) or shares_outstanding <= 0:
        print(f"     ⚠️  Invalid shares outstanding, cannot calculate WACC")
        return None

    total_debt = _get_total_debt_from_bs(balance_sheet)
    market_cap = current_price * shares_outstanding
    enterprise_value = market_cap + total_debt if total_debt >= 0 else market_cap

    equity_weight = market_cap / enterprise_value if enterprise_value > 0 else 1.0
    debt_weight   = total_debt / enterprise_value if enterprise_value > 0 else 0.0

    try:
        income_stmt = data['income_stmt']
        interest_expense = abs(float(income_stmt.loc['Interest Expense'].iloc[0])) if 'Interest Expense' in income_stmt.index else 0.0
        raw_rd = (interest_expense / total_debt) if total_debt > 0 else 0.05
        cost_of_debt = float(np.clip(raw_rd, 0.02, 0.15))
    except Exception:
        cost_of_debt = 0.05

    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - TAX_RATE))
    if np.isnan(wacc) or wacc <= 0:
        print(f"     ⚠️  Invalid WACC calculated → using cost of equity as fallback (review inputs)")
        wacc = cost_of_equity

    return {
        'wacc': wacc,
        'cost_of_equity': cost_of_equity,
        'cost_of_debt': cost_of_debt,
        'beta': beta,
        'equity_weight': equity_weight,
        'debt_weight': debt_weight,
        'total_debt': total_debt,
        'market_cap': market_cap,
        'enterprise_value': enterprise_value
    }



def project_financials(data, horizon=5):
    """Project FCFF with ΔNWC ratio and margin/growth fade"""
    income_stmt = data['income_stmt']
    balance_sheet = data['balance_sheet']
    cash_flow = data['cash_flow']
    try:
        rev = income_stmt.loc['Total Revenue'].astype(float).values[:3]
        if 'EBIT' in income_stmt.index:
            ebit = income_stmt.loc['EBIT'].astype(float).values[:3]
        else:
            ebit = income_stmt.loc['Operating Income'].astype(float).values[:3]
        if 'Capital Expenditure' in cash_flow.index:
            capex = np.abs(cash_flow.loc['Capital Expenditure'].astype(float).values[:3])
        else:
            capex = rev * 0.05
        if 'Depreciation And Amortization' in cash_flow.index:
            da = cash_flow.loc['Depreciation And Amortization'].astype(float).values[:3]
        else:
            da = rev * 0.03

        hist_growth = [(rev[i] - rev[i+1]) / rev[i+1] for i in range(len(rev)-1) if rev[i+1] > 0]
        revenue_growth = float(np.nanmedian(hist_growth)) if hist_growth else 0.03
        ebit_margin = float(np.nanmedian(ebit / rev))
        capex_to_rev = float(np.nanmedian(capex / rev))
        da_to_rev = float(np.nanmedian(da / rev))
        nwc_ratio = _compute_nwc_ratio(balance_sheet, income_stmt)

        target_margin = np.clip(ebit_margin, 0.05, 0.35)
        margin_fade = 0.20
        growth_decay = 0.85

        last_rev = rev[0]
        last_margin = ebit_margin
        projections = []
        # Historical FCF sanity signal
        hist_fcf_positive = _historical_fcf_positive(cash_flow)

        for t in range(1, horizon + 1):
            # Apply growth floor to avoid unrealistic decays
            g = max(revenue_growth * (growth_decay ** (t - 1)), max(GROWTH_FLOOR, TERMINAL_GROWTH))
            next_rev = float(last_rev * (1 + g))

            last_margin = float(last_margin + (target_margin - last_margin) * margin_fade)
            ebit_t = next_rev * last_margin
            nopat = ebit_t * (1 - TAX_RATE)

            da_t = next_rev * da_to_rev
            capex_t = next_rev * (0.5 * capex_to_rev + 0.5 * da_to_rev)
            d_nwc = (next_rev - last_rev) * nwc_ratio

            fcf = float(nopat + da_t - capex_t - d_nwc)
            # Guardrail: if historical FCF was positive but year-1 projection flips negative, soften CapEx/NWC.
            if hist_fcf_positive and t == 1 and fcf < 0:
                capex_t = max(da_t, capex_t * 0.85)
                d_nwc = max(0.0, d_nwc * 0.7)
                fcf = float(nopat + da_t - capex_t - d_nwc)

            projections.append({
                'year': t,
                'revenue': next_rev,
                'ebit': ebit_t,
                'nopat': nopat,
                'depreciation': da_t,
                'capex': capex_t,
                'nwc_change': d_nwc,
                'fcf': fcf
            })
            last_rev = next_rev

        return {
            'projections': projections,
            'historical_revenue_growth': revenue_growth,
            'ebit_margin': ebit_margin,
            'latest_revenue': rev[0],
            'latest_ebit': ebit[0]
        }
    except Exception as e:
        print(f"  ⚠️ Projection error: {e}")
        return None



def calculate_dcf(data, projections, wacc_data):
    """DCF with mid-year convention, guardrails, and terminal fade if TV dominates"""
    if projections is None:
        return None
    wacc = wacc_data['wacc']
    fcf_proj = projections['projections']

    pv_fcf = _pv_of_projections(fcf_proj, wacc)

    last_fcf = fcf_proj[-1]['fcf']
    g = TERMINAL_GROWTH
    if wacc <= g:
        print(f"     ⚠️  Guardrail: WACC ({wacc:.2%}) <= g ({g:.2%}); bumping WACC for TV")
        wacc_eff = g + 0.01
    else:
        wacc_eff = wacc
    tv_fcf = last_fcf * (1 + g)
    tv = tv_fcf / (wacc_eff - g)
    pv_tv = tv / _discount_factor(wacc_eff, len(fcf_proj))

    ev = pv_fcf + pv_tv
    tv_share = pv_tv / ev if ev > 0 else np.nan

    if tv_share == tv_share and tv_share > TV_FADE_TRIGGER:
        steps = FADE_YEARS
        fcf = last_fcf
        pv_fade = 0.0
        start_g = max(projections['historical_revenue_growth'], g)
        for i in range(1, steps + 1):
            fg = start_g + (g - start_g) * (i / steps)
            fcf *= (1 + fg)
            year = len(fcf_proj) + i
            pv_fade += fcf / _discount_factor(wacc, year)
        tv2 = (fcf * (1 + g)) / (wacc_eff - g)
        pv_tv = tv2 / _discount_factor(wacc_eff, len(fcf_proj) + steps)
        ev = pv_fcf + pv_fade + pv_tv
        tv_share = pv_tv / ev if ev > 0 else tv_share

    try:
        cash_bs = float(data['balance_sheet'].loc['Cash And Cash Equivalents'].iloc[0])
    except Exception:
        cash_bs = float(safe_get(data['info'], 'totalCash', 0) or 0.0)
    net_debt = wacc_data['total_debt'] - cash_bs
    equity_value = ev - net_debt
    shares = data['shares_outstanding']
    fv = equity_value / shares if shares and shares > 0 else 0.0

    return {
        'pv_fcf': pv_fcf,
        'terminal_value': tv,
        'pv_terminal_value': pv_tv,
        'enterprise_value': ev,
        'equity_value': equity_value,
        'fair_value_per_share': fv,
        'tv_percentage': float(tv_share * 100) if tv_share == tv_share else np.nan,
        'current_price': data['current_price']
    }



def dcf_sensitivity_analysis(data, projections, wacc_data):
    """DCF sensitivity analysis (mid-year)"""
    if projections is None or wacc_data is None:
        return None
    base_wacc = wacc_data['wacc']
    if np.isnan(base_wacc) or base_wacc <= 0:
        print("     ⚠️  Invalid WACC, skipping sensitivity analysis")
        return None
    wmin = max(base_wacc - WACC_SENSITIVITY, TERMINAL_GROWTH + 0.001)
    wmax = base_wacc + WACC_SENSITIVITY
    try:
        wgrid = np.arange(wmin, wmax + 0.0001, max(0.0025, WACC_SENSITIVITY / 2))
        ggrid = np.arange(TERMINAL_GROWTH - TERMINAL_G_SENSITIVITY, TERMINAL_GROWTH + TERMINAL_G_SENSITIVITY + 0.0001, max(0.001, TERMINAL_G_SENSITIVITY / 2))
    except ValueError:
        return None
    pts = []
    for w in wgrid:
        for g in ggrid:
            if w <= g:
                continue
            pv_fcf = _pv_of_projections(projections['projections'], w)
            last = projections['projections'][-1]['fcf']
            tv = (last * (1 + g)) / (w - g)
            pv_tv = tv / _discount_factor(w, len(projections['projections']))
            ev = pv_fcf + pv_tv
            try:
                cash_bs = float(data['balance_sheet'].loc['Cash And Cash Equivalents'].iloc[0])
            except Exception:
                cash_bs = float(safe_get(data['info'], 'totalCash', 0) or 0.0)
            net_debt = wacc_data['total_debt'] - cash_bs
            eq = ev - net_debt
            fv = eq / data['shares_outstanding'] if data['shares_outstanding'] > 0 else 0.0
            pts.append({'wacc': w, 'terminal_growth': g, 'fair_value': fv})
    if not pts:
        return None
    df = pd.DataFrame(pts)
    return df.pivot(index='wacc', columns='terminal_growth', values='fair_value')


def get_comparable_companies(ticker, info):
    """Find comparable companies by sector"""
    sector = info.get('sector', 'Unknown')

    sector_peers = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'AVGO', 'ORCL', 'ADBE'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'LLY', 'DHR', 'BMY'],
        'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TJX', 'LOW', 'BKNG', 'TGT'],
        'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'CL', 'MDLZ', 'PM', 'EL', 'KMB'],
        'Industrials': ['BA', 'HON', 'UNP', 'CAT', 'GE', 'MMM', 'LMT', 'RTX', 'DE', 'UPS'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'VLO', 'PSX', 'OXY'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'SPG', 'O', 'WELL', 'AVB'],
        'Communication Services': ['META', 'GOOGL', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED'],
        'Basic Materials': ['LIN', 'APD', 'ECL', 'SHW', 'DD', 'NEM', 'FCX', 'NUE', 'DOW'],
    }

    peers = sector_peers.get(sector, [])
    peers = [p for p in peers if p != ticker]
    return peers[:5]


def calculate_relative_valuation(data, peers):
    """Calculate relative valuation vs peers"""
    ticker = data['ticker']
    info = data['info']

    target_pe = info.get('trailingPE', np.nan)
    target_pb = info.get('priceToBook', np.nan)
    target_ps = info.get('priceToSalesTrailing12Months', np.nan)
    target_ev_ebitda = info.get('enterpriseToEbitda', np.nan)

    # Get peer metrics
    peer_metrics = []
    for peer in peers:
        try:
            peer_stock = yf.Ticker(peer)
            peer_info = peer_stock.info
            # Optional normalization: US / USD peers only
            if PEERS_US_ONLY:
                country = peer_info.get('country') or peer_info.get('countryISO')
                currency = peer_info.get('currency')
                if (country and str(country) != 'United States') and (currency and str(currency) != 'USD'):
                    continue
            peer_metrics.append({
                'ticker': peer,
                'pe': peer_info.get('trailingPE', np.nan),
                'pb': peer_info.get('priceToBook', np.nan),
                'ps': peer_info.get('priceToSalesTrailing12Months', np.nan),
                'ev_ebitda': peer_info.get('enterpriseToEbitda', np.nan)
            })
        except:
            continue

    if not peer_metrics:
        return None

    peer_df = pd.DataFrame(peer_metrics)

    # Keep only positive, finite ratios before medians
    for col in ['pe', 'pb', 'ps', 'ev_ebitda']:
        if col in peer_df:
            peer_df[col] = pd.to_numeric(peer_df[col], errors='coerce')
            peer_df.loc[~np.isfinite(peer_df[col]) | (peer_df[col] <= 0), col] = np.nan

    peer_median_pe = peer_df['pe'].median()
    peer_median_pb = peer_df['pb'].median()
    peer_median_ps = peer_df['ps'].median()
    peer_median_ev_ebitda = peer_df['ev_ebitda'].median()

    # Implied values (keep your current logic: PE, PB, PS)
    earnings = info.get('trailingEps', 0) * data['shares_outstanding']
    book_value = info.get('bookValue', 0) * data['shares_outstanding']
    revenue = info.get('totalRevenue', 0)

    implied_pe_value = earnings * peer_median_pe / data['shares_outstanding'] if not np.isnan(peer_median_pe) and earnings > 0 and data['shares_outstanding'] > 0 else np.nan
    implied_pb_value = book_value * peer_median_pb / data['shares_outstanding'] if not np.isnan(peer_median_pb) and book_value > 0 and data['shares_outstanding'] > 0 else np.nan
    implied_ps_value = revenue * peer_median_ps / data['shares_outstanding'] if not np.isnan(peer_median_ps) and revenue > 0 and data['shares_outstanding'] > 0 else np.nan

    # Z-scores
    def calc_z(target, series):
        series = pd.to_numeric(series, errors='coerce').dropna()
        mean, std = series.mean(), series.std()
        return (target - mean) / std if std > 0 and np.isfinite(target) else 0

    pe_zscore = calc_z(target_pe, peer_df['pe'])
    ev_ebitda_zscore = calc_z(target_ev_ebitda, peer_df['ev_ebitda'])

    # Blended comps fair value (unchanged blend of PE/PB/PS)
    comps_values = [v for v in [implied_pe_value, implied_pb_value, implied_ps_value] if np.isfinite(v) and v > 0]
    comps_fair_value = float(np.mean(comps_values)) if comps_values else np.nan

    return {
        'target_pe': target_pe,
        'peer_median_pe': peer_median_pe,
        'target_ev_ebitda': target_ev_ebitda,
        'peer_median_ev_ebitda': peer_median_ev_ebitda,
        'implied_pe_value': implied_pe_value,
        'implied_pb_value': implied_pb_value,
        'implied_ps_value': implied_ps_value,
        'comps_fair_value': comps_fair_value,
        'pe_zscore': pe_zscore,
        'ev_ebitda_zscore': ev_ebitda_zscore,
        'peers': peers
    }



def calculate_eva(data, wacc_data):
    """EVA with cash tax and cleaner invested capital"""
    income_stmt = data['income_stmt']
    balance_sheet = data['balance_sheet']
    try:
        ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else income_stmt.loc['Operating Income'].iloc[0]

        # Effective cash tax (multi-year, bounded)
        eff_tax = _effective_cash_tax(income_stmt)
        nopat = float(ebit) * (1 - eff_tax)

        total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else np.nan
        cash_eq     = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else safe_get(data['info'], 'totalCash', 0)
        cur_liab    = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else np.nan
        cur_debt    = balance_sheet.loc['Current Debt'].iloc[0] if 'Current Debt' in balance_sheet.index else 0.0
        invested_capital = (total_assets - cash_eq) - (cur_liab - cur_debt)
        # Optional adjustments
        try:
            if EVA_INCLUDE_GOODWILL and 'Goodwill' in balance_sheet.index:
                invested_capital += float(balance_sheet.loc['Goodwill'].iloc[0])
            if EVA_INCLUDE_INTANG and 'Intangible Assets' in balance_sheet.index:
                invested_capital += float(balance_sheet.loc['Intangible Assets'].iloc[0])
        except Exception:
            pass

        wacc = wacc_data['wacc']
        capital_charge = wacc * invested_capital
        eva = nopat - capital_charge

        rev = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else np.nan
        eva_margin = (eva / rev) * 100 if np.isfinite(rev) and rev != 0 else np.nan
        roic = (nopat / invested_capital) if (invested_capital is not None and invested_capital != 0 and np.isfinite(invested_capital)) else np.nan
        spread = roic - wacc if np.isfinite(roic) else np.nan

        return {
            'nopat': nopat,
            'invested_capital': invested_capital,
            'capital_charge': capital_charge,
            'eva': eva,
            'eva_margin': eva_margin,
            'roic': roic,
            'spread': spread
        }
    except Exception as e:
        print(f"  ⚠️ EVA calculation error: {e}")
        return None


def calculate_margin_of_safety(dcf_value, comps_value, current_price):
    """Calculate Margin of Safety"""
    dcf_mos = ((dcf_value - current_price) / current_price * 100) if dcf_value and not np.isnan(dcf_value) else np.nan
    comps_mos = ((comps_value - current_price) / current_price * 100) if comps_value and not np.isnan(comps_value) else np.nan

    values, weights = [], []
    if not np.isnan(dcf_mos):
        values.append(dcf_value)
        weights.append(DCF_WEIGHT)
    if not np.isnan(comps_mos):
        values.append(comps_value)
        weights.append(COMPS_WEIGHT)

    if values:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        blended_fair_value = sum(v * w for v, w in zip(values, weights))
        blended_mos = ((blended_fair_value - current_price) / current_price * 100)
    else:
        blended_fair_value = np.nan
        blended_mos = np.nan

    return {
        'dcf_mos': dcf_mos,
        'comps_mos': comps_mos,
        'blended_fair_value': blended_fair_value,
        'blended_mos': blended_mos,
        'current_price': current_price
    }


# ----------------------- MAIN ANALYSIS ----------------------- #
print("\n" + "=" * 70)
print("RUNNING VALUATION ANALYSIS...")
print("=" * 70)

all_results = []

for ticker in tickers:
    print(f"\n{'='*70}")
    print(f"ANALYZING: {ticker}")
    print(f"{'='*70}")

    data = get_financial_data(ticker)
    if data is None:
        continue

    print(f"  → Calculating WACC...")
    wacc_data = calculate_wacc(data, ticker, capm_data)

    if wacc_data is None:
        print(f"     ❌ Cannot calculate WACC, skipping {ticker}")
        continue

    print(f"     WACC: {wacc_data['wacc']:.2%}, Beta: {wacc_data['beta']:.2f}")

    print(f"  → Projecting financials ({DCF_HORIZON} years)...")
    projections = project_financials(data, DCF_HORIZON)
    if projections:
        print(f"     → Drivers: growth≈{projections['historical_revenue_growth']:.2%}, EBIT margin≈{projections['ebit_margin']:.2%}")
        # Quick magnitude sanity check
        y1 = projections['projections'][0]['fcf']
        yN = projections['projections'][-1]['fcf']
        print(f"     → Sanity: last Revenue={fmt_money(projections['latest_revenue'])}, Year1 FCF={fmt_money(y1)}, Year{len(projections['projections'])} FCF={fmt_money(yN)}")

    print(f"  → Running DCF analysis...")
    dcf_result = calculate_dcf(data, projections, wacc_data)
    if dcf_result:
        ev_dcf = (dcf_result['pv_fcf'] or 0) + (dcf_result['pv_terminal_value'] or 0)
        tv_share = (dcf_result['pv_terminal_value'] / ev_dcf) if ev_dcf > 0 else float("nan")
        print(
            f"     → PV(FCF)={fmt_money(dcf_result['pv_fcf'])}, "
            f"PV(TV)={fmt_money(dcf_result['pv_terminal_value'])} — "
            f"TV share {fmt_pct(tv_share)} of DCF EV"
        )
    # --- SOFT CORRECTION: Margin-based shrink toward Yahoo target if deviation is too large ---
    def get_yahoo_target_price(ticker):
        """Get Yahoo target price for a ticker (from yfinance)"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            target = info.get('targetMeanPrice', None)
            return float(target) if target and target > 0 else None
        except Exception:
            return None

    def shrink_towards_target(value, target, threshold=0.5, shrink_factor=0.5):
        """
        If the deviation from the Yahoo target is above the threshold (e.g., 50%),
        apply a shrinkage proportional to the deviation.
        shrink_factor (e.g., 0.5) controls how aggressively we shrink.
        """
        if value and target and value > 0 and target > 0:
            deviation = abs(value - target) / target
            if deviation > threshold:
                value = value - shrink_factor * (value - target)
        return value

    if dcf_result:
        dcf_fv = dcf_result['fair_value_per_share']
        if SHRINK_TO_YAHOO:
            yahoo_target = get_yahoo_target_price(ticker)
            if yahoo_target:
                orig_fv = dcf_fv
                dcf_fv = shrink_towards_target(dcf_fv, yahoo_target, threshold=0.3, shrink_factor=0.4)
                if dcf_fv != orig_fv:
                    dcf_result['fair_value_per_share'] = dcf_fv
                    dcf_result["Note_Yahoo_Corrected"] = f"Margin-shrunk toward Yahoo target: ${yahoo_target:.2f}"

            # Diagnostics for DCF output (always print when we have a result)
            if dcf_result:
                print(
                    f"     DCF Fair Value: ${dcf_result['fair_value_per_share']:.2f} (TV: {dcf_result['tv_percentage']:.1f}%)")
                ymean = data.get('yahoo_target')
                if ymean and np.isfinite(ymean) and ymean > 0:
                    deviation_pct = (dcf_result['fair_value_per_share'] - ymean) / ymean * 100
                    print(
                        f"     📊 Yahoo target: ${ymean:.2f} → Model fair value: ${dcf_result['fair_value_per_share']:.2f} → Deviation: {deviation_pct:+.1f}%")
                else:
                    print("     ⚙️ Yahoo target not available for this ticker.")

            # --- Sensitivity --------------------------------------------------------
            print("  → Running sensitivity analysis...")
            sensitivity = dcf_sensitivity_analysis(data, projections, wacc_data)

            # --- Relative Valuation (Comps) ----------------------------------------
            print("  → Finding comparable companies...")
            peers = get_comparable_companies(ticker, data['info'])
            print(f"     Peers: {', '.join(peers) if peers else 'None found'}")

            comps_result = None
            if peers:
                print("  → Calculating relative valuation...")
                comps_result = calculate_relative_valuation(data, peers)
                if comps_result and np.isfinite(comps_result.get('comps_fair_value', np.nan)):
                    print(f"     Comps Fair Value: ${comps_result['comps_fair_value']:.2f}")
                    if np.isfinite(comps_result.get('peer_median_pe', np.nan)):
                        print(
                            f"     P/E: {comps_result.get('target_pe', np.nan):.1f} vs Peers: {comps_result['peer_median_pe']:.1f}")

            # --- EVA ----------------------------------------------------------------
            print("  → Calculating EVA...")
            eva_result = calculate_eva(data, wacc_data)
            if eva_result:
                eva_b = eva_result['eva'] / 1e9 if np.isfinite(eva_result['eva']) else np.nan
                roic_pct = eva_result['roic'] * 100 if np.isfinite(eva_result['roic']) else np.nan
                spread_pct = eva_result['spread'] * 100 if np.isfinite(eva_result['spread']) else np.nan
                print(f"     EVA: ${eva_b:.2f}B, ROIC: {roic_pct:.2f}%, Spread: {spread_pct:.2f}%")

            # --- Margin of Safety & Signal -----------------------------------------
            print("  → Calculating Margin of Safety...")
            dcf_fv = dcf_result['fair_value_per_share'] if dcf_result else np.nan
            comps_fv = comps_result['comps_fair_value'] if comps_result else np.nan
            mos_result = calculate_margin_of_safety(dcf_fv, comps_fv, data['current_price'])
            print(f"     Blended Fair Value: ${mos_result['blended_fair_value']:.2f}")
            print(f"     Margin of Safety: {mos_result['blended_mos']:.1f}%")

            mos = mos_result['blended_mos']
            if mos > 30:
                signal, color = "STRONG BUY", "🟢"
            elif mos > 15:
                signal, color = "BUY", "🟢"
            elif mos > 0:
                signal, color = "HOLD", "🟡"
            elif mos > -15:
                signal, color = "SELL", "🔴"
            else:
                signal, color = "STRONG SELL", "🔴"
            print(f"     {color} Valuation Signal: {signal}")


            # --- Per-ticker PNGs ----------------------------------------------------
            def __ensure_ticker_image_dir(t: str) -> str:
                p = os.path.join("output", "images", t.lower())
                os.makedirs(p, exist_ok=True)
                return p


            def __plot_per_ticker_charts(ticker, data, dcf_result, projections, sensitivity):
                if dcf_result is None or projections is None:
                    return
                out_dir = __ensure_ticker_image_dir(ticker)

                # 1) FCFF & components
                try:
                    years = [p['year'] for p in projections['projections']]
                    fcf = [p['fcf'] for p in projections['projections']]
                    capex = [p['capex'] for p in projections['projections']]
                    da = [p['depreciation'] for p in projections['projections']]
                    nopat = [p['nopat'] for p in projections['projections']]

                    plt.figure(figsize=(10, 6))
                    plt.plot(years, fcf, marker='o', label='FCFF')
                    plt.plot(years, nopat, marker='o', linestyle='--', label='NOPAT')
                    plt.plot(years, capex, marker='o', linestyle='--', label='CapEx')
                    plt.plot(years, da, marker='o', linestyle='--', label='D&A')
                    plt.title(f"{ticker} — Projected FCFF & Components")
                    plt.xlabel("Year");
                    plt.ylabel("USD")
                    plt.legend();
                    plt.grid(True, alpha=0.3);
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"{ticker}_fcff.png"), dpi=200);
                    plt.close()
                except Exception as e:
                    print(f"     ⚠️  Could not save FCFF chart for {ticker}: {e}")

                # 2) PV breakdown
                try:
                    plt.figure(figsize=(7, 5))
                    pv_fcf = dcf_result['pv_fcf'];
                    pv_tv = dcf_result['pv_terminal_value']
                    plt.bar(['PV of FCF', 'PV of Terminal'], [pv_fcf, pv_tv])
                    plt.title(f"{ticker} — PV Decomposition (TV {dcf_result['tv_percentage']:.1f}%)")
                    plt.ylabel("USD")
                    plt.grid(True, axis='y', alpha=0.3);
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"{ticker}_pv_breakdown.png"), dpi=200);
                    plt.close()
                except Exception as e:
                    print(f"     ⚠️  Could not save PV breakdown chart for {ticker}: {e}")

                # 3) Sensitivity heatmap
                try:
                    if sensitivity is not None and isinstance(sensitivity, pd.DataFrame) and sensitivity.shape[0] > 0:
                        plt.figure(figsize=(8, 6))
                        arr = sensitivity.values
                        plt.imshow(arr, aspect='auto', origin='lower')
                        plt.colorbar(label='Fair Value ($/sh)')
                        plt.xticks(range(arr.shape[1]), [f"{g * 100:.2f}%" for g in sensitivity.columns], rotation=45,
                                   ha='right')
                        plt.yticks(range(arr.shape[0]), [f"{w * 100:.2f}%" for w in sensitivity.index])
                        plt.xlabel("Terminal Growth");
                        plt.ylabel("WACC")
                        plt.title(f"{ticker} — DCF Sensitivity")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"{ticker}_sensitivity.png"), dpi=200);
                        plt.close()
                except Exception as e:
                    print(f"     ⚠️  Could not save sensitivity chart for {ticker}: {e}")


            if PER_TICKER_IMAGES:
                __plot_per_ticker_charts(ticker, data, dcf_result, projections, sensitivity)

            # --- Collect result row --------------------------------------------------
            result = {
                'Ticker': ticker,
                'Sector': data['info'].get('sector', 'N/A'),
                'Current_Price': data['current_price'],
                'DCF_Fair_Value': dcf_fv,
                'DCF_MoS_%': mos_result['dcf_mos'],
                'DCF_TV_%': dcf_result['tv_percentage'] if dcf_result else np.nan,
                'Comps_Fair_Value': comps_fv,
                'Comps_MoS_%': mos_result['comps_mos'],
                'P/E': comps_result['target_pe'] if comps_result else np.nan,
                'Peer_Median_P/E': comps_result['peer_median_pe'] if comps_result else np.nan,
                'P/E_Z-Score': comps_result['pe_zscore'] if comps_result else np.nan,
                'EV/EBITDA': comps_result['target_ev_ebitda'] if comps_result else np.nan,
                'EVA_$M': (eva_result['eva'] / 1e6) if eva_result and np.isfinite(eva_result['eva']) else np.nan,
                'ROIC_%': (eva_result['roic'] * 100) if eva_result and np.isfinite(eva_result['roic']) else np.nan,
                'ROIC-WACC_Spread_%': (eva_result['spread'] * 100) if eva_result and np.isfinite(
                    eva_result['spread']) else np.nan,
                'WACC_%': wacc_data['wacc'] * 100,
                'Beta': wacc_data['beta'],
                'Cost_of_Equity_%': wacc_data['cost_of_equity'] * 100,
                'Blended_Fair_Value': mos_result['blended_fair_value'],
                'Blended_MoS_%': mos_result['blended_mos'],
                'Valuation_Signal': signal,
                'sensitivity': sensitivity,
                'projections': projections
            }
            all_results.append(result)