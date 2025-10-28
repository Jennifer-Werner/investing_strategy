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

print("=" * 70)
print("COMPREHENSIVE VALUATION MODEL")
print("DCF + Relative Valuation + EVA + Margin of Safety")
print("=" * 70)

# ----------------------- LOAD PRE-CALCULATED CAPM ----------------------- #
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
capm_data = load_capm_data()


# ----------------------- DATA COLLECTION ----------------------- #
def get_financial_data(ticker):
    """Fetch comprehensive financial data for valuation"""
    try:
        stock = yf.Ticker(ticker)

        # Get financial statements
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        info = stock.info

        # Current price and shares
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        shares_outstanding = info.get('sharesOutstanding', 0)

        if income_stmt.empty or balance_sheet.empty or cash_flow.empty:
            print(f"❌ {ticker}: Insufficient financial data")
            return None

        print(f"✅ {ticker}: Financial data loaded")

        # Insert Yahoo target mean price
        yahoo_target = info.get('targetMeanPrice', None)

        return {
            'ticker': ticker,
            'info': info,
            'income_stmt': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'current_price': current_price,
            'shares_outstanding': shares_outstanding,
            'yahoo_target': yahoo_target
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
    """
    Calculate WACC using pre-calculated cost of equity from CAPM
    WACC = (E/V) * Re + (D/V) * Rd * (1 - Tax)
    """
    balance_sheet = data['balance_sheet']

    # Get cost of equity from pre-calculated CAPM
    cost_of_equity, beta = get_cost_of_equity(ticker, capm_df, data)

    # Validate and clean price data
    current_price = data.get('current_price', 0)
    if current_price is None or np.isnan(current_price) or current_price <= 0:
        print(f"     ⚠️  Invalid price data, cannot calculate WACC")
        return None

    shares_outstanding = data.get('shares_outstanding', 0)
    if shares_outstanding is None or np.isnan(shares_outstanding) or shares_outstanding <= 0:
        print(f"     ⚠️  Invalid shares outstanding, cannot calculate WACC")
        return None

    # Get debt
    try:
        total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
        if pd.isna(total_debt) or total_debt == 0:
            long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance_sheet.index else 0
            short_term_debt = balance_sheet.loc['Current Debt'].iloc[0] if 'Current Debt' in balance_sheet.index else 0
            long_term_debt = 0 if pd.isna(long_term_debt) else long_term_debt
            short_term_debt = 0 if pd.isna(short_term_debt) else short_term_debt
            total_debt = long_term_debt + short_term_debt
    except:
        total_debt = 0

    # Ensure total_debt is a valid number
    total_debt = 0 if pd.isna(total_debt) else total_debt

    market_cap = current_price * shares_outstanding
    enterprise_value = market_cap + total_debt

    # Weights
    equity_weight = market_cap / enterprise_value if enterprise_value > 0 else 1.0
    debt_weight = total_debt / enterprise_value if enterprise_value > 0 else 0.0

    # Cost of debt
    try:
        income_stmt = data['income_stmt']
        interest_expense = abs(income_stmt.loc['Interest Expense'].iloc[0]) if 'Interest Expense' in income_stmt.index else 0
        interest_expense = 0 if pd.isna(interest_expense) else interest_expense
        cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.05
        cost_of_debt = min(cost_of_debt, 0.12)
    except:
        cost_of_debt = 0.05

    # WACC
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - TAX_RATE))

    # Validate WACC
    if np.isnan(wacc) or wacc <= 0:
        print(f"     ⚠️  Invalid WACC calculated, using cost of equity as fallback")
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
    """Project Free Cash Flow to Firm (FCFF)"""
    income_stmt = data['income_stmt']
    cash_flow = data['cash_flow']

    try:
        revenues = income_stmt.loc['Total Revenue'].iloc[:3].values
        ebit = income_stmt.loc['EBIT'].iloc[:3].values if 'EBIT' in income_stmt.index else income_stmt.loc['Operating Income'].iloc[:3].values
        capex = abs(cash_flow.loc['Capital Expenditure'].iloc[:3].values) if 'Capital Expenditure' in cash_flow.index else revenues * 0.05
        depreciation = cash_flow.loc['Depreciation And Amortization'].iloc[:3].values if 'Depreciation And Amortization' in cash_flow.index else revenues * 0.03

        revenue_growth = np.mean([(revenues[i] - revenues[i+1]) / revenues[i+1] for i in range(len(revenues)-1)])
        ebit_margin = np.mean(ebit / revenues)
        capex_to_revenue = np.mean(capex / revenues)
        da_to_revenue = np.mean(depreciation / revenues)

        projected_fcf = []
        last_revenue = revenues[0]

        for year in range(1, horizon + 1):
            growth_rate = revenue_growth * (0.9 ** (year - 1))
            growth_rate = max(growth_rate, TERMINAL_GROWTH)

            projected_revenue = last_revenue * (1 + growth_rate)
            projected_ebit = projected_revenue * ebit_margin
            nopat = projected_ebit * (1 - TAX_RATE)
            projected_da = projected_revenue * da_to_revenue
            projected_capex = projected_revenue * capex_to_revenue
            nwc_change = (projected_revenue - last_revenue) * 0.02
            fcf = nopat + projected_da - projected_capex - nwc_change

            projected_fcf.append({
                'year': year,
                'revenue': projected_revenue,
                'ebit': projected_ebit,
                'nopat': nopat,
                'depreciation': projected_da,
                'capex': projected_capex,
                'nwc_change': nwc_change,
                'fcf': fcf
            })

            last_revenue = projected_revenue

        return {
            'projections': projected_fcf,
            'historical_revenue_growth': revenue_growth,
            'ebit_margin': ebit_margin,
            'latest_revenue': revenues[0],
            'latest_ebit': ebit[0]
        }

    except Exception as e:
        print(f"  ⚠️ Projection error: {e}")
        return None


def calculate_dcf(data, projections, wacc_data):
    """Calculate DCF valuation with terminal value"""
    if projections is None:
        return None

    wacc = wacc_data['wacc']
    fcf_projections = projections['projections']

    # Discount projected FCF
    pv_fcf = 0
    for proj in fcf_projections:
        year = proj['year']
        fcf = proj['fcf']
        pv = fcf / ((1 + wacc) ** year)
        pv_fcf += pv

    # Terminal value
    last_fcf = fcf_projections[-1]['fcf']
    terminal_fcf = last_fcf * (1 + TERMINAL_GROWTH)
    terminal_value = terminal_fcf / (wacc - TERMINAL_GROWTH)
    pv_terminal_value = terminal_value / ((1 + wacc) ** len(fcf_projections))

    # Enterprise and equity value
    enterprise_value = pv_fcf + pv_terminal_value
    net_debt = wacc_data['total_debt'] - data['info'].get('totalCash', 0)
    equity_value = enterprise_value - net_debt

    shares = data['shares_outstanding']
    fair_value_per_share = equity_value / shares if shares > 0 else 0
    tv_percentage = (pv_terminal_value / enterprise_value) * 100 if enterprise_value > 0 else 0

    return {
        'pv_fcf': pv_fcf,
        'terminal_value': terminal_value,
        'pv_terminal_value': pv_terminal_value,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'fair_value_per_share': fair_value_per_share,
        'tv_percentage': tv_percentage,
        'current_price': data['current_price']
    }


def dcf_sensitivity_analysis(data, projections, wacc_data):
    """DCF sensitivity analysis grid"""
    if projections is None or wacc_data is None:
        return None

    base_wacc = wacc_data['wacc']

    # Validate base_wacc
    if np.isnan(base_wacc) or base_wacc <= 0:
        print("     ⚠️  Invalid WACC, skipping sensitivity analysis")
        return None

    # Ensure WACC range doesn't include values that would cause division by zero or negative denominators
    wacc_min = max(base_wacc - WACC_SENSITIVITY, TERMINAL_GROWTH + 0.001)
    wacc_max = base_wacc + WACC_SENSITIVITY

    try:
        wacc_range = np.arange(wacc_min, wacc_max + 0.001, WACC_SENSITIVITY / 2)
        terminal_g_range = np.arange(TERMINAL_GROWTH - TERMINAL_G_SENSITIVITY, TERMINAL_GROWTH + TERMINAL_G_SENSITIVITY + 0.001, TERMINAL_G_SENSITIVITY / 2)
    except ValueError as e:
        print(f"     ⚠️  Error creating sensitivity ranges: {e}")
        return None

    sensitivity_grid = []

    for w in wacc_range:
        for g in terminal_g_range:
            # Skip if WACC <= terminal growth (would cause division by zero or negative values)
            if w <= g:
                continue

            pv_fcf = 0
            fcf_projections = projections['projections']
            for proj in fcf_projections:
                pv = proj['fcf'] / ((1 + w) ** proj['year'])
                pv_fcf += pv

            last_fcf = fcf_projections[-1]['fcf']
            terminal_fcf = last_fcf * (1 + g)
            terminal_value = terminal_fcf / (w - g) if w > g else 0
            pv_terminal_value = terminal_value / ((1 + w) ** len(fcf_projections))

            enterprise_value = pv_fcf + pv_terminal_value
            net_debt = wacc_data['total_debt'] - data['info'].get('totalCash', 0)
            equity_value = enterprise_value - net_debt
            fair_value = equity_value / data['shares_outstanding'] if data['shares_outstanding'] > 0 else 0

            sensitivity_grid.append({'wacc': w, 'terminal_growth': g, 'fair_value': fair_value})

    if not sensitivity_grid:
        print("     ⚠️  No valid sensitivity data points")
        return None

    sensitivity_df = pd.DataFrame(sensitivity_grid)
    return sensitivity_df.pivot(index='wacc', columns='terminal_growth', values='fair_value')


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

    peer_median_pe = peer_df['pe'].median()
    peer_median_pb = peer_df['pb'].median()
    peer_median_ps = peer_df['ps'].median()
    peer_median_ev_ebitda = peer_df['ev_ebitda'].median()

    # Implied values
    earnings = info.get('trailingEps', 0) * data['shares_outstanding']
    book_value = info.get('bookValue', 0) * data['shares_outstanding']
    revenue = info.get('totalRevenue', 0)

    implied_pe_value = earnings * peer_median_pe / data['shares_outstanding'] if not np.isnan(peer_median_pe) and earnings > 0 else np.nan
    implied_pb_value = book_value * peer_median_pb / data['shares_outstanding'] if not np.isnan(peer_median_pb) and book_value > 0 else np.nan
    implied_ps_value = revenue * peer_median_ps / data['shares_outstanding'] if not np.isnan(peer_median_ps) and revenue > 0 else np.nan

    # Z-scores
    def calc_z(target, series):
        mean, std = series.mean(), series.std()
        return (target - mean) / std if std > 0 and not np.isnan(target) else 0

    pe_zscore = calc_z(target_pe, peer_df['pe'])
    ev_ebitda_zscore = calc_z(target_ev_ebitda, peer_df['ev_ebitda'])

    # Blended comps fair value
    comps_values = [v for v in [implied_pe_value, implied_pb_value, implied_ps_value] if not np.isnan(v) and v > 0]
    comps_fair_value = np.mean(comps_values) if comps_values else np.nan

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
    """Calculate Economic Value Added (EVA)"""
    income_stmt = data['income_stmt']
    balance_sheet = data['balance_sheet']

    try:
        ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else income_stmt.loc['Operating Income'].iloc[0]
        nopat = ebit * (1 - TAX_RATE)

        total_assets = balance_sheet.loc['Total Assets'].iloc[0]
        current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0]
        invested_capital = total_assets - (current_liabilities * 0.5)

        wacc = wacc_data['wacc']
        capital_charge = wacc * invested_capital
        eva = nopat - capital_charge

        revenue = income_stmt.loc['Total Revenue'].iloc[0]
        eva_margin = (eva / revenue) * 100
        roic = nopat / invested_capital if invested_capital > 0 else 0
        spread = roic - wacc

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

    print(f"  → Running DCF analysis...")
    dcf_result = calculate_dcf(data, projections, wacc_data)


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
        yahoo_target = get_yahoo_target_price(ticker)
        if yahoo_target:
            # Margin-based shrink strategy
            orig_fv = dcf_fv
            dcf_fv = shrink_towards_target(dcf_fv, yahoo_target, threshold=0.3, shrink_factor=0.4)
            if dcf_fv != orig_fv:
                dcf_result['fair_value_per_share'] = dcf_fv
                dcf_result["Note_Yahoo_Corrected"] = f"Margin-shrunk toward Yahoo target: ${yahoo_target:.2f}"

        print(f"     DCF Fair Value: ${dcf_result['fair_value_per_share']:.2f} (TV: {dcf_result['tv_percentage']:.1f}%)")
        # Insert Yahoo target comparison
        yahoo_target = data.get('yahoo_target')
        fair_value_per_share = dcf_result['fair_value_per_share']
        if yahoo_target and yahoo_target > 0:
            deviation_pct = (fair_value_per_share - yahoo_target) / yahoo_target * 100
            print(f"     📊 Yahoo target: ${yahoo_target:.2f} → Model fair value: ${fair_value_per_share:.2f} → Deviation: {deviation_pct:+.1f}%")
        else:
            print("     ⚙️ Yahoo target not available for this ticker.")
        if dcf_result['tv_percentage'] > 70:
            print(f"     ⚠️  WARNING: Terminal Value is {dcf_result['tv_percentage']:.1f}% (>70% threshold)")

    print(f"  → Running sensitivity analysis...")
    sensitivity = dcf_sensitivity_analysis(data, projections, wacc_data)

    print(f"  → Finding comparable companies...")
    peers = get_comparable_companies(ticker, data['info'])
    print(f"     Peers: {', '.join(peers) if peers else 'None found'}")

    comps_result = None
    if peers:
        print(f"  → Calculating relative valuation...")
        comps_result = calculate_relative_valuation(data, peers)
        if comps_result:
            print(f"     Comps Fair Value: ${comps_result['comps_fair_value']:.2f}")
            print(f"     P/E: {comps_result['target_pe']:.1f} vs Peers: {comps_result['peer_median_pe']:.1f}")

    print(f"  → Calculating EVA...")
    eva_result = calculate_eva(data, wacc_data)

    if eva_result:
        print(f"     EVA: ${eva_result['eva']/1e9:.2f}B, ROIC: {eva_result['roic']:.2%}, Spread: {eva_result['spread']:.2%}")

    print(f"  → Calculating Margin of Safety...")
    dcf_fv = dcf_result['fair_value_per_share'] if dcf_result else np.nan
    comps_fv = comps_result['comps_fair_value'] if comps_result else np.nan

    mos_result = calculate_margin_of_safety(dcf_fv, comps_fv, data['current_price'])

    print(f"     Blended Fair Value: ${mos_result['blended_fair_value']:.2f}")
    print(f"     Margin of Safety: {mos_result['blended_mos']:.1f}%")

    # Valuation Signal
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

    # Compile results
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
        'EVA_$M': eva_result['eva'] / 1e6 if eva_result else np.nan,
        'ROIC_%': eva_result['roic'] * 100 if eva_result else np.nan,
        'ROIC-WACC_Spread_%': eva_result['spread'] * 100 if eva_result else np.nan,
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

# ----------------------- OUTPUT ----------------------- #
if all_results:
    df = pd.DataFrame(all_results)

    # Save to Excel
    with pd.ExcelWriter(EXPORT_EXCEL, engine='openpyxl') as writer:
        summary_cols = ['Ticker', 'Sector', 'Current_Price', 'DCF_Fair_Value', 'Comps_Fair_Value',
                       'Blended_Fair_Value', 'Blended_MoS_%', 'Valuation_Signal']
        df[summary_cols].to_excel(writer, sheet_name='Summary', index=False)

        df.drop(['sensitivity', 'projections'], axis=1).to_excel(writer, sheet_name='Detailed_Valuation', index=False)

        dcf_cols = ['Ticker', 'DCF_Fair_Value', 'DCF_MoS_%', 'DCF_TV_%', 'WACC_%', 'Beta', 'Cost_of_Equity_%']
        df[dcf_cols].to_excel(writer, sheet_name='DCF_Analysis', index=False)

        comps_cols = ['Ticker', 'Comps_Fair_Value', 'Comps_MoS_%', 'P/E', 'Peer_Median_P/E', 'P/E_Z-Score', 'EV/EBITDA']
        df[comps_cols].to_excel(writer, sheet_name='Relative_Valuation', index=False)

        eva_cols = ['Ticker', 'EVA_$M', 'ROIC_%', 'ROIC-WACC_Spread_%', 'WACC_%']
        df[eva_cols].to_excel(writer, sheet_name='EVA_Analysis', index=False)

        for i, row in df.iterrows():
            if row['sensitivity'] is not None:
                row['sensitivity'].to_excel(writer, sheet_name=f"{row['Ticker']}_Sensitivity")

    print("\n" + "=" * 70)
    print("✅ VALUATION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {EXPORT_EXCEL}")

    # Summary table
    print("\n--- VALUATION SUMMARY ---")
    display_df = df[['Ticker', 'Current_Price', 'Blended_Fair_Value', 'Blended_MoS_%', 'Valuation_Signal']].copy()
    display_df['Current_Price'] = display_df['Current_Price'].apply(lambda x: f"${x:.2f}")
    display_df['Blended_Fair_Value'] = display_df['Blended_Fair_Value'].apply(lambda x: f"${x:.2f}" if not np.isnan(x) else "N/A")
    display_df['Blended_MoS_%'] = display_df['Blended_MoS_%'].apply(lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A")
    print(display_df.to_string(index=False))

    # Investment recommendations
    print("\n--- INVESTMENT RECOMMENDATIONS ---")
    buy_candidates = df[df['Blended_MoS_%'] > 15].sort_values('Blended_MoS_%', ascending=False)
    if len(buy_candidates) > 0:
        print(f"\n🟢 BUY/STRONG BUY ({len(buy_candidates)} stocks with >15% MoS):")
        for _, row in buy_candidates.iterrows():
            print(f"   {row['Ticker']}: ${row['Current_Price']:.2f} → Fair Value: ${row['Blended_Fair_Value']:.2f} (MoS: {row['Blended_MoS_%']:.1f}%)")

    hold_candidates = df[(df['Blended_MoS_%'] > 0) & (df['Blended_MoS_%'] <= 15)]
    if len(hold_candidates) > 0:
        print(f"\n🟡 HOLD ({len(hold_candidates)} stocks with 0-15% MoS)")

    sell_candidates = df[df['Blended_MoS_%'] <= 0].sort_values('Blended_MoS_%')
    if len(sell_candidates) > 0:
        print(f"\n🔴 SELL/STRONG SELL ({len(sell_candidates)} stocks overvalued):")
        for _, row in sell_candidates.iterrows():
            print(f"   {row['Ticker']}: ${row['Current_Price']:.2f} → Fair Value: ${row['Blended_Fair_Value']:.2f} (MoS: {row['Blended_MoS_%']:.1f}%)")

    # Visualizations
    print("\n--- GENERATING VISUALIZATIONS ---")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Current vs Fair Value
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax1.bar(x - width/2, df['Current_Price'], width, label='Current', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, df['Blended_Fair_Value'], width, label='Fair Value', color='coral', alpha=0.8)
    ax1.set_xlabel('Stock', fontweight='bold')
    ax1.set_ylabel('Price ($)', fontweight='bold')
    ax1.set_title('Current Price vs Fair Value', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Ticker'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Chart 2: Margin of Safety
    ax2 = axes[0, 1]
    colors = ['green' if x > 15 else 'orange' if x > 0 else 'red' for x in df['Blended_MoS_%']]
    ax2.barh(df['Ticker'], df['Blended_MoS_%'], color=colors, alpha=0.7)
    ax2.set_xlabel('Margin of Safety (%)', fontweight='bold')
    ax2.set_title('Margin of Safety', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=15, color='green', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    # Chart 3: Valuation Methods
    ax3 = axes[1, 0]
    x = np.arange(len(df))
    width = 0.25
    ax3.bar(x - width, df['DCF_Fair_Value'], width, label='DCF', color='#2ecc71', alpha=0.8)
    ax3.bar(x, df['Comps_Fair_Value'], width, label='Comps', color='#3498db', alpha=0.8)
    ax3.bar(x + width, df['Current_Price'], width, label='Current', color='#e74c3c', alpha=0.8)
    ax3.set_xlabel('Stock', fontweight='bold')
    ax3.set_ylabel('Price ($)', fontweight='bold')
    ax3.set_title('Valuation Methods', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['Ticker'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Chart 4: ROIC vs WACC
    ax4 = axes[1, 1]
    valid = df[~df['ROIC_%'].isna() & ~df['WACC_%'].isna()]
    colors_spread = ['green' if x > 0 else 'red' for x in valid['ROIC-WACC_Spread_%']]
    ax4.scatter(valid['WACC_%'], valid['ROIC_%'], c=colors_spread, s=150, alpha=0.6, edgecolors='black')
    max_val = max(valid[['WACC_%', 'ROIC_%']].max())
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='ROIC = WACC')
    ax4.set_xlabel('WACC (%)', fontweight='bold')
    ax4.set_ylabel('ROIC (%)', fontweight='bold')
    ax4.set_title('Value Creation', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    for _, row in valid.iterrows():
        ax4.annotate(row['Ticker'], (row['WACC_%'], row['ROIC_%']), fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig('output/images/Valuation_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Charts saved to: output/images/Valuation_Analysis.png")
    print("\n" + "=" * 70)

else:
    print("\n❌ No results to save")