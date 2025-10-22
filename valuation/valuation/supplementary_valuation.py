"""
SUPPLEMENTARY VALUATION MODELS
Dividend Discount Model (DDM) + Asset-Based Valuation + Precedent Transactions

Location: valuation/valuation/supplementary_valuation.py
Outputs: valuation/valuation/output/sheets/ and /images/
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ----------------------- DIRECTORY SETUP ----------------------- #
def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("output/sheets", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)


ensure_directories()

# ----------------------- SETTINGS ----------------------- #
EXPORT_EXCEL = "output/sheets/Supplementary_Valuation.xlsx"
CAPM_INPUT_FILE = "../risk-models/output/sheets/Risk_Model_Output.xlsx.xlsx"

print("=" * 70)
print("SUPPLEMENTARY VALUATION MODELS")
print("DDM + Asset-Based + Precedent Transactions")
print("=" * 70)


# ----------------------- LOAD CAPM DATA ----------------------- #
def load_capm_data():
    """Load pre-calculated CAPM for cost of equity"""
    try:
        capm_df = pd.read_excel(CAPM_INPUT_FILE, sheet_name='CAPM_Results')
        print(f"✅ Loaded CAPM data for {len(capm_df)} stocks")
        return capm_df
    except:
        print(f"⚠️  CAPM file not found, will use yfinance data")
        return None


# ----------------------- USER INPUTS ----------------------- #
n = int(input("\nEnter the number of tickers to analyze: "))
tickers = [input(f"Enter ticker {i + 1}: ").upper() for i in range(n)]

print("\n--- Parameters ---")
RISK_FREE_RATE = float(input("Risk-free rate (%, e.g., 4.5): ")) / 100
MARKET_RISK_PREMIUM = float(input("Market risk premium (%, e.g., 7.0): ")) / 100
PERPETUAL_GROWTH = float(input("Perpetual dividend growth rate (%, e.g., 2.5): ")) / 100

print("\n--- Loading Risk Model Data ---")
capm_data = load_capm_data()


# ----------------------- DATA COLLECTION ----------------------- #
def get_data(ticker):
    """Fetch financial data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow

        # Historical data
        hist = stock.history(period="5y")
        dividends = stock.dividends

        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        shares = info.get('sharesOutstanding', 0)

        print(f"✅ {ticker}: Data loaded")

        return {
            'ticker': ticker,
            'info': info,
            'balance_sheet': balance_sheet,
            'income_stmt': income_stmt,
            'cash_flow': cash_flow,
            'hist': hist,
            'dividends': dividends,
            'current_price': current_price,
            'shares_outstanding': shares
        }

    except Exception as e:
        print(f"❌ {ticker}: Error - {e}")
        return None


def get_cost_of_equity(ticker, capm_df, info):
    """Get cost of equity from CAPM"""
    if capm_df is not None and ticker in capm_df['Ticker'].values:
        cost_of_equity = capm_df[capm_df['Ticker'] == ticker].iloc[0]['Expected_Return']
        print(f"     ✅ Using pre-calculated CAPM: {cost_of_equity:.2%}")
        return cost_of_equity
    else:
        beta = info.get('beta', 1.0) or 1.0
        cost_of_equity = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM
        print(f"     ⚠️  Calculated CAPM: {cost_of_equity:.2%}")
        return cost_of_equity


# ----------------------- 1. DIVIDEND DISCOUNT MODEL (DDM) ----------------------- #
def calculate_ddm(data, cost_of_equity):
    """
    Dividend Discount Model (Gordon Growth Model)
    Fair Value = D1 / (r - g)
    Where:
    - D1 = Expected dividend next year
    - r = Required return (cost of equity)
    - g = Perpetual growth rate
    """
    info = data['info']
    dividends = data['dividends']

    # Check if company pays dividends
    if len(dividends) == 0 or info.get('dividendRate', 0) == 0:
        print(f"     ⚠️  No dividends - DDM not applicable")
        return None

    try:
        # Get current annual dividend
        current_dividend = info.get('dividendRate', info.get('trailingAnnualDividendRate', 0))

        if current_dividend == 0:
            return None

        # Calculate historical dividend growth rate
        recent_divs = dividends.resample('Y').sum().tail(5)
        if len(recent_divs) >= 2:
            div_growth_rates = [(recent_divs.iloc[i] - recent_divs.iloc[i - 1]) / recent_divs.iloc[i - 1]
                                for i in range(1, len(recent_divs))]
            historical_growth = np.mean(div_growth_rates)
        else:
            historical_growth = PERPETUAL_GROWTH

        # Use lower of historical growth and perpetual growth (conservative)
        growth_rate = min(historical_growth, PERPETUAL_GROWTH)

        # Ensure growth < discount rate
        if growth_rate >= cost_of_equity:
            growth_rate = cost_of_equity * 0.8  # Cap at 80% of discount rate

        # Calculate D1 (next year's dividend)
        d1 = current_dividend * (1 + growth_rate)

        # DDM Fair Value
        ddm_fair_value = d1 / (cost_of_equity - growth_rate)

        # Dividend yield
        dividend_yield = current_dividend / data['current_price']

        # Payout ratio
        earnings_per_share = info.get('trailingEps', 0)
        payout_ratio = (current_dividend / earnings_per_share) if earnings_per_share > 0 else np.nan

        return {
            'current_dividend': current_dividend,
            'd1': d1,
            'growth_rate': growth_rate,
            'historical_growth': historical_growth,
            'ddm_fair_value': ddm_fair_value,
            'dividend_yield': dividend_yield,
            'payout_ratio': payout_ratio
        }

    except Exception as e:
        print(f"     ⚠️  DDM calculation error: {e}")
        return None


# ----------------------- 2. ASSET-BASED VALUATION ----------------------- #
def calculate_asset_based_valuation(data):
    """
    Asset-Based Valuation (Book Value Approach)

    Methods:
    1. Book Value (Shareholders' Equity)
    2. Tangible Book Value (excludes intangibles)
    3. Liquidation Value (conservative haircuts)
    """
    balance_sheet = data['balance_sheet']

    try:
        # Book Value
        total_assets = balance_sheet.loc['Total Assets'].iloc[0]
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[
            0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else \
        balance_sheet.loc['Total Liabilities'].iloc[0]
        shareholders_equity = total_assets - total_liabilities

        book_value_per_share = shareholders_equity / data['shares_outstanding']

        # Tangible Book Value (exclude intangibles)
        try:
            intangible_assets = balance_sheet.loc['Goodwill And Other Intangible Assets'].iloc[
                0] if 'Goodwill And Other Intangible Assets' in balance_sheet.index else 0
        except:
            intangible_assets = 0

        tangible_equity = shareholders_equity - intangible_assets
        tangible_book_value_per_share = tangible_equity / data['shares_outstanding']

        # Liquidation Value (apply haircuts)
        try:
            cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[
                0] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
            receivables = balance_sheet.loc['Receivables'].iloc[0] if 'Receivables' in balance_sheet.index else 0
            inventory = balance_sheet.loc['Inventory'].iloc[0] if 'Inventory' in balance_sheet.index else 0
            ppe = balance_sheet.loc['Net PPE'].iloc[0] if 'Net PPE' in balance_sheet.index else 0

            # Apply liquidation haircuts
            liquidation_value = (
                                        cash * 1.0 +  # Cash at full value
                                        receivables * 0.80 +  # 80% of receivables
                                        inventory * 0.60 +  # 60% of inventory (fire sale)
                                        ppe * 0.50 +  # 50% of PP&E (used equipment)
                                        (total_assets - cash - receivables - inventory - ppe) * 0.30  # 30% other assets
                                ) - total_liabilities

            liquidation_value_per_share = liquidation_value / data['shares_outstanding']
        except:
            liquidation_value_per_share = tangible_book_value_per_share * 0.7  # Rough estimate

        # Price-to-Book ratios
        current_price = data['current_price']
        pb_ratio = current_price / book_value_per_share if book_value_per_share > 0 else np.nan
        ptb_ratio = current_price / tangible_book_value_per_share if tangible_book_value_per_share > 0 else np.nan

        return {
            'book_value_per_share': book_value_per_share,
            'tangible_book_value_per_share': tangible_book_value_per_share,
            'liquidation_value_per_share': liquidation_value_per_share,
            'pb_ratio': pb_ratio,
            'ptb_ratio': ptb_ratio,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'shareholders_equity': shareholders_equity,
            'tangible_equity': tangible_equity
        }

    except Exception as e:
        print(f"     ⚠️  Asset-based valuation error: {e}")
        return None


# ----------------------- 3. PRECEDENT TRANSACTIONS ----------------------- #
def calculate_precedent_transactions(data):
    """
    Precedent Transactions Analysis

    Uses historical M&A multiples for the sector:
    - EV/Revenue
    - EV/EBITDA
    - P/E

    Note: In practice, you would use a comprehensive M&A database.
    Here we use approximate sector multiples based on historical deals.
    """
    info = data['info']
    sector = info.get('sector', 'Unknown')

    # Historical M&A multiples by sector (approximations)
    # Source: Capital IQ, PitchBook data (2020-2024 averages)
    sector_multiples = {
        'Technology': {'ev_revenue': 5.5, 'ev_ebitda': 18.0, 'pe': 25.0},
        'Healthcare': {'ev_revenue': 4.0, 'ev_ebitda': 16.0, 'pe': 22.0},
        'Financial Services': {'ev_revenue': 3.5, 'ev_ebitda': 12.0, 'pe': 15.0},
        'Consumer Cyclical': {'ev_revenue': 1.8, 'ev_ebitda': 12.0, 'pe': 18.0},
        'Consumer Defensive': {'ev_revenue': 2.0, 'ev_ebitda': 14.0, 'pe': 20.0},
        'Industrials': {'ev_revenue': 1.5, 'ev_ebitda': 11.0, 'pe': 16.0},
        'Energy': {'ev_revenue': 1.2, 'ev_ebitda': 8.0, 'pe': 12.0},
        'Real Estate': {'ev_revenue': 8.0, 'ev_ebitda': 18.0, 'pe': 22.0},
        'Communication Services': {'ev_revenue': 3.0, 'ev_ebitda': 13.0, 'pe': 20.0},
        'Utilities': {'ev_revenue': 2.5, 'ev_ebitda': 10.0, 'pe': 18.0},
        'Basic Materials': {'ev_revenue': 1.8, 'ev_ebitda': 9.0, 'pe': 14.0}
    }

    # Default multiples if sector not found
    default_multiples = {'ev_revenue': 2.5, 'ev_ebitda': 13.0, 'pe': 18.0}

    multiples = sector_multiples.get(sector, default_multiples)

    try:
        # Get company financials
        revenue = info.get('totalRevenue', 0)
        ebitda = info.get('ebitda', 0)
        earnings = info.get('trailingEps', 0) * data['shares_outstanding']

        # Calculate enterprise value based on precedent multiples
        implied_ev_from_revenue = revenue * multiples['ev_revenue']
        implied_ev_from_ebitda = ebitda * multiples['ev_ebitda'] if ebitda > 0 else np.nan

        # Calculate market cap from P/E
        implied_market_cap_from_pe = earnings * multiples['pe'] if earnings > 0 else np.nan

        # Convert EV to equity value (EV - Net Debt)
        total_debt = info.get('totalDebt', 0)
        cash = info.get('totalCash', 0)
        net_debt = total_debt - cash

        equity_value_from_revenue = implied_ev_from_revenue - net_debt
        equity_value_from_ebitda = implied_ev_from_ebitda - net_debt if not np.isnan(implied_ev_from_ebitda) else np.nan

        # Fair value per share
        fv_from_revenue = equity_value_from_revenue / data['shares_outstanding']
        fv_from_ebitda = equity_value_from_ebitda / data['shares_outstanding'] if not np.isnan(
            equity_value_from_ebitda) else np.nan
        fv_from_pe = implied_market_cap_from_pe / data['shares_outstanding'] if not np.isnan(
            implied_market_cap_from_pe) else np.nan

        # Blended precedent fair value
        precedent_values = [v for v in [fv_from_revenue, fv_from_ebitda, fv_from_pe] if not np.isnan(v)]
        precedent_fair_value = np.mean(precedent_values) if precedent_values else np.nan

        # Premium/Discount to current price
        current_price = data['current_price']
        implied_premium = ((precedent_fair_value - current_price) / current_price * 100) if not np.isnan(
            precedent_fair_value) else np.nan

        return {
            'sector': sector,
            'ev_revenue_multiple': multiples['ev_revenue'],
            'ev_ebitda_multiple': multiples['ev_ebitda'],
            'pe_multiple': multiples['pe'],
            'fv_from_revenue': fv_from_revenue,
            'fv_from_ebitda': fv_from_ebitda,
            'fv_from_pe': fv_from_pe,
            'precedent_fair_value': precedent_fair_value,
            'implied_premium_%': implied_premium
        }

    except Exception as e:
        print(f"     ⚠️  Precedent transactions error: {e}")
        return None


# ----------------------- MAIN ANALYSIS ----------------------- #
print("\n" + "=" * 70)
print("RUNNING SUPPLEMENTARY VALUATION...")
print("=" * 70)

all_results = []

for ticker in tickers:
    print(f"\n{'=' * 70}")
    print(f"ANALYZING: {ticker}")
    print(f"{'=' * 70}")

    data = get_data(ticker)
    if data is None:
        continue

    # Get cost of equity
    print(f"  → Getting cost of equity...")
    cost_of_equity = get_cost_of_equity(ticker, capm_data, data['info'])

    # 1. Dividend Discount Model
    print(f"  → Running DDM...")
    ddm_result = calculate_ddm(data, cost_of_equity)

    if ddm_result:
        print(f"     DDM Fair Value: ${ddm_result['ddm_fair_value']:.2f}")
        print(f"     Dividend Yield: {ddm_result['dividend_yield']:.2%}, Growth: {ddm_result['growth_rate']:.2%}")

    # 2. Asset-Based Valuation
    print(f"  → Running Asset-Based Valuation...")
    asset_result = calculate_asset_based_valuation(data)

    if asset_result:
        print(f"     Book Value: ${asset_result['book_value_per_share']:.2f}")
        print(f"     Tangible Book Value: ${asset_result['tangible_book_value_per_share']:.2f}")
        print(f"     Liquidation Value: ${asset_result['liquidation_value_per_share']:.2f}")

    # 3. Precedent Transactions
    print(f"  → Running Precedent Transactions Analysis...")
    precedent_result = calculate_precedent_transactions(data)

    if precedent_result:
        print(f"     Precedent Fair Value: ${precedent_result['precedent_fair_value']:.2f}")
        print(f"     Implied Premium: {precedent_result['implied_premium_%']:.1f}%")

    # Compile results
    current_price = data['current_price']

    result = {
        'Ticker': ticker,
        'Sector': data['info'].get('sector', 'N/A'),
        'Current_Price': current_price,

        # DDM
        'DDM_Fair_Value': ddm_result['ddm_fair_value'] if ddm_result else np.nan,
        'DDM_MoS_%': ((ddm_result['ddm_fair_value'] - current_price) / current_price * 100) if ddm_result else np.nan,
        'Dividend_Yield_%': ddm_result['dividend_yield'] * 100 if ddm_result else np.nan,
        'Dividend_Growth_%': ddm_result['growth_rate'] * 100 if ddm_result else np.nan,
        'Payout_Ratio_%': ddm_result['payout_ratio'] * 100 if ddm_result else np.nan,

        # Asset-Based
        'Book_Value': asset_result['book_value_per_share'] if asset_result else np.nan,
        'Book_Value_MoS_%': ((asset_result[
                                  'book_value_per_share'] - current_price) / current_price * 100) if asset_result else np.nan,
        'Tangible_Book_Value': asset_result['tangible_book_value_per_share'] if asset_result else np.nan,
        'Liquidation_Value': asset_result['liquidation_value_per_share'] if asset_result else np.nan,
        'P/B_Ratio': asset_result['pb_ratio'] if asset_result else np.nan,
        'P/TB_Ratio': asset_result['ptb_ratio'] if asset_result else np.nan,

        # Precedent Transactions
        'Precedent_Fair_Value': precedent_result['precedent_fair_value'] if precedent_result else np.nan,
        'Precedent_MoS_%': ((precedent_result[
                                 'precedent_fair_value'] - current_price) / current_price * 100) if precedent_result else np.nan,
        'Sector_EV/Revenue_Multiple': precedent_result['ev_revenue_multiple'] if precedent_result else np.nan,
        'Sector_EV/EBITDA_Multiple': precedent_result['ev_ebitda_multiple'] if precedent_result else np.nan,
        'Sector_P/E_Multiple': precedent_result['pe_multiple'] if precedent_result else np.nan,
    }

    all_results.append(result)

# ----------------------- OUTPUT ----------------------- #
if all_results:
    df = pd.DataFrame(all_results)

    # Save to Excel
    with pd.ExcelWriter(EXPORT_EXCEL, engine='openpyxl') as writer:
        # Summary
        summary_cols = ['Ticker', 'Sector', 'Current_Price', 'DDM_Fair_Value', 'Book_Value',
                        'Precedent_Fair_Value', 'DDM_MoS_%', 'Book_Value_MoS_%', 'Precedent_MoS_%']
        df[summary_cols].to_excel(writer, sheet_name='Summary', index=False)

        # Full results
        df.to_excel(writer, sheet_name='Detailed_Results', index=False)

        # DDM details
        ddm_cols = ['Ticker', 'DDM_Fair_Value', 'DDM_MoS_%', 'Dividend_Yield_%', 'Dividend_Growth_%', 'Payout_Ratio_%']
        df[ddm_cols].to_excel(writer, sheet_name='DDM_Analysis', index=False)

        # Asset-based details
        asset_cols = ['Ticker', 'Book_Value', 'Tangible_Book_Value', 'Liquidation_Value',
                      'P/B_Ratio', 'P/TB_Ratio', 'Book_Value_MoS_%']
        df[asset_cols].to_excel(writer, sheet_name='Asset_Based', index=False)

        # Precedent transactions details
        precedent_cols = ['Ticker', 'Sector', 'Precedent_Fair_Value', 'Precedent_MoS_%',
                          'Sector_EV/Revenue_Multiple', 'Sector_EV/EBITDA_Multiple', 'Sector_P/E_Multiple']
        df[precedent_cols].to_excel(writer, sheet_name='Precedent_Transactions', index=False)

    print("\n" + "=" * 70)
    print("✅ SUPPLEMENTARY VALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {EXPORT_EXCEL}")

    # Display summary
    print("\n--- SUPPLEMENTARY VALUATION SUMMARY ---")
    display_df = df[['Ticker', 'Current_Price', 'DDM_Fair_Value', 'Book_Value', 'Precedent_Fair_Value']].copy()
    display_df['Current_Price'] = display_df['Current_Price'].apply(lambda x: f"${x:.2f}")
    display_df['DDM_Fair_Value'] = display_df['DDM_Fair_Value'].apply(
        lambda x: f"${x:.2f}" if not np.isnan(x) else "N/A")
    display_df['Book_Value'] = display_df['Book_Value'].apply(lambda x: f"${x:.2f}" if not np.isnan(x) else "N/A")
    display_df['Precedent_Fair_Value'] = display_df['Precedent_Fair_Value'].apply(
        lambda x: f"${x:.2f}" if not np.isnan(x) else "N/A")
    print(display_df.to_string(index=False))

    # Visualizations
    print("\n--- GENERATING VISUALIZATIONS ---")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Fair Value Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.2
    ax1.bar(x - width * 1.5, df['Current_Price'], width, label='Current', color='#e74c3c', alpha=0.8)
    ax1.bar(x - width * 0.5, df['DDM_Fair_Value'], width, label='DDM', color='#3498db', alpha=0.8)
    ax1.bar(x + width * 0.5, df['Book_Value'], width, label='Book Value', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width * 1.5, df['Precedent_Fair_Value'], width, label='Precedent', color='#f39c12', alpha=0.8)
    ax1.set_xlabel('Stock', fontweight='bold')
    ax1.set_ylabel('Price ($)', fontweight='bold')
    ax1.set_title('Fair Value Methods Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Ticker'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Chart 2: Margin of Safety by Method
    ax2 = axes[0, 1]
    valid_ddm = df[~df['DDM_MoS_%'].isna()]
    if len(valid_ddm) > 0:
        colors = ['green' if x > 0 else 'red' for x in valid_ddm['DDM_MoS_%']]
        ax2.barh(valid_ddm['Ticker'], valid_ddm['DDM_MoS_%'], color=colors, alpha=0.7)
        ax2.set_xlabel('DDM Margin of Safety (%)', fontweight='bold')
        ax2.set_title('DDM: Margin of Safety', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')
    else:
        ax2.text(0.5, 0.5, 'No DDM values\n(Stocks may not pay dividends)',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('DDM: Margin of Safety', fontsize=14, fontweight='bold')

    # Chart 3: P/B Ratio
    ax3 = axes[1, 0]
    valid_pb = df[~df['P/B_Ratio'].isna()]
    colors_pb = ['green' if x < 1.5 else 'orange' if x < 3 else 'red' for x in valid_pb['P/B_Ratio']]
    ax3.barh(valid_pb['Ticker'], valid_pb['P/B_Ratio'], color=colors_pb, alpha=0.7)
    ax3.set_xlabel('P/B Ratio', fontweight='bold')
    ax3.set_title('Price-to-Book Ratios', fontsize=14, fontweight='bold')
    ax3.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='P/B = 1.0')
    ax3.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5, label='P/B = 1.5')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.legend()

    # Chart 4: Precedent Transactions Premium
    ax4 = axes[1, 1]
    valid_precedent = df[~df['Precedent_MoS_%'].isna()]
    colors_prec = ['green' if x > 0 else 'red' for x in valid_precedent['Precedent_MoS_%']]
    ax4.barh(valid_precedent['Ticker'], valid_precedent['Precedent_MoS_%'], color=colors_prec, alpha=0.7)
    ax4.set_xlabel('Precedent Transactions MoS (%)', fontweight='bold')
    ax4.set_title('M&A Multiples: Implied Premium/Discount', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('output/images/Supplementary_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Charts saved to: output/images/Supplementary_Analysis.png")

    # Key insights
    print("\n--- KEY INSIGHTS ---")

    # DDM stocks
    dividend_stocks = df[~df['DDM_Fair_Value'].isna()]
    if len(dividend_stocks) > 0:
        undervalued_ddm = dividend_stocks[dividend_stocks['DDM_MoS_%'] > 10]
        if len(undervalued_ddm) > 0:
            print(f"\n💰 Undervalued Dividend Stocks (DDM):")
            for _, row in undervalued_ddm.iterrows():
                print(
                    f"   {row['Ticker']}: ${row['Current_Price']:.2f} → DDM FV: ${row['DDM_Fair_Value']:.2f} (Yield: {row['Dividend_Yield_%']:.2f}%)")

    # Value stocks (P/B < 1)
    value_stocks = df[df['P/B_Ratio'] < 1.0]
    if len(value_stocks) > 0:
        print(f"\n📚 Deep Value Stocks (P/B < 1.0):")
        for _, row in value_stocks.iterrows():
            print(f"   {row['Ticker']}: P/B = {row['P/B_Ratio']:.2f}, Book Value: ${row['Book_Value']:.2f}")

    # M&A targets
    undervalued_precedent = df[df['Precedent_MoS_%'] > 20]
    if len(undervalued_precedent) > 0:
        print(f"\n🎯 Potential M&A Targets (>20% discount to precedent multiples):")
        for _, row in undervalued_precedent.iterrows():
            print(f"   {row['Ticker']}: Trading at {row['Precedent_MoS_%']:.1f}% discount to sector M&A multiples")

    print("\n" + "=" * 70)

else:
    print("\n❌ No results to save")