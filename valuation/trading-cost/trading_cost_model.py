import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


# ----------------------- DIRECTORY SETUP ----------------------- #
def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("output/sheets", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)


ensure_directories()

# ----------------------- SETTINGS ----------------------- #
EXPORT_EXCEL = "output/sheets/Transaction_Cost_Analysis.xlsx"
LOOKBACK_DAYS = 252  # ~1 year of trading days

# ----------------------- USER INPUTS ----------------------- #
print("=" * 70)
print("TRANSACTION COST MODEL (FINAL CORRECTED VERSION)")
print("=" * 70)

n = int(input("\nEnter the number of tickers to analyze: "))
tickers = [input(f"Enter ticker {i + 1}: ").upper() for i in range(n)]

print("\n--- Portfolio Parameters ---")
TYPICAL_POSITION_SIZE = float(input("Enter typical position size ($): "))
TRADING_FREQUENCY = input("Enter trading frequency (daily/weekly/monthly/quarterly/annual): ").lower()
PARTICIPATION_RATE = float(input("Enter target participation rate (% of daily volume, e.g., 5): ")) / 100

# Convert trading frequency to annual trades
FREQ_MAP = {'daily': 252, 'weekly': 52, 'monthly': 12, 'quarterly': 4, 'annual': 1}
ANNUAL_TRADES = FREQ_MAP.get(TRADING_FREQUENCY, 1)

print(f"\nAssuming {ANNUAL_TRADES} round trip(s) per year per position.")


# ----------------------- DATA COLLECTION ----------------------- #
def get_trading_metrics(ticker, lookback_days=252):
    """Download historical data and calculate trading metrics"""
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=int(lookback_days * 1.5))

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty or len(df) < 20:
            print(f"❌ {ticker}: Insufficient data")
            return None

        stock = yf.Ticker(ticker)
        info = stock.info

        avg_volume = df['Volume'].tail(lookback_days).mean()
        avg_price = df['Close'].tail(lookback_days).mean()
        volatility = df['Close'].pct_change().tail(lookback_days).std() * np.sqrt(252)

        market_cap = info.get('marketCap', avg_price * avg_volume * 252)
        current_price = df['Close'].iloc[-1]
        avg_dollar_volume = avg_volume * avg_price

        print(f"✅ {ticker}: Loaded {len(df)} days of data")

        return {
            'ticker': ticker,
            'current_price': float(current_price),
            'avg_daily_volume': float(avg_volume),
            'avg_dollar_volume': float(avg_dollar_volume),
            'volatility': float(volatility),
            'market_cap': float(market_cap),
            'data_points': len(df)
        }

    except Exception as e:
        print(f"❌ {ticker}: Error downloading data - {e}")
        return None


# ----------------------- COST MODELS (FINAL CORRECTED) ----------------------- #

def estimate_bid_ask_spread(ticker_data):
    """
    Estimate bid-ask spread based on liquidity ONLY

    Academic reference:
    - Hasbrouck (2009): "Trading Costs and Returns for U.S. Equities"
    - Holden (2009): "New Low-Frequency Spread Measures"

    Returns spread in basis points (bps)
    """
    dollar_volume = ticker_data['avg_dollar_volume']

    # Liquidity-based spread estimates (empirically validated)
    if dollar_volume > 1_000_000_000:  # >$1B daily (mega-cap)
        spread_bps = 1
    elif dollar_volume > 500_000_000:  # $500M-$1B (large-cap high)
        spread_bps = 2
    elif dollar_volume > 100_000_000:  # $100-500M (large-cap)
        spread_bps = 3
    elif dollar_volume > 50_000_000:  # $50-100M (mid-cap high)
        spread_bps = 5
    elif dollar_volume > 10_000_000:  # $10-50M (mid-cap)
        spread_bps = 10
    elif dollar_volume > 5_000_000:  # $5-10M (small-cap high)
        spread_bps = 15
    elif dollar_volume > 1_000_000:  # $1-5M (small-cap)
        spread_bps = 25
    else:  # <$1M (micro-cap)
        spread_bps = 50

    return {
        'estimated_spread_bps': spread_bps
    }


def calculate_market_impact(ticker_data, position_size, participation_rate):
    """
    CORRECTED: Calculate market impact using proper Almgren-Chriss calibration

    Academic calibration (Almgren & Chriss 2001):
    - η (temporary): 0.01 to 0.1 (NOT 0.2-1.0!)
    - γ (permanent): 0.01 to 0.05 (NOT 0.1-0.3!)

    Formula:
    - Temporary impact: η * σ * sqrt(participation_rate) * 10000
    - Permanent impact: γ * σ * participation_rate * 10000
    """
    vol = ticker_data['volatility']
    avg_volume = ticker_data['avg_daily_volume']
    price = ticker_data['current_price']
    dollar_volume = ticker_data['avg_dollar_volume']

    shares = position_size / price
    days_to_trade = max(1, (shares / avg_volume) / participation_rate)

    # CORRECTED COEFFICIENTS (10x smaller than before!)
    if dollar_volume > 100_000_000:  # High liquidity
        eta = 0.02  # was 0.2
        gamma = 0.01  # was 0.1
    elif dollar_volume > 10_000_000:  # Medium liquidity
        eta = 0.05  # was 0.5
        gamma = 0.02  # was 0.2
    else:  # Low liquidity
        eta = 0.1  # was 1.0
        gamma = 0.05  # was 0.3

    # Calculate impact in basis points
    temp_impact_bps = eta * vol * np.sqrt(participation_rate) * 10000
    perm_impact_bps = gamma * vol * participation_rate * 10000

    total_impact_bps = temp_impact_bps + perm_impact_bps

    return {
        'shares_to_trade': shares,
        'days_to_trade': days_to_trade,
        'temp_impact_bps': temp_impact_bps,
        'perm_impact_bps': perm_impact_bps,
        'total_impact_bps': total_impact_bps,
        'eta_coefficient': eta,
        'gamma_coefficient': gamma
    }


def calculate_trading_capacity(ticker_data, max_participation=0.1):
    """Calculate maximum position size before excessive market impact"""
    avg_volume = ticker_data['avg_daily_volume']
    price = ticker_data['current_price']

    max_shares_per_day = avg_volume * max_participation
    max_position_shares = max_shares_per_day * 5
    max_position_dollars = max_position_shares * price

    return {
        'max_shares_per_day': max_shares_per_day,
        'max_position_shares': max_position_shares,
        'max_position_dollars': max_position_dollars,
        'days_to_enter_exit': 5
    }


def calculate_total_transaction_costs(spread_bps, market_impact_bps, position_size, annual_trades):
    """
    Calculate total transaction costs

    For a round-trip (buy + sell):
    - Spread cost: pay half-spread on entry, half-spread on exit
    - Market impact: impact on entry + impact on exit
    """
    spread_pct = spread_bps / 10000
    impact_pct = market_impact_bps / 10000

    # One-way costs (single direction)
    spread_cost_one_way = (spread_pct / 2) * position_size  # Half-spread
    impact_cost_one_way = impact_pct * position_size
    total_one_way = spread_cost_one_way + impact_cost_one_way

    # Round-trip costs (entry + exit)
    total_round_trip = total_one_way * 2

    # Annual costs
    annual_tc_dollars = total_round_trip * annual_trades

    # As percentages
    tc_pct_one_way = (total_one_way / position_size) * 100
    tc_pct_round_trip = (total_round_trip / position_size) * 100
    tc_pct_annual = (annual_tc_dollars / position_size) * 100

    return {
        'spread_cost_one_way': spread_cost_one_way,
        'impact_cost_one_way': impact_cost_one_way,
        'total_one_way': total_one_way,
        'total_round_trip': total_round_trip,
        'annual_tc_dollars': annual_tc_dollars,
        'tc_pct_one_way': tc_pct_one_way,
        'tc_pct_round_trip': tc_pct_round_trip,
        'tc_pct_annual': tc_pct_annual
    }


def breakeven_analysis(tc_pct_annual):
    """Calculate required alpha to overcome transaction costs"""
    target_return = 10.0  # Target 10% return after costs
    required_alpha_to_breakeven = tc_pct_annual
    required_alpha_for_target = tc_pct_annual + target_return

    return {
        'breakeven_alpha': required_alpha_to_breakeven,
        'required_alpha_for_10pct': required_alpha_for_target
    }


# ----------------------- MAIN ANALYSIS ----------------------- #
print("\n" + "=" * 70)
print("ANALYZING TRADING COSTS...")
print("=" * 70)

results = []

for ticker in tickers:
    print(f"\n--- {ticker} ---")

    ticker_data = get_trading_metrics(ticker, LOOKBACK_DAYS)
    if ticker_data is None:
        continue

    spread = estimate_bid_ask_spread(ticker_data)
    impact = calculate_market_impact(ticker_data, TYPICAL_POSITION_SIZE, PARTICIPATION_RATE)
    capacity = calculate_trading_capacity(ticker_data)
    costs = calculate_total_transaction_costs(
        spread['estimated_spread_bps'],
        impact['total_impact_bps'],
        TYPICAL_POSITION_SIZE,
        ANNUAL_TRADES
    )
    breakeven = breakeven_analysis(costs['tc_pct_annual'])

    result = {
        'Ticker': ticker,
        'Current_Price': ticker_data['current_price'],
        'Avg_Daily_Volume': ticker_data['avg_daily_volume'],
        'Avg_Dollar_Volume_M': ticker_data['avg_dollar_volume'] / 1_000_000,
        'Volatility_pct': ticker_data['volatility'] * 100,
        'Market_Cap_M': ticker_data['market_cap'] / 1_000_000,

        # Spread
        'Estimated_Spread_bps': spread['estimated_spread_bps'],

        # Market Impact
        'Market_Impact_bps': impact['total_impact_bps'],
        'Temp_Impact_bps': impact['temp_impact_bps'],
        'Perm_Impact_bps': impact['perm_impact_bps'],
        'Eta': impact['eta_coefficient'],
        'Gamma': impact['gamma_coefficient'],
        'Days_To_Trade': impact['days_to_trade'],

        # Capacity
        'Max_Position_M': capacity['max_position_dollars'] / 1_000_000,
        'Position_vs_Capacity_pct': (TYPICAL_POSITION_SIZE / capacity['max_position_dollars']) * 100,

        # Transaction Costs
        'TC_One_Way_bps': costs['tc_pct_one_way'] * 100,
        'TC_Round_Trip_pct': costs['tc_pct_round_trip'],
        'TC_Annual_pct': costs['tc_pct_annual'],
        'Annual_TC_Dollars': costs['annual_tc_dollars'],

        # Breakeven
        'Breakeven_Alpha_pct': breakeven['breakeven_alpha'],
        'Required_Alpha_10pct_Return': breakeven['required_alpha_for_10pct'],

        # Tradeable flags (realistic thresholds for buy-and-hold)
        'Tradeable_5pct': 'Yes' if (costs['tc_pct_annual'] < 5.0 and
                                    TYPICAL_POSITION_SIZE < capacity['max_position_dollars']) else 'No',
        'Tradeable_10pct': 'Yes' if (costs['tc_pct_annual'] < 10.0 and
                                     TYPICAL_POSITION_SIZE < capacity['max_position_dollars']) else 'No',
        'Tradeable_15pct': 'Yes' if (costs['tc_pct_annual'] < 15.0 and
                                     TYPICAL_POSITION_SIZE < capacity['max_position_dollars']) else 'No'
    }

    results.append(result)

    # Print summary
    print(f"  Dollar Volume: ${ticker_data['avg_dollar_volume'] / 1_000_000:.1f}M")
    print(f"  Spread: {spread['estimated_spread_bps']:.1f} bps")
    print(
        f"  Market Impact: {impact['total_impact_bps']:.1f} bps (Temp: {impact['temp_impact_bps']:.1f}, Perm: {impact['perm_impact_bps']:.1f})")
    print(f"  Coefficients: η={impact['eta_coefficient']:.3f}, γ={impact['gamma_coefficient']:.3f}")
    print(f"  TC one-way: {costs['tc_pct_one_way']:.3f}%")
    print(f"  TC round-trip: {costs['tc_pct_round_trip']:.2f}%")
    print(f"  TC annual ({ANNUAL_TRADES} trades/yr): {costs['tc_pct_annual']:.2f}%")
    print(f"  Breakeven alpha: {breakeven['breakeven_alpha']:.2f}%")
    print(f"  Max capacity: ${capacity['max_position_dollars'] / 1_000_000:.1f}M")
    print(
        f"  Tradeable: 5%={result['Tradeable_5pct']} | 10%={result['Tradeable_10pct']} | 15%={result['Tradeable_15pct']}")

# ----------------------- SAVE & REPORT ----------------------- #
if results:
    df = pd.DataFrame(results)
    df = df.sort_values('TC_Annual_pct')

    with pd.ExcelWriter(EXPORT_EXCEL, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Transaction_Costs', index=False)

        summary = df[['Ticker', 'Tradeable_5pct', 'Tradeable_10pct', 'Tradeable_15pct',
                      'TC_Annual_pct', 'TC_Round_Trip_pct', 'Breakeven_Alpha_pct',
                      'Estimated_Spread_bps', 'Market_Impact_bps',
                      'Avg_Dollar_Volume_M', 'Max_Position_M']]
        summary.to_excel(writer, sheet_name='Summary', index=False)

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {EXPORT_EXCEL}")

    print("\n--- SUMMARY STATISTICS ---")
    print(f"Average spread: {df['Estimated_Spread_bps'].mean():.1f} bps")
    print(f"Average market impact: {df['Market_Impact_bps'].mean():.1f} bps")
    print(f"Average round-trip TC: {df['TC_Round_Trip_pct'].mean():.2f}%")
    print(f"Average annual TC ({ANNUAL_TRADES} trades/yr): {df['TC_Annual_pct'].mean():.2f}%")
    print(f"Median annual TC: {df['TC_Annual_pct'].median():.2f}%")

    # Count tradeable stocks
    tradeable_5 = df[df['Tradeable_5pct'] == 'Yes']
    tradeable_10 = df[df['Tradeable_10pct'] == 'Yes']
    tradeable_15 = df[df['Tradeable_15pct'] == 'Yes']

    print(f"\n--- TRADEABLE STOCKS ---")
    print(f"5% threshold: {len(tradeable_5)}/{len(df)} stocks ({len(tradeable_5) / len(df) * 100:.1f}%)")
    print(f"10% threshold: {len(tradeable_10)}/{len(df)} stocks ({len(tradeable_10) / len(df) * 100:.1f}%)")
    print(f"15% threshold: {len(tradeable_15)}/{len(df)} stocks ({len(tradeable_15) / len(df) * 100:.1f}%)")

    if len(tradeable_5) > 0:
        print(f"\nTradeable at 5%: {', '.join(tradeable_5['Ticker'].tolist())}")
    if len(tradeable_10) > 0:
        print(f"Tradeable at 10%: {', '.join(tradeable_10['Ticker'].tolist())}")
    if len(tradeable_15) > 0:
        print(f"Tradeable at 15%: {', '.join(tradeable_15['Ticker'].tolist())}")

    print(f"\n--- TOP 10 LOWEST COST STOCKS ---")
    print(df[['Ticker', 'TC_Round_Trip_pct', 'TC_Annual_pct', 'Estimated_Spread_bps',
              'Market_Impact_bps', 'Avg_Dollar_Volume_M']].head(10).to_string(index=False))

    print(f"\n--- TRADING COST BREAKDOWN ---")
    print(f"Note: With {ANNUAL_TRADES} round trip(s) per year")
    print(f"- If you trade less frequently, multiply annual TC by (your_trades/{ANNUAL_TRADES})")
    print(f"- Example: 1 trade/year = Annual TC * (1/{ANNUAL_TRADES})")

    # ----------------------- VISUALIZATIONS ----------------------- #
    print("\n--- GENERATING VISUALIZATIONS ---")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Annual TC vs Market Cap
    ax1 = axes[0, 0]
    colors = df['Tradeable_2pct'].map({'Yes': 'green', 'No': 'red'})
    scatter = ax1.scatter(df['Market_Cap_M'], df['TC_Annual_pct'],
                          c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Market Cap ($M)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Annual Transaction Cost (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Transaction Costs vs Market Cap ({ANNUAL_TRADES} trades/year)',
                  fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=2, color='green', linestyle='--', linewidth=2, label='2% threshold', alpha=0.7)
    ax1.axhline(y=3, color='orange', linestyle='--', linewidth=2, label='3% threshold', alpha=0.7)
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, label='5% threshold', alpha=0.7)
    ax1.legend(fontsize=9)

    # Add ticker labels
    for idx, row in df.iterrows():
        if row['TC_Annual_pct'] > 2.0:  # Only label stocks above 2%
            ax1.annotate(row['Ticker'], (row['Market_Cap_M'], row['TC_Annual_pct']),
                         fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

    # Plot 2: Spread vs Market Impact
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['Estimated_Spread_bps'], df['Market_Impact_bps'],
                           c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Bid-Ask Spread (bps)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Market Impact (bps)', fontsize=11, fontweight='bold')
    ax2.set_title('Spread vs Market Impact Components', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add diagonal reference lines
    max_val = max(df['Estimated_Spread_bps'].max(), df['Market_Impact_bps'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Equal spread/impact')
    ax2.legend(fontsize=9)

    for idx, row in df.iterrows():
        ax2.annotate(row['Ticker'], (row['Estimated_Spread_bps'], row['Market_Impact_bps']),
                     fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

    # Plot 3: Top/Bottom Performers
    ax3 = axes[1, 0]
    top_10 = df.nsmallest(10, 'TC_Annual_pct')
    bottom_10 = df.nlargest(10, 'TC_Annual_pct')
    combined = pd.concat([top_10, bottom_10]).sort_values('TC_Annual_pct')

    bar_colors = ['green' if x == 'Yes' else 'red' for x in combined['Tradeable_2pct']]
    bars = ax3.barh(combined['Ticker'], combined['TC_Annual_pct'],
                    color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Annual Transaction Cost (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Best & Worst: Annual Transaction Costs', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax3.axvline(x=2, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax3.axvline(x=3, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax3.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # Add value labels
    for i, (idx, row) in enumerate(combined.iterrows()):
        ax3.text(row['TC_Annual_pct'] + 0.1, i, f"{row['TC_Annual_pct']:.2f}%",
                 va='center', fontsize=8)

    # Plot 4: Round-Trip TC vs Liquidity
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(df['Avg_Dollar_Volume_M'], df['TC_Round_Trip_pct'],
                           c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Average Daily Dollar Volume ($M)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Round-Trip Transaction Cost (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Transaction Costs vs Liquidity', fontsize=13, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Add reference lines
    ax4.axhline(y=0.5, color='green', linestyle='--', linewidth=1,
                label='0.5% (low cost)', alpha=0.7)
    ax4.axhline(y=1.0, color='orange', linestyle='--', linewidth=1,
                label='1.0% (moderate)', alpha=0.7)
    ax4.axhline(y=2.0, color='red', linestyle='--', linewidth=1,
                label='2.0% (high cost)', alpha=0.7)
    ax4.legend(fontsize=9)

    for idx, row in df.iterrows():
        if row['TC_Round_Trip_pct'] > 1.0:  # Only label high-cost stocks
            ax4.annotate(row['Ticker'], (row['Avg_Dollar_Volume_M'], row['TC_Round_Trip_pct']),
                         fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('output/images/Transaction_Cost_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Charts saved to: output/images/Transaction_Cost_Analysis.png")

    # ----------------------- SUMMARY TABLE ----------------------- #
    print("\n--- COST BREAKDOWN BY LIQUIDITY TIER ---")

    # Create liquidity tiers
    df['Liquidity_Tier'] = pd.cut(df['Avg_Dollar_Volume_M'],
                                  bins=[0, 10, 50, 100, 500, float('inf')],
                                  labels=['<$10M', '$10-50M', '$50-100M', '$100-500M', '>$500M'])

    tier_summary = df.groupby('Liquidity_Tier').agg({
        'Ticker': 'count',
        'Estimated_Spread_bps': 'mean',
        'Market_Impact_bps': 'mean',
        'TC_Round_Trip_pct': 'mean',
        'TC_Annual_pct': 'mean',
        'Tradeable_2pct': lambda x: (x == 'Yes').sum()
    }).round(2)

    tier_summary.columns = ['Count', 'Avg Spread (bps)', 'Avg Impact (bps)',
                            'Avg RT TC (%)', 'Avg Annual TC (%)', 'Tradeable @ 2%']
    print(tier_summary.to_string())

else:
    print("\n❌ No results to save")