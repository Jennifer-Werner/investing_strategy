"""
COMPREHENSIVE VALUATION SYNTHESIS - NORMALIZED SCORING
Pulls data from all computed models and creates unified analysis

Data Sources:
1. Risk Models: CAPM, Fama-French, APT
2. Trading Costs: Break-even alpha, Capacity
3. Valuations: DCF, EVA, Relative, Market-based

Output: Comprehensive investment recommendations with normalized scores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ======================= CONFIGURATION ======================= #
class Config:
    # Input files (relative to valuation/ directory)
    RISK_MODEL_OUTPUT = "risk-models/output/sheets/Risk_Model_Output.xlsx"
    APT_OUTPUT = "risk-models/output/sheets/APT_Model_Output.xlsx"
    MODEL_COMPARISON = "risk-models/output/sheets/Model_Comparison_Report.xlsx"
    TRANSACTION_COST = "trading-cost/output/sheets/Transaction_Cost_Analysis.xlsx"
    VALUATION_MAIN = "valuation/output/sheets/Valuation_Analysis.xlsx"
    VALUATION_SUPP = "valuation/output/sheets/Supplementary_Valuation.xlsx"

    # Output
    OUTPUT_DIR = "comprehensive-analysis/output/sheets"
    OUTPUT_FILE = "Comprehensive_Valuation_Report_Normalized.xlsx"
    CHARTS_DIR = "comprehensive-analysis/output/images"

    # Scoring weights
    WEIGHTS = {
        'alpha_quality': 0.35,      # Quality of alpha across models
        'trading_feasibility': 0.20, # Trading costs vs alpha
        'valuation_upside': 0.30,    # Upside based on valuations
        'risk_adjusted': 0.15        # Risk-adjusted metrics
    }

config = Config()

# Create output directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.CHARTS_DIR, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE VALUATION SYNTHESIS - NORMALIZED SCORING")
print("Analyzing all computed models and creating unified recommendations")
print("=" * 80)

# ======================= DATA LOADING ======================= #
def load_all_data():
    """Load all computed data from spreadsheets"""
    data = {}

    print("\n📂 LOADING DATA SOURCES...")
    print("=" * 80)

    # RISK MODEL OUTPUT
    try:
        filename = config.RISK_MODEL_OUTPUT
        print(f"📥 Loading {filename}... ", end="")
        xls = pd.ExcelFile(filename)
        if 'Summary' in xls.sheet_names:
            data['capm'] = pd.read_excel(xls, sheet_name='Summary')
        else:
            data['capm'] = None
        print("✅")
    except Exception as e:
        print(f"❌ Error: {e}")
        data['capm'] = None

    # APT MODEL OUTPUT
    try:
        filename = config.APT_OUTPUT
        print(f"📥 Loading {filename}... ", end="")
        xls = pd.ExcelFile(filename)
        if 'Summary' in xls.sheet_names:
            data['apt'] = pd.read_excel(xls, sheet_name='Summary')
        else:
            data['apt'] = None
        print("✅")
    except Exception as e:
        print(f"❌ Error: {e}")
        data['apt'] = None

    # TRANSACTION COST ANALYSIS
    try:
        filename = config.TRANSACTION_COST
        print(f"📥 Loading {filename}... ", end="")
        xls = pd.ExcelFile(filename)
        for sheet in ['Summary', 'Transaction_Costs', 'Cost_Analysis']:
            if sheet in xls.sheet_names:
                data['trading_cost'] = pd.read_excel(xls, sheet_name=sheet)
                break
        else:
            data['trading_cost'] = None
        print("✅")
    except Exception as e:
        print(f"❌ Error: {e}")
        data['trading_cost'] = None

    # VALUATION ANALYSIS
    try:
        filename = config.VALUATION_MAIN
        print(f"📥 Loading {filename}... ", end="")
        xls = pd.ExcelFile(filename)
        if 'Summary' in xls.sheet_names:
            data['valuation_summary'] = pd.read_excel(xls, sheet_name='Summary')
        if 'DCF_Analysis' in xls.sheet_names:
            data['dcf'] = pd.read_excel(xls, sheet_name='DCF_Analysis')
        if 'EVA_Analysis' in xls.sheet_names:
            data['eva'] = pd.read_excel(xls, sheet_name='EVA_Analysis')
        if 'Relative_Valuation' in xls.sheet_names:
            data['relative'] = pd.read_excel(xls, sheet_name='Relative_Valuation')
        print("✅")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    return data

# ======================= ALPHA ANALYSIS ======================= #
def analyze_alpha(ticker, data):
    """Analyze alpha across all risk models"""
    alpha_analysis = {
        'ticker': ticker,
        'capm_alpha': np.nan,
        'ff3_alpha': np.nan,
        'ff5_alpha': np.nan,
        'apt_alpha': np.nan,
        'avg_alpha': np.nan,
        'alpha_consistency': 0,
        'best_model': 'N/A',
        'beta': np.nan,
        'sharpe_ratio': np.nan
    }

    # CAPM Alpha
    if data['capm'] is not None and 'Ticker' in data['capm'].columns and ticker in data['capm']['Ticker'].values:
        capm_row = data['capm'][data['capm']['Ticker'] == ticker].iloc[0]
        alpha_analysis['capm_alpha'] = capm_row.get('Mean_Alpha_CAPM', np.nan)
        alpha_analysis['beta'] = capm_row.get('Beta_CAPM', np.nan)
        alpha_analysis['sharpe_ratio'] = capm_row.get('Info_Ratio_CAPM', np.nan)

    # Fama-French Alpha
    if data['capm'] is not None and 'Ticker' in data['capm'].columns and ticker in data['capm']['Ticker'].values:
        ff_row = data['capm'][data['capm']['Ticker'] == ticker].iloc[0]
        alpha_analysis['ff3_alpha'] = ff_row.get('Mean_Alpha_FF3', np.nan)
        alpha_analysis['ff5_alpha'] = ff_row.get('Mean_Alpha_FF5', np.nan)

    # APT Alpha
    if data['apt'] is not None and 'Ticker' in data['apt'].columns and ticker in data['apt']['Ticker'].values:
        apt_row = data['apt'][data['apt']['Ticker'] == ticker].iloc[0]
        alpha_analysis['apt_alpha'] = apt_row.get('Mean_Alpha', np.nan)

    # Calculate average alpha (using all 4 models: CAPM, FF3, FF5, APT)
    alphas = [alpha_analysis['capm_alpha'], alpha_analysis['apt_alpha'],
              alpha_analysis['ff3_alpha'], alpha_analysis['ff5_alpha']]
    valid_alphas = [x for x in alphas if not np.isnan(x)]

    if valid_alphas:
        alpha_analysis['avg_alpha'] = np.mean(valid_alphas)
        alpha_analysis['alpha_std'] = np.std(valid_alphas) if len(valid_alphas) > 1 else 0
        alpha_analysis['alpha_consistency'] = 100 - min(alpha_analysis['alpha_std'] * 1000, 100)

    return alpha_analysis

# ======================= TRADING FEASIBILITY ======================= #
def analyze_trading_feasibility(ticker, alpha_data, data):
    """Analyze if alpha exceeds trading costs"""
    feasibility = {
        'ticker': ticker,
        'can_trade': False,
        'net_alpha': np.nan,
        'alpha_to_cost_ratio': np.nan,
        'break_even_alpha': np.nan,
        'total_cost': np.nan
    }

    if data['trading_cost'] is None or 'Ticker' not in data['trading_cost'].columns:
        return feasibility

    if ticker not in data['trading_cost']['Ticker'].values:
        return feasibility

    tc_row = data['trading_cost'][data['trading_cost']['Ticker'] == ticker].iloc[0]

    # Use correct column names from your file
    break_even_alpha_pct = tc_row.get('Breakeven_Alpha_pct', np.nan)
    total_cost_pct = tc_row.get('TC_Round_Trip_pct', np.nan) * 100  # Convert decimal to %
    annual_cost_pct = tc_row.get('TC_Annual_pct', np.nan) * 100  # Convert decimal to %

    avg_alpha = alpha_data.get('avg_alpha', np.nan)

    if not np.isnan(avg_alpha) and not np.isnan(break_even_alpha_pct):
        avg_alpha_annual_pct = avg_alpha * 12 * 100  # Monthly to annual %
        feasibility['net_alpha'] = avg_alpha_annual_pct - break_even_alpha_pct
        feasibility['can_trade'] = feasibility['net_alpha'] > 0
        feasibility['alpha_to_cost_ratio'] = avg_alpha_annual_pct / break_even_alpha_pct if break_even_alpha_pct != 0 else np.nan
        feasibility['break_even_alpha'] = break_even_alpha_pct
        feasibility['total_cost'] = total_cost_pct
        feasibility['annual_alpha_pct'] = avg_alpha_annual_pct

        # Additional metrics from your trading cost analysis
        feasibility['avg_dollar_volume_m'] = tc_row.get('Avg_Dollar_Volume_M', np.nan)
        feasibility['max_position_m'] = tc_row.get('Max_Position_M', np.nan)

    return feasibility

# ======================= VALUATION ANALYSIS ======================= #
def analyze_valuation(ticker, data):
    """Comprehensive valuation analysis"""
    val_analysis = {
        'ticker': ticker,
        'current_price': np.nan,
        'dcf_fair_value': np.nan,
        'comps_fair_value': np.nan,
        'blended_fair_value': np.nan,
        'upside_pct': np.nan
    }

    # Summary data
    if data.get('valuation_summary') is not None and 'Ticker' in data['valuation_summary'].columns:
        if ticker in data['valuation_summary']['Ticker'].values:
            sum_row = data['valuation_summary'][data['valuation_summary']['Ticker'] == ticker].iloc[0]
            val_analysis['current_price'] = sum_row.get('Current_Price', np.nan)
            val_analysis['blended_fair_value'] = sum_row.get('Blended_Fair_Value', np.nan)
            val_analysis['valuation_signal'] = sum_row.get('Valuation_Signal', 'N/A')

    # DCF Analysis
    if data['dcf'] is not None and 'Ticker' in data['dcf'].columns and ticker in data['dcf']['Ticker'].values:
        dcf_row = data['dcf'][data['dcf']['Ticker'] == ticker].iloc[0]
        val_analysis['dcf_fair_value'] = dcf_row.get('DCF_Fair_Value', np.nan)
        val_analysis['dcf_mos'] = dcf_row.get('DCF_MoS_%', np.nan)
        val_analysis['dcf_tv_pct'] = dcf_row.get('DCF_TV_%', np.nan)
        val_analysis['wacc'] = dcf_row.get('WACC_%', np.nan)

    # EVA Analysis
    if data.get('eva') is not None and 'Ticker' in data['eva'].columns:
        if ticker in data['eva']['Ticker'].values:
            eva_row = data['eva'][data['eva']['Ticker'] == ticker].iloc[0]
            val_analysis['roic'] = eva_row.get('ROIC_%', np.nan)
            val_analysis['roic_wacc_spread'] = eva_row.get('ROIC-WACC_Spread_%', np.nan)

    # Relative Valuation
    if data['relative'] is not None and 'Ticker' in data['relative'].columns and ticker in data['relative']['Ticker'].values:
        rel_row = data['relative'][data['relative']['Ticker'] == ticker].iloc[0]
        val_analysis['comps_fair_value'] = rel_row.get('Comps_Fair_Value', np.nan)
        val_analysis['comps_mos'] = rel_row.get('Comps_MoS_%', np.nan)
        val_analysis['pe_ratio'] = rel_row.get('P/E', np.nan)
        val_analysis['peer_median_pe'] = rel_row.get('Peer_Median_P/E', np.nan)
        val_analysis['pe_zscore'] = rel_row.get('P/E_Z-Score', np.nan)

    # Calculate upside
    if not np.isnan(val_analysis['current_price']) and not np.isnan(val_analysis['blended_fair_value']):
        val_analysis['upside_pct'] = ((val_analysis['blended_fair_value'] - val_analysis['current_price']) /
                                      val_analysis['current_price'] * 100)

    return val_analysis

# ======================= NORMALIZED SCORING ======================= #
def normalize_score(value, percentiles):
    """
    Normalize a value to 0-100 scale based on percentile distribution
    """
    if np.isnan(value):
        return 50  # Neutral score for missing data

    # Map percentiles to scores
    if value <= percentiles['p10']:
        return 10
    elif value <= percentiles['p25']:
        return 10 + (value - percentiles['p10']) / (percentiles['p25'] - percentiles['p10']) * 15
    elif value <= percentiles['p50']:
        return 25 + (value - percentiles['p25']) / (percentiles['p50'] - percentiles['p25']) * 25
    elif value <= percentiles['p75']:
        return 50 + (value - percentiles['p50']) / (percentiles['p75'] - percentiles['p50']) * 25
    elif value <= percentiles['p90']:
        return 75 + (value - percentiles['p75']) / (percentiles['p90'] - percentiles['p75']) * 15
    else:
        return min(90 + (value - percentiles['p90']) / max(percentiles['p90'] - percentiles['p75'], 0.001) * 10, 100)

def calculate_comprehensive_score(ticker, alpha_data, feasibility_data, valuation_data, percentiles=None):
    """Calculate overall investment score with normalized metrics"""
    scores = {
        'ticker': ticker,
        'alpha_score': 50,
        'feasibility_score': 50,
        'valuation_score': 50,
        'risk_score': 50,
        'total_score': 0,
        'recommendation': 'HOLD'
    }

    if percentiles is None:
        # Fallback if no percentiles
        scores['total_score'] = 50
        return scores

    # 1. Alpha Quality Score (normalized)
    avg_alpha = alpha_data.get('avg_alpha', np.nan)
    consistency = alpha_data.get('alpha_consistency', np.nan)

    if not np.isnan(avg_alpha):
        alpha_score = normalize_score(avg_alpha * 100, percentiles['alpha'])
        consistency_score = normalize_score(consistency, percentiles['consistency'])
        scores['alpha_score'] = (alpha_score * 0.7) + (consistency_score * 0.3)

    # 2. Trading Feasibility Score (normalized)
    alpha_cost_ratio = feasibility_data.get('alpha_to_cost_ratio', np.nan)
    if not np.isnan(alpha_cost_ratio):
        scores['feasibility_score'] = normalize_score(alpha_cost_ratio, percentiles['feasibility'])

    # 3. Valuation Upside Score (normalized)
    upside = valuation_data.get('upside_pct', np.nan)
    if not np.isnan(upside):
        scores['valuation_score'] = normalize_score(upside, percentiles['upside'])

    # 4. Risk-Adjusted Score (normalized)
    sharpe = alpha_data.get('sharpe_ratio', np.nan)
    roic_spread = valuation_data.get('roic_wacc_spread', np.nan)

    risk_scores = []
    if not np.isnan(sharpe):
        risk_scores.append(normalize_score(sharpe, percentiles['sharpe']))
    if not np.isnan(roic_spread):
        risk_scores.append(normalize_score(roic_spread, percentiles['roic_spread']))

    scores['risk_score'] = np.mean(risk_scores) if risk_scores else 50

    # Calculate Total Score (weighted)
    scores['total_score'] = (
        scores['alpha_score'] * config.WEIGHTS['alpha_quality'] +
        scores['feasibility_score'] * config.WEIGHTS['trading_feasibility'] +
        scores['valuation_score'] * config.WEIGHTS['valuation_upside'] +
        scores['risk_score'] * config.WEIGHTS['risk_adjusted']
    )

    # Generate Recommendation
    total = scores['total_score']
    if total >= 80:
        scores['recommendation'] = 'STRONG BUY'
        scores['recommendation_emoji'] = '🟢🟢'
    elif total >= 65:
        scores['recommendation'] = 'BUY'
        scores['recommendation_emoji'] = '🟢'
    elif total >= 50:
        scores['recommendation'] = 'HOLD'
        scores['recommendation_emoji'] = '🟡'
    elif total >= 35:
        scores['recommendation'] = 'SELL'
        scores['recommendation_emoji'] = '🔴'
    else:
        scores['recommendation'] = 'STRONG SELL'
        scores['recommendation_emoji'] = '🔴🔴'

    return scores

# ======================= MAIN ANALYSIS ======================= #
print("\n" + "=" * 80)
print("LOADING DATA...")
print("=" * 80)

data = load_all_data()

if data is None:
    print("\n❌ Failed to load data. Exiting.")
    exit(1)

# Get list of all tickers
all_tickers = set()
if data.get('capm') is not None:
    all_tickers.update(data['capm']['Ticker'].values)
if data.get('valuation_summary') is not None:
    all_tickers.update(data['valuation_summary']['Ticker'].values)

if len(all_tickers) == 0:
    print("\n❌ No tickers found. Exiting.")
    exit(1)

all_tickers = sorted(list(all_tickers))

print(f"\n📊 Analyzing {len(all_tickers)} stocks...")
print("=" * 80)

# ============ FIRST PASS: Collect raw metrics ============
print("\n📐 FIRST PASS: Collecting raw metrics for normalization...")
raw_metrics = {
    'alpha': [],
    'consistency': [],
    'feasibility': [],
    'upside': [],
    'sharpe': [],
    'roic_spread': []
}

preliminary_results = []

for ticker in all_tickers:
    alpha_data = analyze_alpha(ticker, data)
    feasibility_data = analyze_trading_feasibility(ticker, alpha_data, data)
    valuation_data = analyze_valuation(ticker, data)

    preliminary_results.append({
        'ticker': ticker,
        'alpha_data': alpha_data,
        'feasibility_data': feasibility_data,
        'valuation_data': valuation_data
    })

    # Collect raw metrics
    avg_alpha = alpha_data.get('avg_alpha', np.nan)
    if not np.isnan(avg_alpha):
        raw_metrics['alpha'].append(avg_alpha * 100)

    consistency = alpha_data.get('alpha_consistency', np.nan)
    if not np.isnan(consistency):
        raw_metrics['consistency'].append(consistency)

    alpha_cost_ratio = feasibility_data.get('alpha_to_cost_ratio', np.nan)
    if not np.isnan(alpha_cost_ratio):
        raw_metrics['feasibility'].append(alpha_cost_ratio)

    upside = valuation_data.get('upside_pct', np.nan)
    if not np.isnan(upside):
        raw_metrics['upside'].append(upside)

    sharpe = alpha_data.get('sharpe_ratio', np.nan)
    if not np.isnan(sharpe):
        raw_metrics['sharpe'].append(sharpe)

    roic_spread = valuation_data.get('roic_wacc_spread', np.nan)
    if not np.isnan(roic_spread):
        raw_metrics['roic_spread'].append(roic_spread)

# Calculate percentiles
print("\n📊 Calculating percentiles for normalization...")
percentiles = {}
for metric_name, values in raw_metrics.items():
    if len(values) > 0:
        percentiles[metric_name] = {
            'p10': np.percentile(values, 10),
            'p25': np.percentile(values, 25),
            'p50': np.percentile(values, 50),
            'p75': np.percentile(values, 75),
            'p90': np.percentile(values, 90)
        }
        print(f"  {metric_name:15s}: p10={percentiles[metric_name]['p10']:7.2f}, "
              f"p50={percentiles[metric_name]['p50']:7.2f}, "
              f"p90={percentiles[metric_name]['p90']:7.2f}")
    else:
        percentiles[metric_name] = {
            'p10': -1, 'p25': -0.5, 'p50': 0, 'p75': 0.5, 'p90': 1
        }

# ============ SECOND PASS: Calculate normalized scores ============
print("\n" + "="*80)
print("🎯 SECOND PASS: Calculating normalized scores...")
print("="*80)

comprehensive_results = []

for item in preliminary_results:
    ticker = item['ticker']
    alpha_data = item['alpha_data']
    feasibility_data = item['feasibility_data']
    valuation_data = item['valuation_data']

    scores = calculate_comprehensive_score(ticker, alpha_data, feasibility_data, valuation_data, percentiles)

    # Compile results
    result = {**alpha_data, **feasibility_data, **valuation_data, **scores}
    comprehensive_results.append(result)

print(f"\n✅ Analyzed {len(comprehensive_results)} stocks")

# ======================= OUTPUT RESULTS ======================= #
print("\n" + "=" * 80)
print("GENERATING COMPREHENSIVE REPORT...")
print("=" * 80)

df_comprehensive = pd.DataFrame(comprehensive_results)
df_comprehensive = df_comprehensive.sort_values('total_score', ascending=False)

# Create Excel output
output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILE)
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Executive Summary
    summary_cols = ['ticker', 'recommendation', 'total_score', 'upside_pct', 'avg_alpha',
                   'net_alpha', 'current_price', 'blended_fair_value']
    ws_name = 'Executive_Summary'
    df_comprehensive[summary_cols].to_excel(writer, sheet_name=ws_name, index=False)
    print(f"✅ Created worksheet: {ws_name}")

    # Alpha Analysis (without ff_alpha)
    alpha_cols = ['ticker', 'capm_alpha', 'ff3_alpha', 'ff5_alpha', 'apt_alpha', 'avg_alpha',
                  'alpha_consistency', 'best_model', 'sharpe_ratio', 'beta']
    ws_name = 'Alpha_Analysis'
    df_comprehensive[alpha_cols].to_excel(writer, sheet_name=ws_name, index=False)
    print(f"✅ Created worksheet: {ws_name}")

    # Trading Feasibility
    trading_cols = ['ticker', 'can_trade', 'net_alpha', 'alpha_to_cost_ratio',
                   'break_even_alpha', 'total_cost', 'annual_alpha_pct']
    # Only add additional columns if they exist
    if 'avg_dollar_volume_m' in df_comprehensive.columns:
        trading_cols.append('avg_dollar_volume_m')
    if 'max_position_m' in df_comprehensive.columns:
        trading_cols.append('max_position_m')
    ws_name = 'Trading_Feasibility'
    df_comprehensive[trading_cols].to_excel(writer, sheet_name=ws_name, index=False)
    print(f"✅ Created worksheet: {ws_name}")

    # Valuation Details
    val_cols = ['ticker', 'current_price', 'dcf_fair_value', 'comps_fair_value',
                'blended_fair_value', 'upside_pct']
    # Add optional columns if they exist
    optional_val_cols = ['dcf_mos', 'comps_mos', 'eva_$m', 'roic', 'roic_wacc_spread',
                        'pe_ratio', 'peer_median_pe']
    for col in optional_val_cols:
        if col in df_comprehensive.columns:
            val_cols.append(col)
    ws_name = 'Valuation_Details'
    df_comprehensive[val_cols].to_excel(writer, sheet_name=ws_name, index=False)
    print(f"✅ Created worksheet: {ws_name}")

    # Score Breakdown
    score_cols = ['ticker', 'alpha_score', 'feasibility_score', 'valuation_score',
                  'risk_score', 'total_score', 'recommendation']
    ws_name = 'Score_Breakdown'
    df_comprehensive[score_cols].to_excel(writer, sheet_name=ws_name, index=False)
    print(f"✅ Created worksheet: {ws_name}")

    # Complete Data
    ws_name = 'Complete_Data'
    df_comprehensive.to_excel(writer, sheet_name=ws_name, index=False)
    print(f"✅ Created worksheet: {ws_name}")

print(f"\n✅ Excel report saved to: {output_path}")

# ======================= VISUALIZATIONS ======================= #
print("\n📊 Generating visualizations...")

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Top Stocks by Total Score
ax1 = fig.add_subplot(gs[0, :])
top_10 = df_comprehensive.head(10)
colors = ['darkgreen' if x >= 65 else 'gold' if x >= 50 else 'darkred' for x in top_10['total_score']]
ax1.barh(top_10['ticker'], top_10['total_score'], color=colors, alpha=0.7)
ax1.set_xlabel('Total Score', fontweight='bold', fontsize=12)
ax1.set_title('Top 10 Stocks by Comprehensive Score', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 100)
ax1.grid(True, alpha=0.3, axis='x')

# 2. Alpha vs Trading Costs
ax2 = fig.add_subplot(gs[1, 0])
tradeable = df_comprehensive[df_comprehensive['can_trade'] == True]
if len(tradeable) > 0:
    ax2.scatter(tradeable['break_even_alpha'], tradeable['avg_alpha']*100,
               c=tradeable['total_score'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    if len(tradeable) > 0 and tradeable['break_even_alpha'].max() > 0:
        ax2.plot([0, tradeable['break_even_alpha'].max()], [0, tradeable['break_even_alpha'].max()],
                'k--', alpha=0.3, label='Break-even')
    ax2.legend()
ax2.set_xlabel('Break-even Alpha (%)', fontweight='bold')
ax2.set_ylabel('Average Alpha (%)', fontweight='bold')
ax2.set_title('Alpha vs Trading Costs', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Valuation Upside Distribution
ax3 = fig.add_subplot(gs[1, 1])
upside_data = df_comprehensive['upside_pct'].dropna()
if len(upside_data) > 0:
    ax3.hist(upside_data, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Fair Value')
    ax3.axvline(x=15, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Buy Threshold')
    ax3.legend()
ax3.set_xlabel('Upside (%)', fontweight='bold')
ax3.set_ylabel('Number of Stocks', fontweight='bold')
ax3.set_title('Valuation Upside Distribution', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Score Components Breakdown
ax4 = fig.add_subplot(gs[1, 2])
score_means = df_comprehensive[['alpha_score', 'feasibility_score', 'valuation_score', 'risk_score']].mean()
colors_comp = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']
ax4.bar(range(len(score_means)), score_means.values, color=colors_comp, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(score_means)))
ax4.set_xticklabels(['Alpha', 'Feasibility', 'Valuation', 'Risk'], rotation=45)
ax4.set_ylabel('Average Score', fontweight='bold')
ax4.set_title('Score Components (Portfolio Average)', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Recommendation Distribution
ax5 = fig.add_subplot(gs[2, 0])
rec_counts = df_comprehensive['recommendation'].value_counts()
colors_rec = {'STRONG BUY': 'darkgreen', 'BUY': 'green', 'HOLD': 'gold',
              'SELL': 'orange', 'STRONG SELL': 'darkred'}
colors_list = [colors_rec.get(x, 'gray') for x in rec_counts.index]
ax5.pie(rec_counts.values, labels=rec_counts.index, autopct='%1.0f%%',
       colors=colors_list, startangle=90)
ax5.set_title('Investment Recommendations', fontsize=12, fontweight='bold')

# 6. Alpha Consistency vs Score
ax6 = fig.add_subplot(gs[2, 1])
valid_consistency = df_comprehensive[~df_comprehensive['alpha_consistency'].isna() &
                                    ~df_comprehensive['total_score'].isna()]
if len(valid_consistency) > 0:
    scatter = ax6.scatter(valid_consistency['alpha_consistency'], valid_consistency['total_score'],
               c=valid_consistency['upside_pct'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Upside %', rotation=270, labelpad=20)
ax6.set_xlabel('Alpha Consistency', fontweight='bold')
ax6.set_ylabel('Total Score', fontweight='bold')
ax6.set_title('Alpha Consistency vs Total Score', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. ROIC vs WACC Spread
ax7 = fig.add_subplot(gs[2, 2])
valid_roic = df_comprehensive[~df_comprehensive['roic_wacc_spread'].isna()]
if len(valid_roic) > 0:
    colors_roic = ['green' if x > 0 else 'red' for x in valid_roic['roic_wacc_spread']]
    top_roic = valid_roic.nlargest(10, 'roic_wacc_spread')
    if len(top_roic) > 0:
        ax7.barh(top_roic['ticker'], top_roic['roic_wacc_spread'],
                color=['green' if x > 0 else 'red' for x in top_roic['roic_wacc_spread']], alpha=0.7)
        ax7.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax7.set_xlabel('ROIC - WACC Spread (%)', fontweight='bold')
ax7.set_title('Value Creation (Top 10)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='x')

plt.suptitle('COMPREHENSIVE VALUATION DASHBOARD', fontsize=16, fontweight='bold', y=0.995)

chart_path = os.path.join(config.CHARTS_DIR, 'Comprehensive_Dashboard.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Dashboard saved to: {chart_path}")

# ======================= RECOMMENDATIONS ======================= #
print("\n" + "=" * 80)
print("INVESTMENT RECOMMENDATIONS")
print("=" * 80)

for rec_type in ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL']:
    subset = df_comprehensive[df_comprehensive['recommendation'] == rec_type]
    if len(subset) > 0:
        emoji = subset.iloc[0]['recommendation_emoji']
        print(f"\n{emoji} {rec_type} ({len(subset)} stocks)")
        print("-" * 80)
        for _, row in subset.head(5).iterrows():
            print(f"  {row['ticker']:6s} | Score: {row['total_score']:.0f} | "
                  f"Alpha: {row['alpha_score']:.0f} | Feas: {row['feasibility_score']:.0f} | "
                  f"Val: {row['valuation_score']:.0f} | Risk: {row['risk_score']:.0f}")

print("\n" + "=" * 80)
print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nAll results saved to: {config.OUTPUT_DIR}/")