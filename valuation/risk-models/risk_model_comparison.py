import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# ----------------------- DIRECTORY SETUP ----------------------- #
def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("output/sheets", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)

ensure_directories()

# ----------------------- SETTINGS ----------------------- #
FF_FILE = "output/sheets/Risk_Model_Output.xlsx"
APT_FILE = "output/sheets/APT_Model_Output.xlsx"
OUTPUT_FILE = "output/sheets/Model_Comparison_Report.xlsx"


# ----------------------- LOAD DATA ----------------------- #
def load_excel_safely(filename):
    """Load Excel file and return summary + all sheets"""
    if not Path(filename).exists():
        print(f"❌ File not found: {filename}")
        return None, {}

    try:
        # Load summary
        summary = pd.read_excel(filename, sheet_name="Summary")

        # Load all other sheets
        xl_file = pd.ExcelFile(filename)
        sheets = {}
        for sheet in xl_file.sheet_names:
            if sheet != "Summary":
                sheets[sheet] = pd.read_excel(filename, sheet_name=sheet, index_col=0)

        return summary, sheets
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None, {}


print("=" * 70)
print("RISK MODEL COMPARISON TOOL")
print("=" * 70)
print("\nLoading Fama-French models...")
ff_summary, ff_sheets = load_excel_safely(FF_FILE)

print("Loading APT models...")
apt_summary, apt_sheets = load_excel_safely(APT_FILE)

if ff_summary is None and apt_summary is None:
    print("\n❌ No model files found. Please run risk_models.py and apt_model.py first.")
    exit()

# Get common tickers
ff_tickers = set(ff_sheets.keys()) if ff_sheets else set()
apt_tickers = set(apt_sheets.keys()) if apt_sheets else set()
common_tickers = ff_tickers.intersection(apt_tickers)

if not common_tickers:
    print("\n⚠️ No common tickers found between the two analyses.")
    print(f"FF tickers: {ff_tickers}")
    print(f"APT tickers: {apt_tickers}")

    # If only one model exists, still proceed with what we have
    if ff_tickers:
        print("\n⚠️ Proceeding with Fama-French models only...")
        common_tickers = ff_tickers
    elif apt_tickers:
        print("\n⚠️ Proceeding with APT models only...")
        common_tickers = apt_tickers
    else:
        exit()

print(f"\n✅ Found {len(common_tickers)} ticker(s): {sorted(common_tickers)}")


# ----------------------- COMPARISON FUNCTIONS ----------------------- #
def extract_model_metrics(df, model_prefix, window):
    """Extract alpha and R² for a specific model and window"""
    alpha_col = f"Alpha_{model_prefix}_{window}M"
    r2_col = f"R2_{model_prefix}_{window}M"

    if alpha_col in df.columns and r2_col in df.columns:
        return df[[alpha_col, r2_col]].dropna()
    return pd.DataFrame()


def compare_alphas(ticker, ff_data, apt_data, window='52'):
    """Compare alphas across all models for a given ticker and window"""
    results = {}

    # Fama-French models
    if not ff_data.empty:
        for model in ['CAPM', 'FF3', 'FF5']:
            alpha_col = f"Alpha_{model}_{window}M"
            r2_col = f"R2_{model}_{window}M"

            # Get beta column (market beta)
            if model == 'CAPM':
                beta_col = f"Beta_Mkt-RF_{window}M"  # Look for beta saved from rolling regression
            else:
                beta_col = f"Beta_Mkt-RF_{window}M"  # FF3 and FF5 also have market beta

            if alpha_col in ff_data.columns and r2_col in ff_data.columns:
                alpha_series = ff_data[alpha_col].dropna()
                r2_series = ff_data[r2_col].dropna()
                beta_series = ff_data[beta_col].dropna() if beta_col in ff_data.columns else pd.Series()

                results[model] = {
                    'alpha': alpha_series,
                    'r2': r2_series,
                    'beta': beta_series
                }

    # APT models - look for flattened column names
    if not apt_data.empty:
        for factor_type in ['PCA', 'Manual']:
            alpha_col = f"Alpha_{factor_type}_{window}M"
            r2_col = f"R_squared_{factor_type}_{window}M"

            # For APT, we'll take the first principal component or first factor as "beta"
            if factor_type == 'PCA':
                beta_col = f"Beta_PC1_{factor_type}_{window}M"
            else:
                beta_col = f"Beta_S&P500_{factor_type}_{window}M"  # Assuming S&P500 is first manual factor

            if alpha_col in apt_data.columns:
                alpha_series = apt_data[alpha_col].dropna()

                if len(alpha_series) > 0:
                    # Get R² if available
                    if r2_col in apt_data.columns:
                        r2_series = apt_data[r2_col].dropna()
                    else:
                        r2_col_alt = f"R2_{factor_type}_{window}M"
                        r2_series = apt_data[r2_col_alt].dropna() if r2_col_alt in apt_data.columns else pd.Series()

                    # Get beta if available
                    beta_series = apt_data[beta_col].dropna() if beta_col in apt_data.columns else pd.Series()

                    results[f"APT_{factor_type}"] = {
                        'alpha': alpha_series,
                        'r2': r2_series if len(r2_series) > 0 else alpha_series * 0,
                        'beta': beta_series
                    }

    return results


def calculate_summary_stats(model_results):
    """Calculate summary statistics for model comparison"""
    summary = []

    for model_name, data in model_results.items():
        alpha = data['alpha']
        r2 = data['r2']
        beta = data.get('beta', pd.Series())  # Get beta if available

        if len(alpha) > 0:
            # Annualize alpha (monthly to annual)
            mean_alpha_monthly = alpha.mean()
            mean_alpha_annual = mean_alpha_monthly * 12

            # Calculate statistical significance (t-stat approximation)
            std_alpha = alpha.std()
            n_obs = len(alpha)
            t_stat = (mean_alpha_monthly / (std_alpha / np.sqrt(n_obs))) if std_alpha > 0 else 0

            # Calculate Information Ratio
            info_ratio = (mean_alpha_monthly / std_alpha) if std_alpha > 0 else 0

            # Get beta if available
            mean_beta = beta.mean() if len(beta) > 0 and not beta.isna().all() else np.nan

            summary.append({
                'Model': model_name,
                'Mean_Alpha_Monthly': mean_alpha_monthly,
                'Mean_Alpha_Annual': mean_alpha_annual,
                'Std_Alpha': std_alpha,
                'Info_Ratio': info_ratio,
                'Mean_Beta': mean_beta,
                't_statistic': t_stat,
                'Mean_R2': r2.mean(),
                'Observations': n_obs,
                'Significant': '✓' if abs(t_stat) > 1.96 else '✗'  # 95% confidence
            })

    return pd.DataFrame(summary)


# ----------------------- ANALYSIS BY TICKER ----------------------- #
writer = pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl', mode='w')
all_comparisons = []

for ticker in sorted(common_tickers):
    print(f"\n{'=' * 70}")
    print(f"Analyzing: {ticker}")
    print('=' * 70)

    ff_data = ff_sheets.get(ticker, pd.DataFrame())
    apt_data = apt_sheets.get(ticker, pd.DataFrame())

    # Skip if both are empty
    if ff_data.empty and apt_data.empty:
        print(f"⚠️ No data available for {ticker}")
        continue

    # Debug: Show what columns we have
    if not apt_data.empty:
        print(f"\nAPT data columns available:")
        apt_cols = [col for col in apt_data.columns if 'Alpha' in str(col) or 'R_squared' in str(col)]
        for col in apt_cols[:10]:  # Show first 10 relevant columns
            print(f"  - {col}")
        if len(apt_cols) > 10:
            print(f"  ... and {len(apt_cols) - 10} more")

    # Create ticker-specific image directory
    ticker_dir = f"output/images/{ticker.lower()}"
    os.makedirs(ticker_dir, exist_ok=True)

    # Try both 36M and 52M windows
    for window in ['36', '52']:
        print(f"\n--- {window}-Month Window ---")

        model_results = compare_alphas(ticker, ff_data, apt_data, window)

        if not model_results:
            print(f"⚠️ No data for {window}-month window")
            continue

        print(f"Models found: {list(model_results.keys())}")

        # Calculate summary stats
        summary_stats = calculate_summary_stats(model_results)
        print("\n" + summary_stats.to_string(index=False))

        # Add ticker and window info
        summary_stats.insert(0, 'Window', f"{window}M")
        summary_stats.insert(0, 'Ticker', ticker)
        all_comparisons.append(summary_stats)

        # Create visualization - Alpha comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Alpha over time
        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))

        for (model_name, data), color in zip(model_results.items(), colors):
            alpha = data['alpha']
            if len(alpha) > 0:
                ax1.plot(alpha.index, alpha.values, label=model_name,
                         linewidth=2, alpha=0.8, color=color)

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.set_title(f"{ticker} - Alpha Comparison ({window}-Month Window)",
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Alpha (Monthly)")
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: R² comparison
        ax2 = axes[1]

        for (model_name, data), color in zip(model_results.items(), colors):
            r2 = data['r2']
            if len(r2) > 0:
                ax2.plot(r2.index, r2.values, label=model_name,
                         linewidth=2, alpha=0.8, color=color)

        ax2.set_title(f"{ticker} - Model Fit (R²) Comparison ({window}-Month Window)",
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("R-squared")
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        filename = f"{ticker_dir}/{ticker}_Comparison_{window}M.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")

        # Create box plot comparison for alpha distribution
        fig, ax = plt.subplots(figsize=(12, 6))

        alpha_data = []
        labels = []
        for model_name, data in model_results.items():
            alpha = data['alpha'].dropna()
            if len(alpha) > 0:
                alpha_data.append(alpha.values)
                labels.append(model_name)

        if alpha_data:
            bp = ax.boxplot(alpha_data, labels=labels, patch_artist=True)

            # Color boxes
            colors = plt.cm.tab10(np.linspace(0, 1, len(alpha_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_title(f"{ticker} - Alpha Distribution Comparison ({window}M)",
                         fontsize=14, fontweight='bold')
            ax.set_ylabel("Alpha (Monthly)")
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            filename = f"{ticker_dir}/{ticker}_Alpha_Distribution_{window}M.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {filename}")

# ----------------------- AGGREGATE ANALYSIS ----------------------- #
if all_comparisons:
    print("\n" + "=" * 70)
    print("CREATING AGGREGATE COMPARISON")
    print("=" * 70)

    # Combine all comparisons
    full_comparison = pd.concat(all_comparisons, ignore_index=True)
    full_comparison.to_excel(writer, sheet_name="All_Comparisons", index=False)

    # Best model by ticker (highest mean alpha)
    best_models = full_comparison.loc[full_comparison.groupby(['Ticker', 'Window'])['Mean_Alpha_Annual'].idxmax()]
    best_models = best_models[['Ticker', 'Window', 'Model', 'Mean_Alpha_Annual', 'Mean_R2', 'Significant']]
    best_models.to_excel(writer, sheet_name="Best_Models", index=False)

    print("\nBest Model by Ticker:")
    print(best_models.to_string(index=False))

    # Average performance by model type across all tickers
    avg_by_model = full_comparison.groupby('Model').agg({
        'Mean_Alpha_Annual': 'mean',
        'Std_Alpha': 'mean',
        'Mean_R2': 'mean',
        't_statistic': 'mean',
        'Observations': 'sum'
    }).round(4)
    avg_by_model.to_excel(writer, sheet_name="Avg_by_Model")

    print("\n" + "=" * 70)
    print("Average Performance by Model Type:")
    print("=" * 70)
    print(avg_by_model.to_string())

    # Create summary heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Heatmap 1: Mean Annual Alpha by Ticker and Model
    pivot_alpha = full_comparison.pivot_table(
        values='Mean_Alpha_Annual',
        index='Ticker',
        columns='Model',
        aggfunc='mean'
    )

    sns.heatmap(pivot_alpha, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=axes[0], cbar_kws={'label': 'Annual Alpha'})
    axes[0].set_title('Mean Annual Alpha by Model', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Ticker')

    # Heatmap 2: Mean R² by Ticker and Model
    pivot_r2 = full_comparison.pivot_table(
        values='Mean_R2',
        index='Ticker',
        columns='Model',
        aggfunc='mean'
    )

    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'R²'})
    axes[1].set_title('Mean R² by Model', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Ticker')

    plt.tight_layout()
    filename = "output/images/Model_Performance_Heatmap.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {filename}")

    # Model ranking chart
    fig, ax = plt.subplots(figsize=(12, 6))

    model_ranking = avg_by_model.sort_values('Mean_Alpha_Annual', ascending=True)
    colors = ['green' if x > 0 else 'red' for x in model_ranking['Mean_Alpha_Annual']]

    ax.barh(model_ranking.index, model_ranking['Mean_Alpha_Annual'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Mean Annual Alpha', fontsize=12)
    ax.set_title('Model Performance Ranking (Across All Tickers)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add R² as text annotations
    for idx, (model, row) in enumerate(model_ranking.iterrows()):
        ax.text(row['Mean_Alpha_Annual'], idx,
                f"  R²={row['Mean_R2']:.3f}",
                va='center', fontsize=9)

    plt.tight_layout()
    filename = "output/images/Model_Ranking.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")

writer.close()

print("\n" + "=" * 70)
print("✅ COMPARISON ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nResults saved to: {OUTPUT_FILE}")
print(f"Per-ticker charts saved to: ../output/images/<ticker>/")
print(f"Aggregate charts saved to: ../output/images/")
print("\n" + "=" * 70)