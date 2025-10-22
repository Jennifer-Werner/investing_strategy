import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import zipfile

# ----------------------- DIRECTORY SETUP ----------------------- #
def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("output/sheets", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)

ensure_directories()

# ----------------------- SETTINGS ----------------------- #
ROLLING_WINDOWS = [36, 52]  # in months1
EXPORT_EXCEL = "output/sheets/APT_Model_Output.xlsx"

# Macro/Market indicators to download (tickers for yfinance)
MACRO_INDICATORS = {
    '^GSPC': 'S&P500',  # Broad market
    '^VIX': 'VIX',  # Volatility
    '^TNX': '10Y_Treasury',  # Interest rates
    'DX-Y.NYB': 'USD_Index',  # Dollar strength
    'CL=F': 'Crude_Oil',  # Oil prices
    'GC=F': 'Gold',  # Gold (inflation hedge)
    '^IXIC': 'NASDAQ',  # Tech sector proxy
    '^DJI': 'DOW',  # Industrial proxy
    'XLF': 'Financials_Sector',  # Financial sector
    'XLE': 'Energy_Sector',  # Energy sector
    'XLK': 'Tech_Sector',  # Technology sector
    'XLV': 'Healthcare_Sector',  # Healthcare sector
    'XLI': 'Industrial_Sector',  # Industrial sector
    'XLP': 'Consumer_Staples',  # Consumer staples
    'XLY': 'Consumer_Discret',  # Consumer discretionary
}

# Number of PCA components to extract
N_PCA_COMPONENTS = 5

# ----------------------- INPUT ----------------------- #
print("=== APT Model Analysis ===\n")
print("Choose factor selection method:")
print("1. PCA-based factors (automatic extraction from market data)")
print("2. Manual macro factors (predefined indicators)")
print("3. Both (PCA + Manual)")
method = input("Enter choice (1/2/3): ").strip()

n = int(input("\nEnter the number of stock tickers to analyze: "))
tickers = [input(f"Enter ticker {i + 1}: ").upper() for i in range(n)]


# ----------------------- FETCH DATA ----------------------- #
def get_monthly_log_returns(tickers, min_months=36, max_months=119):
    """Download monthly log returns for given tickers"""
    from datetime import datetime, timedelta
    end_date = datetime.today()
    attempt_years = [20, 15, 10, 7, 5, 3]

    all_returns = {}

    for ticker in tickers:
        for years in attempt_years:
            start_date = end_date - timedelta(days=365 * years)
            try:
                df = yf.download(ticker, start=start_date, end=end_date, interval='1mo',
                                 auto_adjust=True, progress=False)
                if df.empty:
                    continue

                df = df[['Close']].dropna()
                df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
                df = df[['LogReturn']].dropna()

                if len(df) >= min_months:
                    if len(df) > max_months:
                        df = df.iloc[-max_months:]
                        print(f"✅ {ticker}: Loaded {max_months} months (capped).")
                    else:
                        print(f"✅ {ticker}: Loaded {len(df)} months.")

                    all_returns[ticker] = df['LogReturn']
                    break
                else:
                    print(f"⚠️ {ticker}: Only {len(df)} months. Trying shorter period...")
            except Exception as e:
                print(f"❌ Error downloading {ticker}: {e}")

        if ticker not in all_returns:
            print(f"❌ {ticker}: Failed to download sufficient data.")

    if all_returns:
        combined = pd.concat(all_returns.values(), axis=1)
        combined.columns = list(all_returns.keys())
        return combined
    else:
        return pd.DataFrame()


def get_macro_factors(indicator_dict, min_months=36):
    """Download macro/market indicators for APT analysis"""
    from datetime import datetime, timedelta
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 12)  # Try 12 years

    all_factors = {}

    print("\n=== Downloading Macro/Market Indicators ===")
    for ticker, name in indicator_dict.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1mo',
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"⚠️ {name}: No data available")
                continue

            df = df[['Close']].dropna()
            df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
            df = df[['LogReturn']].dropna()

            if len(df) >= min_months:
                all_factors[name] = df['LogReturn']
                print(f"✅ {name}: Loaded {len(df)} months")
            else:
                print(f"⚠️ {name}: Only {len(df)} months (skipping)")
        except Exception as e:
            print(f"❌ {name}: Download failed - {e}")

    if all_factors:
        combined = pd.concat(all_factors.values(), axis=1)
        combined.columns = list(all_factors.keys())
        return combined
    else:
        return pd.DataFrame()


# ----------------------- PCA FACTOR EXTRACTION ----------------------- #
def extract_pca_factors(factor_data, n_components=5):
    """Extract principal components from factor data"""
    print(f"\n=== Extracting {n_components} PCA Components ===")

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(factor_data.fillna(0))

    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_data)

    # Create DataFrame
    pc_names = [f'PC{i + 1}' for i in range(n_components)]
    pca_factors = pd.DataFrame(components, index=factor_data.index, columns=pc_names)

    # Print explained variance
    print("\nExplained Variance by Component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {var * 100:.2f}%")
    print(f"  Total: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

    # Print component loadings (top contributors)
    print("\nTop Factor Loadings per Component:")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=pc_names,
        index=factor_data.columns
    )

    for pc in pc_names:
        print(f"\n{pc}:")
        top_loadings = loadings[pc].abs().sort_values(ascending=False).head(3)
        for factor, loading in top_loadings.items():
            actual_loading = loadings.loc[factor, pc]
            print(f"  {factor}: {actual_loading:+.3f}")

    return pca_factors, pca, loadings


# ----------------------- MANUAL FACTOR SELECTION ----------------------- #
def select_manual_factors(factor_data, max_factors=8):
    """Allow user to select specific factors or use predefined set"""
    print("\n=== Manual Factor Selection ===")
    print("Available factors:")
    for i, factor in enumerate(factor_data.columns, 1):
        print(f"  {i}. {factor}")

    print("\nOptions:")
    print("  - Enter numbers separated by commas (e.g., 1,2,5,7)")
    print("  - Press ENTER to use default set (S&P500, VIX, 10Y_Treasury, Crude_Oil)")

    selection = input("Your choice: ").strip()

    if not selection:
        # Default factors
        default = ['S&P500', 'VIX', '10Y_Treasury', 'Crude_Oil']
        selected = [f for f in default if f in factor_data.columns]
        print(f"Using default factors: {selected}")
    else:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        selected = [factor_data.columns[i] for i in indices if 0 <= i < len(factor_data.columns)]
        print(f"Selected factors: {selected}")

    return factor_data[selected]


# ----------------------- ROLLING REGRESSION ----------------------- #
def rolling_regress(y, X, window):
    """Run rolling window regression"""
    results = []
    shared_dates = X.index.intersection(y.index)
    if len(shared_dates) < window:
        return pd.DataFrame()

    X = X.loc[shared_dates]
    y = y.loc[shared_dates]

    for i in range(len(y) - window + 1):
        y_win = y.iloc[i:i + window]
        X_win = X.iloc[i:i + window]

        if len(y_win) < window or X_win.isnull().any().any():
            continue

        try:
            model = sm.OLS(y_win, sm.add_constant(X_win)).fit()
            result = {
                "Alpha": float(model.params.get("const", np.nan)),
                "R_squared": model.rsquared,
                "t_stat_alpha": float(model.tvalues.get("const", np.nan)),
                "Adj_R_squared": model.rsquared_adj,
            }
            # Add factor betas
            for factor in X_win.columns:
                result[f"Beta_{factor}"] = float(model.params.get(factor, np.nan))

            results.append(pd.Series(result, name=y_win.index[-1]))
        except Exception:
            continue

    return pd.DataFrame(results) if results else pd.DataFrame()


# ----------------------- MAIN ANALYSIS ----------------------- #
print("\n=== Downloading Stock Data ===")
stock_returns = get_monthly_log_returns(tickers)
stock_returns.index = stock_returns.index.to_period("M").to_timestamp("M")

if stock_returns.empty:
    print("❌ No stock data loaded. Exiting.")
    exit()

# Download macro/market factors
macro_data = get_macro_factors(MACRO_INDICATORS)
macro_data.index = macro_data.index.to_period("M").to_timestamp("M")

if macro_data.empty:
    print("❌ No macro factor data loaded. Exiting.")
    exit()

# Prepare factor sets based on method choice
factor_sets = {}

if method in ['1', '3']:
    # PCA factors
    pca_factors, pca_model, loadings = extract_pca_factors(macro_data, N_PCA_COMPONENTS)
    factor_sets['PCA'] = pca_factors

if method in ['2', '3']:
    # Manual factors
    manual_factors = select_manual_factors(macro_data)
    factor_sets['Manual'] = manual_factors

# ----------------------- ROLLING REGRESSIONS ----------------------- #
if os.path.exists(EXPORT_EXCEL):
    try:
        with zipfile.ZipFile(EXPORT_EXCEL, 'r') as zip_ref:
            pass
    except zipfile.BadZipFile:
        os.remove(EXPORT_EXCEL)

writer = pd.ExcelWriter(EXPORT_EXCEL, engine='openpyxl', mode='w')
summary_rows = []

for ticker in tickers:
    print(f"\n{'=' * 60}")
    print(f"Processing {ticker}")
    print('=' * 60)

    y = stock_returns[ticker].dropna()
    if y.empty:
        print(f"❌ {ticker}: No return data. Skipping.")
        continue

    n_months = len(y)
    print(f"Available data: {n_months} months ({y.index.min()} to {y.index.max()})")

    # Determine available windows
    available_windows = [w for w in ROLLING_WINDOWS if n_months >= w]
    if not available_windows:
        print(f"❌ Not enough data. Need at least {min(ROLLING_WINDOWS)} months.")
        continue

    print(f"Using windows: {available_windows}")

    # Run for each factor set and window
    all_results = {}

    for factor_name, factors in factor_sets.items():
        print(f"\n--- {factor_name} Factors ---")

        for window in available_windows:
            print(f"\nRunning {window}-month rolling regression...")

            rolling_results = rolling_regress(y, factors, window)

            if rolling_results.empty:
                print(f"⚠️ No results for {window}-month window")
                continue

            print(f"✅ Generated {len(rolling_results)} rolling estimates")

            # Store with unique key
            key = f"{factor_name}_{window}M"
            all_results[key] = rolling_results

            # Add to summary
            # Calculate betas from the rolling results (use last window as representative)
            factor_betas = {}
            for factor in factors.columns:
                beta_col = f"Beta_{factor}"
                if beta_col in rolling_results.columns:
                    factor_betas[f"Beta_{factor}"] = rolling_results[beta_col].iloc[-1] if len(
                        rolling_results) > 0 else np.nan
                else:
                    factor_betas[f"Beta_{factor}"] = np.nan

            # Calculate Information Ratio
            info_ratio = (rolling_results["Alpha"].mean() / rolling_results["Alpha"].std()) if rolling_results[
                                                                                                   "Alpha"].std() > 0 else np.nan

            summary_row = {
                "Ticker": ticker,
                "Factor_Set": factor_name,
                "Window": f"{window}M",
                "Mean_Alpha": rolling_results["Alpha"].mean(),
                "Std_Alpha": rolling_results["Alpha"].std(),
                "Info_Ratio": info_ratio,
                "Mean_R2": rolling_results["R_squared"].mean(),
                "Mean_Adj_R2": rolling_results["Adj_R_squared"].mean(),
                "t_stat_Alpha": rolling_results["t_stat_alpha"].mean(),
            }

            # Add factor betas to summary
            summary_row.update(factor_betas)

            summary_rows.append(summary_row)

# Save results to Excel with flattened columns
    if all_results:
        # Flatten the multi-level structure for easy reading
        flattened_dfs = []
        for key, df in all_results.items():
            # Rename columns to include the key (e.g., "Alpha" → "Alpha_PCA_36M")
            df_renamed = df.copy()
            df_renamed.columns = [f"{col}_{key}" for col in df.columns]
            flattened_dfs.append(df_renamed)

        # Combine all into one DataFrame
        combined = pd.concat(flattened_dfs, axis=1)
        combined.index.name = "Date"
        combined.to_excel(writer, sheet_name=ticker[:31])
        print(f"\n✅ Saved {ticker} results to Excel")

    # Create ticker-specific image directory
    ticker_dir = f"output/images/{ticker.lower()}"
    os.makedirs(ticker_dir, exist_ok=True)

    # Create plots
    print("\n--- Creating Visualizations ---")

    for factor_name in factor_sets.keys():
        factor_results = {k: v for k, v in all_results.items() if k.startswith(factor_name)}

        if not factor_results:
            continue

        n_plots = len(factor_results)
        if n_plots == 0:
            continue

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for idx, (key, results) in enumerate(factor_results.items()):
            ax = axes[idx]
            window = key.split('_')[-1]

            # Plot alpha
            alpha_col = f"Alpha_{key}"
            ax.plot(results.index, results["Alpha"],
                    label=f"Alpha ({window})", color='darkblue', linewidth=2)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

            # Shade confidence intervals (±2 std errors)
            alpha_std = results["Alpha"].rolling(12).std()
            ax.fill_between(results.index,
                            results["Alpha"] - 2 * alpha_std,
                            results["Alpha"] + 2 * alpha_std,
                            alpha=0.2, color='blue', label='±2 Std Dev')

            ax.set_title(f"{ticker} - APT Alpha ({factor_name} Factors, {window})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Alpha (Monthly Return)")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{ticker_dir}/{ticker}_APT_Alpha_{factor_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")

        # Plot R-squared evolution
        fig, ax = plt.subplots(figsize=(12, 5))
        for key, results in factor_results.items():
            window = key.split('_')[-1]
            ax.plot(results.index, results["R_squared"],
                    label=f"R² ({window})", linewidth=2, marker='o', markersize=3)

        ax.set_title(f"{ticker} - Model R² over Time ({factor_name} Factors)")
        ax.set_xlabel("Date")
        ax.set_ylabel("R-squared")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        filename = f"{ticker_dir}/{ticker}_APT_Rsquared_{factor_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")

# THIS SHOULD BE OUTSIDE THE FOR TICKER LOOP (UN-INDENT 4 SPACES)
# Save summary
if summary_rows:
    summary_df = pd.DataFrame(summary_rows).round(6)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    print("\n✅ Summary statistics saved")

writer.close()

print("\n" + "=" * 60)
print("✅ APT ANALYSIS COMPLETE")
print("=" * 60)
print(f"Results saved to: {EXPORT_EXCEL}")
print("Charts saved to ticker-specific directories in output/images/")
print("\nSummary Statistics:")
if summary_rows:
    summary_display = pd.DataFrame(summary_rows)
    print(summary_display.to_string(index=False))