import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import openpyxl

# ----------------------- DIRECTORY SETUP ----------------------- #
def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("output/sheets", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)

ensure_directories()

# ----------------------- SETTINGS ----------------------- #
ROLLING_WINDOWS = [36, 52]  # in months
FACTOR_FILE_FF3 = "output/sheets/F-F_Research_Data_Factors.xlsx"
FACTOR_FILE_FF5 = "output/sheets/F-F_Research_Data_5_Factors_2x3.xlsx"
EXPORT_EXCEL = "output/sheets/Risk_Model_Output.xlsx"

# ----------------------- INPUT ----------------------- #
n = int(input("Enter the number of tickers: "))
tickers = [input(f"Enter ticker {i + 1}: ").upper() for i in range(n)]


# ----------------------- FETCH PRICE DATA ----------------------- #
def get_monthly_log_returns(tickers, min_months=36):
    from datetime import datetime, timedelta
    end_date = datetime.today()
    attempt_years = [20, 15, 10, 7, 5, 3]

    all_returns = {}

    for ticker in tickers:
        for years in attempt_years:
            start_date = end_date - timedelta(days=365 * years)
            try:
                df = yf.download(ticker, start=start_date, end=end_date, interval='1mo', auto_adjust=True,
                                 progress=False)
                if df.empty:
                    continue

                df = df[['Close']].dropna()
                df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
                df = df[['LogReturn']].dropna()

                if len(df) >= min_months:
                    if len(df) > 119:
                        df = df.iloc[-119:]
                        print(f"✅ {ticker}: Loaded 119 months of data (capped).")
                    else:
                        print(f"✅ {ticker}: Loaded {len(df)} months of data (last {years} years).")

                    all_returns[ticker] = df['LogReturn']
                    break
                else:
                    print(f"⚠️ {ticker}: Only {len(df)} months from last {years} years. Trying shorter window...")
            except Exception as e:
                print(f"❌ Error downloading {ticker} for {years} years: {e}")

        if ticker not in all_returns:
            print(f"❌ {ticker}: Failed to download sufficient data.")

    if all_returns:
        combined = pd.concat(all_returns.values(), axis=1)
        combined.columns = list(all_returns.keys())
        return combined
    else:
        return pd.DataFrame()


returns = get_monthly_log_returns(tickers)
returns.index = returns.index.to_period("M").to_timestamp("M")


# ----------------------- LOAD FACTORS ----------------------- #
def load_ff_factors(file, expected_cols):
    try:
        if file.endswith(".xlsx") or file.endswith(".xls"):
            try:
                xl = pd.ExcelFile(file)
                if not xl.sheet_names:
                    raise ValueError(f"No sheets found in {file}")
                df = xl.parse(xl.sheet_names[0])
            except Exception as e:
                print(f"⚠️ Excel read failed: {e} — trying CSV fallback.")
                df = pd.read_csv(file, encoding="latin1")
        else:
            df = pd.read_csv(file, encoding="latin1")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load factor file {file}: {e}")

    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df = df[~df["Date"].astype(str).str.contains(r"\s")]
    df = df[df["Date"].astype(str).str.match(r"^\d{6}$")]
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df.set_index("Date", inplace=True)
    df = df[expected_cols]
    df = df.apply(pd.to_numeric, errors="coerce") / 100
    return df.dropna()


ff3 = load_ff_factors(FACTOR_FILE_FF3, ["Mkt-RF", "SMB", "HML", "RF"])
ff5 = load_ff_factors(FACTOR_FILE_FF5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

print(f"FF3 date range: {ff3.index.min()} to {ff3.index.max()} ({len(ff3)} months)")
print(f"FF5 date range: {ff5.index.min()} to {ff5.index.max()} ({len(ff5)} months)")


# ----------------------- ROLLING REGRESSION FUNCTION ----------------------- #
def rolling_regress(y, X, window):
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
            alpha = float(model.params.get("const", np.nan))
            r_squared = model.rsquared
            t_stat = np.nan
            if "const" in model.params.index and "const" in model.tvalues.index:
                t_stat = float(model.tvalues["const"])
            resid_std_err = np.std(model.resid, ddof=model.df_model + 1)
            result = {
                "Alpha": alpha,
                "R_squared": r_squared,
                "t_stat_alpha": t_stat,
                "Residual_SE": resid_std_err
            }

            for factor in X_win.columns:
                result[f"Beta_{factor}"] = float(model.params.get(factor, np.nan))

            results.append(pd.Series(result, name=y_win.index[-1]))
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# ----------------------- MAIN LOOP ----------------------- #
import zipfile

if os.path.exists(EXPORT_EXCEL):
    try:
        with zipfile.ZipFile(EXPORT_EXCEL, 'r') as zip_ref:
            pass
    except zipfile.BadZipFile:
        print(f"⚠️ Existing Excel file '{EXPORT_EXCEL}' is corrupted. Replacing it.")
        os.remove(EXPORT_EXCEL)

writer = pd.ExcelWriter(EXPORT_EXCEL, engine='openpyxl', mode='w')

summary_rows = []

for ticker in tickers:
    print(f"\n===== Processing {ticker} =====")

    y = returns[ticker].dropna()
    if y.empty:
        print(f"❌ {ticker}: No return data. Skipping.")
        continue

    aligned_ff5 = ff5.reindex(y.index).dropna()
    aligned_ff3 = ff3.reindex(y.index).dropna()

    shared_dates = y.index.intersection(aligned_ff5.index)
    if shared_dates.empty:
        print(f"❌ {ticker}: No overlapping dates with factors. Skipping.")
        continue

    y_model = y.loc[shared_dates]
    X_ff5 = aligned_ff5.loc[shared_dates][["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    X_ff3 = aligned_ff3.loc[shared_dates][["Mkt-RF", "SMB", "HML"]]

    rf_for_ticker = ff5.loc[shared_dates]["RF"]
    y_excess = y_model.sub(rf_for_ticker, axis=0)

    n_months = len(y_excess)
    print(f"{ticker}: {n_months} months of data available")

    # Determine which windows to use
    available_windows = [w for w in ROLLING_WINDOWS if n_months >= w]

    if not available_windows:
        print(f"❌ {ticker}: Not enough data (only {n_months} months). Need at least {min(ROLLING_WINDOWS)}. Skipping.")
        continue

    print(f"Using windows: {available_windows}")

    X_capm = X_ff3[["Mkt-RF"]]

    # Store results for all windows
    all_window_results = {}

    for window in available_windows:
        print(f"\n--- Running {window}-month window for {ticker} ---")

        rolling_capm = rolling_regress(y_excess, X_capm, window)
        rolling_ff3 = rolling_regress(y_excess, X_ff3, window)
        rolling_ff5 = rolling_regress(y_excess, X_ff5, window)

        print(f"Rolling CAPM produced {len(rolling_capm)} results")
        print(f"Rolling FF3 produced {len(rolling_ff3)} results")
        print(f"Rolling FF5 produced {len(rolling_ff5)} results")

        # Merge rolling results
        combined_df = pd.DataFrame(index=rolling_capm.index)

        if not rolling_capm.empty:
            combined_df[f"Alpha_CAPM_{window}M"] = rolling_capm["Alpha"]
            combined_df[f"R2_CAPM_{window}M"] = rolling_capm["R_squared"]
            combined_df[f"t_stat_CAPM_{window}M"] = rolling_capm["t_stat_alpha"]
            combined_df[f"Residual_SE_CAPM_{window}M"] = rolling_capm["Residual_SE"]
            # ADD BETA
            combined_df[f"Beta_Mkt-RF_{window}M"] = rolling_capm[
                "Beta_Mkt-RF"] if "Beta_Mkt-RF" in rolling_capm.columns else np.nan

        if not rolling_ff3.empty:
            combined_df[f"Alpha_FF3_{window}M"] = rolling_ff3["Alpha"]
            combined_df[f"R2_FF3_{window}M"] = rolling_ff3["R_squared"]
            combined_df[f"t_stat_FF3_{window}M"] = rolling_ff3["t_stat_alpha"]
            combined_df[f"Residual_SE_FF3_{window}M"] = rolling_ff3["Residual_SE"]
            # ADD BETA (same market beta column name for consistency)
            combined_df[f"Beta_Mkt-RF_FF3_{window}M"] = rolling_ff3[
                "Beta_Mkt-RF"] if "Beta_Mkt-RF" in rolling_ff3.columns else np.nan

        if not rolling_ff5.empty:
            combined_df[f"Alpha_FF5_{window}M"] = rolling_ff5["Alpha"]
            combined_df[f"R2_FF5_{window}M"] = rolling_ff5["R_squared"]
            combined_df[f"t_stat_FF5_{window}M"] = rolling_ff5["t_stat_alpha"]
            combined_df[f"Residual_SE_FF5_{window}M"] = rolling_ff5["Residual_SE"]
            # ADD BETA
            combined_df[f"Beta_Mkt-RF_FF5_{window}M"] = rolling_ff5[
                "Beta_Mkt-RF"] if "Beta_Mkt-RF" in rolling_ff5.columns else np.nan

        all_window_results[window] = combined_df.dropna(how='all')

    # Combine all windows into one DataFrame for Excel export
    if all_window_results:
        # Merge all window results on index (dates may differ)
        combined_all_windows = pd.concat(all_window_results.values(), axis=1)
        combined_all_windows.index.name = "Date"
        combined_all_windows.to_excel(writer, sheet_name=ticker[:31])

        # Add summary statistics for each window
        for window in available_windows:
            window_df = all_window_results[window]

            # Get latest beta values (from the most recent rolling window)
            capm_beta = window_df[f"Beta_Mkt-RF_{window}M"].iloc[
                -1] if f"Beta_Mkt-RF_{window}M" in window_df.columns else np.nan
            ff3_beta = window_df[f"Beta_Mkt-RF_FF3_{window}M"].iloc[
                -1] if f"Beta_Mkt-RF_FF3_{window}M" in window_df.columns else np.nan
            ff5_beta = window_df[f"Beta_Mkt-RF_FF5_{window}M"].iloc[
                -1] if f"Beta_Mkt-RF_FF5_{window}M" in window_df.columns else np.nan

            # Calculate Information Ratios
            capm_ir = (window_df[f"Alpha_CAPM_{window}M"].mean() / window_df[f"Alpha_CAPM_{window}M"].std()) if \
            window_df[f"Alpha_CAPM_{window}M"].std() > 0 else np.nan
            ff3_ir = (window_df[f"Alpha_FF3_{window}M"].mean() / window_df[f"Alpha_FF3_{window}M"].std()) if window_df[
                                                                                                                 f"Alpha_FF3_{window}M"].std() > 0 else np.nan
            ff5_ir = (window_df[f"Alpha_FF5_{window}M"].mean() / window_df[f"Alpha_FF5_{window}M"].std()) if window_df[
                                                                                                                 f"Alpha_FF5_{window}M"].std() > 0 else np.nan

            summary_rows.append({
                "Ticker": ticker,
                "Window": f"{window}M",
                "Mean_Alpha_CAPM": window_df[f"Alpha_CAPM_{window}M"].mean(),
                "Mean_Alpha_FF3": window_df[f"Alpha_FF3_{window}M"].mean(),
                "Mean_Alpha_FF5": window_df[f"Alpha_FF5_{window}M"].mean(),
                "Beta_CAPM": capm_beta,
                "Beta_FF3": ff3_beta,
                "Beta_FF5": ff5_beta,
                "Info_Ratio_CAPM": capm_ir,
                "o_FF3": ff3_ir,
                "Info_Ratio_FF5": ff5_ir,
                "Mean_R2_CAPM": window_df[f"R2_CAPM_{window}M"].mean(),
                "Mean_R2_FF3": window_df[f"R2_FF3_{window}M"].mean(),
                "Mean_R2_FF5": window_df[f"R2_FF5_{window}M"].mean(),
                "Mean_t_stat_CAPM": window_df[f"t_stat_CAPM_{window}M"].mean(),
                "Mean_t_stat_FF3": window_df[f"t_stat_FF3_{window}M"].mean(),
                "Mean_t_stat_FF5": window_df[f"t_stat_FF5_{window}M"].mean(),
            })

        # Create ticker-specific image directory
        ticker_dir = f"output/images/{ticker.lower()}"
        os.makedirs(ticker_dir, exist_ok=True)

        # Plot both windows on same chart if we have enough data
        min_points_to_plot = 10

        plottable_windows = [w for w in available_windows if len(all_window_results[w]) >= min_points_to_plot]

        if plottable_windows:
            try:
                fig, axes = plt.subplots(len(plottable_windows), 1, figsize=(12, 6 * len(plottable_windows)))

                # Handle case of single window (axes is not a list)
                if len(plottable_windows) == 1:
                    axes = [axes]

                for idx, window in enumerate(plottable_windows):
                    window_df = all_window_results[window]
                    ax = axes[idx]

                    ax.plot(window_df.index, window_df[f"Alpha_CAPM_{window}M"],
                            label=f"Alpha CAPM ({window}M)", marker='o', markersize=3)
                    ax.plot(window_df.index, window_df[f"Alpha_FF3_{window}M"],
                            label=f"Alpha FF3 ({window}M)", marker='s', markersize=3)
                    ax.plot(window_df.index, window_df[f"Alpha_FF5_{window}M"],
                            label=f"Alpha FF5 ({window}M)", marker='^', markersize=3)

                    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                    ax.set_title(f"{ticker} Rolling Alpha - {window}-Month Window")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Alpha (Monthly)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                pngfile = f"{ticker_dir}/{ticker}_Rolling_Alpha_All_Windows.png"
                plt.savefig(pngfile, bbox_inches="tight", dpi=150)
                plt.close()
                print(f"✅ Saved plot: {pngfile}")
            except Exception as e:
                print(f"⚠️ Could not save alpha plot for {ticker}: {e}")
        else:
            print(f"⚠️ Not enough data points to plot for {ticker} (need at least {min_points_to_plot})")

# Write consolidated summary sheet
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.round(6)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
else:
    print("⚠️ No summary rows to write.")

writer.close()
print("\n✅ All results saved to Excel and charts exported.")