import re
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import zscore
from tqdm import tqdm
import time
import random

# Header normalization logic (robust, abbreviation-preserving)
def normalize_headers(df):
    """Normalize DataFrame column headers with regex-safe, abbreviation-preserving logic."""
    abbreviations = {"EBIT", "EBITDA", "FCF", "ROE", "ROA", "ROIC", "PB", "EPS", "EV", "COGS", "DCF", "WACC", "PE"}
    def normalize_col(col):
        col_original = str(col)
        col_clean = re.sub(r'[_\-]+', ' ', col_original).strip()
        parts = col_clean.split()
        new_parts = []
        for part in parts:
            cleaned = re.sub(r'[^A-Za-z]', '', part).upper()
            if cleaned in abbreviations:
                new_parts.append(part.upper())
            else:
                new_parts.append(part.title())
        new_col = " ".join(new_parts)
        for abbr in abbreviations:
            new_col = re.sub(rf'\b{abbr}\b', abbr, new_col, flags=re.IGNORECASE)
        return new_col
    old_cols = list(df.columns)
    df.columns = [normalize_col(c) for c in df.columns]
    print("\n[Header Normalization]")
    for old, new in zip(old_cols, df.columns):
        print(f"  {old} → {new}")
    return df

#
# ========== CONFIGURATION ==========
INPUT_FILE = "../output/sp1500_esg_screened.xlsx"
OUTPUT_FILE = "../output/sp1500_factor_quality_ranked.xlsx"
WEIGHTS = {'value': 0.4, 'profitability': 0.3, 'quality': 0.3}
tqdm.pandas()

# Save interval for partial results
SAVE_INTERVAL = 100

# ========== LOAD ESG-SCREENED UNIVERSE ==========
df = pd.read_excel(INPUT_FILE)
old_cols = list(df.columns)
df = normalize_headers(df)
df['Symbol'] = df['Symbol'].str.upper()

print(f"✅ Loaded {len(df)} tickers for factor evaluation.")

# Aliases for row names across Yahoo variants (case-insensitive)
FIN_ALIASES = {
    'revenue': ['Total Revenue', 'Revenue'],
    'ebitda': ['Ebitda', 'EBITDA'],
    'ebit': ['Ebit', 'EBIT'],
    'operating_income': ['Operating Income', 'Operating Income or Loss'],
    'cogs': ['Cost Of Revenue', 'Cost of Revenue', 'Cost of Revenue'],
    'net_income': ['Net Income', 'Net Income To Common', 'Net Income Applicable To Common Shares', 'Net Income Common Stockholders']
}
BS_ALIASES = {
    'equity': ['Total Stockholder Equity', 'Total Equity Gross Minority Interest', 'Total Equity'],
    'total_assets': ['Total Assets'],
    'lt_debt': ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'],
    'st_debt': ['Short Long Term Debt', 'Current Portion Of Long Term Debt', 'Short Term Debt'],
    'current_assets': ['Total Current Assets'],
    'current_liabilities': ['Total Current Liabilities'],
    'cash': ['Cash', 'Cash And Cash Equivalents', 'Cash And Cash Equivalents, Including Restricted Cash']
}
CF_ALIASES = {
    'ocf': ['Total Cash From Operating Activities', 'Net Cash Provided By Operating Activities'],
    'capex': ['Capital Expenditures', 'Investments In Property, Plant, And Equipment']
}

def first_nonnull(*vals):
    for v in vals:
        if pd.notna(v):
            return v
    return np.nan

def extract_row(df, alias_list, col):
    if df is None or df.empty or col is None:
        return np.nan
    # Exact match first
    for name in alias_list:
        if name in df.index:
            try:
                return df.loc[name, col]
            except Exception:
                pass
    # Case-insensitive fallback
    lower_index = {str(idx).lower(): idx for idx in df.index}
    for name in alias_list:
        key = str(name).lower()
        if key in lower_index:
            try:
                return df.loc[lower_index[key], col]
            except Exception:
                pass
    return np.nan

def extract_with_fallback(annual_df, quarterly_df, aliases):
    # Prefer latest annual column; else fallback to latest quarterly column
    annual_col = annual_df.columns[0] if (annual_df is not None and not annual_df.empty) else None
    quarterly_col = quarterly_df.columns[0] if (quarterly_df is not None and not quarterly_df.empty) else None
    val = extract_row(annual_df, aliases, annual_col)
    if pd.isna(val):
        val = extract_row(quarterly_df, aliases, quarterly_col)
    return val

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)

        # Pull statements (annual + quarterly as fallback)
        fin_a = stock.financials
        fin_q = stock.quarterly_financials
        bs_a = stock.balance_sheet
        bs_q = stock.quarterly_balance_sheet
        cf_a = stock.cashflow
        cf_q = stock.quarterly_cashflow

        # fast_info: fewer 404s than .info
        price = np.nan
        market_cap = np.nan
        shares_out = np.nan
        try:
            fi = stock.fast_info
            price = fi.get('last_price', np.nan)
            market_cap = fi.get('market_cap', np.nan)
            shares_out = fi.get('shares', np.nan) or fi.get('shares_outstanding', np.nan)
        except Exception:
            pass

        # Fallback to .info for EPS and anything missing
        eps = np.nan
        try:
            info = stock.info
            eps = info.get('trailingEps', np.nan)
            if pd.isna(price):
                price = info.get('currentPrice', np.nan)
            if pd.isna(market_cap):
                market_cap = info.get('marketCap', np.nan)
            if pd.isna(shares_out):
                shares_out = info.get('sharesOutstanding', np.nan)
        except Exception:
            pass

        # Income statement
        revenue = extract_with_fallback(fin_a, fin_q, FIN_ALIASES['revenue'])
        ebitda = extract_with_fallback(fin_a, fin_q, FIN_ALIASES['ebitda'])
        ebit = extract_with_fallback(fin_a, fin_q, FIN_ALIASES['ebit'])
        operating_income = extract_with_fallback(fin_a, fin_q, FIN_ALIASES['operating_income'])
        cogs = extract_with_fallback(fin_a, fin_q, FIN_ALIASES['cogs'])
        net_income = extract_with_fallback(fin_a, fin_q, FIN_ALIASES['net_income'])

        # Balance sheet
        equity = extract_with_fallback(bs_a, bs_q, BS_ALIASES['equity'])
        total_assets = extract_with_fallback(bs_a, bs_q, BS_ALIASES['total_assets'])
        lt_debt = extract_with_fallback(bs_a, bs_q, BS_ALIASES['lt_debt'])
        st_debt = extract_with_fallback(bs_a, bs_q, BS_ALIASES['st_debt'])
        current_assets = extract_with_fallback(bs_a, bs_q, BS_ALIASES['current_assets'])
        current_liabilities = extract_with_fallback(bs_a, bs_q, BS_ALIASES['current_liabilities'])
        cash = extract_with_fallback(bs_a, bs_q, BS_ALIASES['cash'])

        # Cash flow
        ocf = extract_with_fallback(cf_a, cf_q, CF_ALIASES['ocf'])
        capex = extract_with_fallback(cf_a, cf_q, CF_ALIASES['capex'])
        if pd.notna(capex):
            capex = -abs(capex)  # Yahoo often stores CapEx as negative; ensure subtraction produces FCF
        free_cash_flow = first_nonnull((ocf + capex) if (pd.notna(ocf) and pd.notna(capex)) else np.nan, np.nan)

        # Synthesize totals
        total_debt = first_nonnull(lt_debt + st_debt if pd.notna(lt_debt) or pd.notna(st_debt) else np.nan, np.nan)

        # Gentle pacing to reduce 403/empty frames
        time.sleep(0.3)

        print(
            f"[INFO] {ticker}: TotalAssets={total_assets}, LongTermDebt={lt_debt}, EBIT={ebit}, Equity={equity}, "
            f"COGS={cogs}, OperatingIncome={operating_income}, CurrentAssets={current_assets}, "
            f"CurrentLiabilities={current_liabilities}, PB={np.nan}, ROE={np.nan}, ROA={np.nan}, "
            f"GrossMargin={np.nan}, OperatingMargin={np.nan}, NOPAT={np.nan}, InvestedCapital={np.nan}, ROIC={np.nan}",
            flush=True
        )

        return pd.Series({
            "Price": price,
            "MarketCap": market_cap,
            "SharesOutstanding": shares_out,
            "EPS": eps,

            "Revenue": revenue,
            "EBITDA": ebitda,
            "EBIT": ebit,
            "OperatingIncome": operating_income,
            "COGS": cogs,
            "NetIncome": net_income,

            "Equity": equity,
            "TotalAssets": total_assets,
            "TotalDebt": total_debt,
            "LongTermDebt": lt_debt,
            "ShortTermDebt": st_debt,
            "CurrentAssets": current_assets,
            "CurrentLiabilities": current_liabilities,
            "Cash": cash,

            "OperatingCashFlow": ocf,
            "CapitalExpenditures": capex,
            "FreeCashFlow": free_cash_flow,
        })

    except Exception as e:
        print(f"[WARN] {ticker}: Failed to fetch fundamentals ({e})", flush=True)
        return pd.Series()

# ========== FETCH DATA ==========
total_tickers = len(df)
print_interval = 25 if total_tickers >= 50 else 50

# --- New block: Scraping loop with detailed print and auto-save ---
results = []
output_file = OUTPUT_FILE
for i, ticker in enumerate(tqdm(df['Symbol'], desc="Fetching fundamentals")):
    s = get_fundamentals(ticker)
    # Extract raw values for printing and appending
    total_assets = s.get("TotalAssets", np.nan)
    long_term_debt = s.get("LongTermDebt", np.nan)
    ebit = s.get("EBIT", np.nan)
    equity = s.get("Equity", np.nan)
    cogs = s.get("COGS", np.nan)
    operating_income = s.get("OperatingIncome", np.nan)
    current_assets = s.get("CurrentAssets", np.nan)
    current_liabilities = s.get("CurrentLiabilities", np.nan)

    # Compute interim metrics for print
    # Defensive: avoid division by zero
    price = s.get("Price", np.nan)
    shares_out = s.get("SharesOutstanding", np.nan)
    pb = np.nan
    if pd.notna(price) and pd.notna(equity) and pd.notna(shares_out) and shares_out != 0:
        pb = price / (equity / shares_out) if equity != 0 else np.nan
    roe = s.get("NetIncome", np.nan) / equity if pd.notna(equity) and equity != 0 else np.nan
    roa = s.get("NetIncome", np.nan) / total_assets if pd.notna(total_assets) and total_assets != 0 else np.nan
    gross_margin = (s.get("Revenue", np.nan) - cogs) / s.get("Revenue", np.nan) if pd.notna(s.get("Revenue", np.nan)) and s.get("Revenue", np.nan) != 0 and pd.notna(cogs) else np.nan
    operating_margin = operating_income / s.get("Revenue", np.nan) if pd.notna(operating_income) and pd.notna(s.get("Revenue", np.nan)) and s.get("Revenue", np.nan) != 0 else np.nan
    nopat = ebit * 0.79 if pd.notna(ebit) else np.nan
    invested_capital = s.get("TotalDebt", np.nan) + equity - s.get("Cash", np.nan) if pd.notna(s.get("TotalDebt", np.nan)) and pd.notna(equity) and pd.notna(s.get("Cash", np.nan)) else np.nan
    roic = nopat / invested_capital if pd.notna(nopat) and pd.notna(invested_capital) and invested_capital != 0 else np.nan

    print(
        f"   TotalAssets={total_assets}, LongTermDebt={long_term_debt}, EBIT={ebit}")
    print(
        f"   Equity={equity}, PB={pb}, ROE={roe}, ROA={roa}")
    print(
        f"   GrossMargin={gross_margin}, OperatingMargin={operating_margin}")
    print(
        f"   NOPAT={nopat}, InvestedCapital={invested_capital}, ROIC={roic}\n")

    results.append({
        "Symbol": ticker,
        "TotalAssets": total_assets,
        "LongTermDebt": long_term_debt,
        "EBIT": ebit,
        "Equity": equity,
        "COGS": cogs,
        "OperatingIncome": operating_income,
        "CurrentAssets": current_assets,
        "CurrentLiabilities": current_liabilities,
        "PB": pb,
        "ROE": roe,
        "ROA": roa,
        "GrossMargin": gross_margin,
        "OperatingMargin": operating_margin,
        "NOPAT": nopat,
        "InvestedCapital": invested_capital,
        "ROIC": roic
    })

    if (i + 1) % SAVE_INTERVAL == 0:
        pd.DataFrame(results).to_excel(output_file, index=False)
        print(f"[AUTO-SAVE] Saved partial progress at {i + 1} tickers -> {output_file}\n")

    # Randomized short sleep to reduce blocking
    time.sleep(random.uniform(0.5, 1.5))

    if (i + 1) % print_interval == 0 or (i + 1) == total_tickers:
        print(f"[PROGRESS] Completed {i + 1}/{total_tickers} tickers. Most recent: {ticker}", flush=True)

# After loop, convert results to DataFrame
fundamentals = pd.DataFrame(results)
df = pd.concat([df.reset_index(drop=True), fundamentals], axis=1)

# --- Summary print for key fields ---
key_fields = ["Equity", "TotalAssets", "LongTermDebt", "EBIT", "COGS", "OperatingIncome", "CurrentAssets", "CurrentLiabilities"]
summary_counts = {k: df[k].notnull().sum() for k in key_fields if k in df.columns}
print(f"Fundamental data non-null counts (sample): " +
      ", ".join([f"{k}: {v}" for k, v in summary_counts.items()]))

# ========== COMPUTE FACTOR METRICS ==========

# --- VALUE METRICS ---
df['EarningsYield'] = df['EPS'] / df['Price']

# EV calculation: synthesize TotalDebt from LT+ST if needed
mask_debt_missing = df['TotalDebt'].isna() & (df['LongTermDebt'].notna() | df.get('ShortTermDebt', pd.Series()).notna())
df.loc[mask_debt_missing, 'TotalDebt'] = (
    df.loc[mask_debt_missing, 'LongTermDebt'].fillna(0) +
    df.loc[mask_debt_missing, 'ShortTermDebt'].fillna(0)
)
df['EV'] = df['MarketCap'] + df['TotalDebt'] - df['Cash']
df['EV_EBITDA'] = df['EV'] / df['EBITDA']

# Book Value Per Share and PB: prefer SharesOutstanding if available
def compute_bvps(row):
    if pd.notna(row.get('Equity', np.nan)) and pd.notna(row.get('SharesOutstanding', np.nan)) and row['SharesOutstanding'] != 0:
        return row['Equity'] / row['SharesOutstanding']
    elif pd.notna(row.get('Equity', np.nan)) and pd.notna(row.get('MarketCap', np.nan)) and pd.notna(row.get('Price', np.nan)) and row['Price'] != 0:
        # Fallback: original logic
        shares = row['MarketCap'] / row['Price'] if row['Price'] != 0 else np.nan
        return row['Equity'] / shares if shares else np.nan
    else:
        return np.nan
df['BookValuePerShare'] = df.apply(compute_bvps, axis=1)
df['PB'] = df['Price'] / df['BookValuePerShare']
df['FCFYield'] = df['FreeCashFlow'] / df['MarketCap']

# --- PROFITABILITY METRICS ---
df['ROE'] = df['NetIncome'] / df['Equity']
df['ROA'] = df['NetIncome'] / df['TotalAssets']
df['GrossMargin'] = (df['Revenue'] - df['COGS']) / df['Revenue']
df['OperatingMargin'] = df['OperatingIncome'] / df['Revenue']
df['FCFMargin'] = df['FreeCashFlow'] / df['Revenue']
df['NOPAT'] = df['EBIT'] * 0.79  # assume 21% tax rate
df['InvestedCapital'] = df['TotalDebt'] + df['Equity'] - df['Cash']
df['ROIC'] = df['NOPAT'] / df['InvestedCapital']

# --- QUALITY METRICS ---
df['LT_Debt_to_Equity'] = df['LongTermDebt'] / df['Equity']
df['CurrentRatio'] = df['CurrentAssets'] / df['CurrentLiabilities']
df['AccrualsRatio'] = (df['NetIncome'] - df['FreeCashFlow']) / df['TotalAssets']

# Defensive NaN handling
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ========== Z-SCORE STANDARDIZATION ==========

def standardize(series, inverse=False):
    if series.isna().all():
        return np.nan
    s = zscore(series.replace([np.inf, -np.inf], np.nan), nan_policy='omit')
    return -s if inverse else s

# Value: higher better except EV/EBITDA, P/B
df['z_value'] = (
    standardize(df['EarningsYield']) +
    standardize(df['FCFYield']) -
    standardize(df['EV_EBITDA']) -
    standardize(df['PB'])
) / 4

# Profitability
df['z_profitability'] = (
    standardize(df['ROE']) +
    standardize(df['ROA']) +
    standardize(df['GrossMargin']) +
    standardize(df['OperatingMargin']) +
    standardize(df['FCFMargin']) +
    standardize(df['ROIC'])
) / 6

# Quality (lower LT debt better, lower accruals better)
df['z_quality'] = (
    -standardize(df['LT_Debt_to_Equity']) +
    standardize(df['CurrentRatio']) -
    standardize(df['AccrualsRatio'])
) / 3

# Composite Score
df['CompositeScore'] = (
    WEIGHTS['value'] * df['z_value'] +
    WEIGHTS['profitability'] * df['z_profitability'] +
    WEIGHTS['quality'] * df['z_quality']
)

df.sort_values('CompositeScore', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# ========== EXPORT ==========
final_df = df
final_df.to_excel(output_file, index=False)
print("✅ Factor Tilt + Quality analysis completed.")
print(f"✅ Results saved to: {output_file}")
print(f"Top 5 companies by composite score:\n{final_df[['Symbol', 'CompositeScore']].head()}")