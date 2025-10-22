import pandas as pd
import numpy as np

# =========================
# Helpers
# =========================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column names from different sources into a single canonical set.
    We match on a very forgiving lowercase + stripped key and rename to the canonical label.
    """
    mapping = {
        'symbol': 'Symbol',
        'name': 'Name',
        'sector': 'Sector',
        'industry': 'Industry',

        # Size / price-related
        'marketcap': 'Market Cap',
        'marketcapitalization': 'Market Cap',
        'beta': 'Beta',

        # Value
        'p/ettm': 'P/E TTM',
        'pe': 'P/E TTM',
        'pettm': 'P/E TTM',
        'eps': 'EPS TTM',
        'epsttm': 'EPS TTM',
        'ev': 'EV',
        'enterprisevalue': 'EV',
        'revenue': 'Revenue',
        'totalrevenue': 'Revenue',
        'ebitda': 'EBITDA',
        'freecashflow': 'FCF',
        'fcf': 'FCF',
        'pb': 'P/B',

        # Profitability
        'netincome(a)': 'Net Income (A)',
        'netincome': 'Net Income (A)',
        'eps ttm': 'EPS TTM',
        'ebit': 'EBIT',
        'operatingincome': 'Operating Income',
        'grossmargin': 'Gross Margin',
        'operatingmargin': 'Operating Margin',
        'roe': 'ROE',
        'roa': 'ROA',
        'roic': 'ROIC',
        'nopat': 'NOPAT',
        'cogs': 'COGS',

        # Quality / balance sheet
        'equity': 'Equity',
        'totalassets': 'Total Assets',
        'longtermdebt': 'Long Term Debt',
        'currentassets': 'Current Assets',
        'currentliabilities': 'Current Liabilities',
        'investedcapital': 'Invested Capital',

        # Dividends / meta
        'dividend(a)': 'Dividend (A)',
        'dividenda': 'Dividend (A)',
        'dividend': 'Dividend (A)',
        'divyield(a)': 'Div Yield (A)',
        'divyield': 'Div Yield (A)',
        'dividendyield': 'Div Yield (A)',
        'lastupdatedate': 'Last Update Date',
        'ticker': 'Ticker',
        'latestearnings': 'Latest Earnings',

        # ESG (to drop later)
        'totalesgscore': 'Total ESG Score',
        'environmentriskscore': 'Environment Risk Score',
        'socialriskscore': 'Social Risk Score',
        'governanceriskscore': 'Governance Risk Score',
        'controversylevel': 'Controversy Level',
    }

    def keyize(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("-", "").replace("/", "")

    renames = {}
    for c in df.columns:
        k = keyize(str(c))
        if k in mapping:
            renames[c] = mapping[k]
    return df.rename(columns=renames)

def clean_div_yield_series(s: pd.Series) -> pd.Series:
    """Return a series with dividend yield strings cleaned to a consistent numeric percent (not fraction).
       '2.55%' -> 2.55, '0.7%' -> 0.7, '2.55' assumed already percent -> 2.55.
    """
    def _clean(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            # Assume already in percent values (e.g., 2.55 for 2.55%)
            return float(v)
        v = str(v).strip()
        if v.endswith('%'):
            v = v[:-1].strip()
        try:
            return float(v)
        except Exception:
            return np.nan
    return s.map(_clean)

# =========================
# Load datasets
# =========================
financials = pd.read_excel("sp1500_esg_financials_full.xlsx")
factors = pd.read_excel("sp1500_factor_quality_ranked.xlsx")

# Standardize columns before any operations
financials = standardize_columns(financials)
factors = standardize_columns(factors)

# Ensure Symbol exists and normalized
for df in [financials, factors]:
    if 'Symbol' not in df.columns:
        # Try to detect a ticker-like column
        poss = [c for c in df.columns if c.lower() in ('ticker',)]
        if poss:
            df.rename(columns={poss[0]: 'Symbol'}, inplace=True)
    df['Symbol'] = df['Symbol'].astype(str).str.upper().str.strip()

# =========================
# Remove duplicates and unify
# =========================
# Prefer Market Cap from the ESG/financials file; drop Market Cap if present in factors
if 'Market Cap' in factors.columns:
    factors = factors.drop(columns=['Market Cap'])

# Drop overlapping columns from 'factors' except ID fields
protected = {'Symbol', 'Name', 'Sector', 'Industry'}
overlaps = [c for c in factors.columns if c in financials.columns and c not in protected]
if overlaps:
    print(f"[INFO] Dropping overlapping columns from factors: {overlaps}")
    factors = factors.drop(columns=overlaps, errors="ignore")

# =========================
# Merge
# =========================
merged = pd.merge(financials, factors, on="Symbol", how="outer")

# =========================
# Remove ESG + Meta fields you no longer want
# =========================
esg_cols = [
    "Total ESG Score", "Environment Risk Score", "Social Risk Score",
    "Governance Risk Score", "Controversy Level"
]
drop_meta = ["Latest Earnings", "Ticker"]
merged.drop(columns=[c for c in esg_cols + drop_meta if c in merged.columns], inplace=True, errors="ignore")

# =========================
# Restore/fix dividend yield
# =========================
# If Div Yield (A) exists but is missing, try to backfill from the original financials
if "Div Yield (A)" in merged.columns:
    before_missing = merged["Div Yield (A)"].isna().sum()
    # Look for any dividend yield column in the original (already standardized) financials
    div_src_cols = [c for c in financials.columns if c == "Div Yield (A)"]
    if div_src_cols:
        src = financials[["Symbol", "Div Yield (A)"]].copy()
        # Clean to numeric percent (2.55, not 0.0255)
        src["Div Yield (A)"] = clean_div_yield_series(src["Div Yield (A)"])
        merged = merged.merge(src, on="Symbol", how="left", suffixes=("", "_src"))
        # If merged has missing, fill from src
        mask = merged["Div Yield (A)"].isna() & merged["Div Yield (A)_src"].notna()
        merged.loc[mask, "Div Yield (A)"] = merged.loc[mask, "Div Yield (A)_src"]
        merged.drop(columns=["Div Yield (A)_src"], inplace=True)
        after_missing = merged["Div Yield (A)"].isna().sum()
        print(f"[INFO] Restored {before_missing - after_missing} 'Div Yield (A)' values from financials.")
    else:
        print("[WARN] No 'Div Yield (A)' present in financials to restore from.")
else:
    # If merged doesn't have it but financials do, add it
    if "Div Yield (A)" in financials.columns:
        merged = merged.merge(financials[["Symbol", "Div Yield (A)"]], on="Symbol", how="left")
        print("[INFO] Added 'Div Yield (A)' from financials.")

# =========================
# Convert obviously numeric columns (safe list)
# =========================
numeric_candidates = [
    "Market Cap","Beta","P/E TTM","EPS TTM","EV","EBITDA","FCF","Revenue","COGS",
    "ROE","ROA","ROIC","EBIT","Operating Income","Net Income (A)",
    "Total Assets","Equity","Long Term Debt","Current Assets","Current Liabilities","Invested Capital",
    "P/B","Gross Margin","Operating Margin","NOPAT","Div Yield (A)","Dividend (A)"
]

# --- Fix duplicate column names and multi-dimensional duplicates ---
if merged.columns.duplicated().any():
    print(f"[WARN] Found duplicate column names. Deduplicating...")
    merged = merged.loc[:, ~merged.columns.duplicated()]

# Ensure each target column is a Series (not a DataFrame)
for col in numeric_candidates:
    if col in merged.columns:
        # If col is duplicated (multiple columns with same name), select the first one
        if isinstance(merged[col], pd.DataFrame):
            print(f"[WARN] {col} had multiple sub-columns, keeping first only.")
            merged[col] = merged[col].iloc[:, 0]

for col in numeric_candidates:
    if col in merged.columns:
        # Div Yield and Dividend may be strings; handle gracefully
        if col == "Div Yield (A)":
            merged[col] = clean_div_yield_series(merged[col])
        else:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

# =========================
# Compute ROIC if missing
# =========================
if "ROIC" not in merged.columns or merged["ROIC"].isna().all():
    print("[INFO] Computing ROIC (Return on Invested Capital) from components...")
else:
    print("[INFO] Filling missing ROIC values using available financial data...")

# Ensure the necessary columns exist
required_cols = ["EBIT", "Total Assets", "Current Liabilities", "Long Term Debt", "Equity"]
for col in required_cols:
    if col not in merged.columns:
        merged[col] = np.nan

# Compute Invested Capital as (Total Assets - Current Liabilities)
merged["Invested Capital"] = merged["Total Assets"] - merged["Current Liabilities"]

# Compute ROIC = EBIT / Invested Capital
merged["ROIC_Calc"] = np.where(
    (merged["Invested Capital"].notna()) & (merged["Invested Capital"] != 0),
    merged["EBIT"] / merged["Invested Capital"],
    np.nan
)

# Convert to percent
merged["ROIC_Calc"] = merged["ROIC_Calc"] * 100

# If "ROIC" exists, fill missing values
if "ROIC" in merged.columns:
    before_fill = merged["ROIC"].isna().sum()
    merged.loc[merged["ROIC"].isna(), "ROIC"] = merged.loc[merged["ROIC"].isna(), "ROIC_Calc"]
    after_fill = merged["ROIC"].isna().sum()
    print(f"[INFO] Filled {before_fill - after_fill} missing ROIC values using calculated data.")
else:
    merged["ROIC"] = merged["ROIC_Calc"]

# Drop temporary column
merged.drop(columns=["ROIC_Calc"], inplace=True)

# =========================
# Order columns by logical groups
# =========================
grouped_order = [
    # Identification
    "Symbol", "Name", "Sector", "Industry",

    # Size / Market characteristics
    "Market Cap", "Beta",

    # VALUE
    "P/E TTM", "P/B", "EV", "Revenue", "EBITDA", "FCF",

    # PROFITABILITY
    "EPS TTM", "Net Income (A)", "ROE", "ROA", "ROIC", "Gross Margin", "Operating Margin", "EBIT", "Operating Income", "COGS",

    # QUALITY / BALANCE SHEET
    "Total Assets", "Equity", "Long Term Debt", "Current Assets", "Current Liabilities", "Invested Capital",

    # DIVIDENDS
    "Dividend (A)", "Div Yield (A)",

    # Meta
    "Last Update Date",
]

present = [c for c in grouped_order if c in merged.columns]
rest = [c for c in merged.columns if c not in present]
merged = merged[present + rest]

print(f"[INFO] Final shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
print(f"[INFO] First 5 columns: {merged.columns[:5].tolist()}")
print(f"[INFO] Dividend yield non-null count: {merged['Div Yield (A)'].notna().sum() if 'Div Yield (A)' in merged.columns else 0}")

# =========================
# Save
# =========================
output_file = "../output/sp1500_financials_combined.xlsx"
merged.to_excel(output_file, index=False)

print("\n✅ Combined financial dataset successfully created.")
print(f"   → Saved to: {output_file}")
print(f"   → Rows: {len(merged)} | Columns: {len(merged.columns)}")