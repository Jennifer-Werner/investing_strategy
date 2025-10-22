import re
import pandas as pd
import time
from yahooquery import Ticker

# === CONFIG ===
INPUT_PATH = "../output/sp1500_esg_raw.xlsx"
OUTPUT_PATH = "../output/sp1500_sector_industry.xlsx"
USER_AGENT = "Tina Tran (trantina375657@gmail.com) - personal research"

# === HEADER NORMALIZATION ===
def normalize_headers(df):
    """Normalize column headers for consistency (regex-safe, abbreviation-aware)."""
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
    df.columns = [normalize_col(c) for c in df.columns]
    return df

# === STEP 1 - LOAD & NORMALIZE ===
print("[INFO] Loading tickers from ESG raw dataset...")
df = pd.read_excel(INPUT_PATH)
df = normalize_headers(df)

# Identify Symbol column
symbol_col = next((c for c in df.columns if c.lower() == "symbol" or "symbol" in c.lower()), None)
if not symbol_col:
    raise KeyError("No column named 'Symbol' found in ESG dataset.")

tickers = df[symbol_col].dropna().astype(str).str.upper().unique().tolist()
print(f"[INFO] Loaded {len(tickers)} tickers for sector/industry scraping.\n")

# === STEP 2 - SCRAPE SECTOR & INDUSTRY ===
results = []
for i, ticker in enumerate(tickers, 1):
    try:
        t = Ticker(ticker, asynchronous=False)
        profile_data = t.asset_profile
        if isinstance(profile_data, dict):
            profile = profile_data.get(ticker, {}) or next(iter(profile_data.values()), {})
        else:
            profile = {}

        sector = profile.get("sector")
        industry = profile.get("industry")

        results.append({"Symbol": ticker, "Sector": sector, "Industry": industry})
        print(f"[{i}/{len(tickers)}] ✅ {ticker}: Sector={sector}, Industry={industry}")

    except Exception as e:
        print(f"[{i}/{len(tickers)}] ⚠️ Error retrieving data for {ticker}: {e}")
        results.append({"Symbol": ticker, "Sector": None, "Industry": None})

    time.sleep(0.8)  # Rate-limit friendly

# === STEP 3 - SAVE OUTPUT ===
result_df = pd.DataFrame(results)
result_df = normalize_headers(result_df)
result_df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")

print("\n[✅] Sector & industry data successfully saved.")
print(f"[📊] Output file: {OUTPUT_PATH}")
print(f"[📈] Companies processed: {len(result_df)}")
print(f"[📉] Missing sector info: {result_df['Sector'].isna().sum()} / {len(result_df)}")