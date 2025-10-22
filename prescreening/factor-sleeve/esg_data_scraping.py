import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"
# -*- coding: utf-8 -*-
import time
import random
import pandas as pd
from yahooquery import Ticker

# ---------------------------------------------------------
#
# CONFIGURATION
# ---------------------------------------------------------
USER_AGENT = "Tina Tran (trantina375657@gmail.com) - personal research"
INPUT_FILE = "../output/sp1500_universe.xlsx"
OUTPUT_FILE = "../output/sp1500_esg_raw.xlsx"

# ---------------------------------------------------------
# STEP 1 - Load tickers
# ---------------------------------------------------------
print("[INFO] Loading tickers...", flush=True)
df = pd.read_excel(INPUT_FILE)

import re
def normalize_header(header):
    """
    Normalize column headers:
    - Replace underscores/hyphens with spaces.
    - Trim whitespace.
    - Title Case non-abbreviations.
    - Keep key financial abbreviations fully uppercase.
    """
    abbreviations = {"EBIT", "EBITDA", "FCF", "ROE", "ROA", "ROIC", "PB", "EPS", "EV", "COGS", "DCF", "WACC", "PE"}
    header_clean = re.sub(r'[_\-]+', ' ', str(header)).strip()
    parts = header_clean.split()
    normalized_parts = []
    for part in parts:
        cleaned = re.sub(r'[^A-Za-z]', '', part).upper()
        if cleaned in abbreviations:
            normalized_parts.append(cleaned)
        else:
            normalized_parts.append(part.title())
    new_header = " ".join(normalized_parts)
    for abbr in abbreviations:
        new_header = re.sub(rf'\b{abbr}\b', abbr, new_header, flags=re.IGNORECASE)
    print(f"[Header Normalization] {header} -> {new_header}", flush=True)
    return new_header

df.columns = [normalize_header(str(col)) for col in df.columns]

print(f"[INFO] Normalized headers: {df.columns.tolist()}", flush=True)

df.columns = df.columns.str.strip().str.lower()
symbol_col = None
for candidate in ["symbol", "ticker"]:
    if candidate in df.columns:
        symbol_col = candidate
        break
if symbol_col is None:
    raise KeyError("No column named 'symbol' or 'ticker' found.")

tickers = df[symbol_col].astype(str).str.upper().unique().tolist()
print(f"[OK] Loaded {len(tickers)} tickers.\n", flush=True)


# ---------------------------------------------------------
# STEP 2 - Define ESG Scraping Function (yahooquery-based)
# ---------------------------------------------------------
def fetch_esg_data(ticker, pause=1.5):
    """
    Fetch ESG metrics via yahooquery (uses Yahoo's underlying JSON endpoints; no JS rendering).
    Requires: pip install yahooquery
    """
    import random, time
    ticker_fixed = ticker.replace(".", "-")
    try:
        t = Ticker(ticker_fixed, asynchronous=False)
        data = t.esg_scores
        # yahooquery returns a dict keyed by lowercased symbol
        if isinstance(data, dict):
            # keys are usually lowercased symbols
            data = data.get(ticker_fixed.lower()) or data.get(ticker_fixed) or next(iter(data.values()), {})
        if not isinstance(data, dict):
            print(f"[WARN] {ticker}: esg_scores payload not dict.", flush=True)
            time.sleep(random.uniform(1.0, 2.0))
            return None

        # Try multiple possible field names robustly (Yahoo changes naming)
        def pick(*keys):
            for k in keys:
                if k in data and data[k] is not None:
                    return data[k]
            return None

        total_esg = pick('totalEsg', 'esgScore', 'totalEsgRisk')
        env = pick('environmentScore', 'environmentRiskScore')
        soc = pick('socialScore', 'socialRiskScore')
        gov = pick('governanceScore', 'governanceRiskScore')
        controversy = pick('highestControversy', 'controversyLevel')
        last_update = pick('ratingYear', 'ratingMonth', 'ratingDate', 'lastUpdateDate')

        # Convert numeric-like values safely
        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        result = {
            "ticker": ticker,
            "total_esg_score": to_float(total_esg),
            "environment_risk_score": to_float(env),
            "social_risk_score": to_float(soc),
            "governance_risk_score": to_float(gov),
            "controversy_level": to_float(controversy),
            "last_update_date": str(last_update) if last_update is not None else None
        }

        if all(v is None for k,v in result.items() if k not in ("ticker","last_update_date")):
            print(f"[WARN] {ticker}: no numeric ESG fields found in yahooquery payload.", flush=True)
            time.sleep(random.uniform(1.0, 2.0))
            return None

        print(f"[INFO] {ticker}: ESG={result['total_esg_score']}, Controversy={result['controversy_level']}", flush=True)
        time.sleep(random.uniform(1.0, 2.0))
        return result

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}", flush=True)
        time.sleep(random.uniform(1.0, 2.0))
        return None


# ---------------------------------------------------------
# STEP 3 - Run scraper loop
# ---------------------------------------------------------
print("[INFO] Starting scraping...", flush=True)
print(f"[INFO] Beginning scraping of {len(tickers)} tickers...", flush=True)
esg_data = []
for i, ticker in enumerate(tickers, start=1):
    print(f"[INFO] Processing {ticker} ({i}/{len(tickers)})", flush=True)
    result = fetch_esg_data(ticker)
    if result:
        esg_data.append(result)
    if i % 10 == 0 or i == len(tickers):
        print(f"Progress: {i}/{len(tickers)} tickers processed.", flush=True)

print(f"\n[OK] Scraped data for {len(esg_data)} companies.", flush=True)

# ---------------------------------------------------------
# STEP 4 - Save to Excel
# ---------------------------------------------------------
esg_df = pd.DataFrame(esg_data)
merged = df.merge(esg_df, left_on=symbol_col, right_on="ticker", how="left")
merged.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"[SAVED] ESG data written to {OUTPUT_FILE}", flush=True)
