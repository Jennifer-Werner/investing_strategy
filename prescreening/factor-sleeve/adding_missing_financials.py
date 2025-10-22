import os
import time
import random
import numpy as np
import pandas as pd
import yfinance as yf

# --- Load base ESG dataset ---
file_path = "../output/sp1500_esg_screened.xlsx"
df = pd.read_excel(file_path)
df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()

# --- Normalize headers: abbreviations uppercase, normal words Title Case ---
abbreviations = {"EBIT", "EBITDA", "FCF", "ROE", "ROA", "ROIC", "PB", "EPS", "EV"}
def normalize_header(col):
    parts = col.replace("_", " ").split()
    new_parts = []
    for part in parts:
        up = part.upper()
        if up in abbreviations:
            new_parts.append(up)
        else:
            new_parts.append(part.title())
    return " ".join(new_parts)

df.columns = [normalize_header(str(c)) for c in df.columns]

print(f"[INFO] Loaded {len(df)} tickers from ESG dataset.\n")

# --- Ensure all required columns exist ---
cols_needed = [
    "Market Cap", "Enterprise Value", "EBITDA", "Free Cashflow", "Total Revenue",
    "Current Assets", "Current Liabilities", "Invested Capital", "ROIC"
]
for col in cols_needed:
    if col not in df.columns:
        df[col] = np.nan

progress_file = "../output/sp1500_esg_financials_partial.xlsx"
total = len(df)

# ---------- helpers ----------
def _normalize_index(s):
    """lowercase, strip spaces & punctuation for robust label matching"""
    return (
        s.astype(str)
         .str.lower()
         .str.replace(r"[^a-z0-9]", "", regex=True)
    )

def _pick_latest_numeric(series):
    """pick the newest non-null value across date columns"""
    if series is None or series.empty:
        return np.nan
    # series is a row; columns are dates -> pick last non-null from left->right reversed
    vals = pd.to_numeric(series.dropna(), errors="coerce")
    if vals.empty:
        return np.nan
    # rightmost col is most recent in yfinance; take the last value
    return float(vals.iloc[-1])

def get_bs_items(yf_ticker):
    """
    Return (current_assets, current_liabilities, total_assets) using yfinance
    Prefer quarterly balance sheet; fallback to annual.
    """
    # Try quarterly first
    for source in ("quarterly_balance_sheet", "balance_sheet"):
        try:
            bs = getattr(yf_ticker, source)
            if isinstance(bs, pd.DataFrame) and not bs.empty:
                idx = _normalize_index(bs.index.to_series())
                # Build a map from normalized label -> original label
                lab_map = dict(zip(idx, bs.index))

                # possible label variants
                ca_keys = ["totalcurrentassets", "currentassets"]
                cl_keys = ["totalcurrentliabilities", "currentliabilities"]
                ta_keys = ["totalassets"]

                def _extract(keys):
                    for k in keys:
                        if k in lab_map:
                            return _pick_latest_numeric(bs.loc[lab_map[k]])
                    return np.nan

                current_assets = _extract(ca_keys)
                current_liabilities = _extract(cl_keys)
                total_assets = _extract(ta_keys)

                return current_assets, current_liabilities, total_assets
        except Exception:
            # try next source
            continue

    return np.nan, np.nan, np.nan

# ---------- main loop ----------
for i, ticker in enumerate(df["Symbol"], start=1):
    print(f"[INFO] Processing {ticker} ({i}/{total})", flush=True)

    try:
        stock = yf.Ticker(ticker)

        # ---- base metrics (keep as in your script) ----
        try:
            info = stock.info or {}
        except Exception:
            info = {}

        df.loc[df["Symbol"] == ticker, "Market Cap"] = info.get("marketCap")
        df.loc[df["Symbol"] == ticker, "Enterprise Value"] = info.get("enterpriseValue")
        df.loc[df["Symbol"] == ticker, "EBITDA"] = info.get("ebitda")
        df.loc[df["Symbol"] == ticker, "Free Cashflow"] = info.get("freeCashflow")
        df.loc[df["Symbol"] == ticker, "Total Revenue"] = info.get("totalRevenue")

        print(
            f"   ↳ MarketCap={info.get('marketCap')}, EV={info.get('enterpriseValue')}, "
            f"EBITDA={info.get('ebitda')}, FCF={info.get('freeCashflow')}, Revenue={info.get('totalRevenue')}",
            flush=True
        )

        # ---- balance sheet via yfinance only (no yahooquery) ----
        ca, cl, ta = get_bs_items(stock)

        # ---- invested capital (robust) ----
        if pd.notna(ta) and pd.notna(cl):
            inv_cap = ta - cl
        elif pd.notna(ca) and pd.notna(cl):
            inv_cap = ca - cl
        else:
            inv_cap = np.nan

        df.loc[df["Symbol"] == ticker, ["Current Assets", "Current Liabilities", "Invested Capital"]] = [ca, cl, inv_cap]

        print(f"   ↳ Assets={ta}, Liab={cl}, CurrAssets={ca}, InvCap={inv_cap}", flush=True)

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}", flush=True)

    # auto-save progress every 50 tickers
    if i % 50 == 0:
        try:
            df.to_excel(progress_file, index=False)
            print(f"[AUTO-SAVE] Progress saved after {i} tickers -> {os.path.abspath(progress_file)}\n", flush=True)
        except Exception as e:
            print(f"[AUTO-SAVE ERROR] {e}", flush=True)

    # small jitter to avoid throttling
    time.sleep(random.uniform(0.5, 1.2))

# --- Save final output ---
output_path = "../output/sp1500_esg_financials_full.xlsx"
df.to_excel(output_path, index=False)
print(f"\n✅ Finished scraping all {total} tickers.")
print(f"✅ Saved to: {os.path.abspath(output_path)}")