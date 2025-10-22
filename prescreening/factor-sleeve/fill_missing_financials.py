import random
# --- New helper: fetch from local sources ---
def _scrape_html_table(url: str, pattern_dict: dict) -> dict:
    """
    Fetches HTML from a URL and applies regex patterns to extract numeric values.
    Returns a dict of {key: value}.
    """
    try:
        resp = session.get(url, timeout=TIMEOUT, headers={"User-Agent": UA})
        if resp.status_code != 200:
            return {}
        html = resp.text
    except Exception:
        return {}
    result = {}
    for key, pat in pattern_dict.items():
        matches = re.findall(pat, html, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            # If multiple matches, take the first, flatten tuple if necessary
            m = matches[0]
            if isinstance(m, tuple):
                m = m[0]
            # Remove commas, percent, whitespace
            val_str = str(m).replace(",", "").replace("%", "").strip()
            try:
                val = float(val_str)
                result[key] = val
            except Exception:
                continue
    return result

def fetch_from_alt_sources(symbol: str) -> dict:
    """
    Scrape MarketWatch, MacroTrends, StockAnalysis for key metrics.
    Returns dict of {field: value}.
    """
    out = {}
    # MarketWatch
    mw_url = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}/financials"
    mw_patterns = {
        "EBITDA": r"EBITDA[^\\d\\-]+([\\d,\\.]+)\\s*M?B?\\s*<",
        "Gross Margin": r"Gross Margin[^\\d\\-]+([\\d\\.]+)%",  # percent
        "Operating Margin": r"Operating Margin[^\\d\\-]+([\\d\\.]+)%",  # percent
        "COGS": r"Cost of Goods Sold[^\\d\\-]+([\\d,\\.]+)\\s*M?B?\\s*<",
        "FCF": r"Free Cash Flow[^\\d\\-]+([\\d,\\.]+)\\s*M?B?\\s*<",
    }
    mw = _scrape_html_table(mw_url, mw_patterns)
    # MacroTrends
    mt_url = f"https://www.macrotrends.net/stocks/charts/{symbol.lower()}/{symbol.lower()}/financial-ratios"
    mt_patterns = {
        "ROIC": r"ROIC[^\\d\\-]+([\\-]?[\\d\\.]+)%",  # percent
        "ROE": r"ROE[^\\d\\-]+([\\-]?[\\d\\.]+)%",    # percent
        "ROA": r"ROA[^\\d\\-]+([\\-]?[\\d\\.]+)%",    # percent
        "Gross Margin": r"Gross Margin[^\\d\\-]+([\\d\\.]+)%",  # percent
        "Operating Margin": r"Operating Margin[^\\d\\-]+([\\d\\.]+)%",  # percent
    }
    mt = _scrape_html_table(mt_url, mt_patterns)
    # StockAnalysis
    sa_url = f"https://stockanalysis.com/stocks/{symbol.lower()}/financials/"
    sa_patterns = {
        "EBITDA": r"EBITDA[^\\d\\-]+([\\d,\\.]+)\\s*M?B?\\s*<",
        "Gross Margin": r"Gross Margin[^\\d\\-]+([\\d\\.]+)%",  # percent
        "Operating Margin": r"Operating Margin[^\\d\\-]+([\\d\\.]+)%",  # percent
        "COGS": r"Cost of Goods Sold[^\\d\\-]+([\\d,\\.]+)\\s*M?B?\\s*<",
        "FCF": r"Free Cash Flow[^\\d\\-]+([\\d,\\.]+)\\s*M?B?\\s*<",
    }
    sa = _scrape_html_table(sa_url, sa_patterns)
    # Merge, prefer first non-None
    for k in set(list(mw.keys()) + list(mt.keys()) + list(sa.keys())):
        for src in (mw, mt, sa):
            if k in src:
                out[k] = src[k]
                break
    # Convert percent fields to decimals
    percent_fields = ["ROIC", "ROE", "ROA", "Gross Margin", "Operating Margin"]
    for pf in percent_fields:
        if pf in out:
            try:
                out[pf] = float(out[pf]) / 100.0
            except Exception:
                out[pf] = None
    # Polite sleep to avoid rate-limiting
    time.sleep(random.uniform(2.5, 5.0))
    return {k: v for k, v in out.items() if v is not None}
import os
import time
import math
import json
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import re
from typing import Dict, Any, Optional

# ------------------ Header Normalization & Validation ------------------
def normalize_headers(df):
    """Normalize DataFrame column headers with regex-safe, abbreviation-preserving logic."""
    abbreviations = {"EBIT", "EBITDA", "FCF", "ROE", "ROA", "ROIC", "PB", "EPS", "EV", "COGS", "DCF", "WACC", "PE", "TTM"}
    def normalize_col(col):
        col_original = str(col)
        col_clean = re.sub(r'[_\\-]+', ' ', col_original).strip()
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
            new_col = re.sub(rf'\\b{abbr}\\b', abbr, new_col, flags=re.IGNORECASE)
        return new_col
    df.columns = [normalize_col(c) for c in df.columns]
    return df

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common header variants to canonical names used by this script."""
    alias_map = {
        # EPS TTM variants
        "Eps Ttm": "EPS TTM",
        "EPS Ttm": "EPS TTM",
        "Eps TTM": "EPS TTM",
        "Trailing Eps": "EPS TTM",
        "Trailing EPS": "EPS TTM",
        "Earnings Per Share": "EPS TTM",
        # EV and related
        "Enterprise Value": "EV",
        "EnterpriseValue": "EV",
        # FCF
        "Free Cash Flow": "FCF",
        "Freecashflow": "FCF",
        # Revenue
        "Total Revenue": "Revenue",
        "TotalRevenue": "Revenue",
        # Net Income
        "Net Income": "Net Income (A)",
        "Net Income(A)": "Net Income (A)",
        # Operating Income
        "OperatingIncome": "Operating Income",
        # Costs
        "Cost Of Revenue": "COGS",
        "Cost Of Goods Sold": "COGS",
        "Cost Of Goods": "COGS",
        # Balance sheet
        "Totalassets": "Total Assets",
        "Total Assets": "Total Assets",
        "Totalassets": "Total Assets",
        "Total Stockholder Equity": "Equity",
        "TotalStockholderEquity": "Equity",
        "Long Term Debt": "Long Term Debt",
        "Longtermdebt": "Long Term Debt",
        "Currentassets": "Current Assets",
        "Current Assets": "Current Assets",
        "Currentliabilities": "Current Liabilities",
        "Current Liabilities": "Current Liabilities",
    }
    # Build actual renames present in df
    renames = {src: dst for src, dst in alias_map.items() if src in df.columns}
    if renames:
        df.rename(columns=renames, inplace=True)
    return df

def validate_value(col, val):
    """Simple sanity validation to prevent wrong-scale or bad data."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return False
    if "margin" in col.lower() and not (0 <= val <= 1):
        return False
    if "roe" in col.lower() and abs(val) > 5:
        return False
    if "roic" in col.lower() and abs(val) > 5:
        return False
    if "debt" in col.lower() and val < 0:
        return False
    return True

# ======================= CONFIG =======================
INPUT_FILE  = "../output/sp1500_financials_filled.xlsx"
OUTPUT_FILE = "../output/sp1500_financials_filled.xlsx"
LOCAL_SOURCES = [
    "sp1500_financials_combined.xlsx",
    "sp1500_factor_quality_ranked.xlsx",
    "sp1500_esg_financials_full.xlsx",
    "sp1500_esg_screened.xlsx"
]
SLEEP_BASE  = 0.6          # base delay between symbols
MAX_RETRIES = 3            # per HTTP call
TIMEOUT     = 12           # seconds per HTTP call
UA          = os.getenv("SCRAPER_UA", "Mozilla/5.0 (compatible; ESGBot/1.0; +https://example.org)")
SEC_UA      = os.getenv("SEC_UA", "Your Name email@example.com")  # for SEC API etiquette
# Fields we try to fill
TARGET_COLS = [
    "EV", "Revenue", "EBITDA", "FCF", "EPS TTM", "Net Income (A)",
    "ROE", "ROA", "ROIC", "Gross Margin", "Operating Margin",
    "EBIT", "Operating Income", "COGS", "Total Assets", "Equity", "Long Term Debt",
    "Current Assets", "Current Liabilities", "Invested Capital"
]
# ======================================================

session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*"})


def _backoff_sleep(attempt: int):
    time.sleep(SLEEP_BASE * (1.5 ** attempt))


def _get(url: str, params=None, headers=None) -> Optional[requests.Response]:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            # Some Yahoo JSON endpoints occasionally return 404/502/503—try again
            _backoff_sleep(attempt)
        except requests.RequestException:
            _backoff_sleep(attempt)
    return None


def yahoo_quote_summary(symbol: str, modules: list) -> Dict[str, Any]:
    """Yahoo JSON (undocumented) endpoint. Returns parsed dict or {}."""
    base = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    params = {"modules": ",".join(modules)}
    r = _get(base, params=params)
    if not r:
        return {}
    try:
        js = r.json()
        result = js.get("quoteSummary", {}).get("result", [])
        return result[0] if result else {}
    except (ValueError, KeyError, IndexError):
        return {}


def _extract_latest_from_statement(stmt: Dict[str, Any], field: str) -> Optional[float]:
    """Helper to extract 'raw' from Yahoo JSON statements lists."""
    if not stmt or field not in stmt:
        return None
    items = stmt[field]
    if isinstance(items, list) and items:
        raw = items[0].get("raw")
        return float(raw) if raw is not None else None
    if isinstance(items, dict) and "raw" in items:
        return float(items["raw"])
    return None


def fetch_from_yahoo_json(symbol: str) -> Dict[str, Any]:
    """
    Pull from Yahoo JSON:
      financialData/defaultKeyStatistics/summaryDetail for ebitda, freeCashflow, margins, ratios;
      income/cashflow/balance (annual+quarterly) for totals/missing items.
    """
    out = {}

    # 1) Point-in-time aggregates
    mod1 = ["financialData", "defaultKeyStatistics", "summaryDetail"]
    js1 = yahoo_quote_summary(symbol, mod1)

    fd  = js1.get("financialData", {})
    dks = js1.get("defaultKeyStatistics", {})
    sdy = js1.get("summaryDetail", {})

    # numeric helpers
    def g(container, key):
        v = container.get(key, {})
        if isinstance(v, dict):
            v = v.get("raw")
        return float(v) if isinstance(v, (int, float)) else (float(v) if str(v).replace('.', '', 1).isdigit() else None)

    out["EBITDA"]          = g(fd, "ebitda") or g(dks, "ebitda")
    out["FCF"]             = g(fd, "freeCashflow")
    out["Revenue"]         = g(fd, "totalRevenue")
    out["Operating Margin"]= g(fd, "operatingMargins")
    out["ROE"]             = g(fd, "returnOnEquity")
    out["ROA"]             = g(fd, "returnOnAssets")

    # 2) Full statements (annual + quarterly) → totals we often miss
    mod2 = [
        "incomeStatementHistory","incomeStatementHistoryQuarterly",
        "cashflowStatementHistory","cashflowStatementHistoryQuarterly",
        "balanceSheetHistory","balanceSheetHistoryQuarterly"
    ]
    js2 = yahoo_quote_summary(symbol, mod2)

    incA = js2.get("incomeStatementHistory", {}).get("incomeStatementHistory", [])
    incQ = js2.get("incomeStatementHistoryQuarterly", {}).get("incomeStatementHistory", [])
    cfaA = js2.get("cashflowStatementHistory", {}).get("cashflowStatements", [])
    cfaQ = js2.get("cashflowStatementHistoryQuarterly", {}).get("cashflowStatements", [])
    bsA  = js2.get("balanceSheetHistory", {}).get("balanceSheetStatements", [])
    bsQ  = js2.get("balanceSheetHistoryQuarterly", {}).get("balanceSheetStatements", [])

    def latest(lst, key):
        if not lst:
            return None
        val = lst[0].get(key, {})
        raw = val.get("raw") if isinstance(val, dict) else None
        return float(raw) if raw is not None else None

    # Revenue fallback
    out["Revenue"] = out["Revenue"] or latest(incA, "totalRevenue") or latest(incQ, "totalRevenue")

    # EBITDA fallback (rarely in statements; sometimes in cashflow as "operatingIncome" + D&A… not reliable)
    # Keep as-is; we'll get better from yfinance statements later if still missing.

    # Current Assets / Liabilities
    out["Current Assets"]      = latest(bsA, "totalCurrentAssets")      or latest(bsQ, "totalCurrentAssets")
    out["Current Liabilities"] = latest(bsA, "totalCurrentLiabilities") or latest(bsQ, "totalCurrentLiabilities")

    # Operating Margin fallback: operatingIncome / totalRevenue
    if out.get("Operating Margin") is None:
        op_inc = latest(incA, "operatingIncome") or latest(incQ, "operatingIncome")
        rev    = out.get("Revenue")
        if op_inc is not None and rev and rev != 0:
            out["Operating Margin"] = float(op_inc) / float(rev)

    # FCF fallback: operatingCashflow - capitalExpenditures
    if out.get("FCF") is None:
        ocf = latest(cfaA, "totalCashFromOperatingActivities") or latest(cfaQ, "totalCashFromOperatingActivities")
        capex = latest(cfaA, "capitalExpenditures") or latest(cfaQ, "capitalExpenditures")
        if ocf is not None and capex is not None:
            out["FCF"] = float(ocf) - float(capex)

    # ROE/ROA fallback from statements if missing
    if out.get("ROE") is None or out.get("ROA") is None:
        net = latest(incA, "netIncome") or latest(incQ, "netIncome")
        eq  = latest(bsA, "totalStockholderEquity") or latest(bsQ, "totalStockholderEquity")
        ta  = latest(bsA, "totalAssets") or latest(bsQ, "totalAssets")
        if net is not None:
            if out.get("ROE") is None and eq not in (None, 0):
                out["ROE"] = float(net) / float(eq)
            if out.get("ROA") is None and ta not in (None, 0):
                out["ROA"] = float(net) / float(ta)

    # Additional fields requested
    out["EV"]             = g(fd, "enterpriseValue") or g(dks, "enterpriseValue")
    out["EPS TTM"]        = g(fd, "earningsPerShare") or g(dks, "trailingEps")
    out["Net Income (A)"] = latest(incA, "netIncome") or latest(incQ, "netIncome")
    out["EBIT"]           = latest(incA, "ebit") or latest(incQ, "ebit")
    out["Operating Income"] = latest(incA, "operatingIncome") or latest(incQ, "operatingIncome")
    out["COGS"]           = latest(incA, "costOfRevenue") or latest(incQ, "costOfRevenue")
    out["Total Assets"]   = latest(bsA, "totalAssets") or latest(bsQ, "totalAssets")
    out["Equity"]         = latest(bsA, "totalStockholderEquity") or latest(bsQ, "totalStockholderEquity")
    out["Long Term Debt"] = latest(bsA, "longTermDebt") or latest(bsQ, "longTermDebt")
    out["Gross Margin"]   = g(fd, "grossMargins") or g(dks, "grossMargins")

    return {k: v for k, v in out.items() if v is not None}


def fetch_from_yfinance_statements(symbol: str) -> Dict[str, Any]:
    """
    Use yfinance structured statements (annual & quarterly).
    """
    t = yf.Ticker(symbol)
    out = {}

    def newest(series_like):
        if series_like is None or series_like.empty:
            return None
        try:
            return float(series_like.iloc[0])
        except Exception:
            return None

    # Income statement
    try:
        incA = t.income_stmt
    except Exception:
        incA = None
    try:
        incQ = t.quarterly_income_stmt
    except Exception:
        incQ = None

    # Cashflow
    try:
        cfA = t.cashflow
    except Exception:
        cfA = None
    try:
        cfQ = t.quarterly_cashflow
    except Exception:
        cfQ = None

    # Balance sheet
    try:
        bsA = t.balance_sheet
    except Exception:
        bsA = None
    try:
        bsQ = t.quarterly_balance_sheet
    except Exception:
        bsQ = None

    # Helpers to try annual then quarterly
    def pick(dfA, dfQ, key_variants):
        for key in key_variants:
            val = None
            if dfA is not None and key in dfA.index:
                val = newest(dfA.loc[key])
            if val is None and dfQ is not None and key in dfQ.index:
                val = newest(dfQ.loc[key])
            if val is not None:
                return float(val)
        return None

    # Revenue
    out["Revenue"] = pick(incA, incQ, ["Total Revenue", "TotalRevenue"])

    # EBITDA (yfinance statements may provide; if not, we leave None)
    out["EBITDA"] = pick(incA, incQ, ["Ebitda", "EBITDA"])

    # Operating Margin
    op_inc = pick(incA, incQ, ["Operating Income", "OperatingIncome"])
    if op_inc is not None and out.get("Revenue"):
        out["Operating Margin"] = float(op_inc) / float(out["Revenue"])

    # FCF from OCF - CapEx
    ocf   = pick(cfA, cfQ, ["Total Cash From Operating Activities", "Operating Cash Flow", "TotalCashFromOperatingActivities"])
    capex = pick(cfA, cfQ, ["Capital Expenditures", "CapitalExpenditures"])
    if ocf is not None and capex is not None:
        out["FCF"] = float(ocf) - float(capex)

    # Current assets & liabilities
    out["Current Assets"]      = pick(bsA, bsQ, ["Total Current Assets", "TotalCurrentAssets"])
    out["Current Liabilities"] = pick(bsA, bsQ, ["Total Current Liabilities", "TotalCurrentLiabilities"])

    # Additional fields support
    out["EBIT"]           = pick(incA, incQ, ["Ebit", "EBIT"])
    out["Operating Income"] = pick(incA, incQ, ["Operating Income", "OperatingIncome"])
    out["COGS"]           = pick(incA, incQ, ["Cost Of Revenue", "CostOfRevenue"])
    out["Total Assets"]   = pick(bsA, bsQ, ["Total Assets", "TotalAssets"])
    out["Equity"]         = pick(bsA, bsQ, ["Total Stockholder Equity", "TotalStockholderEquity"])
    out["Long Term Debt"] = pick(bsA, bsQ, ["Long Term Debt", "LongTermDebt"])
    out["Net Income (A)"] = pick(incA, incQ, ["Net Income", "NetIncome"])

    return {k: v for k, v in out.items() if v is not None}


# -------- SEC fallback (for Current Assets/Liabilities) ----------
# Ticker → CIK cache
_cik_cache: Dict[str, str] = {}

def sec_fetch_cik(symbol: str) -> Optional[str]:
    sym = symbol.upper()
    if sym in _cik_cache:
        return _cik_cache[sym]
    r = _get("https://www.sec.gov/files/company_tickers.json",
             headers={"User-Agent": SEC_UA})
    if not r:
        return None
    try:
        js = r.json()
        # file is mapping index -> {cik_str,ticker,title}
        for _, v in js.items():
            if v.get("ticker", "").upper() == sym:
                cik = str(v.get("cik_str")).zfill(10)
                _cik_cache[sym] = cik
                return cik
    except Exception:
        return None
    return None


def sec_company_concept(cik: str, concept: str) -> Optional[float]:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
    r = _get(url, headers={"User-Agent": SEC_UA, "Accept": "application/json"})
    if not r:
        return None
    try:
        js = r.json()
        # pick the latest by 'fy' or 'end'
        facts = js.get("units", {})
        # choose USD if present
        for unit in ("USD", "USD$"):
            if unit in facts and facts[unit]:
                # sort by end date desc
                vals = sorted(facts[unit], key=lambda x: x.get("end", ""), reverse=True)
                raw = vals[0].get("val")
                return float(raw) if raw is not None else None
    except Exception:
        return None
    return None


def compute_invested_capital(row: pd.Series) -> Optional[float]:
    """
    Conservative fallback:
      Invested Capital ≈ Total Assets - Current Liabilities - Cash & Equivalents
    If cash is unavailable, use: Total Assets - Current Liabilities
    """
    ta = row.get("Total Assets")
    cl = row.get("Current Liabilities")
    cash = row.get("Cash And Cash Equivalents") or row.get("Cash")
    try:
        if pd.notna(ta) and pd.notna(cl):
            ic = float(ta) - float(cl)
            if pd.notna(cash):
                ic -= float(cash)
            return ic
    except Exception:
        pass
    return None

# --- New helper: fetch from local sources ---
def fetch_from_local_sources(symbol: str, missing_cols) -> Dict[str, Any]:
    """
    Search local Excel files for missing data by Symbol, but only for missing columns.
    Prefer normalized files if available, fallback to unnormalized if not.
    """
    result = {}
    # For each local source, try normalized version first, then fallback to unnormalized if missing
    for src in LOCAL_SOURCES:
        # Try normalized file
        tried_files = []
        if os.path.exists(src):
            tried_files.append(src)
        # Fallback: try unnormalized version if normalized doesn't exist
        if not os.path.exists(src):
            # Remove _normalized.xlsx if present, else try original
            if src.endswith("_normalized.xlsx"):
                orig_src = src.replace("_normalized.xlsx", ".xlsx")
                if os.path.exists(orig_src):
                    tried_files.append(orig_src)
        for file_path in tried_files:
            try:
                df_local = pd.read_excel(file_path)
                df_local = normalize_headers(df_local)
            except Exception:
                continue
            if "Symbol" not in df_local.columns:
                continue
            df_local["Symbol"] = df_local["Symbol"].astype(str).str.upper().str.strip()
            row = df_local.loc[df_local["Symbol"] == symbol]
            if row.empty:
                continue
            for col in missing_cols:
                if col in row.columns and pd.notna(row.iloc[0][col]) and col not in result:
                    val = row.iloc[0][col]
                    if validate_value(col, val):
                        result[col] = val
            # If all missing_cols are filled, stop searching further files
            if all(col in result for col in missing_cols):
                return result
    return result


 # ==================== MAIN ====================
df = pd.read_excel(INPUT_FILE)
df = normalize_headers(df)
df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()

# Ensure the target columns exist
for col in TARGET_COLS:
    if col not in df.columns:
        df[col] = np.nan

updated_cells = 0

print(f"[INFO] File loaded: {INPUT_FILE}  |  Rows: {len(df)}")
for idx, row in df.iterrows():
    sym = row["Symbol"]
    missing = [c for c in TARGET_COLS if pd.isna(row.get(c))]
    if not missing:
        continue

    print(f"[{idx+1}/{len(df)}] {sym}: Missing {missing}")

    # Layer 0: Local files
    local_data = fetch_from_local_sources(sym, missing)
    layer_updates = {}
    newly_filled_fields = []
    if local_data:
        for k, v in local_data.items():
            if k in missing and pd.notna(v) and validate_value(k, v):
                df.at[idx, k] = v
                layer_updates[k] = ("Local", v)
                newly_filled_fields.append(k)
        if local_data:
            print(f"   [Filled from Local] {sym} → {list(local_data.keys())}")

    # Layer 0.5: AltWeb scraping
    still_missing = [c for c in TARGET_COLS if pd.isna(df.at[idx, c])]
    alt_data = {}
    if still_missing:
        alt_data = fetch_from_alt_sources(sym)
        if alt_data:
            for k, v in alt_data.items():
                if k in still_missing and pd.notna(v) and validate_value(k, v):
                    df.at[idx, k] = v
                    layer_updates[k] = ("AltWeb", v)
                    newly_filled_fields.append(k)
            if alt_data:
                print(f"   [Filled from AltWeb] {sym} → {list(alt_data.keys())}")

    # Layer 1: Yahoo JSON
    still_missing = [c for c in TARGET_COLS if pd.isna(df.at[idx, c])]
    js_data = fetch_from_yahoo_json(sym)
    if js_data:
        for k, v in js_data.items():
            if k in still_missing and pd.notna(v) and validate_value(k, v):
                df.at[idx, k] = v
                layer_updates[k] = ("YahooJSON", v)
                newly_filled_fields.append(k)

    # Layer 2: yfinance statements
    still_missing = [c for c in TARGET_COLS if pd.isna(df.at[idx, c])]
    if still_missing:
        yf_data = fetch_from_yfinance_statements(sym)
        if yf_data:
            for k, v in yf_data.items():
                if k in still_missing and pd.notna(v) and validate_value(k, v):
                    df.at[idx, k] = v
                    layer_updates[k] = ("yfinance_stmt", v)
                    newly_filled_fields.append(k)

    # Layer 3: SEC (only for Current Assets/Liabilities)
    still_missing = [c for c in TARGET_COLS if pd.isna(df.at[idx, c])]
    need_sec = any(x in still_missing for x in ["Current Assets", "Current Liabilities"])
    if need_sec:
        cik = sec_fetch_cik(sym)
        if cik:
            ca = df.at[idx, "Current Assets"]
            cl = df.at[idx, "Current Liabilities"]
            if pd.isna(ca):
                ca_val = sec_company_concept(cik, "AssetsCurrent")
                if ca_val is not None and validate_value("Current Assets", ca_val):
                    df.at[idx, "Current Assets"] = ca_val
                    layer_updates["Current Assets"] = ("SEC", ca_val)
                    newly_filled_fields.append("Current Assets")
            if pd.isna(cl):
                cl_val = sec_company_concept(cik, "LiabilitiesCurrent")
                if cl_val is not None and validate_value("Current Liabilities", cl_val):
                    df.at[idx, "Current Liabilities"] = cl_val
                    layer_updates["Current Liabilities"] = ("SEC", cl_val)
                    newly_filled_fields.append("Current Liabilities")

    # Derived: Invested Capital if missing and we have ingredients
    if pd.isna(df.at[idx, "Invested Capital"]):
        ic_now = compute_invested_capital(df.loc[idx])
        if ic_now is not None and validate_value("Invested Capital", ic_now):
            df.at[idx, "Invested Capital"] = ic_now
            layer_updates["Invested Capital"] = ("derived", ic_now)
            newly_filled_fields.append("Invested Capital")

    # Print status summary for this symbol
    if layer_updates:
        # nice compact print: field -> (source, value)
        parts = [f"{fld}←{src}:{'%.4g' % val if isinstance(val,(int,float)) else val}"
                 for fld,(src,val) in layer_updates.items()]
        print(f"   ↳ Filled: " + ", ".join(parts))
        # Print which of the specific fields were newly filled
        special_fields = [
            "EV", "EPS TTM", "Net Income (A)", "EBIT", "Operating Income", "COGS",
            "Total Assets", "Equity", "Long Term Debt", "Gross Margin"
        ]
        filled_special = [fld for fld in newly_filled_fields if fld in special_fields]
        if filled_special:
            print(f"   [INFO] Newly filled special fields: {filled_special}")
        updated_cells += len(layer_updates)
    else:
        print("   ↳ No updates from any source.")

    # Autosave checkpoint every 100 rows
    if (idx + 1) % 100 == 0:
        backup_path = f"{OUTPUT_FILE.replace('.xlsx', '')}_autosave_{idx+1}.xlsx"
        df.to_excel(backup_path, index=False)
        print(f"[AUTO-SAVE] Saved intermediate progress to {backup_path}")

    # polite pause
    time.sleep(SLEEP_BASE)

print(f"\n✅ Finished. Updated cells: {updated_cells}")
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Saved → {OUTPUT_FILE}")

# Print summary of filled counts per TARGET_COLS
filled_counts = {col: df[col].notna().sum() for col in TARGET_COLS}
print("\n[SUMMARY] Non-null counts per field:")
for k, v in filled_counts.items():
    print(f"   {k:<20} {v}/{len(df)}")