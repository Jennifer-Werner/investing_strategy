# data.py
from __future__ import annotations
import re, time
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple

# --- NEW: ticker sanitizer ---
def _clean_ticker(x: str) -> str:
    # normalize to UPPER, strip whitespace, drop leading '$', keep only A-Z, 0-9, '.', '-'
    s = str(x).upper().strip()
    if s.startswith("$"):
        s = s[1:]
    s = re.sub(r"[^A-Z0-9\.\-]", "", s)
    return s

def load_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        df.columns = ["ticker"]
    if "sector" not in df.columns:
        df["sector"] = None
    df["ticker"] = df["ticker"].map(_clean_ticker)
    return df

def load_holdings(path: str) -> pd.Series:
    df = pd.read_csv(path)
    s = df.set_index("ticker")["weight"].astype(float)
    s.index = s.index.map(_clean_ticker)
    s = s / s.sum() if s.sum() > 0 else s
    return s

# --- NEW: robust chunked downloader with retries ---
def _extract_px_frame(raw: pd.DataFrame, chunk: List[str]) -> pd.DataFrame:
    """
    Convert any yfinance shape to a DataFrame of prices with columns=chunk tickers (some may be missing).
    Prefers 'Adj Close', falls back to 'Close', or first field available.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

    if isinstance(raw, pd.Series):
        # Single ticker; Series of prices already
        # yfinance usually returns a DataFrame even for single ticker, but handle just in case
        df = raw.to_frame(name=chunk[0])
        return df

    if isinstance(raw.columns, pd.MultiIndex):
        # group_by='ticker' gives MultiIndex (ticker, field)
        # Try Adj Close, then Close, else first field
        fields = list(dict.fromkeys(raw.columns.get_level_values(1)))
        field = "Adj Close" if "Adj Close" in fields else ("Close" if "Close" in fields else fields[0])
        pieces = {}
        for t in chunk:
            try:
                s = raw[(t, field)].astype(float)
                s.name = t
                pieces[t] = s
            except Exception:
                # missing one ticker in this chunk — leave it for later fill
                continue
        df = pd.DataFrame(pieces)
        return df

    # Flat DataFrame; assume this *is* the price frame (auto_adjust=True -> Close is adjusted)
    # If it’s a single column, name it the single ticker; otherwise columns should already match tickers.
    df = raw.copy()
    return df

def fetch_prices(
    tickers: List[str],
    start: str,
    end: str,
    chunk_size: int = 20,
    retries: int = 3,
    timeout: int = 30,
    sleep_between: float = 1.0,
) -> pd.DataFrame:
    """
    Robust price fetch:
      - Cleans symbols (drops leading '$', invalid chars)
      - Splits into chunks to avoid throttling
      - Retries with backoff on each chunk
      - Extracts prices from any yfinance shape
      - Returns a DataFrame with requested columns (missing = NaN), sorted index, ffilled
    """
    if not tickers:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

    req = [_clean_ticker(t) for t in tickers if t and t != "BONDS"]
    req = list(dict.fromkeys(req))  # de-dup, preserve order

    frames = []
    failed: List[str] = []
    for i in range(0, len(req), chunk_size):
        chunk = req[i:i + chunk_size]
        ok = False
        last_err = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    chunk, start=start, end=end,
                    auto_adjust=True, progress=False, group_by="ticker",
                    threads=False,  # threads can trigger throttling; keep False when chunking
                    timeout=timeout
                )
                df = _extract_px_frame(raw, chunk)
                frames.append(df)
                ok = True
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep_between * (attempt + 1))  # simple backoff
        if not ok:
            failed.extend(chunk)
            print(f"[warn] chunk failed after {retries} attempts: {chunk} | last_err={last_err}")

        # be nice to Yahoo
        time.sleep(sleep_between)

    # Merge all chunks
    px = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
    px = px.sort_index().ffill().dropna(how="all")

    # Guarantee columns for all requested real tickers (even if missing)
    for t in req:
        if t not in px.columns:
            px[t] = np.nan

    # Return in requested order (excluding synthetic BONDS)
    px = px.reindex(columns=req)

    if failed:
        print(f"[warn] {len(failed)} symbols failed to download (kept as NaN): {failed[:10]}{' ...' if len(failed)>10 else ''}")

    return px

def fetch_dividend_yields(tickers: List[str]) -> pd.Series:
    res: Dict[str, float] = {}
    for t in (tickers or []):
        t_clean = _clean_ticker(t)
        if t_clean == "BONDS":
            res[t_clean] = 0.0425
            continue
        try:
            tk = yf.Ticker(t_clean)
            info = tk.info or {}
            yld = info.get("dividendYield", None)
            if yld is None:
                divs = tk.dividends
                price_hist = tk.history(period="1y")
                if len(divs) > 0 and len(price_hist) > 0:
                    ttm = float(divs.tail(252).sum())
                    last = float(price_hist["Close"].iloc[-1])
                    yld = (ttm / last) if last > 0 else 0.0
                else:
                    yld = 0.0
            res[t_clean] = float(yld)
        except Exception:
            res[t_clean] = 0.0
    # preserve original order, mapped through cleaner
    out = pd.Series({ _clean_ticker(t): res.get(_clean_ticker(t), 0.0) for t in tickers })
    return out

def load_trade_filter_from_excel(
    path: str,
    sheet: str = "Complete_Data",
    min_dollar_vol_m: float = 20.0,
    require_can_trade: bool = True
) -> pd.Index:
    """
    Returns an index of tickers that pass basic tradability filters.
    Expects columns: 'ticker', 'can_trade', 'avg_dollar_volume_m'
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns:
            return pd.Index([])
        df["ticker"] = df["ticker"].map(_clean_ticker)
        if require_can_trade and "can_trade" in df.columns:
            df = df[df["can_trade"].astype(str).str.lower().isin(["1","true","yes","y"])]
        if "avg_dollar_volume_m" in df.columns:
            df = df[df["avg_dollar_volume_m"].astype(float) >= float(min_dollar_vol_m)]
        return pd.Index(df["ticker"].unique())
    except Exception:
        return pd.Index([])