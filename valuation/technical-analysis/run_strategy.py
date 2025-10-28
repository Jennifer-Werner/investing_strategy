"""
run_strategy.py
===============

This script makes use of the `TechnicalAnalysisStrategy` class defined in
`technical_analysis_strategy.py`.  It prompts the user to input one or
more ticker symbols (separated by commas), fetches historical daily data
for those tickers using the `yfinance` library, fits the breakout rule
on the available data, generates entry/exit signals, and outputs both
visualizations and a spreadsheet summarizing the results.  The
visualization for each ticker consists of a simple price chart with the
long‑term SMA overlaid and markers indicating entry and exit points.

The resulting spreadsheet is written in Excel format (.xlsx) with one
sheet per ticker.  Each sheet contains the date, closing price,
volume, indicators (long‑term SMA, RSI, etc.), positions, and entry/exit
flags.  The charts are saved as PNG files in the current working
directory.

Usage:

    python run_strategy.py

The script will prompt for tickers via standard input.  Enter a
comma‑separated list, e.g.:

    AAPL,MSFT,GOOG

Notes:

* The script relies on the `yfinance` package to retrieve data and
  `matplotlib` to produce charts.  Both must be installed in your
  environment.
* The script writes output files in the directory from which it is
  executed.  Existing files with the same names will be overwritten.

"""

import sys
import os
import datetime
from typing import List, Tuple

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # Use non‑interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib must be installed to run this script.")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance must be installed to fetch market data.")

from technical_analysis_strategy import TechnicalAnalysisStrategy


def prompt_for_tickers() -> List[str]:
    """Prompt the user for the number of tickers and then each ticker on a new line.

    The user is first asked to specify how many ticker symbols they wish to analyze.  The
    function then reads that many tickers from standard input, one per line, and
    returns a list of uppercase ticker strings.
    """
    while True:
        try:
            n_tickers_str = input("Enter the number of tickers to analyze: ")
            n_tickers = int(n_tickers_str)
            if n_tickers <= 0:
                print("Please enter a positive integer for the number of tickers.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
    tickers: List[str] = []
    # Collect tickers until the desired number is reached.  If the user pastes
    # multiple tickers separated by whitespace or commas on a single line, they
    # will all be consumed sequentially.
    while len(tickers) < n_tickers:
        remaining = n_tickers - len(tickers)
        prompt = f"Enter ticker {len(tickers) + 1}: " if remaining == 1 else f"Enter up to {remaining} tickers (separated by spaces or commas): "
        line = input(prompt).strip()
        if not line:
            continue
        # Split input by spaces or commas
        tokens = [token.strip().upper() for token in line.replace(',', ' ').split() if token.strip()]
        for token in tokens:
            if len(tickers) < n_tickers:
                tickers.append(token)
            else:
                break
    return tickers


def fetch_data(ticker: str, start_date: str = "2015-01-01", end_date: str = None, debug: bool = False) -> pd.DataFrame:
    """Download historical price and volume data for a single ticker.

    Parameters
    ----------
    ticker : str
        The ticker symbol.
    start_date : str
        The start date for the historical data.
    end_date : str
        The end date.  If None, defaults to today.
    debug : bool, optional
        If True, print debug information about each download attempt.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: open, high, low, close, adj_close, volume.
    """
    """
    Attempt to download historical data for ``ticker`` using yfinance.  If the
    default ``yf.download`` call fails or returns no data, fall back to
    ``yf.Ticker().history`` which sometimes succeeds when the bulk downloader
    times out.  Returns a DataFrame with columns 'open', 'high', 'low', 'close',
    'adj_close', and 'volume'.
    """
    if debug:
        print(f"\n[DEBUG] Fetching data for {ticker}")
    # First attempt: date‑based download
    df = pd.DataFrame()
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         progress=False, threads=False, timeout=30)
        if debug:
            print(f"[DEBUG] date‑range download rows: {len(df)}")
    except Exception as e:
        if debug:
            print(f"[DEBUG] date‑range download exception: {e}")
        df = pd.DataFrame()
    # Second attempt: period‑based download
    if df is None or df.empty:
        try:
            df = yf.download(ticker, period="max",
                             progress=False, threads=False, timeout=30)
            if debug:
                print(f"[DEBUG] period='max' download rows: {len(df)}")
        except Exception as e:
            if debug:
                print(f"[DEBUG] period='max' download exception: {e}")
            df = pd.DataFrame()
    # Third attempt: Ticker.history API
    if df is None or df.empty:
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date,
                                    auto_adjust=False, timeout=30)
            if debug:
                print(f"[DEBUG] history(start,end) rows: {len(df)}")
            if df is None or df.empty:
                df = ticker_obj.history(period="max",
                                        auto_adjust=False, timeout=30)
                if debug:
                    print(f"[DEBUG] history(period='max') rows: {len(df)}")
        except Exception as e:
            if debug:
                print(f"[DEBUG] history API exception: {e}")
            df = pd.DataFrame()
    # If all attempts failed, return empty DataFrame
    if df is None or df.empty:
        if debug:
            print(f"[DEBUG] no data retrieved for {ticker}")
        return pd.DataFrame()
    # Flatten any MultiIndex columns returned by yfinance by selecting the level
    # that contains field names such as 'open', 'high', 'close', etc.
    if isinstance(df.columns, pd.MultiIndex):
        field_level = None
        for level in range(df.columns.nlevels):
            vals = df.columns.get_level_values(level)
            # Normalize candidate values for comparison
            normalized = [str(v).lower().replace(' ', '').replace('_','') for v in vals]
            if any(v in {"open", "high", "low", "close", "adjclose", "volume"} for v in normalized):
                field_level = level
                break
        if field_level is None:
            # Default to first level if we couldn't find field names
            field_level = 0
        df.columns = df.columns.get_level_values(field_level)
    # Standardize column names: lower case and replace spaces with underscores
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
    # Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def plot_signals(ticker: str, df: pd.DataFrame, output_dir: str) -> None:
    """Plot price, long‑term SMA, and entry/exit points for a ticker.

    Parameters
    ----------
    ticker : str
        The ticker symbol.
    df : pd.DataFrame
        DataFrame with signal annotations returned from
        `TechnicalAnalysisStrategy.generate_signals`.
    output_dir : str
        Directory where the plot image will be saved.
    """
    # Use a non‑interactive backend to avoid popping up windows during execution
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot closing price and long‑term SMA
    ax.plot(df.index, df["close"], label="Close", color="blue", linewidth=1)
    # Use the SMA window from the strategy instance (defined globally in main)
    ax.plot(df.index, df["long_sma"], label=f"{strategy.sma_window}-day SMA", color="orange", linewidth=1)
    # Identify entry and exit points
    entries = df[df["entry"]]
    exits = df[df["exit"]]
    ax.scatter(entries.index, entries["close"], marker="^", color="green", label="Entry", zorder=5)
    ax.scatter(exits.index, exits["close"], marker="v", color="red", label="Exit", zorder=5)
    ax.set_title(f"{ticker} Price and Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    # Save the plot as a PNG in the specified output directory
    filename = os.path.join(output_dir, f"{ticker}_signals.png")
    fig.savefig(filename)
    plt.close(fig)


def save_signals_to_excel(signals_dict: dict, output_path: str) -> None:
    """Save all signals DataFrames to a single Excel workbook.

    Each ticker's data is written to a separate sheet.
    """
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for ticker, df in signals_dict.items():
            df.to_excel(writer, sheet_name=ticker)

def summarize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of entry and exit trades.

    This helper takes the signals DataFrame and extracts the dates where entry and exit
    flags are set.  It returns a new DataFrame with columns for the entry date,
    exit date, and holding period (in trading days).  If there are unmatched
    entries (i.e., an open position at the end of the data), the exit date is
    reported as NaT.

    Parameters
    ----------
    df : pd.DataFrame
        Signals DataFrame containing boolean 'entry' and 'exit' columns.

    Returns
    -------
    pd.DataFrame
        Summary of trades with columns ['entry_date', 'exit_date', 'holding_period'].
    """
    entries = df.index[df["entry"]].to_list()
    exits = df.index[df["exit"]].to_list()
    summary_rows = []
    for i, entry_date in enumerate(entries):
        exit_date = exits[i] if i < len(exits) else pd.NaT
        holding_period = (exit_date - entry_date).days if pd.notnull(exit_date) else None
        summary_rows.append({"entry_date": entry_date, "exit_date": exit_date, "holding_period": holding_period})
    return pd.DataFrame(summary_rows)

def backtest_signals(df: pd.DataFrame, entry_exit_summary: pd.DataFrame) -> Tuple[float, float, int]:
    """Compute the cumulative return of the strategy and compare to buy‑and‑hold.

    Parameters
    ----------
    df : pd.DataFrame
        The signals DataFrame with a 'close' column.
    entry_exit_summary : pd.DataFrame
        Output of summarize_trades() for this ticker.

    Returns
    -------
    Tuple[float, float, int]
        A tuple of (strategy_return, buy_and_hold_return, n_trades), where
        each return is expressed as a decimal (e.g., 0.25 for +25%).
    """
    if df.empty or entry_exit_summary.empty:
        # No trades; return zero strategy return and buy‑and‑hold return
        buy_and_hold = df['close'].iloc[-1] / df['close'].iloc[0] - 1 if not df.empty else 0.0
        return 0.0, buy_and_hold, 0
    capital = 1.0  # start with 1 unit of capital
    for row in entry_exit_summary.itertuples(index=False):
        entry_date = row.entry_date
        exit_date = row.exit_date
        entry_price = df.loc[entry_date, 'close']
        if pd.isna(exit_date):
            # Open trade: exit at last close
            exit_price = df['close'].iloc[-1]
        else:
            exit_price = df.loc[exit_date, 'close']
        # Multiply capital by trade return + 1
        capital *= (exit_price / entry_price)
    strategy_return = capital - 1
    buy_and_hold_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
    return strategy_return, buy_and_hold_return, len(entry_exit_summary)


if __name__ == "__main__":
    # Prompt the user for tickers
    tickers = prompt_for_tickers()
    if not tickers:
        print("No tickers provided. Exiting.")
        sys.exit(0)
    # Set up output directory
    output_dir = os.path.abspath("./strategy_outputs")
    os.makedirs(output_dir, exist_ok=True)
    # Initialize strategy with relaxed breakout requirements.  We disable the
    # reality check (test_breakout_rule=False) and lower the volume multiplier
    # and breakout lookback window to make it easier to generate breakouts.
    strategy = TechnicalAnalysisStrategy(
        sma_window=200,              # 10‑month SMA
        breakout_lookback=126,       # 6‑month breakout window
        volume_window=20,
        volume_multiplier=1.1,       # require volume only 10% above average
        test_breakout_rule=True,     # keep the White's Reality Check enabled
        use_breakout_rule=False      # allow entries without breakouts (pullbacks in uptrend)
    )
    # Set a multiplier for ATR‑based stop losses.  A value of 2.0 means the
    # stop loss is placed two ATRs below the entry price.
    stop_loss_multiplier = 2.0
    # Dictionary to store results for each ticker
    signals_dict = {}
    # Fetch data and generate signals
    for ticker in tickers:
        print(f"Processing {ticker}...")
        # Fetch historical data
        start_date = "2015-01-01"
        end_date = datetime.date.today().isoformat()
        try:
            # Pass debug=True to print diagnostic information when fetching data
            data = fetch_data(ticker, start_date=start_date, end_date=end_date, debug=True)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue
        if data.empty:
            print(f"No data retrieved for {ticker}. Skipping.")
            continue
        # Compute Average True Range (ATR) for stop‑loss calculation.  ATR is
        # the rolling average of the true range, which is the maximum of:
        #   * current high minus current low
        #   * absolute difference between current high and previous close
        #   * absolute difference between current low and previous close
        # We'll use a 14‑day window, a common choice among traders.
        if {'high', 'low', 'close'}.issubset(data.columns):
            high = data['high']
            low = data['low']
            close_series = data['close'] if 'close' in data.columns else data.get('adj_close')
            prev_close = close_series.shift(1)
            true_range = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            data['atr'] = true_range.rolling(window=14, min_periods=14).mean()
        else:
            data['atr'] = pd.NA
        # Determine which column to use for prices
        price_col = None
        if "close" in data.columns:
            price_col = "close"
        elif "adj_close" in data.columns:
            # Some downloads only provide adjusted close
            price_col = "adj_close"
        # Ensure both price and volume columns exist
        if price_col is None or "volume" not in data.columns:
            print(f"Required columns missing for {ticker}. Available columns: {list(data.columns)}. Skipping.")
            continue
        # Fit the breakout rule using the entire series
        try:
            strategy.fit_breakout_rule(price=data[price_col], volume=data["volume"])
        except Exception as e:
            print(f"Error fitting breakout rule for {ticker}: {e}")
            continue
        # Generate signals
        # We rename the price column back to 'close' so the strategy sees the
        # expected column name.  We also forward the ATR column so that
        # stop‑loss calculations can be made after signal generation.
        input_df = data[[price_col, "volume", "atr"]].rename(columns={price_col: "close"})
        signals = strategy.generate_signals(input_df)
        # Store signals for output
        signals_dict[ticker] = signals
        # Summarize trades and print to console
        trade_summary = summarize_trades(signals)
        if not trade_summary.empty:
            print(f"\nSummary of trades for {ticker}:")
            # Compute stop loss column in signals for each entry
            signals['stop_loss'] = pd.NA
            for row in trade_summary.itertuples(index=False):
                entry_date = row.entry_date
                exit_date = row.exit_date
                entry_price = signals.loc[entry_date, 'close']
                atr_value = signals.loc[entry_date, 'atr']
                stop_loss = entry_price - stop_loss_multiplier * atr_value if pd.notnull(atr_value) else pd.NA
                signals.loc[entry_date, 'stop_loss'] = stop_loss
                exit_str = exit_date.strftime("%Y-%m-%d") if pd.notnull(exit_date) else "--"
                hp_str = str(row.holding_period) if row.holding_period is not None else "open"
                sl_str = f"{stop_loss:.2f}" if pd.notnull(stop_loss) else "--"
                print(f"  Entry: {entry_date.strftime('%Y-%m-%d')}, Exit: {exit_str}, Holding Period: {hp_str} days, Stop Loss: {sl_str}")
            # Backtest the strategy
            strat_ret, bh_ret, ntr = backtest_signals(signals, trade_summary)
            print(f"  Trades: {ntr}, Strategy Return: {strat_ret:.2%}, Buy‑and‑Hold Return: {bh_ret:.2%}")
        else:
            print(f"No trades generated for {ticker}.")
        # Plot signals
        try:
            plot_signals(ticker, signals, output_dir)
        except Exception as e:
            print(f"Error plotting signals for {ticker}: {e}")
    # Save all signals to Excel workbook
    if signals_dict:
        excel_path = os.path.join(output_dir, "technical_analysis_signals.xlsx")
        save_signals_to_excel(signals_dict, excel_path)
        print(f"Results saved to {excel_path} and individual PNG charts in {output_dir}.")
    else:
        print("No results to save.")