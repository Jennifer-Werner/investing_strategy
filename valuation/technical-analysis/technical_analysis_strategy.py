"""
technical_analysis_strategy.py
=================================

This module provides a high‑level implementation of a technical analysis
engine designed to help determine entry and exit points for
equity positions.  The framework focuses on *trend following* using a
10‑month simple moving average (SMA) filter, breakout detection with
volume confirmation, and momentum reversal cues from the relative
strength index (RSI).  It also incorporates a simplified version of
White’s Reality Check to guard against data‑snooping and spurious
breakouts.  The design follows insights from academic and practitioner
research:

* **Trend following via a 10‑month SMA:**  Mebane Faber’s tactical asset
  allocation model buys an asset when its monthly price sits above its
  10‑month SMA and moves to cash when the price falls below the SMA【59168652912994†L459-L463】.
  This simple rule improves risk‑adjusted returns and reduces drawdowns
  relative to a buy‑and‑hold approach【59168652912994†L480-L523】.

* **Volume confirmation:**  Trading volume gauges the conviction behind
  price moves.  A price breakout on high volume suggests broad market
  participation and increases the likelihood of follow‑through, while
  breakouts on low volume are more likely to be false signals【499583928440180†L360-L370】.
  Volume spikes during rallies or sell‑offs can also indicate the
  strength of a trend【499583928440180†L360-L440】.

* **RSI for reversal warnings:**  The relative strength index measures
  momentum by comparing average gains and losses.  Stocks are
  generally considered overbought above 70 and oversold below 30; RSI
  divergences (price making higher highs while RSI makes lower highs,
  or vice versa) can foreshadow trend reversals【256304209115562†L190-L241】.  In this
  framework the RSI is used only as a secondary filter—signals are
  ignored unless confirmed by the trend and volume filters.

* **White’s Reality Check (simplified):**  Data‑snooping can make a
  poorly performing rule appear profitable when tested on a single
  sample【534647956523512†L108-L115】.  White’s Reality Check (WRC) uses
  bootstrapping to assess whether a trading rule’s profits are
  statistically significant relative to a null of zero mean return.  A
  simplified bootstrap test is provided here to evaluate breakout rules
  before deploying them.

The module exposes a `TechnicalAnalysisStrategy` class that accepts a
time series of prices and volumes and outputs trading signals.  It is
designed to integrate easily into larger projects: no plotting or
external I/O occurs inside the core functions, and users can feed in
their own DataFrames or fetch data with yfinance.  Signals are
produced in a pandas DataFrame with boolean entries indicating long
positions and flags highlighting entry and exit events.

Note: This code provides educational insight into technical analysis
and risk management.  It is **not** financial advice.  Users should
test and adjust all parameters (lookback periods, thresholds, etc.)
according to their own objectives and risk tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Iterable


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute a simple moving average (SMA) over a rolling window.

    Parameters
    ----------
    series : pd.Series
        Time series of values (e.g., closing prices).
    window : int
        Number of periods over which to compute the average.

    Returns
    -------
    pd.Series
        Series of the SMA with the same index as ``series``.
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    The RSI measures momentum by comparing the magnitude of recent gains to
    recent losses.  Values range from 0 to 100.  It can help identify
    overbought/oversold conditions and divergences【256304209115562†L206-L241】.

    Parameters
    ----------
    series : pd.Series
        Price series from which to compute RSI.
    window : int, optional
        Lookback window length, by default 14 (standard in many
        implementations).

    Returns
    -------
    pd.Series
        RSI values.
    """
    delta = series.diff()
    # Separate positive and negative price changes
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Compute the exponentially weighted moving averages of gains and losses
    # Use Wilder's smoothing to emulate typical RSI calculation
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()

    # Avoid division by zero
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def block_bootstrap(series: np.ndarray, block_size: int, n_samples: int) -> np.ndarray:
    """Generate bootstrap samples from a time series using non‑overlapping blocks.

    This helper function is used for the simplified White's Reality Check.
    Blocks preserve some of the serial dependence structure present in the
    original returns sequence.  Each bootstrap sample has length equal to
    ``len(series)``.

    Parameters
    ----------
    series : np.ndarray
        Original time series data (e.g., returns).
    block_size : int
        Length of each block drawn with replacement.
    n_samples : int
        Number of bootstrap resamples to generate.

    Returns
    -------
    np.ndarray
        An array of shape (n_samples, len(series)) containing bootstrap
        samples.
    """
    n = len(series)
    # Compute the number of blocks needed to fill a bootstrap sample
    n_blocks = int(np.ceil(n / block_size))
    # Precompute block start indices
    block_starts = np.arange(0, n - block_size + 1)
    samples = np.empty((n_samples, n), dtype=float)
    for i in range(n_samples):
        # Draw random starting indices for each block
        idx = np.random.randint(low=0, high=len(block_starts), size=n_blocks)
        # Collect the blocks and concatenate
        resampled = np.concatenate([series[start:start + block_size] for start in block_starts[idx]])
        # Truncate to length n
        samples[i, :] = resampled[:n]
    return samples


def reality_check(returns: pd.Series, block_size: int = 10, n_bootstrap: int = 500, alpha: float = 0.10) -> Tuple[bool, float]:
    """Perform a simplified White's Reality Check on a series of trading returns.

    The test evaluates whether the mean of the observed returns is
    significantly greater than zero under a null hypothesis of no predictability.
    It uses a block bootstrap to account for serial correlation【534647956523512†L108-L115】.

    Parameters
    ----------
    returns : pd.Series
        Realized returns from the trading rule.
    block_size : int, optional
        Length of blocks for the bootstrap, by default 10.
    n_bootstrap : int, optional
        Number of bootstrap samples, by default 500.  Larger values yield
        more precise p‑values at greater computational cost.
    alpha : float, optional
        Significance level.  Returns True if the rule is significant at
        this level.

    Returns
    -------
    Tuple[bool, float]
        A tuple ``(significant, p_value)`` indicating whether the trading
        rule is statistically significant and the associated p‑value.
    """
    # Drop NaN values
    ret = returns.dropna().to_numpy()
    if len(ret) < 20:
        # Not enough data to perform a meaningful test
        return False, np.nan
    observed_mean = ret.mean()
    # Generate bootstrap samples
    samples = block_bootstrap(ret, block_size=block_size, n_samples=n_bootstrap)
    # Compute mean for each bootstrap sample
    bootstrap_means = samples.mean(axis=1)
    # p‑value: proportion of bootstrap means greater than or equal to observed mean
    p_value = (bootstrap_means >= observed_mean).mean()
    return p_value < alpha, p_value


@dataclass
class TechnicalAnalysisStrategy:
    """A class implementing a multifactor technical analysis strategy.

    Parameters
    ----------
    breakout_lookback : int
        Number of days to look back when determining a breakout.  A common
        choice is 252 trading days (~1 year).
    sma_window : int
        Window length in trading days for the long‑term SMA.  A 200‑day
        window approximates a 10‑month SMA【59168652912994†L459-L463】.
    rsi_window : int
        Lookback window for the RSI (default 14 days).
    volume_window : int
        Window length for the moving average of volume used to detect
        unusual activity.
    volume_multiplier : float
        Minimum ratio of current volume to average volume required to
        confirm a breakout【499583928440180†L360-L370】.
    test_breakout_rule : bool
        Whether to apply the simplified White’s Reality Check to the breakout
        rule.  When ``True``, the strategy computes historical returns from
        breakout signals during initialization and deactivates the rule if it
        fails the test.
    bootstrap_samples : int
        Number of bootstrap samples for the reality check.
    bootstrap_block : int
        Block size for the bootstrap.
    bootstrap_alpha : float
        Significance level for the reality check.
    """

    breakout_lookback: int = 252
    sma_window: int = 200
    rsi_window: int = 14
    volume_window: int = 20
    volume_multiplier: float = 1.5
    test_breakout_rule: bool = True
    bootstrap_samples: int = 500
    bootstrap_block: int = 10
    bootstrap_alpha: float = 0.10

    # Internal flag set after performing the reality check
    breakout_significant: bool = field(init=False, default=True)
    breakout_pvalue: Optional[float] = field(init=False, default=None)

    # Determines whether breakout signals are required for entry.  When
    # ``True``, entries only occur if a volume‑confirmed breakout is present
    # (and passes the reality check).  When ``False``, breakout signals are
    # ignored and the strategy can enter based solely on the trend and RSI
    # filters.  This is useful for long‑term investors who wish to time
    # entries on pullbacks rather than waiting for breakouts.
    use_breakout_rule: bool = True

    def fit_breakout_rule(self, price: pd.Series, volume: pd.Series) -> None:
        """Evaluate the breakout rule on historical data and perform WRC.

        This method identifies historical breakout trades based on the
        configured lookback period and volume filter, computes the returns from
        those trades (holding them for a fixed horizon or until the breakout
        fails), and then applies the reality check.  It sets
        ``breakout_significant`` and ``breakout_pvalue`` accordingly.

        Parameters
        ----------
        price : pd.Series
            Daily closing prices indexed by datetime.
        volume : pd.Series
            Daily trading volumes indexed by datetime.
        """
        # Ensure inputs are pandas Series with matching indices.  If price or volume
        # is not a Series (e.g., a scalar or array), convert it to one.  If there
        # are fewer than two observations, skip evaluation.
        if not isinstance(price, pd.Series) or not isinstance(volume, pd.Series):
            try:
                price = pd.Series(price)
                volume = pd.Series(volume)
            except Exception:
                # Fallback: not enough data
                self.breakout_significant = False
                self.breakout_pvalue = np.nan
                return
        # Drop missing data and check length
        df = pd.DataFrame({"close": price, "volume": volume}).dropna()
        if len(df) < max(self.breakout_lookback, self.volume_window) + 1:
            # Insufficient data to compute lookback highs and averages
            self.breakout_significant = False
            self.breakout_pvalue = np.nan
            return
        # Compute rolling maximum of past prices (exclude current day)
        df["lookback_high"] = df["close"].shift(1).rolling(window=self.breakout_lookback).max()
        # Compute volume moving average
        df["avg_volume"] = df["volume"].rolling(window=self.volume_window).mean()
        # Identify breakout points with volume confirmation
        df["breakout"] = (df["close"] > df["lookback_high"]) & (df["volume"] > self.volume_multiplier * df["avg_volume"])
        # Determine simple returns one month (20 trading days) after the breakout
        holding_period = 20
        returns = []
        indices = df.index[df["breakout"] == True]
        for idx in indices:
            entry_price = df.loc[idx, "close"]
            # Determine exit price after holding_period or at end of data
            try:
                exit_idx = df.index.get_loc(idx) + holding_period
            except KeyError:
                continue
            if exit_idx < len(df):
                exit_price = df.iloc[exit_idx]["close"]
                returns.append((exit_price - entry_price) / entry_price)
        if returns and self.test_breakout_rule:
            returns_series = pd.Series(returns)
            sig, pval = reality_check(returns_series, block_size=self.bootstrap_block,
                                      n_bootstrap=self.bootstrap_samples, alpha=self.bootstrap_alpha)
            self.breakout_significant = sig
            self.breakout_pvalue = pval
        else:
            # Not enough breakout trades to evaluate; deactivate the rule
            self.breakout_significant = False
            self.breakout_pvalue = np.nan

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate entry/exit signals for a single stock based on price and volume.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing at least two columns: ``close`` and ``volume``.
            The index should be datetime‑like and sorted in ascending order.

        Returns
        -------
        pd.DataFrame
            A copy of the input ``data`` with additional columns:

            * ``trend_up`` – boolean flag indicating the price is above its
              long‑term SMA (trend is bullish).
            * ``breakout_signal`` – boolean flag for volume‑confirmed breakouts.
            * ``rsi`` – computed RSI values.
            * ``rsi_overbought`` – RSI ≥ 70.
            * ``rsi_oversold`` – RSI ≤ 30.
            * ``position`` – binary series (1 for long, 0 for cash) after
              applying the combined rules.
            * ``entry`` – True where a new long position is initiated.
            * ``exit`` – True where a long position is closed.
            * ``signal_reason`` – text describing why an entry or exit occurred.
        """
        # Make a defensive copy and ensure datetime index
        df = data.copy().dropna(subset=["close", "volume"])
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")
        # Compute long‑term SMA and trend flag
        df["long_sma"] = compute_sma(df["close"], window=self.sma_window)
        df["trend_up"] = df["close"] > df["long_sma"]
        # Compute RSI
        df["rsi"] = compute_rsi(df["close"], window=self.rsi_window)
        df["rsi_overbought"] = df["rsi"] >= 70
        df["rsi_oversold"] = df["rsi"] <= 30
        # Compute volume average and breakout signals
        df["avg_volume"] = df["volume"].rolling(window=self.volume_window).mean()
        # Use the same lookback definition as in fit_breakout_rule but compute on the fly
        df["lookback_high"] = df["close"].shift(1).rolling(window=self.breakout_lookback).max()
        df["breakout_raw"] = df["close"] > df["lookback_high"]
        df["volume_confirmation"] = df["volume"] > self.volume_multiplier * df["avg_volume"]
        df["breakout_signal"] = df["breakout_raw"] & df["volume_confirmation"]
        # If the breakout rule is deemed insignificant, disable it
        if not self.breakout_significant:
            df["breakout_signal"] = False
        # Initialize position series
        position = np.zeros(len(df), dtype=int)
        signal_reason = [None] * len(df)
        # Flags to manage open position
        in_position = False
        last_entry_index: Optional[int] = None
        # Loop through rows to assign positions
        for i in range(len(df)):
            price = df.iloc[i]["close"]
            trend = df.iloc[i]["trend_up"]
            breakout = df.iloc[i]["breakout_signal"]
            rsi_overbought = df.iloc[i]["rsi_overbought"]
            rsi_oversold = df.iloc[i]["rsi_oversold"]
            # Entry logic
            if not in_position:
                # Determine whether breakout is required.  If
                # ``use_breakout_rule`` is False, we treat the breakout
                # condition as always satisfied.  Otherwise, we require
                # ``breakout_signal`` to be True.  Entries also require
                # the long‑term trend to be up and RSI not overbought.
                effective_breakout = breakout if self.use_breakout_rule else True
                if trend and effective_breakout and not rsi_overbought:
                    in_position = True
                    position[i] = 1
                    last_entry_index = i
                    # Determine entry reason
                    if breakout and self.use_breakout_rule:
                        reason = "trend_up & breakout"
                    else:
                        reason = "trend_up"
                    signal_reason[i] = f"Entry: {reason}"
                else:
                    position[i] = 0
            else:
                # Already in a position
                position[i] = 1
                # Exit conditions
                exit_condition = False
                reasons: List[str] = []
                # Trend reversal: price falls below long SMA
                if not trend:
                    exit_condition = True
                    reasons.append("price_below_SMA")
                # RSI indicates overbought and subsequently crosses back below 70
                # Use cross detection: previous overbought and current < 70
                if i > 0:
                    prev_overbought = df.iloc[i - 1]["rsi_overbought"]
                    if prev_overbought and not rsi_overbought:
                        exit_condition = True
                        reasons.append("RSI_overbought_cross")
                # Loss of volume momentum: price is rising but volume is declining
                # (Possible weakening of trend).  For simplicity, check if
                # current volume < average volume during the holding period.
                if last_entry_index is not None and i - last_entry_index >= 5:
                    # Compute average volume since entry
                    avg_vol_since_entry = df.iloc[last_entry_index:i + 1]["volume"].mean()
                    if df.iloc[i]["volume"] < avg_vol_since_entry:
                        exit_condition = True
                        reasons.append("volume_dropping")
                # Exit if condition triggered
                if exit_condition:
                    in_position = False
                    position[i] = 0
                    signal_reason[i] = "Exit: " + ", ".join(reasons)
                    last_entry_index = None
        # Build result DataFrame
        df["position"] = position
        df["entry"] = (df["position"].diff().fillna(df["position"]).astype(int) == 1)
        df["exit"] = (df["position"].diff().fillna(0).astype(int) == -1)
        df["signal_reason"] = signal_reason
        return df


def fetch_yfinance(ticker: str, start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical price and volume data using yfinance.

    This convenience function allows quick retrieval of daily data to test
    the strategy.  It requires the optional `yfinance` package.  If
    `yfinance` is unavailable or the internet is disabled, an exception
    will be raised.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'AAPL').
    start : str, optional
        Start date in 'YYYY-MM-DD' format, by default '2010-01-01'.
    end : str, optional
        End date in 'YYYY-MM-DD' format.  If None, the current date is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['open', 'high', 'low', 'close', 'adj close', 'volume'].
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance must be installed to fetch data.") from e
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    return df


__all__ = [
    "TechnicalAnalysisStrategy",
    "compute_sma",
    "compute_rsi",
    "reality_check",
    "fetch_yfinance",
]