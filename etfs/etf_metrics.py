import yfinance as yf
import pandas as pd
import numpy as np

# List of ETF tickers (shortened here for readability — use your full list)
etf_tickers = [
    "ARKK", "GRID", "FAN", "PAVE", "TAN", "PHO", "PBW", "IBB", "DGRO", "ESGU",
    "ICLN", "INDA", "EWW", "EWT", "ITA", "BBCA", "XLU", "ESGV", "VWO", "VHT"
]

# Parameters
lookback_years = 3
risk_free_rate = 0.02  # 2% annualized
trading_days = 252
threshold_return = 0  # for Omega ratio


def get_metrics(etf):
    try:
        data = yf.download(etf, period=f"{lookback_years}y", interval="1d")['Adj Close']
        returns = data.pct_change().dropna()

        # Mean and downside deviation
        mean_return = np.mean(returns) * trading_days
        downside_std = np.std(returns[returns < threshold_return]) * np.sqrt(trading_days)

        # Sortino Ratio
        sortino = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else np.nan

        # Omega Ratio
        gains = returns[returns > threshold_return] - threshold_return
        losses = threshold_return - returns[returns < threshold_return]
        omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.nan

        return {
            "Ticker": etf,
            "Sortino Ratio": round(sortino, 4),
            "Omega Ratio": round(omega, 4)
        }
    except Exception as e:
        print(f"Error fetching {etf}: {e}")
        return {"Ticker": etf, "Sortino Ratio": np.nan, "Omega Ratio": np.nan}


# Run analysis
results = pd.DataFrame([get_metrics(ticker) for ticker in etf_tickers])
results_sorted = results.sort_values(by=["Sortino Ratio", "Omega Ratio"], ascending=False)

# Output
print(results_sorted.head(10))  # Top 10 ETFs
results_sorted.to_excel("ETF_Sortino_Omega_Ranked.xlsx", index=False)
print("\n✔ Done. Results saved to 'ETF_Sortino_Omega_Ranked.xlsx'")