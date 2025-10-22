import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# --- SETTINGS ---
tickers = ['GRID', 'ITA', 'SMH', 'QQQ', 'IWF', 'XLK', 'VUG']
base_ticker = 'VOO'
start_date = '2018-01-01'
end_date = '2024-10-01'
num_portfolios = 3000

# --- DOWNLOAD DATA ---
all_tickers = tickers + [base_ticker]
price_data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=False)

expense_ratios = {
    'GRID': 0.0065,
    'ITA': 0.0040,
    'SMH': 0.0035,
    'QQQ': 0.0020,
    'IWF': 0.0019,
    'XLK': 0.0010,
    'VUG': 0.0004,
    'VOO': 0.0003
}

# Extract Adj Close first
price_data = price_data['Adj Close']

# Check which tickers actually have data
available_tickers = [ticker for ticker in all_tickers if ticker in price_data.columns]
missing_tickers = set(all_tickers) - set(available_tickers)

if missing_tickers:
    print(f"Warning: The following tickers had no data: {missing_tickers}")
    # Remove missing tickers from your lists
    tickers = [t for t in tickers if t in available_tickers]
    all_tickers = [t for t in all_tickers if t in available_tickers]

# Keep only available tickers
price_data = price_data[available_tickers]
returns = price_data.pct_change().dropna()

# --- HELPER FUNCTIONS ---
def get_portfolio_performance(weights, returns_df):
    portfolio_returns = (returns_df * weights).sum(axis=1)
    mean_return = np.mean(portfolio_returns) * 252
    std_dev = np.std(portfolio_returns) * np.sqrt(252)
    downside_returns = portfolio_returns[portfolio_returns < 0]

    # Adjust return by weighted average expense ratio
    weighted_expense_ratio = sum(weights[i] * expense_ratios[i] for i in returns_df.columns if i in weights)
    net_return = mean_return - weighted_expense_ratio

    sortino = net_return / (np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 0 else 0
    sharpe = net_return / std_dev if std_dev != 0 else 0

    return net_return, std_dev, sharpe, sortino


# --- SIMULATE PORTFOLIOS ---
results = []
for _ in range(num_portfolios):
    random_weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0] * 0.4
    weights = dict(zip(tickers, random_weights))
    weights[base_ticker] = 0.6  # Lock VOO at 60%

    full_weights = np.array([weights.get(t, 0) for t in all_tickers])
    ret, vol, sharpe, sortino = get_portfolio_performance(full_weights, returns[all_tickers])

    results.append({
        'Returns': ret,
        'Volatility': vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        **weights
    })

df = pd.DataFrame(results)

# --- MACHINE LEARNING MODEL ---
features = df[tickers]
target = df['Sortino']  # You can change to 'Sharpe' or 'Returns'

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# --- OPTIMAL PORTFOLIO ---
best_weights = features.iloc[model.predict(features).argmax()]
best_weights[base_ticker] = 0.6
print("\n📈 Recommended Allocation (Including VOO):")
print(best_weights.round(4) * 100)  # in %
# Convert weights to full format (including all tickers)
recommended_weights = np.array([best_weights.get(t, 0) for t in all_tickers])

# Recalculate metrics for the recommended portfolio
ret, vol, sharpe, sortino = get_portfolio_performance(recommended_weights, returns[all_tickers])

# Print performance
print("\n📊 Performance of Recommended Portfolio:")
print(f"Net Annualized Return: {ret:.4f}")
print(f"Volatility: {vol:.4f}")
print(f"Sortino Ratio: {sortino:.4f}")

# --- Optional: Save to Excel ---
df.to_excel("Simulated_ETF_Portfolios.xlsx", index=False)