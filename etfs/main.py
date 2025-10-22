import yfinance as yf
import pandas as pd

# Full list of ETF tickers you provided
etfs = [
    "ARKK","GRID","FAN","PAVE","TAN","PHO","PBW","IBB","DGRO","ESGU","ICLN","INDA","EWW","EWT","ITA",
    "BBCA","XLU","ESGV","VWO","VHT","VNQ","VTI","VT","VSS","VNQI","MSOS","IPO","JEPI","COWZ","LCTU",
    "XLC","XLP","XLE","XLF","XLV","XLI","RSP","ESGE","EWA","EWZ","MCHI","DSI","USMV","QUAL","ESGD",
    "MOAT","VEU","VEA","VGK","VOO","VXUS","XLB","XLY","QQQ","MTUM","IWF","XLK","SMH","VUG","VGT","IWD",
    "NOBL","SCHD","VIG","VYM","VTV","IJH","VO","IJR","IWM","AVUV","AGG","JPST","BSV","BND","BNDX",
    "FTSL","HYLS","IGSB","FALN","HYG","LQD","JNK","VCIT","VCSH","SHY","TLT","IEF","EMB","SHV","GOVT",
    "BIL","VGSH","USFR","EMLC","VTIP","TIP","MBB","MUB","VTEB"
]

# Prepare results list
results = []

for symbol in etfs:
    try:
        etf = yf.Ticker(symbol)
        hist = etf.history(period="10y")
        one_year_hist = etf.history(period="1y")

        # 10-year annualized return
        if not hist.empty:
            first_price = hist['Close'].iloc[0]
            last_price = hist['Close'].iloc[-1]
            total_return = (last_price / first_price) - 1
            cagr_10y = (1 + total_return) ** (1/10) - 1
        else:
            cagr_10y = None

        # 1-year return
        if not one_year_hist.empty:
            one_year_return = (one_year_hist['Close'].iloc[-1] / one_year_hist['Close'].iloc[0]) - 1
        else:
            one_year_return = None

        # Volatility (std dev of daily returns annualized)
        daily_returns = one_year_hist['Close'].pct_change().dropna()
        if not daily_returns.empty:
            volatility = daily_returns.std() * (252**0.5)
        else:
            volatility = None

        info = etf.info

        results.append({
            "Ticker": symbol,
            "1Y Return (%)": round(one_year_return * 100, 2) if one_year_return is not None else None,
            "10Y CAGR (%)": round(cagr_10y * 100, 2) if cagr_10y is not None else None,
            "Volatility (Ann. %)": round(volatility * 100, 2) if volatility is not None else None,
            "52W Low": info.get("fiftyTwoWeekLow", None),
            "52W High": info.get("fiftyTwoWeekHigh", None),
            "Beta": info.get("beta", None),
            "Expense Ratio": info.get("expenseRatio", None),
            "Total Assets (B)": info.get("totalAssets", None),
            "Holdings Count": info.get("holdings", None),
        })

    except Exception as e:
        results.append({
            "Ticker": symbol,
            "Error": str(e)
        })

# Convert to DataFrame
df = pd.DataFrame(results)

# You can add a growth scoring formula, e.g.:
# df["Growth Score"] = df["10Y CAGR (%)"] * 0.6 + df["1Y Return (%)"] * 0.4 - df["Volatility (Ann. %)"] * 0.1

# Sort by growth score or by 1Y Return
# df_sorted = df.sort_values(by="Growth Score", ascending=False)

# Save to Excel
df.to_excel("ETF_Metrics_FullList.xlsx", index=False)

print("Scraping complete — data written to ETF_Metrics_FullList.xlsx")