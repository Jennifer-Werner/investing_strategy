import pandas as pd

print("🚀 Combining S&P 500, 400, and 600 datasets...")

# Load your three CSV files from Barchart
sp500 = pd.read_csv("../output/sp500.csv")
sp400 = pd.read_csv("../output/sp400.csv")
sp600 = pd.read_csv("../output/sp600.csv")

# Standardize column names (trim spaces, fix capitalization)
for df in [sp500, sp400, sp600]:
    df.columns = df.columns.str.strip().str.title()
    if 'Symbol' not in df.columns:
        df.rename(columns={'Ticker': 'Symbol'}, inplace=True)

# Combine into one DataFrame
sp1500 = pd.concat([sp500, sp400, sp600], ignore_index=True)
sp1500.drop_duplicates(subset='Symbol', inplace=True)

# Save as Excel file
output_path = "../output/sp1500_universe.xlsx"
sp1500.to_excel(output_path, index=False)
print(f"✅ Combined dataset saved to {output_path} with {len(sp1500)} tickers.")