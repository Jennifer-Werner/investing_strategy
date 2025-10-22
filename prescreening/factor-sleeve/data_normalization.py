# standardize_headers.py
import os
import re
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)

# === Define standardized header mapping ===
HEADER_MAP = {
    # identification
    r'\bsym\b|\bsymbol\b': 'Symbol',
    r'\bname\b': 'Name',
    r'\bsector\b': 'Sector',
    r'\bindustry\b': 'Industry',

    # valuation
    r'market\s*cap': 'Market Cap',
    r'\bp\/?e\b|\bp[ ]?e ttm': 'P/E TTM',
    r'\bp\/?b\b': 'P/B',
    r'\bbeta\b': 'Beta',
    r'enterprise.*value': 'Enterprise Value (EV)',

    # income
    r'\brevenue\b|total\s*revenue': 'Revenue',
    r'ebitda': 'EBITDA',
    r'ebit': 'EBIT',
    r'net\s*income': 'Net Income (A)',
    r'eps': 'EPS TTM',
    r'cogs|cost of goods': 'COGS',
    r'operating\s*income': 'Operating Income',
    r'gross\s*margin': 'Gross Margin',
    r'operating\s*margin': 'Operating Margin',

    # balance sheet
    r'total\s*assets': 'Total Assets',
    r'equity': 'Equity',
    r'long\s*term\s*debt': 'Long Term Debt',
    r'current\s*assets': 'Current Assets',
    r'current\s*liabilities': 'Current Liabilities',
    r'invested\s*capital': 'Invested Capital',

    # cash flow
    r'free\s*cashflow|fcf': 'Free Cashflow (FCF)',
    r'dividend\s*\(a\)|dividend\(a\)': 'Dividend (A)',
    r'div\s*yield': 'Dividend Yield (A)',

    # performance
    r'roe': 'ROE',
    r'roa': 'ROA',
    r'roic': 'ROIC',
    r'nopat': 'NOPAT',

    # esg
    r'total\s*esg': 'Total ESG Score',
    r'environment.*risk': 'Environment Risk Score',
    r'social.*risk': 'Social Risk Score',
    r'governance.*risk': 'Governance Risk Score',
    r'controversy': 'Controversy Level',
    r'last.*date': 'Last Update Date',
}


def normalize_header(header):
    """Convert header to standard naming format."""
    h = str(header).strip().lower()
    h = re.sub(r'[_\-]+', ' ', h)
    for pattern, standard in HEADER_MAP.items():
        if re.search(pattern, h):
            return standard
    # Capitalize if not matched
    return ' '.join(word.capitalize() for word in h.split())


def process_file(file):
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Processing {file}")
    try:
        xl = pd.ExcelFile(file)
    except Exception as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to read {file}: {e}")
        return

    new_sheets = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        old_cols = df.columns
        df.columns = [normalize_header(c) for c in old_cols]

        # Log changes
        changes = [(o, n) for o, n in zip(old_cols, df.columns) if o != n]
        if changes:
            print(f"   {Fore.YELLOW}[HEADERS UPDATED]{Style.RESET_ALL} in sheet {sheet}")
            for old, new in changes:
                print(f"      {old} → {new}")

        new_sheets[sheet] = df

    # Save normalized version
    new_path = file.replace(".xlsx", "_normalized.xlsx")
    with pd.ExcelWriter(new_path, engine="openpyxl") as writer:
        for name, frame in new_sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    print(f"{Fore.GREEN}[DONE]{Style.RESET_ALL} Saved → {new_path}\n")


# === Run for all ===
for file in os.listdir("../.."):
    if file.startswith("sp1500_") and file.endswith(".xlsx") and not file.endswith("_normalized.xlsx"):
        process_file(file)

print(f"{Fore.GREEN}✅ All spreadsheet headers standardized successfully!{Style.RESET_ALL}")