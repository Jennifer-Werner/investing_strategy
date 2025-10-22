import re
import pandas as pd

def normalize_headers(df):
    """Normalize DataFrame column headers with regex-safe, case-insensitive logic."""
    abbreviations = {"EBIT", "EBITDA", "FCF", "ROE", "ROA", "ROIC", "PB", "EPS", "EV", "COGS", "DCF", "WACC", "PE"}
    def normalize_col(col):
        col_original = str(col)
        col_clean = re.sub(r'[_\-]+', ' ', col_original).strip()
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
            new_col = re.sub(rf'\b{abbr}\b', abbr, new_col, flags=re.IGNORECASE)
        return new_col
    old_cols = list(df.columns)
    df.columns = [normalize_col(c) for c in df.columns]
    print("\n[Header Normalization]")
    for old, new in zip(old_cols, df.columns):
        print(f"  {old} → {new}")
    return df

# === Step 1: Load both datasets ===
sector_df = pd.read_excel("sp1500_sector_industry.xlsx")
esg_df = pd.read_excel("sp1500_esg_raw.xlsx")

# === Step 2: Normalize column headers ===
sector_df = normalize_headers(sector_df)
esg_df = normalize_headers(esg_df)

# Rename the symbol column to ensure consistency
if 'Symbol' not in sector_df.columns:
    sector_df.rename(columns={col: 'Symbol' for col in sector_df.columns if 'Symbol' in col.title()}, inplace=True)
if 'Symbol' not in esg_df.columns:
    esg_df.rename(columns={col: 'Symbol' for col in esg_df.columns if 'Symbol' in col.title()}, inplace=True)

# Capitalize symbols
sector_df['Symbol'] = sector_df['Symbol'].str.upper()
esg_df['Symbol'] = esg_df['Symbol'].str.upper()

# === Step 3: Merge both datasets on Symbol ===
merged_df = pd.merge(esg_df, sector_df, on='Symbol', how='left')
print(f"Total companies after merge: {len(merged_df)}")

# === Step 4: Apply ESG hard constraints ===
# Only keep rows with valid ESG + Controversy data
filtered_df = merged_df.copy()

# Clean numeric fields
for col in ['Total Esg Score', 'Esg Risk', 'Controversy', 'Controversy Level']:
    if col in filtered_df.columns:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

# ESG Risk < 30 and Controversy ≤ 3
if 'Esg Risk' in filtered_df.columns:
    before_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['Esg Risk'] < 30]
    after_count = len(filtered_df)
    print(f"Companies filtered out by ESG risk < 30: {before_count - after_count}")
elif 'Total Esg Score' in filtered_df.columns:
    before_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['Total Esg Score'] < 30]
    after_count = len(filtered_df)
    print(f"Companies filtered out by total ESG score < 30: {before_count - after_count}")

if 'Controversy' in filtered_df.columns:
    before_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['Controversy'] <= 3]
    after_count = len(filtered_df)
    print(f"Companies filtered out by controversy ≤ 3: {before_count - after_count}")
elif 'Controversy Level' in filtered_df.columns:
    before_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['Controversy Level'] <= 3]
    after_count = len(filtered_df)
    print(f"Companies filtered out by controversy level ≤ 3: {before_count - after_count}")

# === Step 5: Exclude banned industries ===
banned_keywords = [
    'fossil', 'coal', 'oil', 'gas',
    'tobacco', 'weapon', 'arms',
    'gambling', 'casino', 'betting',
    'adult', 'porn', 'escort', 'alcoholic'
]

def exclude_banned_industries(industry):
    if not isinstance(industry, str):
        return True
    text = industry.lower()
    return not any(keyword in text for keyword in banned_keywords)

before_count = len(filtered_df)
filtered_df = filtered_df[filtered_df['Industry'].apply(exclude_banned_industries)]
after_count = len(filtered_df)
print(f"Companies filtered out by banned industries: {before_count - after_count}")

# === Step 6: Rename 'Symbol' → 'Symbol' for consistency (already capitalized) ===
# No renaming needed since column is already 'Symbol'

# === Step 7: Save cleaned dataset ===
filtered_df.to_excel("sp1500_esg_screened.xlsx", index=False)

print(f"✅ Successfully merged and filtered dataset saved to: sp1500_esg_screened.xlsx")
print(f"✅ Final dataset size: {len(filtered_df)} companies after screening.")