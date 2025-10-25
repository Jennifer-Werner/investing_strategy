import re
import pandas as pd

def normalize_headers(df): #old formatting
    """Normalize DataFrame column headers to snake_case lowercase."""
    def normalize_col(col):
        col = str(col).strip().lower()
        col = re.sub(r'[\s\-\/]+', '_', col)
        col = re.sub(r'[()\[\]]', '', col)
        return col
    old_cols = list(df.columns)
    df.columns = [normalize_col(c) for c in df.columns]
    print("\n[Header Normalization]")
    for old, new in zip(old_cols, df.columns):
        print(f"  {old} → {new}")
    return df

# === Step 1: Load both datasets ===
sector_df = pd.read_excel("../output/sp1500_sector_industry.xlsx")
esg_df = pd.read_excel("../output/sp1500_esg_raw.xlsx")

# === Step 2: Normalize column headers ===
sector_df = normalize_headers(sector_df)
esg_df = normalize_headers(esg_df)

# Rename the symbol column to ensure consistency
if 'symbol' not in sector_df.columns:
    sector_df.rename(columns={col: 'symbol' for col in sector_df.columns if 'symbol' in col.lower()}, inplace=True)
if 'symbol' not in esg_df.columns:
    esg_df.rename(columns={col: 'symbol' for col in esg_df.columns if 'symbol' in col.lower()}, inplace=True)

# Capitalize symbols
sector_df['symbol'] = sector_df['symbol'].str.upper()
esg_df['symbol'] = esg_df['symbol'].str.upper()

# === Step 3: Merge both datasets on symbol ===
merged_df = pd.merge(esg_df, sector_df, on='symbol', how='left')
print(f"Total companies after merge: {len(merged_df)}")

# # === Debug Coca-Cola (KO) presence ===
# print("\n[Debug KO Status]")
# if 'KO' in merged_df['symbol'].values:
#     print("KO found in merged dataset!")
#     available_cols = [c for c in ['esg_risk', 'total_esg_score', 'controversy', 'controversy_level', 'industry'] if c in merged_df.columns]
#     print("Available columns for KO debug:", available_cols)
#     print(merged_df.loc[merged_df['symbol'] == 'KO', available_cols])
# else:
#     print("\u26a0\ufe0f KO not found in merged dataset. Check symbol names in source files.")

# === Step 4: Apply ESG hard constraints ===
filtered_df = merged_df.copy()

# Clean numeric fields
for col in ['total_esg_score', 'esg_risk', 'controversy', 'controversy_level']:
    if col in filtered_df.columns:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

# ESG Risk < 30 and Controversy ≤ 3
if 'esg_risk' in filtered_df.columns:
    before_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['esg_risk'] < 30]
    after_count = len(filtered_df)
    print(f"Companies filtered out by ESG risk < 30: {before_count - after_count}")
elif 'total_esg_score' in filtered_df.columns:
    before_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['total_esg_score'] < 30]
    after_count = len(filtered_df)
    print(f"Companies filtered out by total ESG score < 30: {before_count - after_count}")

if 'controversy' in filtered_df.columns:
    before_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['controversy'] <= 3]
    after_count = len(filtered_df)
    print(f"Companies filtered out by controversy ≤ 3: {before_count - after_count}")
elif 'controversy_level' in filtered_df.columns:
    before_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['controversy_level'] <= 3]
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
    if 'non-alcoholic' in text:
        return True  # explicitly allow non-alcoholic industries
    return not any(keyword in text for keyword in banned_keywords)

before_count = len(filtered_df)
filtered_df = filtered_df[filtered_df['industry'].apply(exclude_banned_industries)]
after_count = len(filtered_df)
print(f"Companies filtered out by banned industries: {before_count - after_count}")

# === Step 6: Save cleaned dataset ===
filtered_df.to_excel("../output/test_sp1500_esg_screened.xlsx", index=False)

print(f"✅ Successfully merged and filtered dataset saved to: sp1500_esg_screened.xlsx")
print(f"✅ Final dataset size: {len(filtered_df)} companies after screening.")