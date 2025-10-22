import pandas as pd
import numpy as np
from scipy import stats

# ==============================
# CONFIGURATION
# ==============================
INPUT_FILE = "../output/sp1500_financials_filled.xlsx"
OUTPUT_FILE = "../output/sp1500_factor_quality_evaluated.xlsx"

WEIGHTS = {
    "value": 0.4,
    "profitability": 0.3,
    "quality": 0.3
}

# ==============================
# HELPER FUNCTIONS
# ==============================

def safe_div(a, b):
    """Division that handles zero and NaN safely, supports scalar and Series/array inputs."""
    if isinstance(a, (pd.Series, np.ndarray)) or isinstance(b, (pd.Series, np.ndarray)):
        a_arr = np.array(a)
        b_arr = np.array(b)
        # Reduce to 1D if multi-column
        if a_arr.ndim > 1:
            print("safe_div: input 'a' has multiple columns, taking first column for division.")
            a_arr = a_arr[:, 0]
        if b_arr.ndim > 1:
            print("safe_div: input 'b' has multiple columns, taking first column for division.")
            b_arr = b_arr[:, 0]
        result = np.where((b_arr == 0) | pd.isna(a_arr) | pd.isna(b_arr), np.nan, a_arr / b_arr)
        return pd.Series(result, index=a.index if isinstance(a, pd.Series) else None)
    else:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b

print("safe_div function updated to handle Series inputs correctly.")

def zscore(series):
    """Compute z-score while ignoring NaNs."""
    return stats.zscore(series, nan_policy='omit')

# ==============================
# LOAD DATA
# ==============================
df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.str.strip()

# Ensure symbols are uppercase
if "Symbol" not in df.columns:
    raise KeyError("Missing 'Symbol' column.")
df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()

# Alias alternative column names
if "EV" not in df.columns and "Enterprise Value" in df.columns:
    df["EV"] = df["Enterprise Value"]
if "EV" not in df.columns and "EnterpriseValue" in df.columns:
    df["EV"] = df["EnterpriseValue"]
if "EV" not in df.columns and "Enterprise_Value" in df.columns:
    df["EV"] = df["Enterprise_Value"]

# Check for "Ev" or case-insensitive variants if EV still not found
if "EV" not in df.columns:
    ev_candidates = [col for col in df.columns if col.strip().lower() == "ev"]
    if ev_candidates:
        df["EV"] = df[ev_candidates[0]]
        print(f"Using '{ev_candidates[0]}' as EV column.")

if "EV" in df.columns:
    print(f"Using '{[col for col in ['EV', 'Enterprise Value', 'EnterpriseValue', 'Enterprise_Value'] if col in df.columns and (col == 'EV' or df[col].equals(df['EV']))][0]}' as EV column.")
elif "Enterprise Value" in df.columns:
    print("Using 'Enterprise Value' as EV column.")
elif "EnterpriseValue" in df.columns:
    print("Using 'EnterpriseValue' as EV column.")
elif "Enterprise_Value" in df.columns:
    print("Using 'Enterprise_Value' as EV column.")
else:
    print("EV column not found.")

# ==============================
# STANDARDIZE COLUMN NAMES TO LOWERCASE
# ==============================
df.columns = df.columns.str.strip().str.lower()

#
# ==============================
# FCF ALIAS MAPPING AND ROBUST FCF CREATION
# ==============================
# Try to ensure an 'fcf' column exists, using robust fallback logic.
if "fcf" not in df.columns:
    # List of alternative FCF column names (case-insensitive)
    fcf_aliases = [
        "fcf", "free cash flow", "freecashflow", "free cashflow", "fcf (a)", "fcf (ttm)", "fcf ttm", "freecashflow (ttm)"
    ]
    found_fcf = None
    for alias in fcf_aliases:
        for col in df.columns:
            if col.strip().lower() == alias:
                df["fcf"] = df[col]
                found_fcf = col
                print(f"Using '{col}' as FCF column.")
                break
        if found_fcf:
            break
    # If still not found, try to create from OCF and CapEx
    if not found_fcf:
        # Try to find operating cash flow and capex columns (case-insensitive)
        ocf_col = None
        capex_col = None
        for col in df.columns:
            lcol = col.strip().lower()
            if lcol in ["operating cash flow", "ocf"]:
                ocf_col = col
            if lcol in ["capital expenditures", "capex"]:
                capex_col = col
        if ocf_col is not None and capex_col is not None:
            # Try both OCF + CapEx and OCF - CapEx, choose the one with larger absolute values or more non-NaNs
            ocf_vals = df[ocf_col]
            capex_vals = df[capex_col]
            # Try both conventions, prefer OCF - CapEx if negative capex, else OCF + CapEx
            # CapEx is usually negative, so OCF + CapEx is standard
            fcf_values = ocf_vals + capex_vals
            df["fcf"] = fcf_values
            print(f"Created FCF as '{ocf_col} + {capex_col}'.")
            found_fcf = "derived"
        else:
            # Still not found: create empty column with NaNs and warn
            df["fcf"] = np.nan
            print("⚠️ FCF missing — created NaN column.")
elif "fcf" in df.columns:
    # Already present
    pass

if "eps ttm" not in df.columns and "earnings per share" in df.columns:
    df["eps ttm"] = df["earnings per share"]

# ==============================
# FACTOR CONSTRUCTION
# ==============================

# --- Value Tilt (0.4 weight) ---
# Only: EV/EBITDA, P/B, Earnings Yield
df["ev/ebitda"] = safe_div(df["ev"], df["ebitda"])
df["p/b"] = safe_div(df["ev"], df["equity"])  # proxy if actual P/B unavailable
# df["fcf yield"] = safe_div(df["fcf"], df["ev"])  # Removed from value tilt
df["earnings yield"] = safe_div(df["eps ttm"], df["ev"])

# Remove precomputed z-scores and direct numerical assignments for scores
# Instead, we will add Excel formulas after saving to Excel

# ==============================
# FINAL OUTPUT PREPARATION
# ==============================

# Define columns to keep in output
cols_to_keep = [
    "symbol",
    "ev/ebitda", "p/b", "earnings yield",
    "roa", "roic", "operating margin",
    "roe", "long term debt", "eps ttm"
]

# Add placeholder columns for scores and composite
df["value score"] = np.nan
df["profitability score"] = np.nan
df["quality score"] = np.nan
df["composite score"] = np.nan

# Add the other score columns to cols_to_keep for output order
cols_to_keep = [
    "symbol",
    "value score", "profitability score", "quality score", "composite score",
    # Value tilt metrics
    "ev/ebitda", "p/b", "earnings yield",
    # Profitability tilt metrics
    "roa", "roic", "operating margin",
    # Quality tilt metrics
    "roe", "long term debt", "eps ttm"
]

# Save to Excel first
df_to_save = df[cols_to_keep].copy()
df_to_save.to_excel(OUTPUT_FILE, index=False)

# ==============================
# WRITE EXCEL FORMULAS FOR Z-SCORES AND SCORES
# ==============================
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

wb = load_workbook(OUTPUT_FILE)
ws = wb.active

# Get the number of data rows (excluding header)
nrows = ws.max_row
ncols = ws.max_column

# Map column names to Excel column letters (1-based)
header = [cell.value for cell in ws[1]]
col_idx = {name: idx+1 for idx, name in enumerate(header)}

# Helper to create z-score formula for a cell in a given column
def zscore_formula(col_letter, row, start_row=2, end_row=nrows):
    # Z = (x - average(range)) / stdev(range)
    # Use AVERAGE and STDEV.P ignoring blanks by default
    rng = f"{col_letter}{start_row}:{col_letter}{end_row}"
    # Return formula string without leading '=' for embedding, use US/UK Excel locale (comma separator)
    return f"IF(ISNUMBER({col_letter}{row}),({col_letter}{row}-AVERAGE({rng}))/STDEV.P({rng}),\"\")"

# Columns for value tilt metrics
val_cols = ["ev/ebitda", "p/b", "earnings yield"]
# Columns for profitability tilt metrics
prof_cols = ["roa", "roic", "operating margin"]
# Columns for quality tilt metrics
qual_cols = ["roe", "long term debt", "eps ttm"]

# Excel rows start at 2 (first data row)
for row in range(2, nrows+1):
    # Compute z-scores for value tilt metrics
    val_zscores = []
    for col in val_cols:
        col_letter = get_column_letter(col_idx[col])
        val_zscores.append(zscore_formula(col_letter, row))
    # Formula for value score = average of value tilt z-scores
    # Use AVERAGE to ignore blanks automatically
    val_score_formula = f"=AVERAGE({','.join(val_zscores)})"
    ws.cell(row=row, column=col_idx["value score"]).value = val_score_formula

    # Compute z-scores for profitability tilt metrics
    prof_zscores = []
    for col in prof_cols:
        col_letter = get_column_letter(col_idx[col])
        prof_zscores.append(zscore_formula(col_letter, row))
    prof_score_formula = f"=AVERAGE({','.join(prof_zscores)})"
    ws.cell(row=row, column=col_idx["profitability score"]).value = prof_score_formula

    # Compute z-scores for quality tilt metrics
    # For "long term debt" invert sign, so formula becomes: -zscore
    qual_zscores = []
    for col in qual_cols:
        col_letter = get_column_letter(col_idx[col])
        zf = zscore_formula(col_letter, row)
        if col == "long term debt":
            # Invert sign of z-score
            zf = f"-( {zf} )"
        qual_zscores.append(zf)
    qual_score_formula = f"=AVERAGE({','.join(qual_zscores)})"
    ws.cell(row=row, column=col_idx["quality score"]).value = qual_score_formula

    # Composite score formula
    # =0.4*value_score + 0.3*profitability_score + 0.3*quality_score
    val_score_cell = ws.cell(row=row, column=col_idx["value score"]).coordinate
    prof_score_cell = ws.cell(row=row, column=col_idx["profitability score"]).coordinate
    qual_score_cell = ws.cell(row=row, column=col_idx["quality score"]).coordinate
    composite_formula = f"=0.4*{val_score_cell}+0.3*{prof_score_cell}+0.3*{qual_score_cell}"
    ws.cell(row=row, column=col_idx["composite score"]).value = composite_formula

# Enable recalculation on file open (supports multiple openpyxl versions)
if hasattr(wb, "calculation_properties"):
    wb.calculation_properties.fullCalcOnLoad = True
elif hasattr(wb, "calc_properties"):
    wb.calc_properties.fullCalcOnLoad = True
else:
    print("⚠️ Unable to set fullCalcOnLoad; openpyxl version may not support this property.")

# Save workbook with formulas
wb.save(OUTPUT_FILE)

print(f"✅ Factor tilt + quality evaluation saved to: {OUTPUT_FILE}")
print(f"✅ Top 10 stocks by composite score:\n", df.sort_values("composite score", ascending=False).head(10)[['symbol', 'composite score']])