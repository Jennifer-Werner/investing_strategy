#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, io, base64, html, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: Pillow for single-PNG dashboard export
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False
    Image = ImageDraw = ImageFont = None

# ---------------- small utils ----------------
def _ann_return(r: pd.Series, periods: int = 252) -> float:
    r = pd.Series(r).dropna()
    if r.empty:
        return float("nan")
    return float((1 + r).prod() ** (periods / max(len(r), 1)) - 1)

# ---------------- IO helpers ----------------
def _read_weights(path: str) -> pd.Series:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if df.ndim == 2:
            s = df.iloc[-1].copy()
            s.name = "weight"
            s.index = s.index.astype(str)
            return s.astype(float)
    else:
        df = pd.read_csv(path)
        if {"asset","weight"} <= set(df.columns):
            s = df.set_index("asset")["weight"].astype(float)
        else:
            s = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
            s.name = "weight"
        s.index = s.index.astype(str)
        return s
    raise SystemExit(f"Could not parse weights from {path}")

def _read_mu(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if {"asset","mu"} <= set(df.columns):
        s = df.set_index("asset")["mu"].astype(float)
    else:
        s = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
        s.name = "mu"
    s.index = s.index.astype(str)
    return s

def _read_cov(path: str) -> pd.DataFrame:
    cov = pd.read_csv(path, index_col=0).astype(float)
    cov.index = cov.index.astype(str)
    cov.columns = cov.columns.astype(str)
    return (cov + cov.T) / 2.0

def _read_sectors(path: str | None) -> pd.Series:
    if not path or not os.path.exists(path):
        return pd.Series(dtype=object)
    df = pd.read_csv(path)
    if "ticker" in df.columns: kcol = "ticker"
    elif "asset" in df.columns: kcol = "asset"
    else: kcol = df.columns[0]
    scol = "sector" if "sector" in df.columns else [c for c in df.columns if c != kcol][0]
    s = df.set_index(kcol)[scol].astype(str)
    s.index = s.index.astype(str)
    return s

def _read_returns_cache(path: str | None) -> pd.Series | None:
    if not path or not os.path.exists(path):
        return None
    obj = pd.read_parquet(path)
    if isinstance(obj, pd.DataFrame):
        if "Strategy" in obj.columns:
            s = obj["Strategy"]
        else:
            s = obj.iloc[:, 0]
    else:
        s = obj
    return pd.Series(s).astype(float).sort_index()

# ---------------- core helpers ----------------
def _clean_index(s: pd.Index) -> pd.Index:
    return pd.Index(x.strip().upper() for x in s.astype(str))

def _align(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, verbose: bool = True):
    # normalize/clean ticker keys
    w.index  = _clean_index(w.index)
    mu.index = _clean_index(mu.index)
    cov.index = _clean_index(cov.index)
    cov.columns = _clean_index(cov.columns)

    common = w.index.intersection(mu.index).intersection(cov.index).intersection(cov.columns)
    dropped = sorted(set(w.index) - set(common))
    if verbose:
        print(json.dumps({
            "ALIGN": {
                "weights_n": int(len(w)),
                "mu_n": int(len(mu)),
                "cov_n": int(len(cov)),
                "common_n": int(len(common)),
                "dropped_from_weights": dropped[:10] + (["..."] if len(dropped) > 10 else [])
            }
        }, indent=2))

    if len(common) == 0:
        raise SystemExit("No overlapping tickers among weights, mu, and cov.")
    w = w.loc[common].clip(lower=0)
    if w.sum() == 0:
        raise SystemExit("All weights are zero after clipping.")
    w = w / w.sum()
    mu = mu.loc[common]
    cov = cov.loc[common, common]
    return w, mu, cov

def _risk_contributions(w: pd.Series, cov_annual: pd.DataFrame):
    Sigma = cov_annual.values
    wv = w.values.reshape(-1, 1)
    port_var = float((wv.T @ Sigma @ wv).item())
    port_vol = np.sqrt(max(port_var, 0.0))
    if port_vol > 0:
        mcr = (Sigma @ wv)[:, 0] / port_vol
        rc = w.values * mcr
    else:
        mcr = np.zeros_like(w.values); rc = np.zeros_like(w.values)
    asset_vol = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    return port_vol, pd.Series(mcr, index=w.index, name="marg_risk"), pd.Series(rc, index=w.index, name="risk_contrib"), pd.Series(asset_vol, index=w.index, name="asset_vol")

def _labels(contrib_ret: pd.Series, risk_contrib: pd.Series) -> pd.Series:
    r_med = contrib_ret.median()
    rc_hi = risk_contrib.quantile(0.80)
    rc_lo = risk_contrib.quantile(0.20)
    lab = []
    top_ret_cut = contrib_ret.quantile(0.70)
    for t in contrib_ret.index:
        cr = contrib_ret.loc[t]; rc = risk_contrib.loc[t]
        if cr >= top_ret_cut:
            lab.append("Strong")
        elif (rc >= rc_hi) and (cr < r_med):
            lab.append("Risky")
        elif (rc <= rc_lo) and (cr > 0):
            lab.append("Safe")
        else:
            lab.append("Weak")
    return pd.Series(lab, index=contrib_ret.index, name="label")

# ---------------- plotting -> base64 helpers ----------------
def _fig_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, dpi=150, format="png", bbox_inches="tight")
    plt.close()
    enc = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{enc}"

def _pie_assets(w: pd.Series) -> str:
    plt.figure(figsize=(7.5, 7.5))
    w_sorted = w.sort_values(ascending=False)
    big = w_sorted[w_sorted >= 0.005]  # ≥0.5%
    small = w_sorted[w_sorted < 0.005]
    if small.sum() > 0:
        big = pd.concat([big, pd.Series({"Other": small.sum()})])
    plt.pie(big.values, labels=big.index, autopct="%1.1f%%", startangle=90)
    plt.title("Portfolio Weights (Assets)")
    return _fig_to_base64()

def _pie_sectors(by_sector: pd.DataFrame) -> str:
    plt.figure(figsize=(7.5, 7.5))
    s = by_sector["weight"].sort_values(ascending=False)
    plt.pie(s.values, labels=s.index, autopct="%1.1f%%", startangle=90)
    plt.title("Portfolio Weights (Sectors)")
    return _fig_to_base64()

def _bar(series: pd.Series, title: str, ylabel: str) -> str:
    plt.figure(figsize=(11, 4.8))
    s = series.sort_values(ascending=False)
    plt.bar(range(len(s)), s.values)
    plt.xticks(range(len(s)), s.index, rotation=90)
    plt.title(title)
    plt.ylabel(ylabel)
    return _fig_to_base64()

def _bar_pct(series: pd.Series, title: str, ylabel: str = "Percent") -> str:
    plt.figure(figsize=(11, 4.8))
    s = (series * 100.0).sort_values(ascending=False)
    plt.bar(range(len(s)), s.values)
    plt.xticks(range(len(s)), s.index, rotation=90)
    plt.title(title)
    plt.ylabel(ylabel)
    return _fig_to_base64()

def _scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str) -> str:
    plt.figure(figsize=(7, 6))
    plt.scatter(x.values, y.values)
    plt.axvline(np.nanmedian(x.values), linestyle="--")
    plt.axhline(np.nanmedian(y.values), linestyle="--")
    for t in x.index:
        try:
            plt.annotate(t, (float(x.loc[t]), float(y.loc[t])), fontsize=7)
        except Exception:
            pass
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    return _fig_to_base64()

def _label_counts(labels: pd.Series) -> str:
    plt.figure(figsize=(6.5, 4.2))
    counts = labels.value_counts().reindex(["Strong","Safe","Weak","Risky"]).fillna(0)
    plt.bar(range(len(counts)), counts.values)
    plt.xticks(range(len(counts)), counts.index, rotation=0)
    plt.title("Label Counts")
    plt.ylabel("# Assets")
    return _fig_to_base64()

def _sleeves_vs_active(w: pd.Series, sleeves: set[str] | None = None) -> str:
    if sleeves is None:
        sleeves = {"VOO","SMH","IWF","QQQ","BONDS"}
    sleeves_w = float(w.reindex(list(sleeves)).fillna(0.0).sum())
    active_w = 1.0 - sleeves_w
    plt.figure(figsize=(6.5, 4.2))
    plt.bar([0,1], [sleeves_w*100.0, active_w*100.0])
    plt.xticks([0,1], ["Sleeves+Bonds", "Active Book"])
    plt.title("Sleeves vs Active Allocation")
    plt.ylabel("Percent")
    return _fig_to_base64()

# ---------------- HTML table helper ----------------
def _df_to_html_table(df: pd.DataFrame, max_rows: int = 20, pct_cols: list[str] | None = None, ndp: int = 2) -> str:
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    if pct_cols is None:
        pct_cols = []
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            if col in pct_cols:
                d[col] = d[col].map(lambda v: f"{v:.{ndp}%}")
            else:
                d[col] = d[col].map(lambda v: f"{v:,.6f}")
    d = d.reset_index().rename(columns={"index": "asset"})
    return d.to_html(index=False, border=0, classes="table")

# ---------------- HTML dashboard writer ----------------
def _write_dashboard(html_path: str, title: str, kpis: dict[str,str], panels: list[tuple[str,str]], tables: list[tuple[str,str]]):
    def kpi_card(k, v):
        return f'<div class="kpi"><div class="kv">{html.escape(v)}</div><div class="kk">{html.escape(k)}</div></div>'
    def img_card(caption, uri):
        return f'<div class="card"><div class="cap">{html.escape(caption)}</div><img src="{uri}" alt="{html.escape(caption)}"></div>'
    def tbl_card(caption, tbl_html):
        return f'<div class="card"><div class="cap">{html.escape(caption)}</div>{tbl_html}</div>'
    kpi_html = "\n".join(kpi_card(k, v) for k, v in kpis.items())
    imgs_html = "\n".join(img_card(cap, uri) for cap, uri in panels)
    tbls_html = "\n".join(tbl_card(cap, t) for cap, t in tables)

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 16px; }}
    h1 {{ margin: 0 0 8px 0; }}
    .kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin: 8px 0 16px 0; }}
    .kpi {{ border:1px solid #ddd; border-radius:10px; padding:10px 14px; min-width:160px; }}
    .kpi .kv {{ font-size: 20px; font-weight: 600; }}
    .kpi .kk {{ font-size: 12px; color:#555; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap:16px; }}
    .card {{ border:1px solid #ddd; border-radius:10px; padding:10px; background:#fff; }}
    .card .cap {{ font-size: 14px; font-weight:600; margin-bottom:8px; }}
    .card img {{ width:100%; height:auto; display:block; border-radius:6px; }}
    .table {{ width:100%; border-collapse: collapse; font-size: 12px; }}
    .table th, .table td {{ border-bottom:1px solid #eee; padding:6px 8px; text-align: right; }}
    .table th:first-child, .table td:first-child {{ text-align:left; }}
    .section {{ margin-top: 18px; margin-bottom: 8px; font-weight:700; }}
    .note {{ color:#666; font-size: 12px; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="kpis">{kpi_html}</div>
  <div class="section">Charts</div>
  <div class="grid">{imgs_html}</div>
  <div class="section">Tables</div>
  <div class="grid">{tbls_html}</div>
  <div class="note">Tip: This HTML embeds images inline (base64). Save or share as a single file.</div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(doc)

# ---------------- PNG dashboard composition (Pillow) ----------------
def _img_from_data_uri(uri: str):
    if not uri.startswith("data:image"):
        raise ValueError("Expected data URI")
    b64 = uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

def _compose_png(out_png: str, title: str, kpis: dict[str,str], panels: list[tuple[str,str]],
                 width: int = 1600, col_gap: int = 18, row_gap: int = 18,
                 left: int = 20, top: int = 20, right: int = 20, bottom: int = 20):
    if not PIL_OK:
        print("[warn] Pillow not installed; skip PNG export. pip install pillow")
        return
    title_h = 56; kpi_h = 86; kpi_pad_x = 16; kpi_pad_y = 10; kpi_box_w = 260; kpi_box_gap = 12; cols = 2
    inner_w = width - left - right
    col_w = (inner_w - col_gap) // cols
    decoded: list[tuple[str, Image.Image]] = []
    for cap, uri in panels:
        try:
            img = _img_from_data_uri(uri)
        except Exception:
            continue
        scale = col_w / img.width
        h = int(img.height * scale)
        decoded.append((cap, img.resize((col_w, h), Image.LANCZOS)))
    nrows = (len(decoded) + cols - 1) // cols
    imgs_h = 0
    for r in range(nrows):
        row_h = 0
        for c in range(cols):
            i = r*cols + c
            if i >= len(decoded): break
            row_h = max(row_h, decoded[i][1].height)
        imgs_h += row_h
    if nrows > 0:
        imgs_h += row_gap * (nrows - 1)
    height = top + title_h + 10 + kpi_h + 20 + imgs_h + bottom
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 28)
        kpi_val_font = ImageFont.truetype("Arial.ttf", 22)
        kpi_key_font = ImageFont.truetype("Arial.ttf", 12)
        caption_font = ImageFont.truetype("Arial.ttf", 13)
    except Exception:
        title_font = ImageFont.load_default()
        kpi_val_font = ImageFont.load_default()
        kpi_key_font = ImageFont.load_default()
        caption_font = ImageFont.load_default()
    draw.text((left, top), title, fill=(0,0,0), font=title_font)
    y = top + title_h + 10
    x = left
    for k, v in kpis.items():
        box = [x, y, x + kpi_box_w, y + kpi_h]
        draw.rounded_rectangle(box, radius=10, outline=(220,220,220), width=1, fill=(250,250,250))
        draw.text((x + kpi_pad_x, y + kpi_pad_y), str(v), fill=(0,0,0), font=kpi_val_font)
        draw.text((x + kpi_pad_x, y + kpi_pad_y + 30), k, fill=(90,90,90), font=kpi_key_font)
        x += kpi_box_w + kpi_box_gap
        if x + kpi_box_w > width - right:
            x = left
            y += kpi_h + 8
    y += kpi_h + 20
    x0 = left
    r_start_y = y
    for r in range(nrows):
        row_imgs = []; row_caps = []; max_h = 0
        for c in range(cols):
            idx = r*cols + c
            if idx >= len(decoded): break
            cap, im = decoded[idx]
            row_imgs.append(im); row_caps.append(cap)
            max_h = max(max_h, im.height)
        x = x0
        for c, im in enumerate(row_imgs):
            canvas.paste(im, (x, r_start_y), im)
            cap = row_caps[c]
            draw.text((x, r_start_y + im.height + 4), cap, fill=(0,0,0), font=caption_font)
            x += col_w + col_gap
        r_start_y += max_h + row_gap + 20
    canvas.convert("RGB").save(out_png, "PNG")
    print(f"Saved dashboard image: {out_png}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",   default="outputs/weights_latest_full.csv")
    ap.add_argument("--mu",        default="outputs/mu_bl.csv")
    ap.add_argument("--cov",       default="outputs/cov_for_mc.csv")
    ap.add_argument("--sectors",   default="data/universe.csv", help="CSV with columns ticker,sector (or asset,sector)")
    ap.add_argument("--out_html",  default="outputs/portfolio_dashboard.html")
    ap.add_argument("--out_png",   default="outputs/portfolio_dashboard.png")  # single-image export
    ap.add_argument("--cov_is_daily",  type=lambda s: s.lower() in {"1","true","yes","y"}, default=True)
    ap.add_argument("--mu_is_annual",  type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                    help="Set false if mu in CSV is DAILY; we will multiply by 252.")
    ap.add_argument("--returns_cache", default="outputs/port_daily_returns.parquet",
                    help="(optional) Strategy daily returns for realized KPI")
    ap.add_argument("--top_n", type=int, default=20, help="Rows to show in Top holdings table")
    args = ap.parse_args()

    # Load & align
    w = _read_weights(args.weights)
    mu = _read_mu(args.mu)
    cov = _read_cov(args.cov)
    w, mu, cov = _align(w, mu, cov, verbose=True)

    # Annualize mu if needed
    if not args.mu_is_annual:
        mu = mu * 252.0

    # Sectors (canonical + sleeves)
    sectors_map = _read_sectors(args.sectors)
    sectors_map.index = _clean_index(sectors_map.index)
    sectors = sectors_map.reindex(w.index).fillna("Unknown")
    sleeves_map = {"VOO":"ETF","SMH":"ETF","IWF":"ETF","QQQ":"ETF","BONDS":"Bonds"}
    for k,v in sleeves_map.items():
        if k in sectors.index:
            sectors.loc[k] = v

    # Annualize covariance if needed
    cov_annual = cov * (252.0 if args.cov_is_daily else 1.0)

    # Contributions & risk
    contrib_ret = w * mu                         # decimals (annual)
    port_mu = float(contrib_ret.sum())          # decimal expected return
    port_vol, mcr, rc, asset_vol = _risk_contributions(w, cov_annual)
    labels = _labels(contrib_ret, rc)

    # Also compute realized (if returns cache is present)
    realized_ann = None
    r_cache = _read_returns_cache(args.returns_cache)
    if r_cache is not None and not r_cache.empty:
        realized_ann = _ann_return(r_cache)

    # Master table
    df = pd.concat([
        sectors.rename("sector"),
        w.rename("weight"),
        mu.rename("mu_annual"),
        contrib_ret.rename("contrib_return"),
        asset_vol.rename("asset_vol_annual"),
        mcr,
        rc
    ], axis=1)
    rc_sum = df["risk_contrib"].sum()
    df["risk_contrib_pct"] = df["risk_contrib"] / (rc_sum if abs(rc_sum) > 0 else 1.0)
    df["label"] = labels

    # Sector rollup
    by_sector = df.groupby("sector").agg(
        weight=("weight","sum"),
        contrib_return=("contrib_return","sum"),
        risk_contrib=("risk_contrib","sum"),
        n_names=("weight","size")
    ).sort_values("weight", ascending=False)
    rc_sec_sum = by_sector["risk_contrib"].sum()
    by_sector["risk_contrib_pct"] = by_sector["risk_contrib"] / (rc_sec_sum if abs(rc_sec_sum) > 0 else 1.0)

    # Top holdings table (hide tiny weights)
    min_display_weight = 1e-4  # 0.01% cutoff for display
    top_holdings = df[df["weight"] >= min_display_weight] \
        .sort_values("weight", ascending=False)[
            ["sector","weight","mu_annual","contrib_return","risk_contrib_pct","label"]
        ] \
        .rename(columns={
            "mu_annual":"mu (annual)",
            "contrib_return":"return contrib",
            "risk_contrib_pct":"risk contrib %"
        })

    # Build charts (each -> base64)
    panels: list[tuple[str,str]] = []
    panels.append(("Weights (Assets)", _pie_assets(w)))
    panels.append(("Weights (Sectors)", _pie_sectors(by_sector)))
    panels.append(("Sector Weights", _bar_pct(by_sector["weight"], "Sector Weights", "Percent")))
    panels.append(("Sector Return Contributions", _bar_pct(by_sector["contrib_return"], "Expected Return Contribution by Sector", "Percent")))
    panels.append(("Sector Risk Contributions", _bar_pct(by_sector["risk_contrib_pct"], "Risk Contribution by Sector", "Percent")))
    panels.append(("Asset Return Contributions", _bar_pct(df["contrib_return"], "Expected Return Contribution by Asset", "Percent")))
    panels.append(("Asset Risk Contributions", _bar_pct(df["risk_contrib_pct"], "Risk Contribution by Asset", "Percent")))
    panels.append(("Return vs Risk (Assets)", _scatter(df["contrib_return"]*100.0, df["risk_contrib_pct"]*100.0,
                        "Return vs Risk Contribution (Assets)", "Return Contribution (%)", "Risk Contribution (%)")))
    panels.append(("Labels Distribution", _label_counts(labels)))
    panels.append(("Sleeves vs Active", _sleeves_vs_active(w)))

    # KPIs (as percent)
    kpis = {
        "Expected Return (Model, annual)": f"{port_mu:.2%}",
        "Volatility (annual)": f"{port_vol:.2%}",
        "Assets": f"{len(df)}",
        "Sectors": f"{df['sector'].nunique()}",
    }
    if realized_ann is not None and np.isfinite(realized_ann):
        kpis["Realized Return (Backtest, annual)"] = f"{realized_ann:.2%}"

    # Tables to embed (format selected columns as percent)
    tables = [
        ("Top Holdings (by weight)",
         _df_to_html_table(top_holdings, max_rows=args.top_n,
                           pct_cols=["weight","mu (annual)","return contrib","risk contrib %"], ndp=2)),
        ("By Sector (weights / contribs / names)",
         _df_to_html_table(
             by_sector.rename(columns={"contrib_return":"return contrib","risk_contrib_pct":"risk contrib %"}),
             max_rows=50,
             pct_cols=["weight","return contrib","risk contrib %"], ndp=2
         )),
    ]

    # Write HTML
    os.makedirs(os.path.dirname(args.out_html) or ".", exist_ok=True)
    _write_dashboard(args.out_html, "Portfolio Dashboard", kpis, panels, tables)
    print(f"Saved dashboard: {args.out_html}")
    print(json.dumps({
        "KPI_DEBUG": {
            "port_mu_decimal": port_mu,
            "port_vol_decimal": port_vol,
            "mu_is_annual": bool(args.mu_is_annual),
            "has_returns_cache": bool(r_cache is not None and not r_cache.empty),
            "realized_ann": realized_ann
        }
    }, indent=2))

    # Also emit PNG (optional)
    if args.out_png:
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        _compose_png(args.out_png, "Portfolio Dashboard", kpis, panels)

if __name__ == "__main__":
    main()