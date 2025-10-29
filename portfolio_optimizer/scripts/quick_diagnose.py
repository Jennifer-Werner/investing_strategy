# quick_diagnose_return_drag.py
import pandas as pd, numpy as np, json
from pathlib import Path

out = Path("outputs")

# 1) What portion is sleeves vs active?
w = pd.read_csv(out/"weights_latest_full.csv", index_col=0)["weight"]
sleeves = {"BONDS","VOO","SMH","IWF","QQQ"}
print("Sleeves weight =", float(w.reindex(sleeves).fillna(0).sum()))

# 2) Which constraints bind? (ex-post proxies)
cov = pd.read_csv(out/"cov_for_mc.csv", index_col=0)
cov = (cov + cov.T)/2
wb = w.copy()
act = [t for t in w.index if t not in sleeves]
wb.loc[act] = (1.0 - float(w.reindex(sleeves).fillna(0).sum()))/len(act)  # your TE benchmark

d = (w - wb).reindex(cov.index).fillna(0).values.reshape(-1,1)
te_ann = float(np.sqrt(d.T @ cov.values @ d) * np.sqrt(252))
print("Realized TE vs benchmark (annual) ≈", round(te_ann, 4), "(target = 0.04)")

div = pd.read_csv(out/"dividend_yields.csv", index_col=0)["div_yield"]
initial_nav = 1.0  # scale-free
with open("config.yml","r") as f:
    pass
# If you know floor/slack from rc, check:
required = 9000*(1-0.0)  # example: your floor; replace with rc.div_floor_abs*(1-rc.div_slack)
achieved = float((div.reindex(w.index).fillna(0).values * w.values).sum()*500000)  # NAV=500k
print("Dividend $ achieved ~", round(achieved,2))
print("Dividend $ required ~", required)