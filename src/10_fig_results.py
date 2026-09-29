"""Step 10 - build the five result figures from the pipeline's CSV tables."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TAB = "results/tables"
FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

GRP = {"HAR": "HAR", "LASSO": "ML", "RF": "ML", "GARCH": "GARCH", "EGARCH": "GARCH",
       "EWMA": "Naive", "HV": "Naive"}
COL = {"HAR": "#1f4e79", "ML": "#2e8b57", "GARCH": "#d68910", "Naive": "#888888"}
c = lambda ns: [COL[GRP[n]] for n in ns]
LEG = [Patch(color=COL["HAR"], label="HAR-RV"), Patch(color=COL["ML"], label="ML (LASSO, RF)"),
       Patch(color=COL["GARCH"], label="GARCH family"), Patch(color=COL["Naive"], label="Naive (HV, EWMA)")]

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG}/{name}.png")

#Loading data from CSV files.
#1 Avg Qlike
QFILES = {"HAR": f"{TAB}/har_oos_by_stock.csv", "GARCH": f"{TAB}/garch_oos_by_stock.csv",
          "EGARCH": f"{TAB}/egarch_oos_by_stock.csv", "LASSO": f"{TAB}/lasso_oos_by_stock.csv",
          "RF": f"{TAB}/randomforest_oos_by_stock.csv", "HV": f"{TAB}/hv_oos_by_stock.csv",
          "EWMA": f"{TAB}/ewma_oos_by_stock.csv"}
qlike = {}
for m, p in QFILES.items():
    try:
        qlike[m] = float(pd.read_csv(p)["qlike"].mean())
    except Exception as e:
        print(f"[warn] QLIKE {m}: {e}")

#2 DM:stocks where HAR significantly wins.
dm = pd.read_csv(f"{TAB}/dm_summary.csv")
dm["model"] = dm["comparison"].str.split("_vs_").str[1]
dmw = dict(zip(dm["model"], dm["HAR_better_sig"]))
lasso_wins = int(dm.loc[dm.model == "LASSO", "comp_better_sig"].iloc[0]) if (dm.model == "LASSO").any() else None

#3 Regime QLIKE
reg = pd.read_csv(f"{TAB}/h3_regime_qlike_condrv.csv")
if "model" not in reg.columns:
    reg = reg.rename(columns={reg.columns[0]: "model"})
regime = {r["model"]: (r["Normal"], r["High"]) for _, r in reg.iterrows()
          if r["model"] in ("HAR", "LASSO", "RF", "GARCH", "EGARCH")}

#4 VaR pass counts (Christoffersen cc, Student-t) at 95% and 99%
var = pd.read_csv(f"{TAB}/var_backtest_summary.csv")
v99 = dict(zip(var[(var.level == 0.99) & (var.dist == "t")]["model"],
               var[(var.level == 0.99) & (var.dist == "t")]["pass_cc"]))
v95 = dict(zip(var[(var.level == 0.95) & (var.dist == "t")]["model"],
               var[(var.level == 0.95) & (var.dist == "t")]["pass_cc"]))

print("avg QLIKE:", {k: round(v, 3) for k, v in qlike.items()})
print("99% VaR pass_cc:", v99, "\n95% VaR pass_cc:", v95)

#GRAPH 1: Accuracy
f, a = plt.subplots(figsize=(7, 4.6))
ns = sorted(qlike, key=qlike.get); vs = [qlike[n] for n in ns]
bb = a.bar(ns, vs, color=c(ns))
for b, v in zip(bb, vs):
    a.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5)
a.set_title("Forecast accuracy: average QLIKE (lower = better)")
a.set_ylabel("avg QLIKE"); a.set_ylim(0, max(vs) * 1.18)
a.legend(handles=LEG, frameon=False, fontsize=8, loc="upper left")
save(f, "fig1_accuracy")

#GRAPH 2: DM
f, a = plt.subplots(figsize=(7, 4.6))
ns = sorted(dmw, key=dmw.get); vs = [dmw[n] for n in ns]
bb = a.bar(ns, vs, color=c(ns))
for b, v in zip(bb, vs):
    a.text(b.get_x() + b.get_width() / 2, v + 0.6, str(int(v)), ha="center", fontsize=9)
a.set_title("Diebold–Mariano: stocks (of 39) where HAR significantly wins")
a.set_ylabel("# stocks (of 39)"); a.set_ylim(0, 43)
a.legend(handles=LEG, frameon=False, fontsize=8, loc="upper left")
save(f, "fig2_dm")

#Graph 3: Regime QLIKE
f, a = plt.subplots(figsize=(7, 4.6))
order = [m for m in ["HAR", "LASSO", "RF", "GARCH", "EGARCH"] if m in regime]
nrm = [regime[m][0] for m in order]; hi = [regime[m][1] for m in order]
x = np.arange(len(order)); w = 0.38
a.bar(x - w / 2, nrm, w, label="Normal", color="#9ecae1")
a.bar(x + w / 2, hi, w, label="High-vol", color="#de2d26")
a.set_xticks(x); a.set_xticklabels(order)
a.set_title("QLIKE by regime — RF collapses in high volatility")
a.set_ylabel("avg QLIKE"); a.legend(frameon=False, fontsize=9); a.set_ylim(0, max(hi) * 1.22)
save(f, "fig3_regime")

#Graph 4: Disconnect in H4
f, a = plt.subplots(figsize=(7, 4.8))
off = {"HAR": (6, -12), "LASSO": (-12, 8), "RF": (7, 4), "GARCH": (8, -3),
       "EGARCH": (7, 6), "EWMA": (7, 4), "HV": (7, 4)}
for n in set(qlike) & set(v99):
    a.scatter(qlike[n], v99[n], s=110, color=COL[GRP[n]], zorder=3, edgecolor="white", linewidth=0.9)
    a.annotate(n, (qlike[n], v99[n]), textcoords="offset points", xytext=off.get(n, (6, 5)), fontsize=9)
a.set_title("The disconnect: forecast accuracy vs 99% VaR adequacy")
a.set_xlabel("avg QLIKE   (worse forecast →)")
a.set_ylabel("99% VaR: stocks passing of 39   (better risk →)")
xs = list(qlike.values()); a.set_xlim(min(xs) * 0.95, max(xs) * 1.05)
a.set_ylim(0, max(v99.values()) * 1.15)
a.legend(handles=LEG, frameon=False, fontsize=8, loc="upper left")
save(f, "fig4_disconnect")

#Graph 5: VaR level flip
f, a = plt.subplots(figsize=(7.4, 4.6))
order = [m for m in ["HAR", "LASSO", "RF", "GARCH", "EGARCH", "EWMA", "HV"] if m in v95 and m in v99]
b95 = [v95[m] for m in order]; b99 = [v99[m] for m in order]
x = np.arange(len(order)); w = 0.38
a.bar(x - w / 2, b95, w, label="95% VaR", color="#74a9cf")
a.bar(x + w / 2, b99, w, label="99% VaR", color="#045a8d")
a.set_xticks(x); a.set_xticklabels(order)
a.set_title("VaR adequacy flips with the confidence level (pass count, of 39)")
a.set_ylabel("# stocks passing (cc, Student-t)")
a.legend(frameon=False, fontsize=9); a.set_ylim(0, 40)
save(f, "fig5_var_levels")

print("\nAll 5 figures written to", FIG)
