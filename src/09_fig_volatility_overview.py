"""Figure - cross-sectional average annualized volatility across the 39 stocks (2015-2025).
Reads data/volare_clean.csv, writes results/figures/fig0_volatility_overview.png."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA  = "data/volare_clean.csv"
FIG   = "results/figures"
SPLIT = pd.Timestamp("2020-01-01")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

NAVY, GREY, RED, ORANGE = "#1f4e79", "#888888", "#b22222", "#d68910"

df = pd.read_csv(DATA, parse_dates=["date"])
df = df[df["rv5"] > 0]

# Cross-sectional mean daily RV -> annualized volatility in %
ann_vol = np.sqrt(252 * df.groupby("date")["rv5"].mean()) * 100
# 21-day rolling mean to show the trend through the daily noise
trend = ann_vol.rolling(21, min_periods=5).mean()

fig, ax = plt.subplots(figsize=(9.2, 4.4))
ax.plot(ann_vol.index, ann_vol.values, color=NAVY, linewidth=0.6, alpha=0.45,
        label="Daily cross-sectional average")
ax.plot(trend.index, trend.values, color=NAVY, linewidth=1.8,
        label="21-day moving average")

# Train / out-of-sample split
ax.axvline(SPLIT, color=GREY, linestyle="--", linewidth=1)
ax.text(SPLIT - pd.Timedelta(days=20), ann_vol.max() * 0.93, "train | out-of-sample",
        color="#555555", fontsize=8.5, ha="right")

# Stress periods
ax.annotate("COVID-19\n(Mar 2020)",
            xy=(pd.Timestamp("2020-03-18"), ann_vol.max()),
            xytext=(pd.Timestamp("2021-02-01"), ann_vol.max() * 0.92),
            arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=8.5)
ax.annotate("2022 Rate Shock",
            xy=(pd.Timestamp("2022-06-15"), ann_vol.loc["2022-01-01":"2022-12-31"].max()),
            xytext=(pd.Timestamp("2022-09-01"), ann_vol.max() * 0.62),
            arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE, fontsize=8.5)

ax.set_title("Cross-sectional average annualized volatility, 39 large-cap stocks (2015-2025)")
ax.set_ylabel("Annualized volatility (%)")
ax.set_xlabel("Date")
ax.set_ylim(0, ann_vol.max() * 1.08)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(frameon=False, fontsize=9, loc="upper right")

fig.tight_layout()
fig.savefig(f"{FIG}/fig0_volatility_overview.png", bbox_inches="tight")
plt.close(fig)
print(f"saved {FIG}/fig0_volatility_overview.png")
