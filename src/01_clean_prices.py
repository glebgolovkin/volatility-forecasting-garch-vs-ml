"""Step 1 - clean the raw VOLARE file.

Adjusts close/open/high/low prices for stock splits (VOLARE prices are unadjusted),
drops META (ticker-change data break, see README) and keeps 2015-2025.
Output: data/volare_clean.csv (39 stocks)."""

import numpy as np
import pandas as pd

CSV_IN  = "data/realized_variance_stocks.csv"
CSV_OUT = "data/volare_clean.csv"

SPLITS = [
    ("AAPL", "2020-08-31", 1/4),  ("AMZN",  "2022-06-06", 1/20),
    ("GOOGL","2022-07-18", 1/20), ("NVDA",  "2021-07-20", 1/4),
    ("NVDA", "2024-06-10", 1/10), ("TSLA",  "2020-08-31", 1/5),
    ("TSLA", "2022-08-25", 1/3),  ("NFLX",  "2015-07-15", 1/7),
    ("NFLX", "2025-11-17", 1/10), ("V",     "2015-03-19", 1/4),
    ("WMT",  "2024-02-26", 1/3),  ("SHW",   "2021-04-01", 1/3),
    ("NKE",  "2015-12-24", 1/2),  ("GE",    "2021-08-02", 8.0),  # 1:8 reverse split
]

df = pd.read_csv(CSV_IN, parse_dates=["date"])
df = df[df.symbol != "META"].copy()
df = df[df.date.dt.year <= 2025].copy()
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

df["adj_factor"] = 1.0
for sym, dt, mult in SPLITS:
    mask = (df.symbol == sym) & (df.date < pd.Timestamp(dt))
    df.loc[mask, "adj_factor"] *= mult
for col in ["open_price", "close_price", "high_price", "low_price"]:
    df[col] = df[col] * df["adj_factor"]
df = df.drop(columns="adj_factor")

ret = df.groupby("symbol")["close_price"].apply(lambda s: np.log(s/s.shift(1))).reset_index(level=0, drop=True)
print("Remaining |log return| > 0.4 after adjustment (only genuine AMD moves expected):")
print(df.loc[ret.abs() > 0.4, ["symbol", "date", "close_price"]].assign(ret=ret[ret.abs() > 0.4]).to_string(index=False))

df.to_csv(CSV_OUT, index=False)
print("\nSaved:", CSV_OUT, "| stocks:", df.symbol.nunique(), "| rows:", len(df))
