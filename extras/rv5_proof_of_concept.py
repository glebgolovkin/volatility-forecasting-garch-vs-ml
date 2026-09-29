"""Proof of concept: compute 5-minute realized variance (RV5) for one stock from
intraday Yahoo Finance data, to show how the VOLARE rv5 measure is built."""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Settings for the stock and proof.
Ticker = 'AAPL'
Interval = '5m'
Period = '1mo'

print(f"Downloading {Interval} data for {Ticker} over the last {Period}")

#Download the data
data = yf.download(Ticker, interval = Interval, period = Period, progress = False)
if data.empty:
    raise SystemExit("No data downloaded. Try period= '5d' or a different ticker.")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

prices = data['Close'].dropna().copy()
prices.index = pd.to_datetime(prices.index)
print(f"Downloaded {len(prices)} five-minute price observations.")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")

#Then we compute 5min log returns (each day).
df = prices.to_frame(name="price")
df["date"]=df.index.date
df["log_price"]=np.log(df["price"])
df["ret"]=df.groupby("date")["log_price"].diff()

# Then aggregate into daily realized variance. Our RV5.
df["ret_squared"]=df["ret"]**2
rv5 = df.groupby("date")["ret_squared"].sum()
rv5 = rv5[rv5 > 0]
ann_vol = np.sqrt(rv5 * 252) * 100
results = pd.DataFrame({"rv5": rv5, "ann_vol_%": ann_vol})
print("\n===Daily Realized Variance (RV5)===")
print(results.round(6).to_string())
print(f"\nMean rv5: {rv5.mean():.6f}")
print(f"Mean annualized volatility: {ann_vol.mean():.6f}%")

results.to_csv(f"rv5_proof_of_concept_{Ticker}.csv")
print(f"\nSaved results to rv5_proof_of_concept_{Ticker}.csv")

#Last: Save and plot the results.
fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
ax[0].plot(rv5.index, rv5.values, marker="o", markersize=3, color="#0f6e56", label="RV5")
ax[0].set_title(f"{Ticker} - Daily Realized Variance (RV5)")
ax[0].set_ylabel("RV5")
ax[0].grid(alpha=0.3)
ax[1].plot(ann_vol.index, ann_vol.values, marker="o", markersize=3, color="#185fa5")
ax[1].set_title(f"{Ticker} - Annualized Volatility")
ax[1].set_ylabel("Annualized Volatility (%)")
ax[1].set_xlabel("Date")
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"rv5_proof_of_concept_{Ticker}.png", dpi=120)
print(f"saved plot to rv5_proof_of_concept_{Ticker}.png")
print("\nDone - RV5 computed and plotted from intraday data!")
