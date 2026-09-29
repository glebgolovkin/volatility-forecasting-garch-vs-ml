"""Step 5 - naive benchmarks: rolling historical variance (HV) and EWMA / RiskMetrics.

Both forecast next-day variance from daily returns only (no intraday data) and are
saved in the same format as the other models."""

import os
import numpy as np
import pandas as pd

# Config
DATA    = "data/volare_clean.csv"
FCAST_DIR = "results/forecasts"
OUT_DIR   = "results/tables"
SPLIT   = pd.Timestamp("2020-01-01")
W       = 22 #HV rolling window (approx 1 trading month)
LAMBDA  = 0.94 #EWMA/RiskMetrics decay
EPS     = 1e-12

os.makedirs(FCAST_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def qlike(actual, forecast):
    a = np.maximum(np.asarray(actual, float), EPS)
    f = np.maximum(np.asarray(forecast, float), EPS)
    return a / f - np.log(a / f) - 1

def build(g, kind):
    """One stock -> daily forecasts of rv5_{t+1} from HV or EWMA. Same columns as HAR file."""
    g = g.sort_values("date").reset_index(drop=True)
    g["r"]  = np.log(g["close_price"] / g["close_price"].shift(1))
    g["r2"] = g["r"] ** 2
    # forecast made at t for day t+1's variance (built on the FULL series so EWMA is burned in)
    if kind == "HV":
        g["fcast"] = g["r2"].rolling(W).mean() #Mean of last W squared returns
    else:
        g["fcast"] = g["r2"].ewm(alpha=1 - LAMBDA, adjust=False).mean()
    g["rv_next"] = g["rv5"].shift(-1)

    d = g[(g["date"] >= SPLIT)].dropna(subset=["fcast", "rv_next"]).copy()
    d = d[(d["fcast"] > 0) & (d["rv_next"] > 0)]
    if len(d) < 100:
        return None, None

    out = pd.DataFrame({"date": d["date"].values})
    out["forecast_rv"] = d["fcast"].values
    out["actual_rv"]   = d["rv_next"].values
    out["forecast_log"] = np.log(out["forecast_rv"])
    out["y_log"]        = np.log(out["actual_rv"])
    out["error_log"]    = out["y_log"] - out["forecast_log"]
    out["error_rv"]     = out["actual_rv"] - out["forecast_rv"]
    out["qlike"]        = qlike(out["actual_rv"], out["forecast_rv"])

    metrics = {
        "mse_log":  float(np.mean(out["error_log"] ** 2)),
        "mae_log":  float(np.mean(np.abs(out["error_log"]))),
        "rmse_log": float(np.sqrt(np.mean(out["error_log"] ** 2))),
        "qlike":    float(np.mean(out["qlike"])),
        "n":        len(out),
    }
    return metrics, out

def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["symbol", "date"])
    symbols = sorted(df["symbol"].unique())
    print(f"Loaded {len(df)} rows | {len(symbols)} stocks | split {SPLIT.date()} | HV W={W} | EWMA lambda={LAMBDA}\n")

    agg = []
    for kind in ["HV", "EWMA"]:
        rows, fcasts = [], []
        for sym in symbols:
            metrics, out = build(df[df.symbol == sym].copy(), kind)
            if metrics is None:
                print(f"[{kind}] {sym}: skipped")
                continue
            rows.append({"symbol": sym, "model": kind, **metrics})
            out["symbol"] = sym
            out["model"]  = kind
            fcasts.append(out)
            print(f"[{kind}] {sym}: RMSE_log={metrics['rmse_log']:.4f}, "
                  f"MAE_log={metrics['mae_log']:.4f}, QLIKE={metrics['qlike']:.4f}")

        tag = kind.lower()
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{tag}_oos_by_stock.csv"), index=False)
        pd.concat(fcasts, ignore_index=True).to_csv(
            os.path.join(FCAST_DIR, f"{tag}_oos_forecasts.csv"), index=False)
        print(f"Saved {FCAST_DIR}/{tag}_oos_forecasts.csv\n")

        res_df = pd.DataFrame(rows)
        agg.append({
            "model":        kind,
            "avg_mse_log":  res_df["mse_log"].mean(),
            "avg_mae_log":  res_df["mae_log"].mean(),
            "avg_rmse_log": res_df["rmse_log"].mean(),
            "avg_qlike":    res_df["qlike"].mean(),
            "median_qlike": res_df["qlike"].median(),
            "n_stocks":     len(res_df),
        })

    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(os.path.join(OUT_DIR, "naive_summary.csv"), index=False)
    print("Aggregate summary (naive floor):")
    print(agg_df.to_string(index=False))

if __name__ == "__main__":
    main()
