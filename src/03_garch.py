"""Step 3 - GARCH(1,1) and EGARCH(1,1,1) benchmarks.

Fitted on daily log returns with the `arch` package, expanding window, parameters
re-estimated every 21 trading days. The one-day-ahead conditional variance is
compared with realized variance rv5_{t+1} using the same losses as the other models.
Daily forecasts are saved for the Diebold-Mariano, VaR and regime scripts."""

import os
import warnings
import pandas as pd
import numpy as np
from arch import arch_model

warnings.filterwarnings("ignore")

# Config
CSV          = "data/volare_clean.csv"
OUT_DIR      = "results/tables"
FCAST_DIR    = "results/forecasts"
SPLIT        = pd.Timestamp("2020-01-01")
REFIT_EVERY  = 21
SCALE        = 100.0   # arch is numerically more stable on percentage returns
BIAS_CORRECT = False   # True rescales GARCH variance to the rv5 level (robustness check)

MODELS = [
    ("GARCH",  "Garch",  0),
    ("EGARCH", "EGARCH", 1),
]
DIST = "normal"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FCAST_DIR, exist_ok=True)

# Core function
def garch_oos_one(g, vol, o):
    """One stock, one model. Returns (metrics_dict, forecasts_df) or (None, None)."""
    g = g.sort_values("date").reset_index(drop=True)
    g["ret"]     = np.log(g["close_price"] / g["close_price"].shift(1)) * SCALE
    g["rv_next"] = g["rv5"].shift(-1)
    g = g.dropna(subset=["ret", "rv_next"]).reset_index(drop=True)

    r       = pd.Series(g["ret"].values,     index=g["date"])
    rv_next = pd.Series(g["rv_next"].values, index=g["date"])

    oos_dates = r.index[r.index >= SPLIT]
    if len(oos_dates) < 50:
        return None, None

    preds = pd.Series(index=oos_dates, dtype=float)

    c = 1.0
    if BIAS_CORRECT:
        ins = r.index < SPLIT
        c = np.nanmean(g.loc[g["date"] < SPLIT, "rv5"].values) / \
            np.nanmean((r[ins].values / SCALE) ** 2)

    anchors = list(range(0, len(oos_dates), REFIT_EVERY))
    for ai, a in enumerate(anchors):
        seg_start = oos_dates[a]
        seg_stop  = oos_dates[anchors[ai + 1]] if ai + 1 < len(anchors) else None
        try:
            am  = arch_model(r, mean="Constant", vol=vol, p=1, o=o, q=1, dist=DIST)
            res = am.fit(last_obs=seg_start, disp="off")
            fc  = res.forecast(start=seg_start, horizon=1, reindex=False)
            var_pct2 = fc.variance.iloc[:, 0]
        except Exception:
            continue
        keep = var_pct2.loc[seg_start:] if seg_stop is None \
            else var_pct2.loc[seg_start:seg_stop].iloc[:-1]
        preds.loc[keep.index] = keep.values

    fcast  = preds / (SCALE ** 2) * c
    actual = rv_next.loc[oos_dates]

    d = pd.DataFrame({"forecast_rv": fcast, "actual_rv": actual}).dropna()
    d = d[(d.forecast_rv > 0) & (d.actual_rv > 0)]
    if len(d) < 50:
        return None, None

    d = d.reset_index().rename(columns={"index": "date"})
    d["forecast_log"] = np.log(d["forecast_rv"])
    d["y_log"]        = np.log(d["actual_rv"])
    d["error_log"]    = d["y_log"] - d["forecast_log"]
    d["error_rv"]     = d["actual_rv"] - d["forecast_rv"]
    d["qlike"]        = d["actual_rv"] / d["forecast_rv"] - np.log(d["actual_rv"] / d["forecast_rv"]) - 1

    metrics = {
        "mse_log":  float(np.mean(d["error_log"] ** 2)),
        "mae_log":  float(np.mean(np.abs(d["error_log"]))),
        "rmse_log": float(np.sqrt(np.mean(d["error_log"] ** 2))),
        "qlike":    float(np.mean(d["qlike"])),
        "n":        len(d),
    }
    return metrics, d

# Driver
def main():
    suffix = "_bc" if BIAS_CORRECT else ""
    df = pd.read_csv(CSV, parse_dates=["date"]).sort_values(["symbol", "date"])
    symbols = sorted(df["symbol"].unique())
    print(f"Loaded {len(df)} rows | {len(symbols)} stocks | split {SPLIT.date()} | "
          f"expanding window | refit every {REFIT_EVERY}d | bias_correct={BIAS_CORRECT}")

    agg_rows = []
    for label, vol, o in MODELS:
        rows, fcasts = [], []
        for sym in symbols:
            metrics, d = garch_oos_one(df[df.symbol == sym].copy(), vol, o)
            if metrics is None:
                print(f"[{label}] {sym}: skipped")
                continue
            rows.append({"symbol": sym, "model": label, **metrics})
            d["symbol"] = sym
            d["model"]  = label
            fcasts.append(d)
            print(f"[{label}] {sym}: RMSE_log={metrics['rmse_log']:.4f}, "
                  f"MAE_log={metrics['mae_log']:.4f}, QLIKE={metrics['qlike']:.4f}")

        res_df = pd.DataFrame(rows)
        res_df.to_csv(f"{OUT_DIR}/{label.lower()}_oos_by_stock{suffix}.csv", index=False)
        pd.concat(fcasts, ignore_index=True).to_csv(
            f"{FCAST_DIR}/{label.lower()}_oos_forecasts{suffix}.csv", index=False)
        print(f"Saved {OUT_DIR}/{label.lower()}_oos_by_stock{suffix}.csv and "
              f"{FCAST_DIR}/{label.lower()}_oos_forecasts{suffix}.csv")

        agg_rows.append({
            "model":        label,
            "avg_mse_log":  res_df["mse_log"].mean(),
            "avg_mae_log":  res_df["mae_log"].mean(),
            "avg_rmse_log": res_df["rmse_log"].mean(),
            "avg_qlike":    res_df["qlike"].mean(),
            "median_qlike": res_df["qlike"].median(),
            "n_stocks":     len(res_df),
        })

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(f"{OUT_DIR}/garch_family_summary{suffix}.csv", index=False)
    print("\nAggregate summary:")
    print(agg.to_string(index=False))

if __name__ == "__main__":
    main()
