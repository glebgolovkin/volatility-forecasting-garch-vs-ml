"""Step 4 - LASSO and Random Forest (H2).

Aligned 1:1 with the HAR-RV script: same target, split, expanding window, QLIKE and
Jensen back-transform. Features: the HAR components plus two extra lags of log RV,
log bipower variation and log realized quarticity. Refit every 21 trading days."""

import os
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Config

INPUT_FILE  = "data/volare_clean.csv"
TABLE_DIR   = "results/tables"
FCAST_DIR   = "results/forecasts"
SPLIT_DATE  = pd.Timestamp("2020-01-01")
MIN_TRAIN   = 250
MIN_TEST    = 100
REFIT_EVERY = 21                             # monthly refit (HAR is refit daily)
FEATURE_SET = "extended"                     # "extended" or HAR core only
EPS         = 1e-12

MODELS = ["LASSO", "RandomForest"]

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FCAST_DIR, exist_ok=True)

def qlike(actual, forecast, eps=EPS):
    actual = np.maximum(actual, eps)
    forecast = np.maximum(forecast, eps)
    return actual / forecast - np.log(actual / forecast) - 1

def make_ml_data(stock_df, feature_set):
    """Same target/HAR core as HAR script, with optional extended features."""
    s = stock_df.sort_values("date").set_index("date").copy()
    s = s[s["rv5"] > 0].copy()
    log_rv = np.log(s["rv5"])

    d = pd.DataFrame(index=s.index)
    d["y_log"]     = log_rv.shift(-1)
    d["actual_rv"] = s["rv5"].shift(-1)

    # HAR core

    d["d"] = log_rv
    d["w"] = log_rv.rolling(5).mean()
    d["m"] = log_rv.rolling(22).mean()
    feats = ["d", "w", "m"]

    if feature_set == "extended":
        d["d_lag1"] = log_rv.shift(1)
        d["d_lag2"] = log_rv.shift(2)
        feats += ["d_lag1", "d_lag2"]
        if "bv5" in s.columns:
            d["log_bv5"] = np.log(np.maximum(s["bv5"].values, EPS))
            feats.append("log_bv5")
        if "rq5" in s.columns:
            d["log_rq5"] = np.log(np.maximum(s["rq5"].values, EPS))
            feats.append("log_rq5")

    return d.dropna(), feats

def fit_predict_block(model_name, Xtr, ytr, Xblock):
    """Fit on the expanding training window, predict the whole block. Returns (preds_log, resid_var)."""
    if model_name == "LASSO":
        scaler = StandardScaler().fit(Xtr)              # scaler fit on TRAIN only (no leakage)
        Xtr_s = scaler.transform(Xtr)
        est = LassoCV(cv=5, max_iter=10000, n_jobs=-1, random_state=0).fit(Xtr_s, ytr)
        resid = ytr - est.predict(Xtr_s)
        resid_var = float(resid @ resid / max(len(ytr) - 1, 1))
        preds_log = est.predict(scaler.transform(Xblock))
    else:  # Random Forest
        est = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10,
            bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
        ).fit(Xtr, ytr)
        oob = est.oob_prediction_                        # honest residuals (out-of-bag)
        ok = np.isfinite(oob)
        resid = ytr[ok] - oob[ok]
        resid_var = float(resid @ resid / max(ok.sum() - 1, 1))
        preds_log = est.predict(Xblock)
    return preds_log, resid_var

def estimate_ml_oos(data, feats, split_date, model_name, refit_every):
    data = data.sort_index()
    dates     = data.index.values
    X         = data[feats].values
    y         = data["y_log"].values
    actual_rv = data["actual_rv"].values

    test_pos = np.where(data.index >= split_date)[0]
    test_pos = test_pos[test_pos >= MIN_TRAIN]
    if len(test_pos) < MIN_TEST:
        return None, None
    rows = []
    for a in range(0, len(test_pos), refit_every):
        block = test_pos[a:a + refit_every]
        i0 = block[0]
        Xtr, ytr = X[:i0], y[:i0]
        try:
            preds_log, resid_var = fit_predict_block(model_name, Xtr, ytr, X[block])
        except Exception:
            continue
        fc_rv = np.exp(preds_log + 0.5 * resid_var)        # Jensen / log-normal correction
        for j, i in enumerate(block):
            rows.append((dates[i], y[i], preds_log[j], actual_rv[i], fc_rv[j]))

    if len(rows) < MIN_TEST:
        return None, None

    test = pd.DataFrame(rows, columns=["date", "y_log", "forecast_log", "actual_rv", "forecast_rv"])
    test["error_log"] = test["y_log"] - test["forecast_log"]
    test["error_rv"]  = test["actual_rv"] - test["forecast_rv"]
    test["qlike"]     = qlike(test["actual_rv"], test["forecast_rv"])

    results = {
        "n_test":   len(test),
        "mse_log":  np.mean(test["error_log"] ** 2),
        "mae_log":  np.mean(np.abs(test["error_log"])),
        "rmse_log": np.sqrt(np.mean(test["error_log"] ** 2)),
        "mse_rv":   np.mean(test["error_rv"] ** 2),
        "mae_rv":   np.mean(np.abs(test["error_rv"])),
        "qlike":    np.mean(test["qlike"]),
    }
    return results, test


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    symbols = sorted(df["symbol"].dropna().unique())
    print(f"Loaded {len(df)} rows | {len(symbols)} stocks | split {SPLIT_DATE.date()} | "
          f"expanding window | refit every {REFIT_EVERY}d | features={FEATURE_SET}\n")

    agg_rows = []
    for model_name in MODELS:
        all_results, all_forecasts = [], []
        for symbol in symbols:
            data, feats = make_ml_data(df[df["symbol"] == symbol], FEATURE_SET)
            results, forecasts = estimate_ml_oos(data, feats, SPLIT_DATE, model_name, REFIT_EVERY)
            if results is None:
                print(f"[{model_name}] {symbol}: skipped")
                continue
            results["symbol"] = symbol
            all_results.append(results)
            forecasts["symbol"] = symbol
            forecasts["model"] = model_name
            all_forecasts.append(forecasts)
            print(f"[{model_name}] {symbol}: RMSE_log={results['rmse_log']:.4f}, "
                  f"MAE_log={results['mae_log']:.4f}, QLIKE={results['qlike']:.4f}")

        res_df = pd.DataFrame(all_results)
        tag = model_name.lower()
        res_df.to_csv(os.path.join(TABLE_DIR, f"{tag}_oos_by_stock.csv"), index=False)
        pd.concat(all_forecasts, ignore_index=True).to_csv(
            os.path.join(FCAST_DIR, f"{tag}_oos_forecasts.csv"), index=False)
        print(f"Saved {TABLE_DIR}/{tag}_oos_by_stock.csv and {FCAST_DIR}/{tag}_oos_forecasts.csv")

        agg_rows.append({
            "model":        model_name,
            "avg_mse_log":  res_df["mse_log"].mean(),
            "avg_mae_log":  res_df["mae_log"].mean(),
            "avg_rmse_log": res_df["rmse_log"].mean(),
            "avg_qlike":    res_df["qlike"].mean(),
            "median_qlike": res_df["qlike"].median(),
            "n_stocks":     len(res_df),
        })

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(os.path.join(TABLE_DIR, f"ml_summary_{FEATURE_SET}.csv"), index=False)
    print("\nAggregate summary:")
    print(agg.to_string(index=False))

if __name__ == "__main__":
    main()
