"""Step 2 - HAR-RV benchmark (Corsi, 2009).

One-day-ahead forecasts of log rv5_{t+1} from daily, weekly (5d) and monthly (22d)
averages of log RV. Expanding window, re-estimated every day from 2020-01-01 on,
with a log-normal (Jensen) correction when mapping back to variance levels."""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT_FILE = "data/volare_clean.csv"
TABLE_DIR = "results/tables"
FCAST_DIR = "results/forecasts"
SPLIT_DATE = pd.Timestamp("2020-01-01")
MIN_TRAIN = 250
MIN_TEST = 100

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FCAST_DIR, exist_ok=True)


def qlike(actual, forecast, eps=1e-12):
    actual = np.maximum(actual, eps)
    forecast = np.maximum(forecast, eps)
    return actual / forecast - np.log(actual / forecast) - 1


def make_har_data(stock_df):
    stock_df = stock_df.sort_values("date").set_index("date").copy()
    stock_df = stock_df[stock_df["rv5"] > 0].copy()
    log_rv = np.log(stock_df["rv5"])

    data = pd.DataFrame(index=stock_df.index)
    data["y_log"] = log_rv.shift(-1)              # target: log rv5_{t+1}
    data["d"] = log_rv                            # daily component
    data["w"] = log_rv.rolling(5).mean()          # weekly (5d)
    data["m"] = log_rv.rolling(22).mean()         # monthly (22d)
    data["actual_rv"] = stock_df["rv5"].shift(-1) # next-day rv5 in levels
    return data.dropna()


def estimate_har_oos(data, split_date):
    """Expanding-window one-day-ahead HAR-RV forecasts (log space) with Jensen correction."""
    data = data.sort_index()
    dates = data.index.values
    X = np.column_stack([np.ones(len(data)), data["d"].values, data["w"].values, data["m"].values])
    y = data["y_log"].values
    actual_rv = data["actual_rv"].values

    test_pos = np.where(data.index >= split_date)[0]
    test_pos = test_pos[test_pos >= MIN_TRAIN]            # ensure enough history for the first fit
    if len(test_pos) < MIN_TEST:
        return None, None

    rows = []
    for i in test_pos:
        Xtr, ytr = X[:i], y[:i]                            # expanding window: everything strictly before t
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        resid = ytr - Xtr @ beta
        resid_var = resid @ resid / (len(ytr) - Xtr.shape[1])   # unbiased residual variance
        fc_log = X[i] @ beta
        fc_rv = np.exp(fc_log + 0.5 * resid_var)           # Jensen / log-normal correction
        rows.append((dates[i], y[i], fc_log, actual_rv[i], fc_rv))

    test = pd.DataFrame(rows, columns=["date", "y_log", "forecast_log", "actual_rv", "forecast_rv"])
    test["error_log"] = test["y_log"] - test["forecast_log"]
    test["error_rv"] = test["actual_rv"] - test["forecast_rv"]
    test["qlike"] = qlike(test["actual_rv"], test["forecast_rv"])

    # representative in-sample coefficients (initial training window) with Newey-West SEs
    train0 = data.loc[data.index < split_date]
    Xt = sm.add_constant(train0[["d", "w", "m"]], has_constant="add")
    fit0 = sm.OLS(train0["y_log"], Xt).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    results = {
        "n_train_init": len(train0),
        "n_test": len(test),
        "r2_train": fit0.rsquared,
        "const": fit0.params["const"],
        "beta_d": fit0.params["d"],
        "beta_w": fit0.params["w"],
        "beta_m": fit0.params["m"],
        "sum_beta": fit0.params[["d", "w", "m"]].sum(),
        "mse_log": np.mean(test["error_log"] ** 2),
        "mae_log": np.mean(np.abs(test["error_log"])),
        "rmse_log": np.sqrt(np.mean(test["error_log"] ** 2)),
        "mse_rv": np.mean(test["error_rv"] ** 2),
        "mae_rv": np.mean(np.abs(test["error_rv"])),
        "qlike": np.mean(test["qlike"]),
    }
    return results, test


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    symbols = sorted(df["symbol"].dropna().unique())

    print(f"Loaded {len(df)} rows | {len(symbols)} stocks | split {SPLIT_DATE.date()} | expanding window\n")

    all_results, all_forecasts = [], []
    for symbol in symbols:
        try:
            data = make_har_data(df[df["symbol"] == symbol])
            results, forecasts = estimate_har_oos(data, SPLIT_DATE)
            if results is None:
                print(f"Skipping {symbol}: not enough observations")
                continue
            results["symbol"] = symbol
            all_results.append(results)
            forecasts["symbol"] = symbol
            all_forecasts.append(forecasts)
            print(f"{symbol}: RMSE_log={results['rmse_log']:.4f}, "
                  f"MAE_log={results['mae_log']:.4f}, QLIKE={results['qlike']:.4f}")
        except Exception as e:
            print(f"Error for {symbol}: {e}")

    if not all_results:
        raise RuntimeError("No stock results produced. Check data/volare_clean.csv.")

    cols = ["symbol", "n_train_init", "n_test", "r2_train", "const",
            "beta_d", "beta_w", "beta_m", "sum_beta",
            "mse_log", "mae_log", "rmse_log", "mse_rv", "mae_rv", "qlike"]
    results_df = pd.DataFrame(all_results)[cols].sort_values("qlike")
    results_df.to_csv(os.path.join(TABLE_DIR, "har_oos_by_stock.csv"), index=False)

    pd.concat(all_forecasts, ignore_index=True).to_csv(
        os.path.join(FCAST_DIR, "har_oos_forecasts.csv"), index=False)

    summary = pd.DataFrame({
        "model": ["HAR-RV"],
        "avg_mse_log": [results_df["mse_log"].mean()],
        "avg_mae_log": [results_df["mae_log"].mean()],
        "avg_rmse_log": [results_df["rmse_log"].mean()],
        "avg_qlike": [results_df["qlike"].mean()],
        "median_qlike": [results_df["qlike"].median()],
        "n_stocks": [results_df["symbol"].nunique()],
    })
    summary.to_csv(os.path.join(TABLE_DIR, "har_oos_summary.csv"), index=False)

    print("\nAggregate summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
