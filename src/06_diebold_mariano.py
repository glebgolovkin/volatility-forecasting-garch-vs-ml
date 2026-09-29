"""Step 6 - Diebold-Mariano tests: HAR-RV (benchmark) vs every other model.

Reads the daily forecast files produced by the model scripts, recomputes a single canonical QLIKE loss for every model, and runs a HAC (Newey-West) DM test per stock
with the Harvey-Leybourne-Newbold small-sample correction.
The sign convention is as follows: loss_diff = L_competitor - L_HAR.
DM > 0  => competitor has higher loss => HAR is better.
DM < 0  => competitor is better. """

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

RES_DIR   = "results/forecasts"
OUT_DIR   = "results/tables"
ALPHA     = 0.05
EPS       = 1e-12
HORIZON   = 1   # one-day-ahead forecasts

BENCH_NAME = "HAR"
BENCH_FILE = "har_oos_forecasts.csv"
COMPETITORS = {
    "GARCH":  "garch_oos_forecasts.csv",
    "EGARCH": "egarch_oos_forecasts.csv",
    "LASSO":  "lasso_oos_forecasts.csv",
    "RF":     "randomforest_oos_forecasts.csv",
    "HV":   "hv_oos_forecasts.csv",
    "EWMA": "ewma_oos_forecasts.csv",
}

os.makedirs(OUT_DIR, exist_ok=True)

# Core functions
def qlike_loss(actual, forecast):
    a = np.maximum(np.asarray(actual, float), EPS)
    f = np.maximum(np.asarray(forecast, float), EPS)
    return a / f - np.log(a / f) - 1


def nw_maxlags(T):
    """Newey-West (1994) automatic bandwidth."""
    return max(int(np.floor(4 * (T / 100.0) ** (2 / 9))), 1)

def dm_test(loss_comp, loss_bench):
    """Returns (dm_stat, p_value, mean_loss_diff, T). DM>0 favours the benchmark (HAR)."""
    d = np.asarray(loss_comp, float) - np.asarray(loss_bench, float)
    T = len(d)
    if T < 30:
        return np.nan, np.nan, float(np.mean(d)) if T else np.nan, T

    X = np.ones((T, 1))
    fit = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_maxlags(T)})
    dm = float(fit.tvalues[0])
    # Harvey-Leybourne-Newbold small-sample correction (where h = HORIZON)
    h = HORIZON
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * corr
    p_hln = 2 * (1 - stats.t.cdf(abs(dm_hln), df=T - 1))
    return dm_hln, p_hln, float(np.mean(d)), T

def load_losses(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["loss"] = qlike_loss(df["actual_rv"], df["forecast_rv"])
    return df[["symbol", "date", "loss"]]

def main():
    bench = load_losses(os.path.join(RES_DIR, BENCH_FILE)).rename(columns={"loss": "loss_bench"})

    by_stock, summary = [], []
    for cname, cfile in COMPETITORS.items():
        comp = load_losses(os.path.join(RES_DIR, cfile)).rename(columns={"loss": "loss_comp"})
        merged = comp.merge(bench, on=["symbol", "date"], how="inner")

        har_sig = comp_sig = nsig = 0
        dms, diffs = [], []
        for sym, g in merged.groupby("symbol"):
            dm, p, dbar, T = dm_test(g["loss_comp"].values, g["loss_bench"].values)
            dms.append(dm); diffs.append(dbar)
            sig = (p < ALPHA)
            if sig and dm > 0:
                har_sig += 1; favors = BENCH_NAME
            elif sig and dm < 0:
                comp_sig += 1; favors = cname
            else:
                nsig += 1; favors = "tie"
            by_stock.append({
                "comparison": f"{BENCH_NAME}_vs_{cname}", "symbol": sym,
                "dm_stat": dm, "p_value": p, "mean_loss_diff": dbar,
                "n": T, "significant": sig, "favors": favors,
            })

        summary.append({
            "comparison":      f"{BENCH_NAME}_vs_{cname}",
            "n_stocks":        har_sig + comp_sig + nsig,
            "HAR_better_sig":  har_sig,
            f"{cname}_better_sig": comp_sig,
            "not_significant": nsig,
            "mean_dm":         float(np.nanmean(dms)),
            "mean_loss_diff":  float(np.nanmean(diffs)),
        })

    by_stock_df = pd.DataFrame(by_stock)
    by_stock_df.to_csv(os.path.join(OUT_DIR, "dm_by_stock.csv"), index=False)

    # Tidy summary
    summ = pd.DataFrame([{
        "comparison":      r["comparison"],
        "n_stocks":        r["n_stocks"],
        "HAR_better_sig":  r["HAR_better_sig"],
        "comp_better_sig": r.get(f"{r['comparison'].split('_vs_')[1]}_better_sig", 0),
        "not_significant": r["not_significant"],
        "mean_dm":         round(r["mean_dm"], 3),
        "mean_loss_diff":  round(r["mean_loss_diff"], 5),
    } for r in summary])
    summ.to_csv(os.path.join(OUT_DIR, "dm_summary.csv"), index=False)

    print("Diebold-Mariano vs HAR  (DM>0 => HAR better; sig at 5%)\n")
    print(summ.to_string(index=False))
    print("\nSaved results/tables/dm_summary.csv and dm_by_stock.csv")


if __name__ == "__main__":
    main()
