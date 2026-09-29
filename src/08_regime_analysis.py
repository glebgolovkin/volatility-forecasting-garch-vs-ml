"""Step 8 (H3) - do ML models do better when volatility is high (COVID-2020, 2022)?

Splits each stock's out-of-sample days into a high-volatility regime (top third of
rv5_t, known at forecast time) and a normal regime, then tests with a Wilcoxon
signed-rank test whether each competitor closes its gap to HAR in high-vol days."""

import os
import pandas as pd
import numpy as np
from scipy import stats

# Config
RES_DIR    = "results/forecasts"
OUT_DIR    = "results/tables"
INPUT_FILE = "data/volare_clean.csv"   #raw rv5 series, needed for the forecast_time regime
EPS        = 1e-12
HIGH_Q     = 2 / 3 # Top third of a stocks oos days = high vol regime
BENCH      = "HAR" #Everyone is compared to HAR
REGIME_BY  = "forecast_time"  #Forecast_time (clean) or outcome (old)

#New (forecast_time) tables get a _condrv suffix so the old outcome-based tables are not overwritten
SUFFIX = "_condrv" if REGIME_BY == "forecast_time" else ""

#Daily forecast files that we saved earlier (one per model)
FILES = {
    "HAR":    "har_oos_forecasts.csv",
    "GARCH":  "garch_oos_forecasts.csv",
    "EGARCH": "egarch_oos_forecasts.csv",
    "LASSO":  "lasso_oos_forecasts.csv",
    "RF":     "randomforest_oos_forecasts.csv",
    "HV":   "hv_oos_forecasts.csv",
    "EWMA": "ewma_oos_forecasts.csv",
}
COMPETITORS = ["GARCH", "EGARCH", "LASSO", "RF", "HV", "EWMA"]

#Named crisis windows => just for nice numbers in the text:
WINDOWS = {
    "COVID-2020": ("2020-02-20", "2020-06-30"),
    "Rate-2022":  ("2022-01-01", "2022-12-31"),
}

os.makedirs(OUT_DIR, exist_ok=True)

#Our loss. It means lower = better forecast. Floor tiny values so logs dont blow up
def qlike(actual, forecast):
    a = np.maximum(np.asarray(actual, float), EPS)
    f = np.maximum(np.asarray(forecast, float), EPS)
    return a / f - np.log(a / f) - 1

def main():
    #It loads every models daily forecasts into one long table and computes its loss
    frames = []
    for m, fn in FILES.items():
        df = pd.read_csv(os.path.join(RES_DIR, fn), parse_dates=["date"])
        df["loss"] = qlike(df["actual_rv"], df["forecast_rv"])
        df["model"] = m
        frames.append(df[["symbol", "date", "actual_rv", "loss", "model"]])
    allm = pd.concat(frames, ignore_index=True)

    #Conditioning variable that defines the high/normal-vol regime per (symbol, date)
    reg = allm[allm.model == BENCH][["symbol", "date", "actual_rv"]].copy()
    if REGIME_BY == "outcome":
        #Old behaviour: split on rv5_{t+1}, i.e. the realised outcome (selection-on-outcome artifact)
        reg["cond"] = reg["actual_rv"]
    elif REGIME_BY == "forecast_time":
        #Clean: split on rv5_t, the volatility known when the forecast is made
        raw = pd.read_csv(INPUT_FILE, parse_dates=["date"])[["symbol", "date", "rv5"]]
        reg = reg.merge(raw, on=["symbol", "date"], how="left")
        reg["cond"] = reg["rv5"]
    else:
        raise ValueError(f"Unknown REGIME_BY={REGIME_BY!r} (use 'forecast_time' or 'outcome')")

    reg = reg.dropna(subset=["cond"])
    reg["thr"] = reg.groupby("symbol")["cond"].transform(lambda s: s.quantile(HIGH_Q))
    reg["regime"] = np.where(reg["cond"] >= reg["thr"], "High", "Normal")
    allm = allm.merge(reg[["symbol", "date", "regime"]], on=["symbol", "date"], how="left")
    allm = allm.dropna(subset=["regime"])

    #Average loss for each model, on each stock, in each regime:
    g = allm.groupby(["model", "symbol", "regime"])["loss"].mean().reset_index()

    tabA = g.groupby(["model", "regime"])["loss"].mean().unstack("regime")
    tabA = tabA.reindex(["HAR", "GARCH", "EGARCH", "LASSO", "RF"])[["Normal", "High"]]
    tabA["High_minus_Normal"] = tabA["High"] - tabA["Normal"]
    tabA.round(4).to_csv(os.path.join(OUT_DIR, f"h3_regime_qlike{SUFFIX}.csv"))

    w = g.pivot_table(index=["symbol", "regime"], columns="model", values="loss")
    hi = w.xs("High", level="regime")
    no = w.xs("Normal", level="regime")
    rowsB = []
    for comp in COMPETITORS:
        gap_hi = (hi[comp] - hi[BENCH]).dropna() #Gap in high volatility days
        gap_no = (no[comp] - no[BENCH]).dropna() #Gap in normal days
        common = gap_hi.index.intersection(gap_no.index)
        gh, gn = gap_hi.loc[common], gap_no.loc[common]
        delta = (gh - gn).values #If <0 = competitor improves in high volatility

        #We perform a Wilcoxon signed-rank test to check if the competitor improves in high volatility
        try:
            _, p_less = stats.wilcoxon(delta, alternative="less")
        except ValueError:
            p_less = np.nan
        rowsB.append({
            "competitor":            comp,
            "mean_gap_high":         round(float(gh.mean()), 5),
            "mean_gap_normal":       round(float(gn.mean()), 5),
            "mean_delta":            round(float(np.mean(delta)), 5),
            "comp_beats_HAR_high":   int((gh < 0).sum()),
            "comp_beats_HAR_normal": int((gn < 0).sum()),
            "n_stocks":              len(common),
            "wilcoxon_p_delta<0":    round(float(p_less), 4), #Small p => our H3 is supported
        })
    tabB = pd.DataFrame(rowsB)
    tabB.to_csv(os.path.join(OUT_DIR, f"h3_relative_test{SUFFIX}.csv"), index=False)

    # C) just the avg qlike inside covid and rate 2022, pooled over all stocks
    rowsC = []
    for wname, (s, e) in WINDOWS.items():
        mask = (allm.date >= pd.Timestamp(s)) & (allm.date <= pd.Timestamp(e))
        sub = allm[mask].groupby("model")["loss"].mean()
        rowsC.append({"window": wname, **{m: round(float(sub.get(m, np.nan)), 4)
                                          for m in ["HAR", "GARCH", "EGARCH", "LASSO", "RF"]}})
    tabC = pd.DataFrame(rowsC)
    tabC.to_csv(os.path.join(OUT_DIR, f"h3_crisis_windows{SUFFIX}.csv"), index=False)


    cond_desc = "rv5_t (known at forecast time)" if REGIME_BY == "forecast_time" else "rv5_{t+1} (outcome)"
    print(f"High-vol regime = top third of each stock's OOS days by {cond_desc} "
          f"(quantile {HIGH_Q:.2f}) | REGIME_BY={REGIME_BY}\n")
    print("(A) Avg QLIKE by model and regime (mean across 39 stocks):")
    print(tabA.round(4).to_string())
    print("\n(B) H3 relative test  (delta = gap_high - gap_normal; delta<0 => ML better in high-vol):")
    print(tabB.to_string(index=False))
    print("\n(C) Pooled QLIKE inside named crisis windows:")
    print(tabC.to_string(index=False))
    print(f"\nSaved h3_regime_qlike{SUFFIX}.csv, h3_relative_test{SUFFIX}.csv, "
          f"h3_crisis_windows{SUFFIX}.csv in results/tables/")

if __name__ == "__main__":
    main()
