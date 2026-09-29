"""Step 7 (H4) - VaR backtest. Turn each model's volatility forecast into a 1-day VaR, then test it.
Tests: Kupiec (1995) unconditional coverage, Christoffersen (1998) conditional coverage."""

import os
import pandas as pd
import numpy as np
from scipy import stats

# Config
RES_DIR = "results/forecasts"
OUT_DIR = "results/tables"
DATA    = "data/volare_clean.csv"
SPLIT   = pd.Timestamp("2020-01-01")
EPS     = 1e-12
LEVELS  = [0.95, 0.99] #VaR confidence levels
DISTS   = ["normal", "t"] #Distributional assumption (H4 lever)

FILES = {
    "HAR":    "har_oos_forecasts.csv",
    "GARCH":  "garch_oos_forecasts.csv",
    "EGARCH": "egarch_oos_forecasts.csv",
    "LASSO":  "lasso_oos_forecasts.csv",
    "RF":     "randomforest_oos_forecasts.csv",
    "HV":   "hv_oos_forecasts.csv",
    "EWMA": "ewma_oos_forecasts.csv",
}
os.makedirs(OUT_DIR, exist_ok=True)

#Kupiec (1995) unconditional coverage test
def kupiec_pof(n, x, p):
    #Are there the right NUMBER of violations?
    if n == 0:
        return np.nan, np.nan
    pi = x / n
    ll_p = x * np.log(p) + (n - x) * np.log(1 - p)
    if x == 0:
        ll_pi = (n - x) * np.log(1 - pi + EPS)
    elif x == n:
        ll_pi = x * np.log(pi)
    else:
        ll_pi = x * np.log(pi) + (n - x) * np.log(1 - pi)
    lr = max(-2 * (ll_p - ll_pi), 0.0)
    return lr, float(stats.chi2.sf(lr, 1))   # sf = 1 - cdf, but accurate in the far tail

#Christoffersen (1998) conditional coverage test.
def christoffersen_cc(viol, p):
    #Conditional coverage = right number of violations AND not clustered
    v = np.asarray(viol).astype(int)
    n = len(v)
    lr_uc, _ = kupiec_pof(n, int(v.sum()), p)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        a, b = v[i - 1], v[i]
        if   a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else:                   n11 += 1
    d0, d1 = n00 + n01, n10 + n11
    pi01 = n01 / d0 if d0 else 0.0
    pi11 = n11 / d1 if d1 else 0.0
    pi   = (n01 + n11) / (d0 + d1) if (d0 + d1) else 0.0
    term = lambda c, pr: c * np.log(pr) if (c > 0 and pr > 0) else 0.0
    ll_ind  = term(n01, pi01) + term(n00, 1 - pi01) + term(n11, pi11) + term(n10, 1 - pi11)
    ll_null = term(n01 + n11, pi) + term(n00 + n10, 1 - pi)
    lr_ind = max(-2 * (ll_null - ll_ind), 0.0)
    lr_cc = lr_uc + lr_ind
    p_cc  = float(stats.chi2.sf(lr_cc, 2))    # joint coverage+independence
    p_ind = float(stats.chi2.sf(lr_ind, 1))   # independence-only (clustering) diagnostic
    return lr_cc, p_cc, p_ind

# Main
def main():
    #Daily returns + next-day return (the thing VaR is tested against)
    raw = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["symbol", "date"])
    raw["ret"]      = raw.groupby("symbol")["close_price"].transform(lambda s: np.log(s / s.shift(1)))
    raw["ret_next"] = raw.groupby("symbol")["ret"].shift(-1)   # return on day t+1, matches forecast made at t

    #Per-stock in-sample distribution of z = r / sqrt(rv5) (2015-2019)
    ins = raw[(raw.date < SPLIT) & (raw.rv5 > 0) & raw.ret.notna()].copy()
    ins["z"] = ins["ret"] / np.sqrt(ins["rv5"])
    params = {}
    for sym, g in ins.groupby("symbol"):
        z = g["z"].values
        z = z[np.isfinite(z)]
        s = float(np.std(z, ddof=1))
        try:
            df_t, _, scale_t = stats.t.fit(z, floc=0.0)
            df_t = float(np.clip(df_t, 3.0, 100.0))
        except Exception:
            df_t, scale_t = 6.0, s
        params[sym] = {"s": s, "df": df_t, "scale": float(scale_t)}

    rn = raw[["symbol", "date", "ret_next"]]

    rows, summ = [], []
    for model, fn in FILES.items():
        fc = pd.read_csv(os.path.join(RES_DIR, fn), parse_dates=["date"])
        fc = fc.merge(rn, on=["symbol", "date"], how="left").dropna(subset=["ret_next", "forecast_rv"])
        fc = fc[fc.forecast_rv > 0]

        for dist in DISTS:
            for lvl in LEVELS:
                a = 1 - lvl
                pass_k = pass_c = nstk = 0
                rates = []
                for sym, g in fc.groupby("symbol"):
                    if sym not in params:
                        continue
                    pr = params[sym]
                    sigma = np.sqrt(g["forecast_rv"].values) #Volatility forecast
                    if dist == "normal":
                        q = pr["s"] * stats.norm.ppf(a) #Left-tail quantile of z
                    else:
                        q = stats.t.ppf(a, pr["df"], loc=0, scale=pr["scale"])
                    var = sigma * q #VaR (negative number)
                    r = g["ret_next"].values
                    viol = (r < var).astype(int)
                    n, x = len(viol), int(viol.sum())
                    if n < 100:
                        continue
                    _, pk = kupiec_pof(n, x, a)
                    _, pc, pind = christoffersen_cc(viol, a)
                    okk, okc = pk > 0.05, pc > 0.05
                    pass_k += int(okk); pass_c += int(okc); nstk += 1
                    rates.append(x / n)
                    rows.append({"model": model, "dist": dist, "level": lvl, "symbol": sym,
                                 "n": n, "n_viol": x, "viol_rate": round(x / n, 4),
                                 "expected": a, "kupiec_p": round(pk, 4), "cc_p": round(pc, 4),
                                 "cc_ind_p": round(pind, 4),
                                 "pass_kupiec": okk, "pass_cc": okc})
                summ.append({"model": model, "dist": dist, "level": lvl,
                             "mean_viol_rate": round(float(np.mean(rates)), 4) if rates else np.nan,
                             "expected_rate": a,
                             "pass_kupiec": pass_k, "pass_cc": pass_c, "n_stocks": nstk})

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "var_backtest_by_stock.csv"), index=False)
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(os.path.join(OUT_DIR, "var_backtest_summary.csv"), index=False)

    print("VaR backtest  (pass = p>0.05; out of 39 stocks)")
    print("ideal mean_viol_rate ~ expected_rate; pass counts higher = better\n")
    for lvl in LEVELS:
        print(f"--- {int(lvl*100)}% VaR ---")
        print(summ_df[summ_df.level == lvl].to_string(index=False))
        print()
    print("Saved var_backtest_summary.csv and var_backtest_by_stock.csv in results/tables/")

if __name__ == "__main__":
    main()
