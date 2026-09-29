# Data

The raw data is not included in this repository. The file is about 130 MB, and VOLARE asks users to download it from its own site.

## How to get it

1. Go to the VOLARE database: http://volare.unime.it
2. Download **stocks → volatility** data for these 40 tickers, 2015-01-01 to 2025-12-31:

   AAPL, ADBE, AMD, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, GE, GOOGL, GS, HD, HON, IBM, JNJ, JPM,
   KO, MCD, META, MMM, MRK, MSFT, NFLX, NKE, NVDA, ORCL, PG, PM, SHW, TRV, TSLA, UNH, V, VZ, WMT, XOM

3. Save the file as `data/realized_variance_stocks.csv`.
4. Run `python src/01_clean_prices.py`. It creates `data/volare_clean.csv`, which every other script reads.

## Columns used

`date`, `symbol`, `open_price`, `high_price`, `low_price`, `close_price`, `rv5` (5-minute realized variance), `bv5` (bipower variation), `rq5` (realized quarticity).

## Citation

Cipollini, F., Cruciani, G., Gallo, G. M., Insana, A., Otranto, E., & Spagnolo, F. (2026). *VOLatility Archive for Realized Estimates (VOLARE).* https://doi.org/10.48550/arXiv.2602.19732
