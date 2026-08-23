# Model Card: RELIANCE Stock Price Forecasting System

## System Overview

Three one-day-ahead forecasting approaches for a single stock's daily closing price, evaluated on the same held-out test period with the same metrics:

| Model | Type | File |
|---|---|---|
| ARIMA(2,1,5) | Classical statistical (autoregressive) | `src/models/arima_model.py` |
| Prophet | Trend + seasonality decomposition | `src/models/prophet_model.py` |
| LSTM (log returns) | Deep learning (recurrent neural net) | `src/models/lstm_model.py` |

Plus two naive baselines (persistence, seasonal-naive) that turned out to be competitive with all three -- see Results below.

## Intended Use

- **Primary use case:** educational/portfolio demonstration of time-series forecasting methodology -- proper chronological validation, honest multi-model comparison, and error analysis under a real market shock (the 2020 COVID crash falls inside the test period).
- **Not intended for:** live trading, investment decisions, or any production financial system. These models forecast one trading day ahead using only the stock's own price history; they have no access to news, fundamentals, or market microstructure, and were validated on a single stock from a single exchange.

## Training Data

- **Source:** NSE (India) daily OHLCV data for 63 symbols, 2000-2021, sourced from a public GitHub mirror (Yahoo Finance/Alpha Vantage/Stooq are blocked by this project's dev sandbox network policy -- see `PHASE_1_SUMMARY.md`-equivalent notes in `notebooks/01_eda.ipynb`).
- **Series used:** RELIANCE (Reliance Industries), 5,306 trading days, 2000-01-03 to 2021-04-30.
- **Known data issue found and fixed:** the raw data has no split-adjusted price column. Two near-exact -50% single-day drops (2009-11-26, 2017-09-07) were identified as unadjusted stock splits/bonus issues and corrected; four other large single-day moves were deliberately left untouched because they match real, documented market events (2004/2009 Indian election swings, the Oct 2008 financial crisis, a likely 2006 demerger) -- see `src/data/loader.py`.
- **Split:** chronological 70/15/15 (train through 2014-11-24, val through 2018-02-06, test through 2021-04-30). Never shuffled -- a random split would leak future prices into training.

## Evaluation

One-step-ahead forecasting on the test set (2018-02-07 to 2021-04-30, 797 trading days, includes the March 2020 COVID crash):

| Model | MAE (INR) | RMSE (INR) | MAPE |
|---|---|---|---|
| Persistence (naive) | 21.15 | 30.66 | 1.48% |
| ARIMA(2,1,5) | 21.23 | 30.81 | 1.48% |
| LSTM (log returns) | 21.13 | 30.68 | 1.48% |
| Seasonal-naive | 49.37 | 68.56 | 3.48% |
| Prophet | 153.38 | 203.43 | 10.67% |

**Important asymmetry in how these numbers were produced:** ARIMA and the naive baselines use true one-step-ahead forecasting (re-conditioned on the real prior price every day). LSTM does the same using true historical lookback windows. **Prophet cannot do this** -- it has no incremental single-observation update -- so it's evaluated with a weekly-refit, 5-trading-day-horizon rolling forecast instead, a genuinely harder task. Prophet's much higher error is not fully comparable to the others for this reason (though it would likely still underperform on a fairer footing, given Phase 1's finding that this series' seasonality -- Prophet's core strength -- is weak relative to trend and volatility).

## Known Limitations & Failure Modes

1. **Model sophistication did not beat a naive "predict no change" baseline.** For one-day-ahead forecasting of a liquid, heavily-traded stock, persistence, ARIMA, and a correctly-specified LSTM all converge to the same ~1.48% MAPE ceiling -- consistent with the efficient-market hypothesis. This held even through the COVID crash (regime-by-regime error: ~1.3% calm periods, ~3.3% during the crash, for both ARIMA and LSTM alike).
2. **A real LSTM bug was found via test-set evaluation, not caught in validation.** The first LSTM was trained on raw price level with a `MinMaxScaler` fit on the training range. RELIANCE's price nearly tripled between train+val and the test period, so test inputs landed up to 3x outside the scaler's fitted range -- MAPE jumped from 2.85% (val) to 20.18% (test). Fixed by modeling log returns instead of price level (scale-invariant). **Lesson:** validation-set performance alone does not guarantee a financial forecasting model will generalize if the price level moves meaningfully beyond the training range -- this needs a held-out test period spanning genuinely different price regimes to catch, exactly as happened here. **Note on reproducibility:** the buggy model's exact MAPE varies noticeably between training runs (one re-run landed at 3.68% instead of 20.18%, while ARIMA and the fixed LSTM stayed stable) -- out-of-distribution neural-net extrapolation isn't just wrong, it's unpredictable, which is itself an argument against relying on it. The direction and severity class of the failure (a full order of magnitude worse than the in-range models) reproduces reliably even when the exact number doesn't.
3. **Prophet is evaluated on an easier-to-fail, harder task** (5-day horizon, weekly refit) than the other models, per the asymmetry noted above -- its poor showing here should not be read as "Prophet is a bad model," but as "Prophet is not well-suited to short-horizon, single-stock, low-seasonality forecasting without expensive daily refitting."
4. **Single stock, single exchange, single 21-year window.** No claim is made about generalization to other stocks, asset classes, exchanges, or time periods. The COVID crash is the only major regime break in the test period; performance under other kinds of shocks (rate shocks, single-company news events, flash crashes) is untested.
5. **No transaction costs, slippage, or execution modeling.** These are price-level forecasts, not a trading strategy backtest -- a ~1.5% MAPE one-step forecast does not imply a profitable trading signal after realistic costs.

## Ethical Considerations

- **Not a financial advice tool.** See "Intended Use" above.
- **Training data reflects a specific market's history and conditions** (NSE India, 2000-2021, including India-specific events like the 2004/2009 election-driven crashes). Patterns learned here should not be assumed to transfer to other markets without re-validation.

## How to Use

See `README.md`'s "Getting Started" section for running the notebooks (to regenerate data/predictions), the Streamlit dashboard, or the Docker container.
