# Time Series Forecasting: Stock Price Prediction

## 📊 Project Overview

A forecasting system comparing three fundamentally different approaches on the same real-world financial time series: a classical statistical model (ARIMA), a decomposition-based model (Prophet), and a deep learning model (LSTM). Built end-to-end with proper time-series validation (no random shuffling -- that would leak the future into training), honest error metrics, and an interactive dashboard.

**Dataset:** Daily OHLCV price history for **Reliance Industries (RELIANCE)**, one of the largest and most liquid stocks on the Indian NSE exchange -- 5,306 trading days from 2000-01-03 to 2021-04-30, sourced from a public GitHub mirror of a Kaggle NSE dataset (see "Data Source" below).

---

## 🎯 Core Objectives

1. **Proper time-series methodology**: chronological train/val/test split (no shuffling), stationarity testing, seasonal decomposition -- not just "fit a model and report a number."
2. **Honest model comparison**: ARIMA, Prophet, and LSTM evaluated on the *same* held-out period with the *same* metrics (MAE, RMSE, MAPE), so the comparison is fair.
3. **Understand what each model captures**: ARIMA (autocorrelation structure), Prophet (trend + seasonality decomposition), LSTM (learned nonlinear temporal patterns) -- and where each one breaks down.
4. **Production-adjacent delivery**: an interactive dashboard to explore predictions, not just a notebook.

---

## 🛠️ Tech Stack

| Component | Tool | Reason |
|-----------|------|--------|
| **Statistical model** | Statsmodels (ARIMA/SARIMA) | Classical baseline, interpretable |
| **Decomposition model** | Prophet | Handles trend/seasonality/holidays explicitly |
| **Deep learning** | TensorFlow/Keras (LSTM) | Learns nonlinear temporal dependencies |
| **Data processing** | Pandas, NumPy, PyArrow | Time-indexed data manipulation |
| **Evaluation** | Scikit-learn | MAE/RMSE/MAPE, consistent across all 3 models |
| **Visualization** | Matplotlib, Seaborn, Plotly | EDA and interactive dashboard |
| **Dashboard** | Streamlit + Plotly | Interactive exploration of forecasts |

---

## 📊 Data Source

**Note on sourcing (same constraint as Project 1):** this dev environment's network policy blocks Yahoo Finance, Alpha Vantage, and Stooq directly. The dataset was sourced from a public GitHub mirror instead:
`https://raw.githubusercontent.com/dheeraj5988/stock_market_dataset/main/combined_stock_data.csv`

This is a 63-symbol NSE (India) daily OHLCV dataset, 2000-2021. **RELIANCE** was selected for the main forecasting task: longest available history (5,306 rows), zero nulls, zero duplicate dates, and a large, liquid, well-known stock -- a clean, realistic single time series with both strong trend and meaningful volatility.

On a machine with normal internet access, the equivalent real-time data is trivially available via the `yfinance` package (`yfinance.download("RELIANCE.NS")` or any other ticker) -- the pipeline code doesn't depend on this specific mirror.

---

## 🚀 Development Plan

### Phase 1: EDA & Baselines
- Load and validate the raw series; build a clean daily-indexed DataFrame
- Seasonal decomposition (trend / seasonality / residual)
- Stationarity testing (ADF test), autocorrelation (ACF/PACF) analysis
- Chronological train/val/test split
- Naive baselines (last-value / seasonal-naive) to establish a floor to beat

### Phase 2: Model Training
- ARIMA/SARIMA (order selection via ACF/PACF + `auto_arima`-style search)
- Prophet (with trend changepoints, seasonality)
- LSTM (windowed sequences, trained on the same chronological split)

### Phase 3: Evaluation & Comparison
- MAE / RMSE / MAPE for all 3 models on the same held-out test period
- Residual analysis: where does each model fail (trend breaks, volatility spikes)?
- Comparison table + plots (actual vs. predicted for each model)

### Phase 4: Dashboard
- Streamlit + Plotly interactive app: pick a date range, compare model forecasts visually

---

## 🔧 Getting Started

```bash
cd project-02-timeseries-forecasting
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Data is fetched during Phase 1 (see notebooks/01_eda.ipynb for the exact source)
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📚 References

- **ARIMA:** Box & Jenkins, *Time Series Analysis: Forecasting and Control*
- **Prophet:** [Taylor & Letham, 2017](https://peerj.com/preprints/3190/)
- **NSE Dataset (original):** [Kaggle - jacksoncrow/stock-market-dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

---

## 📞 Author

Manuel Corona

*Last Updated: 2026-08-23*
