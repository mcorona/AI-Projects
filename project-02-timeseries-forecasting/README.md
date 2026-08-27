# Time Series Forecasting: Stock Price Prediction

## 📊 Project Overview

A forecasting system comparing three fundamentally different approaches on the same real-world financial time series: a classical statistical model (ARIMA), a decomposition-based model (Prophet), and a deep learning model (LSTM). Built end-to-end with proper time-series validation (no random shuffling -- that would leak the future into training), honest error metrics, and an interactive dashboard.

**Dataset:** Daily OHLCV price history for **Reliance Industries (RELIANCE)**, one of the largest and most liquid stocks on the Indian NSE exchange -- 5,306 trading days from 2000-01-03 to 2021-04-30, sourced from a public GitHub mirror of a Kaggle NSE dataset (see "Data Source" below).

---

## 📈 Results

Final one-step-ahead evaluation on the held-out test set (2018-02-07 to 2021-04-30, includes the March 2020 COVID crash):

| Model | MAPE |
|---|---|
| Persistence (naive) | 1.48% |
| **ARIMA(2,1,5)** | **1.48%** |
| **LSTM (log returns)** | **1.48%** |
| Seasonal-naive | 3.48% |
| Prophet (5-day-ahead rolling) | 10.67% |

**Headline finding:** for one-day-ahead forecasting of a liquid single stock, model sophistication was not the bottleneck. Persistence, ARIMA, and a correctly-specified LSTM all land at the same ~1.48% MAPE ceiling -- including through a real market crash. Along the way, test-set evaluation caught a genuine bug: an LSTM trained on raw price level collapsed to 20.18% MAPE on test because the price moved far outside its training range; retraining on log returns (scale-invariant) fixed it completely. Full writeup: `MODEL_CARD.md`, and the notebooks in `notebooks/`.

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

### Phase 1: EDA & Baselines -- Complete
- Data quality checks, unadjusted-split detection and correction (see `MODEL_CARD.md`)
- Seasonal decomposition (trend / seasonality / residual): trend-dominated, weak seasonality
- Stationarity testing (ADF), ACF/PACF analysis: price non-stationary, returns stationary
- Chronological 70/15/15 train/val/test split
- Naive baselines (persistence, seasonal-naive) -- persistence proved a very strong floor

### Phase 2: Model Training -- Complete
- ARIMA (AIC-based order search: (2,1,5)), one-step-ahead via Kalman filter re-conditioning
- Prophet (changepoint-prior-scale search, weekly-refit rolling forecast)
- LSTM (30-day lookback window, one-step-ahead, chronological validation split)

### Phase 3: Evaluation & Comparison -- Complete
- Final test-set MAE/RMSE/MAPE for all models (see Results above)
- Found and fixed a real LSTM scaling bug via test-set evaluation (see `MODEL_CARD.md`)
- Regime analysis (pre-COVID / crash / recovery) and a COVID-crash zoom plot

### Phase 4: Dashboard -- Complete
- `app.py`: Streamlit + Plotly interactive app with 3 tabs (forecast explorer, model comparison, COVID crash case study)

### Phase 5: Deployment -- Complete
- Dockerized dashboard (`Dockerfile`, minimal `requirements-dashboard.txt`)
- `MODEL_CARD.md` with scope, limitations, and known failure modes

---

## 🔧 Getting Started

```bash
cd project-02-timeseries-forecasting
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Get the dataset and build the chronological train/val/test splits
mkdir -p data/raw/nse_stocks
curl -L https://raw.githubusercontent.com/dheeraj5988/stock_market_dataset/main/combined_stock_data.csv \
  -o data/raw/nse_stocks/combined_stock_data.csv
python -m src.data.loader

# Run the notebooks in order -- each one generates the artifacts the next needs
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_evaluation.ipynb

# Explore the dashboard
streamlit run app.py

# Or run the dashboard in Docker
docker build -t reliance-forecast-dashboard .
docker run -p 8501:8501 \
  -v "$(pwd)/data/processed:/app/data/processed" \
  -v "$(pwd)/output/reports:/app/output/reports" \
  reliance-forecast-dashboard
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
