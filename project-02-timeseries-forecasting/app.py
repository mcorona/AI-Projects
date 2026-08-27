"""
Streamlit dashboard: RELIANCE stock price forecasting (ARIMA vs Prophet vs LSTM).

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluation.metrics import evaluate_forecast

st.set_page_config(page_title="RELIANCE Price Forecasting", page_icon="\U0001F4C8", layout="wide")
st.title("\U0001F4C8 RELIANCE Stock Price Forecasting")
st.caption("ARIMA vs. Prophet vs. LSTM, evaluated one-step-ahead on a held-out test set (2018-2021)")

REPORTS_DIR = Path("output/reports")
DATA_DIR = Path("data/processed")

MODEL_COLORS = {
    "Actual": "#141413",
    "ARIMA": "#4c72b0",
    "LSTM (log returns)": "#55a868",
    "LSTM (raw price, buggy)": "#c44e52",
    "Prophet": "#dd8452",
}


@st.cache_data
def load_data():
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    actual = test_df["Close"]

    preds = {
        "ARIMA": pd.read_parquet(REPORTS_DIR / "arima_test_predictions.parquet")["pred"],
        "LSTM (log returns)": pd.read_parquet(REPORTS_DIR / "lstm_returns_test_predictions.parquet")["pred"],
        "LSTM (raw price, buggy)": pd.read_parquet(REPORTS_DIR / "lstm_test_predictions.parquet")["pred"],
        "Prophet": pd.read_parquet(REPORTS_DIR / "prophet_test_predictions.parquet")["pred"],
    }

    with open(REPORTS_DIR / "phase3_comparison.json") as f:
        metrics = json.load(f)

    return actual, preds, metrics


try:
    actual, preds, metrics = load_data()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}. Run notebooks 01-03 first to generate processed data and predictions.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["\U0001F4CA Forecast Explorer", "\U0001F3C6 Model Comparison", "\U0001F9A0 COVID Crash Case Study"])

with tab1:
    st.subheader("Actual vs. Forecasted Price")

    presets = {
        "Full test period": (actual.index.min(), actual.index.max()),
        "Pre-COVID (2018-02 to 2020-01)": (actual.index.min(), pd.Timestamp("2020-01-31")),
        "COVID crash (2020-02 to 2020-04)": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
        "Recovery (2020-05 to 2021-04)": (pd.Timestamp("2020-05-01"), actual.index.max()),
    }
    preset_choice = st.selectbox("Date range", list(presets.keys()))
    start, end = presets[preset_choice]

    model_choices = st.multiselect(
        "Models to show", list(preds.keys()), default=["ARIMA", "LSTM (log returns)"]
    )

    mask = (actual.index >= start) & (actual.index <= end)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual.index[mask], y=actual[mask], name="Actual",
        line=dict(color=MODEL_COLORS["Actual"], width=2),
    ))
    for name in model_choices:
        pred = preds[name]
        pred_mask = (pred.index >= start) & (pred.index <= end)
        fig.add_trace(go.Scatter(
            x=pred.index[pred_mask], y=pred[pred_mask], name=name,
            line=dict(color=MODEL_COLORS.get(name, "#999999"), width=1.5),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Close Price (INR)",
        hovermode="x unified", height=550,
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Look up a specific date")
    valid_dates = actual.index[mask]
    if len(valid_dates) > 0:
        picked = st.select_slider("Trading day", options=valid_dates, format_func=lambda d: d.strftime("%Y-%m-%d"))
        cols = st.columns(len(model_choices) + 1)
        cols[0].metric("Actual", f"₹{actual.loc[picked]:.2f}")
        for i, name in enumerate(model_choices, start=1):
            if picked in preds[name].index:
                pred_val = preds[name].loc[picked]
                delta = pred_val - actual.loc[picked]
                cols[i].metric(name, f"₹{pred_val:.2f}", delta=f"{delta:+.2f}")

with tab2:
    st.subheader("Test Set Metrics")
    metrics_df = pd.DataFrame(metrics).T[["mae", "rmse", "mape"]].sort_values("mape")
    metrics_df.columns = ["MAE (INR)", "RMSE (INR)", "MAPE (%)"]
    st.dataframe(metrics_df.style.format("{:.2f}"), width='stretch')

    fig2 = go.Figure(go.Bar(
        x=metrics_df.index, y=metrics_df["MAPE (%)"],
        marker_color=[MODEL_COLORS.get(
            {"persistence": "Actual", "seasonal_naive": "Actual", "arima": "ARIMA",
             "lstm_log_returns": "LSTM (log returns)", "lstm_raw_price_BUGGY": "LSTM (raw price, buggy)",
             "prophet": "Prophet"}.get(i, ""), "#999999") for i in metrics_df.index],
    ))
    fig2.update_layout(yaxis_title="MAPE (%)", height=400)
    st.plotly_chart(fig2, width='stretch')

    st.info(
        "**Key finding:** persistence, ARIMA, and a correctly-specified LSTM (trained on log returns) "
        "are statistically tied at ~1.48% MAPE. Added model complexity did not translate into added "
        "accuracy for one-day-ahead forecasting of this liquid stock -- see `PHASE_3_SUMMARY.md` for the full writeup."
    )

with tab3:
    st.subheader("The LSTM Scaling Bug")
    st.markdown(
        "The first LSTM attempt was trained on **raw price level**. RELIANCE's price nearly tripled "
        "between the training period and the test period, so test inputs landed far outside the range "
        "the model's scaler was fit on -- it had never seen normalized values that extreme. "
        "Retraining on **log returns** (scale-invariant: day-over-day % change stays in a similar range "
        "regardless of the price level) fixed it completely."
    )

    col1, col2 = st.columns(2)
    col1.metric("LSTM (raw price) MAPE", f"{metrics['lstm_raw_price_BUGGY']['mape']:.2f}%")
    col2.metric("LSTM (log returns) MAPE", f"{metrics['lstm_log_returns']['mape']:.2f}%", delta="fixed", delta_color="off")

    st.subheader("Error by Market Regime")
    regimes = {
        "Pre-COVID": (actual.index.min(), pd.Timestamp("2020-01-31")),
        "COVID crash": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
        "Recovery": (pd.Timestamp("2020-05-01"), actual.index.max()),
    }
    rows = []
    for regime_name, (r_start, r_end) in regimes.items():
        r_mask = (actual.index >= r_start) & (actual.index <= r_end)
        actual_slice = actual[r_mask]
        row = {"Regime": regime_name, "n": int(r_mask.sum())}
        for name in ["ARIMA", "LSTM (log returns)", "LSTM (raw price, buggy)"]:
            pred_slice = preds[name].reindex(actual_slice.index).dropna()
            common = actual_slice.loc[pred_slice.index]
            row[name] = evaluate_forecast(common, pred_slice)["mape"]
        rows.append(row)
    regime_df = pd.DataFrame(rows).set_index("Regime")
    st.dataframe(regime_df.style.format({c: "{:.2f}%" for c in regime_df.columns if c != "n"}), width='stretch')

    st.markdown(
        "Notice the buggy model's error climbs steadily (pre-COVID -> crash -> recovery) as the price "
        "moves further from its training range -- it's not the crash's volatility that breaks it, "
        "it's the price level itself being unfamiliar."
    )
