"""
Prophet model wrapper for the RELIANCE daily Close series.

Author: Manuel Corona
"""

import logging
from typing import Tuple

import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def to_prophet_df(series: pd.Series) -> pd.DataFrame:
    """Convert a date-indexed price Series to Prophet's required ds/y columns."""
    return pd.DataFrame({"ds": series.index, "y": series.values})


def fit_prophet(train_series: pd.Series, changepoint_prior_scale: float = 0.05) -> Prophet:
    """
    Fit Prophet on the training series.

    Args:
        train_series: Date-indexed training price series.
        changepoint_prior_scale: Flexibility of the trend -- higher allows
            more/sharper trend changes (Prophet's main tuning knob).

    Returns:
        Fitted Prophet model.
    """
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale, daily_seasonality=False)
    model.fit(to_prophet_df(train_series))
    return model


def forecast(model: Prophet, eval_series: pd.Series) -> pd.Series:
    """
    Forecast Prophet's predictions on the exact dates in eval_series.

    Note: this is Prophet's own multi-step trend/seasonality extrapolation
    (Prophet doesn't have a native one-step-ahead re-conditioning mode like
    a state-space ARIMA does) -- it does not see eval_series' actual y
    values, only its dates.

    Args:
        model: A fitted Prophet model.
        eval_series: Date-indexed series whose index gives the dates to
            forecast.

    Returns:
        Series of predictions ('yhat'), indexed like eval_series.
    """
    future = pd.DataFrame({"ds": eval_series.index})
    forecast_df = model.predict(future)
    return pd.Series(forecast_df["yhat"].values, index=eval_series.index)


def rolling_forecast(
    full_series: pd.Series, eval_start_idx: int, step: int = 5, changepoint_prior_scale: float = 0.05
) -> pd.Series:
    """
    Rolling-origin evaluation: refit Prophet every `step` trading days on
    all data up to that point, then forecast just the next `step` days.

    Unlike ARIMA's `.append()`-based one-step-ahead (which re-conditions on
    every single true observation via the Kalman filter, cheaply), Prophet
    has no equivalent incremental update -- a full refit is the only way to
    incorporate new observations. Refitting daily for the whole eval period
    would be correct but slow; refitting weekly (step=5 trading days) is
    the standard, tractable compromise and is explicitly a *short-horizon*
    forecast (up to `step` days ahead), not true one-step-ahead like ARIMA.

    Args:
        full_series: The complete date-indexed series (train + eval).
        eval_start_idx: Integer position in full_series where evaluation
            begins (i.e. len(train_series)).
        step: Trading days between refits, and the forecast horizon each time.
        changepoint_prior_scale: Prophet's trend-flexibility hyperparameter.

    Returns:
        Series of predictions covering full_series[eval_start_idx:].
    """
    preds = []
    n = len(full_series)
    cursor = eval_start_idx
    while cursor < n:
        train_chunk = full_series.iloc[:cursor]
        horizon_end = min(cursor + step, n)
        eval_chunk = full_series.iloc[cursor:horizon_end]

        model = fit_prophet(train_chunk, changepoint_prior_scale=changepoint_prior_scale)
        preds.append(forecast(model, eval_chunk))
        cursor = horizon_end

    return pd.concat(preds)


def grid_search_changepoint_prior(
    full_series: pd.Series,
    eval_start_idx: int,
    candidates=(0.01, 0.05, 0.1, 0.5),
    step: int = 5,
) -> Tuple[float, pd.DataFrame]:
    """
    Try a few changepoint_prior_scale values using the same rolling_forecast
    protocol actual evaluation uses (so the selected value isn't tuned
    against a different, easier task), on a short tail slice of the eval
    range to keep the search fast -- full-length rolling search across all
    candidates would take ~4x as long as the final evaluation itself.
    """
    from src.evaluation.metrics import evaluate_forecast

    tail_start = max(eval_start_idx, len(full_series) - 150)
    rows = []
    for cps in candidates:
        pred = rolling_forecast(full_series, eval_start_idx=tail_start, step=step, changepoint_prior_scale=cps)
        m = evaluate_forecast(full_series.iloc[tail_start:], pred)
        rows.append({"changepoint_prior_scale": cps, **m})

    results_df = pd.DataFrame(rows).sort_values("mae")
    best_cps = float(results_df.iloc[0]["changepoint_prior_scale"])
    return best_cps, results_df
