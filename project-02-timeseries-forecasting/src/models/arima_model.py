"""
ARIMA model: order search (by AIC) and one-step-ahead forecasting.

Author: Manuel Corona
"""

import warnings
from typing import List, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def grid_search_order(
    train_series: pd.Series,
    p_range: range = range(0, 6),
    d_values: List[int] = [1],
    q_range: range = range(0, 6),
) -> Tuple[Tuple[int, int, int], pd.DataFrame]:
    """
    Fit ARIMA(p, d, q) for every combination in the given ranges and pick
    the one with the lowest AIC on the training set.

    Args:
        train_series: Training series (e.g. Close price level -- ARIMA's
            own differencing handles making it stationary via `d`).
        p_range, d_values, q_range: Orders to search.

    Returns:
        (best_order, results_df) -- results_df has one row per (p,d,q)
        tried, with its AIC (or NaN if that combination failed to
        converge).
    """
    rows = []
    for d in d_values:
        for p in p_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(train_series, order=(p, d, q)).fit()
                    rows.append({"p": p, "d": d, "q": q, "aic": model.aic})
                except Exception:
                    rows.append({"p": p, "d": d, "q": q, "aic": float("nan")})

    results_df = pd.DataFrame(rows).sort_values("aic")
    best_row = results_df.iloc[0]
    best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
    return best_order, results_df


def fit_and_forecast_one_step(
    train_series: pd.Series, eval_series: pd.Series, order: Tuple[int, int, int]
) -> pd.Series:
    """
    Fit ARIMA on train_series, then produce one-step-ahead forecasts for
    every point in eval_series -- each forecast uses the true prior value
    (not the model's own previous prediction), for a fair comparison
    against the naive baselines evaluated the same way.

    Args:
        train_series: Data the model is fit on.
        eval_series: Data to forecast one step ahead over (e.g. val or test).
        order: (p, d, q).

    Returns:
        Series of one-step-ahead predictions, indexed like eval_series.
    """
    # statsmodels' `.append()` needs the eval index to "extend" the train
    # index at an inferable, consistent frequency; our DatetimeIndex has
    # irregular gaps (weekends/holidays), so use a plain integer index
    # internally and reattach the real dates to the result afterward.
    train_int = train_series.reset_index(drop=True)
    eval_int = eval_series.reset_index(drop=True)
    eval_int.index = range(len(train_int), len(train_int) + len(eval_int))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train_int, order=order).fit()
        extended = model.append(eval_int, refit=False)
        pred = extended.get_prediction(
            start=len(train_int), end=len(train_int) + len(eval_int) - 1, dynamic=False
        )
    return pd.Series(pred.predicted_mean.values, index=eval_series.index)
