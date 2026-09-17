"""
Regression metrics and evaluation helpers.

Metrics
-------
MAE  : mean absolute error, in watts. Average size of the miss - easy to
       explain to a non-specialist ("on average we are off by X W").
RMSE : root mean squared error, in watts. Squares the errors before
       averaging, so a few large misses dominate. This is the model-selection
       metric because a forecasting system for peak-load planning must be
       punished for big errors, not just for being slightly off often.
R2   : coefficient of determination, dimensionless. Share of the variance in
       the load explained by the model. 1.0 is perfect, 0.0 means the model is
       no better than always predicting the mean.
MAPE : mean absolute percentage error, computed only over samples where the
       true value is comfortably non-zero (division by ~0 makes MAPE explode).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true, y_pred) -> float:
    """Mean absolute error (watts)."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (watts).

    Computed as sqrt(MSE) rather than via ``squared=False``, which was removed
    in scikit-learn 1.6 - this form works on every supported version.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination."""
    return float(r2_score(y_true, y_pred))


def mape(y_true, y_pred, floor: float = 1e-6) -> float:
    """Mean absolute percentage error (%), ignoring near-zero true values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > max(floor, 1.0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Return every metric as a plain dict (JSON-serialisable)."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def evaluate_model(model, X, y_true) -> dict[str, float]:
    """Fit-free evaluation of an already-trained estimator."""
    return evaluate_predictions(y_true, model.predict(X))


def format_comparison_table(results: dict[str, dict[str, float]]) -> str:
    """Render a model-comparison dictionary as a monospaced table."""
    header = f"{'Model':<22}{'MAE (W)':>12}{'RMSE (W)':>12}{'R2':>10}{'MAPE (%)':>12}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(
            f"{name:<22}{m['mae']:>12.2f}{m['rmse']:>12.2f}{m['r2']:>10.4f}{m['mape']:>12.2f}"
        )
    return "\n".join(lines)


def select_best_model(
    results: dict[str, dict[str, float]], metric: str = "rmse"
) -> str:
    """
    Return the winning model name under a documented criterion.

    Lower is better for mae/rmse/mape; higher is better for r2.
    """
    if not results:
        raise ValueError("No model results to select from")
    higher_is_better = metric in {"r2"}
    return (max if higher_is_better else min)(results, key=lambda name: results[name][metric])
