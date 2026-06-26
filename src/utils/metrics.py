from __future__ import annotations

import numpy as np


def _as_text(value: object) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def summarize_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(prediction, target),
        "mae": mae(prediction, target),
    }


def inverse_target_scale(
    values: np.ndarray,
    feature_names: np.ndarray | list[str],
    scaler_mean: np.ndarray | list[float],
    scaler_scale: np.ndarray | list[float],
    target_name: str = "global_active_power",
) -> np.ndarray:
    """Convert scaled target predictions back to the original target unit."""
    names = [_as_text(name) for name in feature_names]
    if target_name not in names:
        raise ValueError(f"Target {target_name!r} is not present in feature_names.")
    target_index = names.index(target_name)
    mean = np.asarray(scaler_mean, dtype=np.float64)[target_index]
    scale = np.asarray(scaler_scale, dtype=np.float64)[target_index]
    return np.asarray(values, dtype=np.float64) * scale + mean


def summarize_forecast_metrics(
    prediction_scaled: np.ndarray,
    target_scaled: np.ndarray,
    feature_names: np.ndarray | list[str],
    scaler_mean: np.ndarray | list[float],
    scaler_scale: np.ndarray | list[float],
    target_name: str = "global_active_power",
) -> dict[str, float]:
    """Return metrics in both scaled space and original target units."""
    prediction_original = inverse_target_scale(
        prediction_scaled, feature_names, scaler_mean, scaler_scale, target_name
    )
    target_original = inverse_target_scale(
        target_scaled, feature_names, scaler_mean, scaler_scale, target_name
    )
    scaled = summarize_metrics(prediction_scaled, target_scaled)
    original = summarize_metrics(prediction_original, target_original)
    return {
        "mse_scaled": scaled["mse"],
        "mae_scaled": scaled["mae"],
        "mse_original": original["mse"],
        "mae_original": original["mae"],
    }
