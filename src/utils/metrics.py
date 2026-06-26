from __future__ import annotations

import numpy as np


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def summarize_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(prediction, target),
        "mae": mae(prediction, target),
    }
