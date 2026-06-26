from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {
    "date",
    "observed_minutes",
    "valid_global_active_power_minutes",
    "is_complete_day",
}


@dataclass
class FeatureScaler:
    feature_names: list[str]
    mean: list[float]
    scale: list[float]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_names].to_numpy(dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        return (values - mean) / scale

    def inverse_target(self, values: np.ndarray, target_name: str) -> np.ndarray:
        target_index = self.feature_names.index(target_name)
        return values * self.scale[target_index] + self.mean[target_index]


def infer_feature_names(daily: pd.DataFrame, target_name: str) -> list[str]:
    numeric_columns = daily.select_dtypes(include=[np.number]).columns.tolist()
    features = [column for column in numeric_columns if column not in DEFAULT_EXCLUDE_COLUMNS]
    if target_name not in features:
        raise ValueError(f"Target column {target_name!r} is not numeric or does not exist.")
    return features


def fit_scaler(
    daily: pd.DataFrame,
    feature_names: list[str],
    train_ratio: float,
) -> FeatureScaler:
    train_rows = max(1, int(len(daily) * train_ratio))
    train_values = daily.iloc[:train_rows][feature_names].to_numpy(dtype=np.float32)
    mean = np.nanmean(train_values, axis=0)
    scale = np.nanstd(train_values, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return FeatureScaler(
        feature_names=feature_names,
        mean=mean.astype(float).tolist(),
        scale=scale.astype(float).tolist(),
    )


def make_supervised_windows(
    values: np.ndarray,
    dates: np.ndarray,
    target_index: int,
    input_length: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    total = len(values) - input_length - horizon + 1
    if total <= 0:
        raise ValueError(
            f"Not enough daily rows ({len(values)}) for input_length={input_length} "
            f"and horizon={horizon}."
        )

    x = np.empty((total, input_length, values.shape[1]), dtype=np.float32)
    y = np.empty((total, horizon), dtype=np.float32)
    input_start_dates = []
    target_start_dates = []
    target_end_dates = []

    for i in range(total):
        input_start = i
        input_end = i + input_length
        target_start = input_end
        target_end = target_start + horizon

        x[i] = values[input_start:input_end]
        y[i] = values[target_start:target_end, target_index]
        input_start_dates.append(dates[input_start])
        target_start_dates.append(dates[target_start])
        target_end_dates.append(dates[target_end - 1])

    return {
        "x": x,
        "y": y,
        "input_start_dates": np.asarray(input_start_dates),
        "target_start_dates": np.asarray(target_start_dates),
        "target_end_dates": np.asarray(target_end_dates),
    }


def split_windows(
    windows: dict[str, np.ndarray],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, np.ndarray]:
    sample_count = len(windows["x"])
    train_end = int(sample_count * train_ratio)
    val_end = int(sample_count * (train_ratio + val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= sample_count:
        raise ValueError(
            "Invalid split. Need non-empty train, validation, and test window sets."
        )

    split_indices = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, sample_count),
    }

    output: dict[str, np.ndarray] = {}
    for split, idx in split_indices.items():
        for key, value in windows.items():
            output[f"{key}_{split}"] = value[idx]
    return output


def save_scaler(scaler: FeatureScaler, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(scaler), indent=2), encoding="utf-8")


def load_scaler(path: Path) -> FeatureScaler:
    data = json.loads(path.read_text(encoding="utf-8"))
    return FeatureScaler(**data)


def create_window_file(
    daily_path: Path,
    output_path: Path,
    scaler_path: Path,
    input_length: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    target_name: str = "global_active_power",
    feature_names: list[str] | None = None,
    fit_new_scaler: bool = True,
) -> dict[str, tuple[int, ...]]:
    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    if feature_names is None:
        feature_names = infer_feature_names(daily, target_name)

    # Fill any remaining feature gaps after optional weather merge.
    daily[feature_names] = daily[feature_names].interpolate(limit_direction="both")
    daily[feature_names] = daily[feature_names].ffill().bfill()

    if fit_new_scaler or not scaler_path.exists():
        scaler = fit_scaler(daily, feature_names, train_ratio=train_ratio)
        save_scaler(scaler, scaler_path)
    else:
        scaler = load_scaler(scaler_path)

    values = scaler.transform(daily)
    dates = daily["date"].dt.strftime("%Y-%m-%d").to_numpy()
    target_index = scaler.feature_names.index(target_name)

    windows = make_supervised_windows(
        values=values,
        dates=dates,
        target_index=target_index,
        input_length=input_length,
        horizon=horizon,
    )
    split = split_windows(windows, train_ratio=train_ratio, val_ratio=val_ratio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **split,
        feature_names=np.asarray(scaler.feature_names),
        target_name=np.asarray(target_name),
        input_length=np.asarray(input_length),
        horizon=np.asarray(horizon),
        scaler_mean=np.asarray(scaler.mean, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale, dtype=np.float32),
    )

    return {key: tuple(value.shape) for key, value in split.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create supervised forecasting windows.")
    parser.add_argument("--daily", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scaler", required=True, type=Path)
    parser.add_argument("--input-length", type=int, default=90)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--target", default="global_active_power")
    parser.add_argument("--reuse-scaler", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = create_window_file(
        daily_path=args.daily,
        output_path=args.output,
        scaler_path=args.scaler,
        input_length=args.input_length,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_name=args.target,
        fit_new_scaler=not args.reuse_scaler,
    )
    print(f"Wrote {args.output}")
    for name, shape in shapes.items():
        print(f"{name}: {shape}")


if __name__ == "__main__":
    main()
