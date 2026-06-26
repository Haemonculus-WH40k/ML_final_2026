from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from preprocess_power import preprocess_power
from preprocess_weather import merge_monthly_weather, normalize_weather
from make_windows import create_window_file


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline(config_path: Path) -> None:
    root = config_path.resolve().parents[1]
    config = load_config(config_path)

    raw_power_path = _resolve(root, config["raw_power_path"])
    weather_path = _resolve(root, config.get("weather_path"))
    daily_power_path = _resolve(root, config["daily_power_path"])
    daily_merged_path = _resolve(root, config["daily_merged_path"])
    short_samples_path = _resolve(root, config["short_samples_path"])
    long_samples_path = _resolve(root, config["long_samples_path"])
    scaler_path = _resolve(root, config["scaler_path"])

    metadata_path = daily_power_path.with_suffix(".metadata.json")
    daily = preprocess_power(
        raw_path=raw_power_path,
        output_path=daily_power_path,
        metadata_path=metadata_path,
        min_minutes_per_day=int(config.get("min_minutes_per_day", 1000)),
    )
    print(f"Daily power rows: {len(daily)}")

    if weather_path is not None and weather_path.exists():
        weather_normalized_path = daily_merged_path.with_name("weather_monthly_normalized.csv")
        weather = normalize_weather(
            weather_path,
            weather_normalized_path,
            metadata_path=weather_normalized_path.with_suffix(".metadata.json"),
        )
        merged = merge_monthly_weather(daily, weather)
        print(f"Merged weather columns from {weather_path}")
    else:
        merged = daily.copy()
        print("Weather path is not set or file is missing; continuing with power features only.")

    daily_merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(daily_merged_path, index=False)
    print(f"Merged daily data: {daily_merged_path}")

    split_ratios = config["split_ratios"]
    train_ratio = float(split_ratios["train"])
    val_ratio = float(split_ratios["val"])

    short_shapes = create_window_file(
        daily_path=daily_merged_path,
        output_path=short_samples_path,
        scaler_path=scaler_path,
        input_length=int(config["input_length"]),
        horizon=int(config["short_horizon"]),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        fit_new_scaler=True,
    )
    print(f"Short horizon samples: {short_samples_path}")
    for key, shape in short_shapes.items():
        print(f"  {key}: {shape}")

    long_shapes = create_window_file(
        daily_path=daily_merged_path,
        output_path=long_samples_path,
        scaler_path=scaler_path,
        input_length=int(config["input_length"]),
        horizon=int(config["long_horizon"]),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        fit_new_scaler=False,
    )
    print(f"Long horizon samples: {long_samples_path}")
    for key, shape in long_shapes.items():
        print(f"  {key}: {shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data preprocessing pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
