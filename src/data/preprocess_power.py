from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

RENAME_COLUMNS = {
    "Global_active_power": "global_active_power",
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3",
}

SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
]

MEAN_COLUMNS = ["voltage", "global_intensity"]


def read_power_data(raw_path: Path) -> pd.DataFrame:
    """Read the minute-level power file and normalize column types."""
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    missing_columns = {"Date", "Time", *RAW_NUMERIC_COLUMNS} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    timestamp = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.assign(timestamp=timestamp)
    df = df.dropna(subset=["timestamp"])

    for column in RAW_NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[["timestamp", *RAW_NUMERIC_COLUMNS]]
    df = df.rename(columns=RENAME_COLUMNS)
    df = df.sort_values("timestamp")
    df = df.groupby("timestamp", as_index=False).mean(numeric_only=True)
    return df


def build_complete_minute_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindex to one-minute frequency and interpolate numeric gaps."""
    df = df.set_index("timestamp").sort_index()

    daily_quality = pd.DataFrame(
        {
            "observed_minutes": df["global_active_power"].groupby(pd.Grouper(freq="D")).size(),
            "valid_global_active_power_minutes": df["global_active_power"]
            .notna()
            .groupby(pd.Grouper(freq="D"))
            .sum(),
        }
    )

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="min")
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    numeric_columns = list(RENAME_COLUMNS.values())
    df[numeric_columns] = df[numeric_columns].interpolate(method="time", limit_direction="both")

    df["sub_metering_remainder"] = (
        df["global_active_power"] * 1000.0 / 60.0
        - (df["sub_metering_1"] + df["sub_metering_2"] + df["sub_metering_3"])
    )
    return df, daily_quality


def aggregate_daily(
    minute_df: pd.DataFrame,
    daily_quality: pd.DataFrame,
    min_minutes_per_day: int,
    keep_incomplete_days: bool = False,
) -> pd.DataFrame:
    """Aggregate minute-level records to daily features required by the task."""
    sum_daily = minute_df[SUM_COLUMNS].resample("D").sum(min_count=1)
    mean_daily = minute_df[MEAN_COLUMNS].resample("D").mean()
    daily = pd.concat([sum_daily, mean_daily], axis=1)

    daily = daily.join(daily_quality, how="left")
    daily["observed_minutes"] = daily["observed_minutes"].fillna(0).astype(int)
    daily["valid_global_active_power_minutes"] = (
        daily["valid_global_active_power_minutes"].fillna(0).astype(int)
    )
    daily["is_complete_day"] = daily["observed_minutes"] >= min_minutes_per_day

    if not keep_incomplete_days:
        daily = daily[daily["is_complete_day"]].copy()

    daily = daily.reset_index().rename(columns={"index": "date", "timestamp": "date"})
    daily.insert(0, "date", pd.to_datetime(daily.pop("date")).dt.strftime("%Y-%m-%d"))
    return daily


def preprocess_power(
    raw_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    min_minutes_per_day: int = 1000,
    keep_incomplete_days: bool = False,
) -> pd.DataFrame:
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = read_power_data(raw_path)
    minute_df, quality = build_complete_minute_frame(raw_df)
    daily = aggregate_daily(
        minute_df,
        quality,
        min_minutes_per_day=min_minutes_per_day,
        keep_incomplete_days=keep_incomplete_days,
    )
    daily.to_csv(output_path, index=False)

    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "raw_path": str(raw_path),
            "output_path": str(output_path),
            "raw_rows": int(len(raw_df)),
            "daily_rows": int(len(daily)),
            "first_day": str(daily["date"].iloc[0]) if len(daily) else None,
            "last_day": str(daily["date"].iloc[-1]) if len(daily) else None,
            "min_minutes_per_day": int(min_minutes_per_day),
            "keep_incomplete_days": bool(keep_incomplete_days),
            "columns": list(daily.columns),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess household power data.")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--min-minutes-per-day", type=int, default=1000)
    parser.add_argument("--keep-incomplete-days", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily = preprocess_power(
        raw_path=args.raw,
        output_path=args.output,
        metadata_path=args.metadata,
        min_minutes_per_day=args.min_minutes_per_day,
        keep_incomplete_days=args.keep_incomplete_days,
    )
    print(f"Wrote {len(daily)} daily rows to {args.output}")


if __name__ == "__main__":
    main()
