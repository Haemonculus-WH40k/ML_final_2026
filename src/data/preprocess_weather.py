from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
STATION_COLUMNS = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, sep=";", compression="infer")
    except pd.errors.ParserError:
        return pd.read_csv(path, compression="infer")


def _find_month_column(df: pd.DataFrame) -> str:
    candidates = ["MONTH", "DATE", "AAAAMM", "YYYYMM", "年月", "MOIS"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not infer month column. Expected one of: "
        + ", ".join(candidates)
    )


def _parse_month(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace(r"[^0-9]", "", regex=True)

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    compact_mask = compact.str.len().ge(6)
    parsed.loc[compact_mask] = pd.to_datetime(
        compact.loc[compact_mask].str.slice(0, 6) + "01",
        format="%Y%m%d",
        errors="coerce",
    )
    fallback_mask = parsed.isna()
    parsed.loc[fallback_mask] = pd.to_datetime(text.loc[fallback_mask], errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def normalize_weather(
    weather_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
) -> pd.DataFrame:
    """Normalize monthly weather data to one row per month."""
    weather_path = Path(weather_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _read_table(weather_path)
    upper_map = {column: str(column).upper() for column in df.columns}
    df = df.rename(columns=upper_map)

    month_column = _find_month_column(df)
    available = [column for column in WEATHER_COLUMNS if column in df.columns]
    missing = sorted(set(WEATHER_COLUMNS) - set(available))
    if missing:
        raise ValueError(f"Missing weather columns: {missing}")

    keep_columns = [month_column, *WEATHER_COLUMNS]
    available_station_columns = [column for column in STATION_COLUMNS if column in df.columns]
    normalized = df[[*available_station_columns, *keep_columns]].copy()
    normalized = normalized.rename(columns={month_column: "month"})
    normalized["month"] = _parse_month(normalized["month"])
    normalized = normalized.dropna(subset=["month"])

    for column in WEATHER_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    station_summary = None
    if available_station_columns:
        station_summary = (
            normalized[available_station_columns]
            .drop_duplicates()
            .sort_values(available_station_columns[0])
            .to_dict(orient="records")
        )

    # Météo-France stores RR with tenths precision. The course handout asks
    # to divide the recorded value by 10 before using it.
    normalized["RR"] = normalized["RR"] / 10.0

    monthly = (
        normalized.groupby("month", as_index=False)[WEATHER_COLUMNS]
        .mean(numeric_only=True)
        .sort_values("month")
    )
    missing_before_fill = monthly[WEATHER_COLUMNS].isna().sum().to_dict()
    monthly[WEATHER_COLUMNS] = monthly[WEATHER_COLUMNS].interpolate(limit_direction="both")
    monthly[WEATHER_COLUMNS] = monthly[WEATHER_COLUMNS].ffill().bfill()
    missing_after_fill = monthly[WEATHER_COLUMNS].isna().sum().to_dict()

    monthly["month"] = monthly["month"].dt.strftime("%Y-%m")
    monthly.to_csv(output_path, index=False)

    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source_path": str(weather_path),
            "output_path": str(output_path),
            "aggregation": "mean across available stations in the monthly department file",
            "rr_transform": "RR divided by 10 according to the course handout",
            "missing_before_fill": missing_before_fill,
            "missing_after_fill": missing_after_fill,
            "station_count": len(station_summary or []),
            "stations": station_summary,
            "columns": ["month", *WEATHER_COLUMNS],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return monthly


def merge_monthly_weather(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    weather = weather.copy()
    daily["month"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m")
    weather["month"] = weather["month"].astype(str).str.slice(0, 7)
    merged = daily.merge(weather, on="month", how="left")
    return merged.drop(columns=["month"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize monthly weather data.")
    parser.add_argument("--weather", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized = normalize_weather(args.weather, args.output)
    print(f"Wrote {len(normalized)} monthly weather rows to {args.output}")


if __name__ == "__main__":
    main()
