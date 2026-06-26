from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.metrics import summarize_metrics


def evaluate_predictions(prediction_file: Path, output_file: Path | None = None) -> dict[str, float]:
    data = np.load(prediction_file)
    metrics = summarize_metrics(data["prediction"], data["target"])
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved prediction arrays.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_predictions(args.predictions, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
