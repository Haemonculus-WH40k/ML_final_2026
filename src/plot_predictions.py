from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def plot_prediction(
    prediction_file: Path,
    output_file: Path,
    sample_index: int = 0,
    title: str | None = None,
    scaled: bool = False,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(prediction_file)
    if scaled and {"prediction_scaled", "target_scaled"}.issubset(data.files):
        prediction = data["prediction_scaled"][sample_index]
        target = data["target_scaled"][sample_index]
        ylabel = "Scaled global active power"
    elif {"prediction", "target"}.issubset(data.files):
        prediction = data["prediction"][sample_index]
        target = data["target"][sample_index]
        ylabel = "Global active power"
    else:
        raise ValueError("Prediction file does not contain plottable prediction arrays.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(target, label="Ground Truth", linewidth=2)
    plt.plot(prediction, label="Prediction", linewidth=2)
    plt.xlabel("Forecast day")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prediction and ground truth curves.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--title", default=None)
    parser.add_argument("--scaled", action="store_true", help="Plot scaled arrays instead of original units.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_prediction(args.predictions, args.output, args.sample_index, args.title, args.scaled)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
