from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [42, 2026, 7, 13, 99]
TASKS = {
    "short": {
        "samples": Path("data/processed/short_horizon_samples.npz"),
        "horizon": 90,
    },
    "long": {
        "samples": Path("data/processed/long_horizon_samples.npz"),
        "horizon": 365,
    },
}
MODELS = ["lstm", "transformer", "conv_transformer"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def collect_results(results_path: Path, summary_path: Path, checkpoints_root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_file in sorted(checkpoints_root.glob("*_*_seed*/metrics.json")):
        run_dir = metrics_file.parent
        name = run_dir.name
        try:
            task, model_part = name.split("_", 1)
            model, seed_text = model_part.rsplit("_seed", 1)
            seed = int(seed_text)
        except ValueError:
            continue

        metrics = _read_json(metrics_file)
        history_path = run_dir / "history.json"
        history = _read_json(history_path) if history_path.exists() else []
        figure_path = Path("experiments/figures") / f"{name}_sample0.png"

        rows.append(
            {
                "task": task,
                "horizon": metrics.get("horizon", TASKS.get(task, {}).get("horizon")),
                "model": model,
                "seed": seed,
                "epochs_ran": metrics.get("epochs_ran", len(history)),
                "best_val_loss": metrics.get("best_val_loss", ""),
                "test_mse_scaled": metrics.get("mse_scaled", ""),
                "test_mae_scaled": metrics.get("mae_scaled", ""),
                "test_mse_original": metrics.get("mse_original", ""),
                "test_mae_original": metrics.get("mae_original", ""),
                "checkpoint_dir": str(run_dir).replace("\\", "/"),
                "figure_path": str(figure_path).replace("\\", "/") if figure_path.exists() else "",
            }
        )

    fieldnames = [
        "task",
        "horizon",
        "model",
        "seed",
        "epochs_ran",
        "best_val_loss",
        "test_mse_scaled",
        "test_mae_scaled",
        "test_mse_original",
        "test_mae_original",
        "checkpoint_dir",
        "figure_path",
    ]
    _write_csv(results_path, rows, fieldnames)

    summary_rows = []
    groups = sorted({(row["task"], row["model"]) for row in rows})
    for task, model in groups:
        group_rows = [row for row in rows if row["task"] == task and row["model"] == model]
        mse_original = [float(row["test_mse_original"]) for row in group_rows]
        mae_original = [float(row["test_mae_original"]) for row in group_rows]
        mse_scaled = [float(row["test_mse_scaled"]) for row in group_rows]
        mae_scaled = [float(row["test_mae_scaled"]) for row in group_rows]
        summary_rows.append(
            {
                "task": task,
                "horizon": group_rows[0]["horizon"] if group_rows else "",
                "model": model,
                "runs": len(group_rows),
                "mse_original_mean": _mean(mse_original),
                "mse_original_std": _std(mse_original),
                "mae_original_mean": _mean(mae_original),
                "mae_original_std": _std(mae_original),
                "mse_scaled_mean": _mean(mse_scaled),
                "mse_scaled_std": _std(mse_scaled),
                "mae_scaled_mean": _mean(mae_scaled),
                "mae_scaled_std": _std(mae_scaled),
            }
        )

    summary_fieldnames = [
        "task",
        "horizon",
        "model",
        "runs",
        "mse_original_mean",
        "mse_original_std",
        "mae_original_mean",
        "mae_original_std",
        "mse_scaled_mean",
        "mse_scaled_std",
        "mae_scaled_mean",
        "mae_scaled_std",
    ]
    _write_csv(summary_path, summary_rows, summary_fieldnames)
    return rows


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_train_command(command: list[str], metrics_file: Path, prediction_file: Path) -> None:
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode == 0:
        return
    if metrics_file.exists() and prediction_file.exists():
        print(
            "Training subprocess returned "
            f"{completed.returncode}, but expected outputs exist; continuing.",
            flush=True,
        )
        return
    raise subprocess.CalledProcessError(completed.returncode, command)


def run_experiments(args: argparse.Namespace) -> None:
    selected_tasks = args.tasks or list(TASKS)
    selected_models = args.models or MODELS
    selected_seeds = args.seeds or DEFAULT_SEEDS

    for task in selected_tasks:
        task_info = TASKS[task]
        for model in selected_models:
            for seed in selected_seeds:
                run_name = f"{task}_{model}_seed{seed}"
                output_dir = args.checkpoints_root / run_name
                metrics_file = output_dir / "metrics.json"
                prediction_file = output_dir / "test_predictions.npz"
                if args.skip_existing and metrics_file.exists() and prediction_file.exists():
                    print(f"Skip existing run: {run_name}", flush=True)
                else:
                    command = [
                        sys.executable,
                        "src/train.py",
                        "--samples",
                        str(task_info["samples"]),
                        "--model",
                        model,
                        "--output-dir",
                        str(output_dir),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--learning-rate",
                        str(args.learning_rate),
                        "--weight-decay",
                        str(args.weight_decay),
                        "--patience",
                        str(args.patience),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        "--hidden-size",
                        str(args.hidden_size),
                        "--d-model",
                        str(args.d_model),
                        "--nhead",
                        str(args.nhead),
                        "--num-layers",
                        str(args.num_layers),
                        "--dropout",
                        str(args.dropout),
                    ]
                    run_train_command(command, metrics_file, prediction_file)

                if prediction_file.exists() and not args.no_plots:
                    figure_path = args.figures_root / f"{run_name}_sample0.png"
                    if args.skip_existing and figure_path.exists():
                        print(f"Skip existing figure: {figure_path}", flush=True)
                    else:
                        run_command(
                            [
                                sys.executable,
                                "src/plot_predictions.py",
                                "--predictions",
                                str(prediction_file),
                                "--output",
                                str(figure_path),
                                "--sample-index",
                                "0",
                                "--title",
                                f"{task} {model} seed {seed}",
                            ]
                        )

    rows = collect_results(args.results, args.summary, args.checkpoints_root)
    print(f"Wrote {len(rows)} run rows to {args.results}", flush=True)
    print(f"Wrote grouped summary to {args.summary}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize forecasting experiments.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASKS), default=None)
    parser.add_argument("--models", nargs="*", choices=MODELS, default=None)
    parser.add_argument("--checkpoints-root", type=Path, default=Path("experiments/checkpoints"))
    parser.add_argument("--figures-root", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--results", type=Path, default=Path("experiments/tables/results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("experiments/tables/summary.csv"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        rows = collect_results(args.results, args.summary, args.checkpoints_root)
        print(f"Wrote {len(rows)} run rows to {args.results}", flush=True)
        print(f"Wrote grouped summary to {args.summary}", flush=True)
        raise SystemExit(0)
    return args


if __name__ == "__main__":
    run_experiments(parse_args())
