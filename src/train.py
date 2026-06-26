from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for training. Install torch first.") from exc

from models.improved_model import ConvTransformerForecaster
from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster
from utils.metrics import inverse_target_scale, summarize_forecast_metrics
from utils.seed import set_seed


MODEL_REGISTRY = {
    "lstm": LSTMForecaster,
    "transformer": TransformerForecaster,
    "conv_transformer": ConvTransformerForecaster,
}


def _serializable_args(args: argparse.Namespace) -> dict:
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def _tensor_pair(data: np.lib.npyio.NpzFile, split: str) -> TensorDataset:
    x = torch.tensor(data[f"x_{split}"], dtype=torch.float32)
    y = torch.tensor(data[f"y_{split}"], dtype=torch.float32)
    return TensorDataset(x, y)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = criterion(prediction, y)
        if is_train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    for x, y in loader:
        prediction = model(x.to(device)).cpu().numpy()
        predictions.append(prediction)
        targets.append(y.numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.samples, allow_pickle=True)
    feature_dim = data["x_train"].shape[-1]
    horizon = int(data["horizon"])
    feature_names = data["feature_names"]
    target_name = str(np.asarray(data["target_name"]).item())
    scaler_mean = data["scaler_mean"]
    scaler_scale = data["scaler_scale"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_class = MODEL_REGISTRY[args.model]
    if args.model == "lstm":
        model = model_class(
            feature_dim=feature_dim,
            horizon=horizon,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        model = model_class(
            feature_dim=feature_dim,
            horizon=horizon,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    model = model.to(device)

    train_loader = DataLoader(
        _tensor_pair(data, "train"),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(_tensor_pair(data, "val"), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(_tensor_pair(data, "test"), batch_size=args.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = output_dir / "best_model.pt"
    history = []
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_left = args.patience
            torch.save({"model_state": model.state_dict(), "args": _serializable_args(args)}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_prediction, test_target = predict(model, test_loader, device)
    test_prediction_original = inverse_target_scale(
        test_prediction, feature_names, scaler_mean, scaler_scale, target_name
    )
    test_target_original = inverse_target_scale(
        test_target, feature_names, scaler_mean, scaler_scale, target_name
    )
    metrics = summarize_forecast_metrics(
        test_prediction,
        test_target,
        feature_names,
        scaler_mean,
        scaler_scale,
        target_name,
    )
    metrics.update(
        {
            "best_val_loss": best_val,
            "epochs_ran": len(history),
            "horizon": horizon,
            "model": args.model,
            "seed": args.seed,
        }
    )

    np.savez_compressed(
        output_dir / "test_predictions_scaled.npz",
        prediction=test_prediction,
        target=test_target,
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        prediction=test_prediction_original,
        target=test_target_original,
        prediction_scaled=test_prediction,
        target_scaled=test_target,
        feature_names=feature_names,
        target_name=np.asarray(target_name),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        horizon=np.asarray(horizon),
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
