from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from tqdm.auto import tqdm

from wafer_vision.data import compute_class_weights, make_dataloaders
from wafer_vision.model import WaferCNN, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Empty config: {path}")
    return config


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = []

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_targets.extend(y.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_targets, all_preds),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[list[float]] = []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        probs = logits.softmax(dim=1)

        losses.append(loss.item())
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_targets, all_preds),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "num_examples": len(all_targets),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WaferVision CNN baseline on WM-811K.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config.")
    parser.add_argument("--data-path", default=None, help="Override dataset path.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Quick experiment subset size.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path is not None:
        config["data_path"] = args.data_path
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = resolve_project_path(config["data_path"])
    output_dir = resolve_project_path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    report_dir = output_dir / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    loaders, splits = make_dataloaders(
        data_path=data_path,
        input_size=int(config["input_size"]),
        include_none=bool(config["include_none"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        val_size=float(config["val_size"]),
        test_size=float(config["test_size"]),
        seed=int(config["seed"]),
        max_samples=config.get("max_samples"),
        augment_train=bool(config["augment_train"]),
    )

    class_names = splits.class_names
    model = WaferCNN(num_classes=len(class_names), dropout=float(config["dropout"])).to(device)
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Class counts: {splits.class_counts}")
    print(f"Train/Val/Test: {len(splits.train)}/{len(splits.val)}/{len(splits.test)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    class_weights = None
    if bool(config["use_class_weights"]):
        class_weights = compute_class_weights(splits.train, class_names).to(device)
        print(f"Class weights: {[round(float(w), 4) for w in class_weights]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_steps = max(1, len(loaders["train"]) * int(config["epochs"]))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["learning_rate"]),
        total_steps=total_steps,
        pct_start=0.2,
    )

    best_macro_f1 = -1.0
    history: list[dict[str, Any]] = []
    best_path = checkpoint_dir / "wafer_cnn_best.pt"

    for epoch in range(1, int(config["epochs"]) + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device, scheduler)
        val_metrics = evaluate(model, loaders["val"], criterion, device, class_names)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": {k: v for k, v in val_metrics.items() if k not in {"classification_report", "confusion_matrix"}},
        }
        history.append(row)
        print(
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "input_size": int(config["input_size"]),
                    "config": config,
                    "val_metrics": val_metrics,
                    "class_counts": splits.class_counts,
                },
                best_path,
            )
            print(f"Saved best checkpoint -> {best_path}")

    # Final test on best checkpoint.
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], criterion, device, class_names)

    summary = {
        "best_val_macro_f1": best_macro_f1,
        "test": test_metrics,
        "history": history,
        "class_names": class_names,
        "class_counts": splits.class_counts,
        "checkpoint": str(best_path),
    }
    save_json(report_dir / "metrics.json", summary)
    np.savetxt(report_dir / "confusion_matrix.csv", np.array(test_metrics["confusion_matrix"]), delimiter=",", fmt="%d")
    print("\nFinal test metrics")
    print(f"accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"Reports saved -> {report_dir}")


if __name__ == "__main__":
    main()
