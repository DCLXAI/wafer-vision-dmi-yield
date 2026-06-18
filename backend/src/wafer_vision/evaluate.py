from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from wafer_vision.data import make_dataloaders
from wafer_vision.model import WaferCNN
from wafer_vision.train import evaluate, resolve_project_path


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Empty config: {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained WaferVision checkpoint.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/wafer_cnn_best.pt")
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path is not None:
        config["data_path"] = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    input_size = int(checkpoint["input_size"])

    loaders, _ = make_dataloaders(
        data_path=resolve_project_path(config["data_path"]),
        input_size=input_size,
        include_none=("None" in class_names),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        val_size=float(config["val_size"]),
        test_size=float(config["test_size"]),
        seed=int(config["seed"]),
        max_samples=config.get("max_samples"),
        augment_train=False,
    )

    model = WaferCNN(num_classes=len(class_names), dropout=float(config["dropout"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loaders["test"], criterion, device, class_names)

    report_dir = resolve_project_path(config["output_dir"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.savetxt(report_dir / "eval_confusion_matrix.csv", np.array(metrics["confusion_matrix"]), delimiter=",", fmt="%d")

    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"Saved evaluation report -> {report_dir}")


if __name__ == "__main__":
    main()
