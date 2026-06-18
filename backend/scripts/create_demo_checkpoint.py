from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wafer_vision.data import ALL_CLASSES_9
from wafer_vision.model import WaferCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an untrained demo checkpoint for API smoke tests.")
    parser.add_argument("--output", default="artifacts/checkpoints/wafer_cnn_demo.pt")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = WaferCNN(num_classes=len(ALL_CLASSES_9), dropout=0.25)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ALL_CLASSES_9,
            "input_size": args.input_size,
            "model_version": "wafer-cnn-demo-untrained-v0.2",
            "config": {"dropout": 0.25, "note": "Untrained checkpoint for API smoke testing only."},
            "val_metrics": {"macro_f1": None},
        },
        output,
    )
    print(f"Saved demo checkpoint -> {output}")
    print("Use this only to verify API wiring. Replace it with wafer_cnn_best.pt for real predictions.")


if __name__ == "__main__":
    main()
