from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from wafer_vision.data import wafer_map_to_tensor
from wafer_vision.model import WaferCNN


def load_wafer_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            value: Any = pickle.load(f)
        return np.asarray(value)
    if suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        image = Image.open(path).convert("L")
        return np.asarray(image)
    raise ValueError(f"Unsupported input file type: {path.suffix}")


@torch.inference_mode()
def predict_one(checkpoint_path: str | Path, wafer_path: str | Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    input_size = int(checkpoint["input_size"])

    model = WaferCNN(num_classes=len(class_names), dropout=float(checkpoint.get("config", {}).get("dropout", 0.25))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wafer_array = load_wafer_array(wafer_path)
    x = wafer_map_to_tensor(wafer_array, input_size=input_size).unsqueeze(0).to(device)
    logits = model(x)
    probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()
    order = np.argsort(probs)[::-1]
    return {
        "prediction": class_names[int(order[0])],
        "confidence": float(probs[order[0]]),
        "top_k": [
            {"label": class_names[int(i)], "probability": float(probs[i])}
            for i in order[: min(5, len(order))]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict wafer defect type from one wafer map file.")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/wafer_cnn_best.pt")
    parser.add_argument("--input", required=True, help="Path to .npy/.pkl/.csv/.png wafer map")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args()

    result = predict_one(args.checkpoint, args.input)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
