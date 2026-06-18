from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from wafer_vision.data import DEFECT_CLASSES_8
from wafer_vision.features import FEATURE_SCHEMA, extract_kaggle_feature_vector


def synthetic_pattern(label: str, size: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(abs(hash((label, seed))) % (2**32))
    m = np.zeros((size, size), dtype=np.uint8)
    c = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    wafer = dist <= size * 0.46
    m[wafer] = 1
    if label == "Center":
        defect = dist < size * rng.uniform(0.10, 0.16)
    elif label == "Donut":
        defect = (dist > size * 0.16) & (dist < size * 0.29)
    elif label == "Edge-Loc":
        defect = (dist > size * 0.34) & (xx > size * rng.uniform(0.58, 0.70)) & (yy < size * rng.uniform(0.35, 0.50))
    elif label == "Edge-Ring":
        defect = (dist > size * 0.37) & (dist < size * 0.46)
    elif label == "Loc":
        cx, cy = size * rng.uniform(0.25, 0.72), size * rng.uniform(0.25, 0.72)
        defect = ((xx - cx) ** 2 + (yy - cy) ** 2) < (size * rng.uniform(0.06, 0.12)) ** 2
    elif label == "Near-full":
        defect = wafer & (rng.random((size, size)) > 0.08)
    elif label == "Scratch":
        slope, intercept = rng.uniform(-0.7, 0.7), rng.uniform(size * 0.18, size * 0.72)
        defect = np.abs(yy - (slope * xx + intercept)) < rng.uniform(1.2, 2.4)
    else:
        defect = wafer & (rng.random((size, size)) > 0.955)
    m[defect & wafer] = 2
    noise = wafer & (rng.random((size, size)) > 0.992)
    m[noise] = 2
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small demo SVM bundle without downloading WM-811K.")
    parser.add_argument("--output", default="artifacts/checkpoints/kaggle_svm_demo.joblib")
    parser.add_argument("--samples-per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    X, y = [], []
    for label_idx, label in enumerate(DEFECT_CLASSES_8):
        for i in range(args.samples_per_class):
            X.append(extract_kaggle_feature_vector(synthetic_pattern(label, seed=args.seed + i)))
            y.append(label_idx)
    X, y = np.vstack(X).astype(np.float32), np.asarray(y)
    model = Pipeline([("scaler", StandardScaler()), ("ovo_svm", OneVsOneClassifier(LinearSVC(random_state=args.seed, max_iter=5000, dual="auto")))])
    model.fit(X, y)
    pred = model.predict(X)
    metrics = {
        "train_accuracy": float(accuracy_score(y, pred)),
        "test_accuracy": None,
        "test_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "num_samples": int(len(y)),
        "warning": "Synthetic demo bundle only. Do not report this as WM-811K model performance.",
    }
    bundle = {
        "model_kind": "kaggle_svm_ovo", "model_version": "kaggle-59d-ovo-linear-svm-demo", "model": model,
        "class_names": DEFECT_CLASSES_8, "feature_schema": FEATURE_SCHEMA, "feature_dim": len(FEATURE_SCHEMA),
        "metrics": metrics, "source": {"notebook": "loozin/wm-811k-wafermap", "dataset": "synthetic demo only"},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved demo SVM bundle -> {out}")


if __name__ == "__main__":
    main()
