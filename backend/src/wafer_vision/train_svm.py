from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from wafer_vision.data import DEFECT_CLASSES_8, load_lswmd_dataframe
from wafer_vision.features import FEATURE_SCHEMA, extract_kaggle_feature_vector


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_feature_matrix(wafer_maps: list[Any], desc: str = "features") -> np.ndarray:
    return np.vstack([extract_kaggle_feature_vector(w) for w in tqdm(wafer_maps, desc=desc)]).astype(np.float32)


def balanced_sample(frame, per_class: int | None, seed: int):
    if per_class is None or per_class <= 0:
        return frame.reset_index(drop=True)
    sampled = (
        frame.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_class), random_state=seed))
        .reset_index(drop=True)
    )
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kaggle notebook One-vs-One SVM baseline on WM-811K.")
    parser.add_argument("--data-path", default="data/raw/LSWMD.pkl")
    parser.add_argument("--output-dir", default="artifacts/kaggle_svm_baseline")
    parser.add_argument("--balanced-per-class", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=8000)
    parser.add_argument("--no-scale", action="store_true", help="Disable StandardScaler to match the notebook more closely.")
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, class_names = load_lswmd_dataframe(args.data_path, include_none=False)
    frame = frame[frame["label"].isin(DEFECT_CLASSES_8)].copy().reset_index(drop=True)
    frame = balanced_sample(frame, args.balanced_per_class, args.seed)

    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = frame["label"].map(label_to_idx).astype(int).to_numpy()
    X = build_feature_matrix(frame["wafer_map"].to_list(), desc="extracting 59D Kaggle features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )

    clf = OneVsOneClassifier(LinearSVC(random_state=args.seed, max_iter=args.max_iter, dual="auto"))
    model = clf if args.no_scale else Pipeline([("scaler", StandardScaler()), ("ovo_svm", clf)])
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_test_pred, labels=list(range(len(class_names))))
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_macro_f1": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "num_samples": int(len(frame)),
        "test_size": float(args.test_size),
    }
    bundle = {
        "model_kind": "kaggle_svm_ovo",
        "model_version": "kaggle-59d-ovo-linear-svm-v1",
        "model": model,
        "class_names": class_names,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": len(FEATURE_SCHEMA),
        "metrics": metrics,
        "source": {
            "dataset": "WM-811K / LSWMD.pkl",
            "notebook": "loozin/wm-811k-wafermap",
            "notes": "13 region density + 40 Radon + 6 geometry features, One-vs-One LinearSVC.",
        },
        "created_seconds": float(time.perf_counter() - started),
    }
    bundle_path = output_dir / "kaggle_svm_ovo.joblib"
    joblib.dump(bundle, bundle_path)
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "feature_schema.json", {"feature_schema": FEATURE_SCHEMA})
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    print(f"Classes: {class_names}")
    print(f"Samples: {len(frame):,} | Feature dim: {X.shape[1]}")
    print(f"One-vs-One SVM train_acc={metrics['train_accuracy']:.4f} test_acc={metrics['test_accuracy']:.4f} macro_f1={metrics['test_macro_f1']:.4f}")
    print(f"Saved bundle -> {bundle_path}")


if __name__ == "__main__":
    main()
