from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from wafer_vision.data import wafer_map_to_tensor
from wafer_vision.features import FEATURE_SCHEMA, extract_kaggle_feature_vector
from wafer_vision.model import WaferCNN
from wafer_vision_api.schemas import ModelMetadata, TopKPrediction


@dataclass(frozen=True)
class PredictionOutput:
    label: str
    confidence: float
    top_k: list[TopKPrediction]
    inference_ms: float


class ModelNotLoadedError(RuntimeError):
    pass


class WaferModelService:
    """Serve wafer-map classifiers from either image checkpoints or DMI feature bundles.

    v0.8 adds batch inference because simulator lots commonly contain hundreds
    of wafers. The public predict() method still serves single-upload API calls;
    simulator code should prefer predict_batch() to avoid 1000 separate model
    forwards.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "auto", top_k: int = 5, model_kind: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.requested_device = device
        self.top_k = int(top_k)
        self.requested_model_kind = model_kind
        self.device: torch.device | None = None
        self.model: Any | None = None
        self.class_names: list[str] = []
        self.input_size: int | None = None
        self.model_kind: str = "unloaded"
        self.model_version: str = "unloaded"
        self.validation_macro_f1: float | None = None
        self.feature_dim: int | None = None
        self.feature_schema: list[str] = []
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not self.checkpoint_path.exists():
            self.load_error = f"Model artifact not found: {self.checkpoint_path}"
            return
        try:
            kind = self._detect_model_kind()
            if kind == "cnn":
                self._load_cnn()
            elif kind in {"kaggle_svm", "kaggle_svm_ovo", "svm"}:
                self._load_svm()
            else:
                raise ValueError(f"Unsupported model kind: {kind}")
            self.load_error = None
        except Exception as exc:  # pragma: no cover
            self.model = None
            self.load_error = f"Failed to load model artifact: {exc}"

    def _detect_model_kind(self) -> str:
        requested = (self.requested_model_kind or "auto").lower()
        if requested != "auto":
            return requested
        if self.checkpoint_path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
            return "kaggle_svm"
        return "cnn"

    def _load_cnn(self) -> None:
        device = self._resolve_device(self.requested_device)
        if device.type == "cpu":
            # Small wafer CNN batches are latency-bound; many OpenMP threads can
            # be 50x slower than one thread in API/test environments.
            torch.set_num_threads(1)
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        class_names = list(checkpoint["class_names"])
        input_size = int(checkpoint["input_size"])
        dropout = float(checkpoint.get("config", {}).get("dropout", 0.25))
        model = WaferCNN(num_classes=len(class_names), dropout=dropout).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.device = device
        self.model = model
        self.class_names = class_names
        self.input_size = input_size
        self.model_kind = "cnn"
        self.model_version = str(checkpoint.get("model_version") or self.checkpoint_path.stem)
        self.validation_macro_f1 = _safe_float((checkpoint.get("val_metrics") or {}).get("macro_f1"))
        self.feature_dim = None
        self.feature_schema = []

    def _load_svm(self) -> None:
        bundle = joblib.load(self.checkpoint_path)
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        class_names = list(bundle.get("class_names", [])) if isinstance(bundle, dict) else []
        if not class_names:
            raise ValueError("SVM bundle must include class_names.")
        metrics = bundle.get("metrics", {}) if isinstance(bundle, dict) else {}
        self.device = None
        self.model = model
        self.class_names = class_names
        self.input_size = None
        self.model_kind = str(bundle.get("model_kind", "kaggle_svm_ovo")) if isinstance(bundle, dict) else "kaggle_svm_ovo"
        self.model_version = str(bundle.get("model_version", self.checkpoint_path.stem)) if isinstance(bundle, dict) else self.checkpoint_path.stem
        self.validation_macro_f1 = _safe_float(metrics.get("test_macro_f1") or metrics.get("macro_f1"))
        self.feature_schema = list(bundle.get("feature_schema", FEATURE_SCHEMA)) if isinstance(bundle, dict) else list(FEATURE_SCHEMA)
        self.feature_dim = int(bundle.get("feature_dim", len(self.feature_schema))) if isinstance(bundle, dict) else len(self.feature_schema)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        value = requested.lower()
        if value == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(value)

    def predict(self, wafer_map: np.ndarray) -> PredictionOutput:
        if not self.loaded or self.model is None:
            raise ModelNotLoadedError(self.load_error or "Model artifact is not loaded.")
        return self._predict_cnn(wafer_map) if self.model_kind == "cnn" else self._predict_svm(wafer_map)

    def batch_predict(self, wafer_maps: list[np.ndarray], batch_size: int = 128) -> list[PredictionOutput]:
        """Alias for callers that prefer verb-object naming."""
        return self.predict_batch(wafer_maps, batch_size=batch_size)

    def predict_batch(self, wafer_maps: list[np.ndarray], batch_size: int = 128) -> list[PredictionOutput]:
        """Predict a lot of wafer maps with one model path.

        CNN mode batches tensors before forward(); SVM mode stacks all 59D feature
        vectors before decision_function()/predict_proba(). This keeps simulator
        latency close to one batched inference instead of N single calls.
        """
        if not wafer_maps:
            return []
        if not self.loaded or self.model is None:
            raise ModelNotLoadedError(self.load_error or "Model artifact is not loaded.")
        if self.model_kind == "cnn":
            return self._predict_cnn_batch(wafer_maps, batch_size=batch_size)
        return self._predict_svm_batch(wafer_maps)

    @torch.inference_mode()
    def _predict_cnn(self, wafer_map: np.ndarray) -> PredictionOutput:
        return self._predict_cnn_batch([wafer_map], batch_size=1)[0]

    @torch.inference_mode()
    def _predict_cnn_batch(self, wafer_maps: list[np.ndarray], batch_size: int = 128) -> list[PredictionOutput]:
        if self.model is None or self.device is None or self.input_size is None:
            raise ModelNotLoadedError(self.load_error or "CNN checkpoint is not loaded.")
        started = time.perf_counter()
        tensors = [wafer_map_to_tensor(wafer_map, input_size=self.input_size) for wafer_map in wafer_maps]
        batch_size = max(1, int(batch_size or 128))
        prob_rows: list[np.ndarray] = []
        for start in range(0, len(tensors), batch_size):
            x = torch.stack(tensors[start : start + batch_size]).to(self.device, non_blocking=self.device.type == "cuda")
            logits = self.model(x)
            probs = logits.softmax(dim=1).detach().cpu().numpy()
            prob_rows.extend([np.asarray(row, dtype=np.float64) for row in probs])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_item_ms = elapsed_ms / max(1, len(prob_rows))
        return [self._make_output_from_probs(row, per_item_ms) for row in prob_rows]

    def _predict_svm(self, wafer_map: np.ndarray) -> PredictionOutput:
        return self._predict_svm_batch([wafer_map])[0]

    def _predict_svm_batch(self, wafer_maps: list[np.ndarray]) -> list[PredictionOutput]:
        started = time.perf_counter()
        features = np.vstack([extract_kaggle_feature_vector(wafer_map).reshape(1, -1) for wafer_map in wafer_maps])
        class_count = len(self.class_names)
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(features), dtype=np.float64)
            if scores.ndim == 1:
                # Binary estimators may return one margin. Expand into two-class logits.
                if class_count == 2:
                    scores = np.column_stack([-scores, scores])
                else:
                    fixed = np.zeros((features.shape[0], class_count), dtype=np.float64)
                    fixed[:, 0] = -scores
                    fixed[:, min(1, class_count - 1)] = scores
                    scores = fixed
            if scores.ndim == 2 and scores.shape[1] == class_count:
                prob_matrix = np.vstack([_softmax(row) for row in scores])
            else:
                preds = np.asarray(self.model.predict(features)).reshape(-1)
                prob_matrix = np.zeros((len(preds), class_count), dtype=np.float64)
                label_to_idx = {label: i for i, label in enumerate(self.class_names)}
                for row_idx, pred in enumerate(preds):
                    pred_idx = int(pred) if isinstance(pred, (int, np.integer)) else label_to_idx.get(str(pred), 0)
                    prob_matrix[row_idx, int(np.clip(pred_idx, 0, class_count - 1))] = 1.0
        elif hasattr(self.model, "predict_proba"):
            prob_matrix = _align_sklearn_probabilities(self.model, self.class_names, np.asarray(self.model.predict_proba(features), dtype=np.float64))
        else:
            preds = np.asarray(self.model.predict(features)).reshape(-1)
            prob_matrix = _prob_matrix_from_predictions(preds, self.class_names, confidence=0.92)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_item_ms = elapsed_ms / max(1, prob_matrix.shape[0])
        return [self._make_output_from_probs(row, per_item_ms) for row in prob_matrix]

    def _make_output_from_probs(self, probs: np.ndarray, inference_ms: float) -> PredictionOutput:
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if probs.size != len(self.class_names):
            fixed = np.zeros(len(self.class_names), dtype=np.float64)
            idx = int(np.argmax(probs)) if probs.size else 0
            fixed[int(np.clip(idx, 0, max(0, len(self.class_names) - 1)))] = 1.0
            probs = fixed
        total = probs.sum()
        probs = (np.ones(len(self.class_names)) / max(1, len(self.class_names))) if (not np.isfinite(total) or total <= 0) else probs / total
        order = np.argsort(probs)[::-1]
        top_k = [TopKPrediction(label=self.class_names[int(i)], probability=float(probs[int(i)])) for i in order[: min(self.top_k, len(order))]]
        return PredictionOutput(label=top_k[0].label, confidence=top_k[0].probability, top_k=top_k, inference_ms=float(inference_ms))

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            loaded=self.loaded,
            model_version=self.model_version,
            checkpoint_path=str(self.checkpoint_path),
            input_size=self.input_size,
            class_names=self.class_names,
            device=str(self.device) if self.device is not None else None,
            validation_macro_f1=self.validation_macro_f1,
            model_kind=self.model_kind,
            feature_dim=self.feature_dim,
            feature_schema=self.feature_schema,
            load_error=self.load_error,
        )


def _align_sklearn_probabilities(model: Any, class_names: list[str], probs: np.ndarray) -> np.ndarray:
    """Align scikit-learn probability columns with exported class_names."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != probs.shape[1]:
        return probs
    if [str(item) for item in classes] == [str(item) for item in class_names]:
        return probs
    fixed = np.zeros((probs.shape[0], len(class_names)), dtype=np.float64)
    label_to_idx = {str(label): idx for idx, label in enumerate(class_names)}
    for source_idx, label in enumerate(classes):
        target_idx = label_to_idx.get(str(label))
        if target_idx is not None:
            fixed[:, target_idx] = probs[:, source_idx]
    return fixed


def _prob_matrix_from_predictions(predictions: np.ndarray, class_names: list[str], confidence: float = 0.92) -> np.ndarray:
    class_count = max(1, len(class_names))
    preds = np.asarray(predictions).reshape(-1)
    floor = (1.0 - confidence) / max(1, class_count - 1)
    probs = np.full((len(preds), class_count), floor, dtype=np.float64)
    label_to_idx = {label: i for i, label in enumerate(class_names)}
    for row_idx, pred in enumerate(preds):
        if isinstance(pred, (int, np.integer)):
            pred_idx = int(pred)
        else:
            value = str(pred)
            try:
                pred_idx = int(float(value))
            except Exception:
                pred_idx = label_to_idx.get(value, 0)
        probs[row_idx, int(np.clip(pred_idx, 0, class_count - 1))] = confidence
    return probs


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    scores = scores - np.nanmax(scores)
    exp = np.exp(scores)
    denom = np.sum(exp)
    return np.ones_like(scores) / len(scores) if (not np.isfinite(denom) or denom <= 0) else exp / denom


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except Exception:
        return None
