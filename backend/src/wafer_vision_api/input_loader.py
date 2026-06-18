from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".npy", ".csv", ".png", ".jpg", ".jpeg", ".bmp"}


class InputDecodeError(ValueError):
    pass


def _extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


def load_wafer_array_from_upload(content: bytes, filename: str | None, content_type: str | None = None) -> np.ndarray:
    """Decode uploaded wafer map bytes into a 2D numpy array.

    Supported portfolio/demo formats:
    - .npy: numpy array saved with np.save
    - .csv: numeric 2D grid
    - .png/.jpg/.jpeg/.bmp: grayscale image exported from a wafer map
    """
    if not content:
        raise InputDecodeError("Uploaded file is empty.")

    suffix = _extension(filename)
    ctype = (content_type or "").lower()

    try:
        if suffix == ".npy" or "numpy" in ctype:
            value = np.load(io.BytesIO(content), allow_pickle=False)
            return _ensure_2d(value)

        if suffix == ".csv" or "csv" in ctype:
            # np.loadtxt is strict and fast; fallback gives a clearer error for ragged rows.
            try:
                value = np.loadtxt(io.BytesIO(content), delimiter=",")
            except Exception:
                text = content.decode("utf-8-sig")
                rows = [[float(cell) for cell in row] for row in csv.reader(io.StringIO(text)) if row]
                value = np.asarray(rows, dtype=np.float32)
            return _ensure_2d(value)

        if suffix in {".png", ".jpg", ".jpeg", ".bmp"} or ctype.startswith("image/"):
            image = Image.open(io.BytesIO(content)).convert("L")
            return _ensure_2d(np.asarray(image))
    except InputDecodeError:
        raise
    except Exception as exc:  # pragma: no cover - exact PIL/numpy exceptions differ by version
        raise InputDecodeError(f"Could not decode {filename or 'upload'}: {exc}") from exc

    raise InputDecodeError(
        f"Unsupported upload type. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )


def load_wafer_array_from_inline(value: list[list[float]]) -> np.ndarray:
    return _ensure_2d(np.asarray(value, dtype=np.float32))


def _ensure_2d(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise InputDecodeError(f"Expected a 2D wafer map, got shape {arr.shape}.")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise InputDecodeError(f"Wafer map is too small: shape={arr.shape}.")
    if not np.isfinite(arr.astype(float)).all():
        raise InputDecodeError("Wafer map contains NaN or infinite values.")
    return arr.astype(np.float32)


def describe_array(arr: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(arr)
    return {
        "rows": int(arr.shape[0]) if arr.ndim >= 1 else None,
        "cols": int(arr.shape[1]) if arr.ndim >= 2 else None,
        "min_value": float(np.min(arr)) if arr.size else None,
        "max_value": float(np.max(arr)) if arr.size else None,
    }
