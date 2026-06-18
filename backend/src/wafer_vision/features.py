from __future__ import annotations

"""Kaggle WM-811K feature engineering baseline.

Production-safe adaptation of the public Kaggle notebook `loozin/wm-811k-wafermap`.
It extracts the notebook's 59-dimensional feature vector:
13 region-density + 20 Radon mean + 20 Radon std + 6 geometry features.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import interpolate
from skimage import measure
from skimage.transform import radon

REGION_FEATURE_NAMES = [
    "region_top_density", "region_right_density", "region_bottom_density", "region_left_density",
    "region_inner_r1_c1_density", "region_inner_r1_c2_density", "region_inner_r1_c3_density",
    "region_inner_r2_c1_density", "region_inner_r2_c2_density", "region_inner_r2_c3_density",
    "region_inner_r3_c1_density", "region_inner_r3_c2_density", "region_inner_r3_c3_density",
]
RADON_MEAN_FEATURE_NAMES = [f"radon_mean_{i:02d}" for i in range(20)]
RADON_STD_FEATURE_NAMES = [f"radon_std_{i:02d}" for i in range(20)]
GEOMETRY_FEATURE_NAMES = [
    "geom_area_norm", "geom_perimeter_norm", "geom_major_axis_norm", "geom_minor_axis_norm", "geom_eccentricity", "geom_solidity",
]
FEATURE_SCHEMA = REGION_FEATURE_NAMES + RADON_MEAN_FEATURE_NAMES + RADON_STD_FEATURE_NAMES + GEOMETRY_FEATURE_NAMES


@dataclass(frozen=True)
class FeatureGroups:
    region_density: list[float]
    radon_mean: list[float]
    radon_std: list[float]
    geometry: list[float]

    @property
    def vector(self) -> list[float]:
        return [*self.region_density, *self.radon_mean, *self.radon_std, *self.geometry]

    @property
    def named_vector(self) -> dict[str, float]:
        return dict(zip(FEATURE_SCHEMA, self.vector, strict=True))


def _ensure_2d_array(wafer_map: Any) -> np.ndarray:
    arr = np.asarray(wafer_map)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D wafer map, got shape {arr.shape}.")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"Wafer map is too small: shape={arr.shape}.")
    arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError("Wafer map contains NaN or infinite values.")
    return arr


def to_categorical_wafer(wafer_map: Any) -> np.ndarray:
    """Normalize common encodings to WM-811K categorical values {0,1,2}."""
    arr = _ensure_2d_array(wafer_map)
    if float(np.nanmax(arr)) > 2.0:
        arr = np.rint((arr / max(float(np.nanmax(arr)), 1.0)) * 2.0)
    return np.clip(np.rint(arr), 0, 2).astype(np.uint8)


def defect_only_map(wafer_map: Any) -> np.ndarray:
    arr = to_categorical_wafer(wafer_map)
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr >= 2] = 2
    return out


def _density(region: np.ndarray) -> float:
    return 0.0 if region.size == 0 else float(100.0 * np.sum(region >= 2) / region.size)


def _cuts(n: int) -> np.ndarray:
    out = np.rint(np.linspace(0, n, 6)).astype(int)
    out[0] = 0
    out[-1] = n
    return out


def extract_region_density(wafer_map: Any) -> np.ndarray:
    x = defect_only_map(wafer_map)
    rows, cols = x.shape
    r, c = _cuts(rows), _cuts(cols)
    regions = [
        x[r[0]:r[1], :], x[:, c[4]:c[5]], x[r[4]:r[5], :], x[:, c[0]:c[1]],
        x[r[1]:r[2], c[1]:c[2]], x[r[1]:r[2], c[2]:c[3]], x[r[1]:r[2], c[3]:c[4]],
        x[r[2]:r[3], c[1]:c[2]], x[r[2]:r[3], c[2]:c[3]], x[r[2]:r[3], c[3]:c[4]],
        x[r[3]:r[4], c[1]:c[2]], x[r[3]:r[4], c[2]:c[3]], x[r[3]:r[4], c[3]:c[4]],
    ]
    return np.asarray([_density(region) for region in regions], dtype=np.float32)


def _interp_20(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(20, dtype=np.float32)
    if values.size == 1:
        return np.repeat(values[0] / 100.0, 20).astype(np.float32)
    x = np.linspace(1, values.size, values.size)
    kind = "cubic" if values.size >= 4 else "linear"
    f = interpolate.interp1d(x, values, kind=kind, fill_value="extrapolate", bounds_error=False)
    return np.asarray(f(np.linspace(1, values.size, 20)) / 100.0, dtype=np.float32)


def extract_radon_features(wafer_map: Any) -> tuple[np.ndarray, np.ndarray]:
    x = defect_only_map(wafer_map).astype(np.float32)
    theta = np.linspace(0.0, 180.0, max(x.shape), endpoint=False)
    sinogram = radon(x, theta=theta, circle=False)
    return _interp_20(np.mean(sinogram, axis=1)), _interp_20(np.std(sinogram, axis=1))


def extract_geometry_features(wafer_map: Any) -> np.ndarray:
    binary = (defect_only_map(wafer_map) >= 2).astype(np.uint8)
    if not binary.any():
        return np.zeros(6, dtype=np.float32)
    labels = measure.label(binary, connectivity=1, background=0)
    props = measure.regionprops(labels)
    if not props:
        return np.zeros(6, dtype=np.float32)
    p = max(props, key=lambda item: item.area)
    norm_area = float(binary.shape[0] * binary.shape[1])
    norm_perimeter = float(np.sqrt(binary.shape[0] ** 2 + binary.shape[1] ** 2))
    major_axis = p.axis_major_length if hasattr(p, "axis_major_length") else p.major_axis_length
    minor_axis = p.axis_minor_length if hasattr(p, "axis_minor_length") else p.minor_axis_length
    return np.asarray([
        p.area / max(norm_area, 1.0), p.perimeter / max(norm_perimeter, 1.0),
        major_axis / max(norm_perimeter, 1.0), minor_axis / max(norm_perimeter, 1.0),
        p.eccentricity, p.solidity,
    ], dtype=np.float32)


def extract_feature_groups(wafer_map: Any) -> FeatureGroups:
    region = extract_region_density(wafer_map)
    radon_mean, radon_std = extract_radon_features(wafer_map)
    geom = extract_geometry_features(wafer_map)
    return FeatureGroups(
        region_density=[float(v) for v in region],
        radon_mean=[float(v) for v in radon_mean],
        radon_std=[float(v) for v in radon_std],
        geometry=[float(v) for v in geom],
    )


def extract_kaggle_feature_vector(wafer_map: Any) -> np.ndarray:
    vector = np.asarray(extract_feature_groups(wafer_map).vector, dtype=np.float32)
    if vector.shape != (59,):
        raise RuntimeError(f"Expected 59 Kaggle features, got shape={vector.shape}.")
    return vector
