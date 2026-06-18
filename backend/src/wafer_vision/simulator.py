from __future__ import annotations

"""High-performance synthetic wafer-map simulator.

This module powers the portfolio cockpit. It is deliberately honest: it is not a
fab-grade physics simulator, but it does model the *visual analytics* workflow a
yield/process engineer would use: lot metadata, spatial defect signatures,
risk scoring, excursion hints, and scalable wafer-map generation.

v0.6 upgrades over the first Spotfire-style version:
- cached NumPy grids instead of recomputing meshgrids for every wafer
- faster scratch/radial metrics suitable for hundreds of wafer maps per request
- downsampling helpers for UI payload control
- severity/root-cause helpers used by the API and React cockpit
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

try:  # scipy is already used by the Kaggle feature path; keep a tiny fallback for minimal installs.
    from scipy.ndimage import uniform_filter
except Exception:  # pragma: no cover
    uniform_filter = None  # type: ignore[assignment]

LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "None"]
PROCESS_STEPS = ["Lithography", "Etch", "Deposition", "CMP", "Clean", "Metrology"]
TOOLS = ["ETCH-01", "ETCH-02", "LITHO-03", "CMP-04", "CVD-05", "CLEAN-06"]
CHAMBERS = ["A", "B", "C", "D"]

DEFAULT_PATTERN_MIX: dict[str, float] = {
    "Center": 0.10,
    "Donut": 0.07,
    "Edge-Loc": 0.13,
    "Edge-Ring": 0.20,
    "Loc": 0.16,
    "Near-full": 0.04,
    "Random": 0.16,
    "Scratch": 0.10,
    "None": 0.04,
}

PATTERN_STEP_HINTS: dict[str, str] = {
    "Center": "Etch",
    "Donut": "Deposition",
    "Edge-Loc": "Clean",
    "Edge-Ring": "Etch",
    "Loc": "Lithography",
    "Near-full": "Metrology",
    "Random": "CMP",
    "Scratch": "Clean",
    "None": "Metrology",
}

ROOT_CAUSE_HINTS: dict[str, tuple[str, str]] = {
    "Center": ("Center zone process bias", "Check etch/deposition center uniformity, chuck contact, and center gas flow."),
    "Donut": ("Radial ring non-uniformity", "Review deposition/CVD radial profile, temperature ring, and recipe transitions."),
    "Edge-Loc": ("Localized edge excursion", "Inspect edge exclusion, bevel clean, clamp ring, and chamber edge effects."),
    "Edge-Ring": ("Edge ring excursion", "Check etch edge uniformity, focus ring wear, and edge gas flow calibration."),
    "Loc": ("Localized contamination or lithography issue", "Review reticle/track contamination, local particles, and lot handling records."),
    "Near-full": ("Global catastrophic excursion", "Hold lot, verify metrology, and check upstream process/tool recipe integrity."),
    "Random": ("Random particle signature", "Compare particle monitor, CMP/clean logs, and recent maintenance events."),
    "Scratch": ("Mechanical handling scratch", "Inspect robot arm, FOUP/slot, clean/CMP handling path, and wafer transfer logs."),
    "None": ("No dominant spatial pattern", "Continue monitoring; compare with control lot and baseline SPC limits."),
}

# Risk scoring is intentionally empirical. These weights are not claimed to be
# calibrated fab physics; they encode visual-inspection heuristics for portfolio
# simulation: global density dominates yield risk, scratch/radial signatures are
# strong excursion signals, and low model confidence slightly increases review
# priority. Keeping them named/configurable avoids unexplained magic constants.
DEFAULT_RISK_WEIGHTS: dict[str, float] = {
    "defect_density": 1.35,
    "zone_concentration": 0.25,
    "scratch_score": 0.58,
    "radial_non_uniformity": 0.55,
    "cluster_intensity": 0.12,
    "uncertainty": 0.12,
}

DEFAULT_SEVERITY_THRESHOLDS: dict[str, float] = {
    "monitor": 24.0,
    "warning": 45.0,
    "critical": 72.0,
}


def normalize_risk_weights(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    weights = dict(DEFAULT_RISK_WEIGHTS)
    if overrides:
        for key, value in overrides.items():
            if key in weights:
                weights[key] = float(np.clip(value, 0.0, 5.0))
    return weights


def normalize_severity_thresholds(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    thresholds = dict(DEFAULT_SEVERITY_THRESHOLDS)
    if overrides:
        for key, value in overrides.items():
            if key in thresholds:
                thresholds[key] = float(np.clip(value, 0.0, 100.0))
    # Preserve ordered bands even if user supplied inverted numbers.
    monitor = min(thresholds["monitor"], thresholds["warning"], thresholds["critical"])
    critical = max(thresholds["monitor"], thresholds["warning"], thresholds["critical"])
    warning = float(np.median([thresholds["monitor"], thresholds["warning"], thresholds["critical"]]))
    return {"monitor": monitor, "warning": warning, "critical": critical}


@dataclass(frozen=True)
class SpatialMetrics:
    defect_density: float
    yield_estimate: float
    edge_concentration: float
    center_concentration: float
    scratch_score: float
    radial_non_uniformity: float
    risk_score: float
    edge_center_delta: float = 0.0
    cluster_intensity: float = 0.0


@lru_cache(maxsize=32)
def _square_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = int(np.clip(size, 8, 256))
    radius = size * 0.465
    center = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    dx = xx - center
    dy = yy - center
    dist = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    wafer = dist <= radius
    norm_r = np.hypot(dx / max(size, 1), dy / max(size, 1))
    return yy, xx, dist, angle, wafer, norm_r


@lru_cache(maxsize=32)
def _metric_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    norm_r = np.hypot((xx - cx) / max(width, 1), (yy - cy) / max(height, 1))
    edge_zone = norm_r > 0.36
    center_zone = norm_r < 0.16
    return yy, xx, norm_r, edge_zone, center_zone


def normalize_pattern_mix(pattern_mix: Mapping[str, float] | None) -> tuple[list[str], np.ndarray]:
    source = dict(DEFAULT_PATTERN_MIX if not pattern_mix else pattern_mix)
    weights: list[float] = []
    labels: list[str] = []
    for label in LABELS:
        value = float(source.get(label, 0.0))
        if value > 0:
            labels.append(label)
            weights.append(value)
    if not weights:
        labels = list(DEFAULT_PATTERN_MIX)
        weights = list(DEFAULT_PATTERN_MIX.values())
    arr = np.asarray(weights, dtype=np.float64)
    arr = arr / arr.sum()
    return labels, arr


def generate_wafer_map(
    label: str,
    *,
    size: int = 64,
    noise_level: float = 0.04,
    defect_density_scale: float = 1.0,
    rng: np.random.Generator | None = None,
    secondary_label: str | None = None,
) -> np.ndarray:
    """Generate a WM-811K-style categorical wafer map with values 0/1/2.

    0 = outside wafer, 1 = good die, 2 = defective die.
    """
    rng = rng or np.random.default_rng()
    size = int(np.clip(size, 24, 160))
    label = label if label in LABELS else "Random"
    _, _, _, _, wafer, _ = _square_grid(size)
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[wafer] = 1

    mask = _pattern_mask(label, size, rng)
    if secondary_label and secondary_label in LABELS and secondary_label != "None":
        secondary = _pattern_mask(secondary_label, size, rng)
        mask = mask | (secondary & (rng.random((size, size)) > 0.38))

    mask = mask & wafer
    if label == "None":
        mask = np.zeros_like(mask, dtype=bool)

    if label not in {"Near-full", "None"}:
        keep = np.clip(0.72 * defect_density_scale, 0.15, 1.0)
        mask = mask & (rng.random((size, size)) < keep)
    elif label == "Near-full":
        keep = np.clip(0.90 * defect_density_scale, 0.45, 1.0)
        mask = mask & (rng.random((size, size)) < keep)

    noise_probability = float(np.clip(noise_level, 0.0, 0.45)) * np.clip(defect_density_scale, 0.25, 3.0)
    random_noise = wafer & (rng.random((size, size)) < noise_probability)
    if label == "None":
        random_noise = wafer & (rng.random((size, size)) < min(noise_probability * 0.22, 0.02))

    arr[mask | random_noise] = 2
    return arr


def _pattern_mask(label: str, size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx, dist, angle, _, _ = _square_grid(size)
    center = (size - 1) / 2.0
    dx = xx - center
    dy = yy - center

    if label == "Center":
        jitter_x, jitter_y = rng.normal(0, size * 0.018, 2)
        return np.hypot(dx - jitter_x, dy - jitter_y) < size * rng.uniform(0.095, 0.155)
    if label == "Donut":
        return (dist > size * rng.uniform(0.17, 0.22)) & (dist < size * rng.uniform(0.27, 0.33))
    if label == "Edge-Ring":
        return (dist > size * rng.uniform(0.36, 0.40)) & (dist < size * rng.uniform(0.445, 0.468))
    if label == "Edge-Loc":
        theta = rng.uniform(-np.pi, np.pi)
        angular = np.abs(np.angle(np.exp(1j * (angle - theta)))) < rng.uniform(0.34, 0.62)
        return angular & (dist > size * rng.uniform(0.33, 0.39)) & (dist < size * 0.468)
    if label == "Loc":
        cx = center + rng.uniform(-0.26, 0.26) * size
        cy = center + rng.uniform(-0.26, 0.26) * size
        blob = np.hypot(xx - cx, yy - cy) < size * rng.uniform(0.075, 0.135)
        if rng.random() < 0.35:
            cx2 = center + rng.uniform(-0.28, 0.28) * size
            cy2 = center + rng.uniform(-0.28, 0.28) * size
            blob |= np.hypot(xx - cx2, yy - cy2) < size * rng.uniform(0.045, 0.085)
        return blob
    if label == "Near-full":
        return dist < size * rng.uniform(0.39, 0.455)
    if label == "Scratch":
        slope = rng.uniform(-0.85, 0.85)
        intercept = rng.uniform(size * 0.18, size * 0.76)
        thickness = rng.uniform(1.1, 2.8)
        line = np.abs(yy - (slope * xx + intercept)) < thickness
        if rng.random() < 0.25:
            line |= np.abs(yy - ((slope + rng.normal(0, 0.09)) * xx + intercept + rng.normal(0, size * 0.08))) < thickness * 0.75
        return line
    if label == "Random":
        return np.zeros((size, size), dtype=bool)
    return np.zeros((size, size), dtype=bool)


def calculate_spatial_metrics(wafer_map: np.ndarray, confidence: float = 0.75, risk_weights: Mapping[str, float] | None = None) -> SpatialMetrics:
    arr = np.asarray(wafer_map)
    wafer = arr > 0
    bad = arr >= 2
    h, w = arr.shape
    _, _, norm_r, edge_zone_base, center_zone_base = _metric_grid(h, w)
    defect_density = float(bad.sum() / max(1, wafer.sum()))
    yield_estimate = float(np.clip(1.0 - defect_density * 0.86, 0.0, 1.0))

    edge_zone = wafer & edge_zone_base
    center_zone = wafer & center_zone_base
    edge_concentration = float(bad[edge_zone].mean()) if edge_zone.any() else 0.0
    center_concentration = float(bad[center_zone].mean()) if center_zone.any() else 0.0
    edge_center_delta = edge_concentration - center_concentration

    # Fast scratch proxy: max occupancy of rows/columns and diagonals. This is
    # less expensive than scanning many continuous lines and scales better for
    # hundreds of wafer maps in a single simulator request.
    denom = max(1, min(h, w))
    row_score = float(bad.sum(axis=1).max() / denom) if h else 0.0
    col_score = float(bad.sum(axis=0).max() / denom) if w else 0.0
    diag_score = 0.0
    offsets = range(-h + 1, w, max(1, min(h, w) // 18))
    for offset in offsets:
        d1 = np.diagonal(bad, offset=offset)
        d2 = np.diagonal(np.fliplr(bad), offset=offset)
        if d1.size:
            diag_score = max(diag_score, float(d1.sum() / denom))
        if d2.size:
            diag_score = max(diag_score, float(d2.sum() / denom))
    scratch_score = max(row_score, col_score, diag_score)

    rings: list[float] = []
    for lo, hi in [(0.0, 0.12), (0.12, 0.22), (0.22, 0.32), (0.32, 0.42), (0.42, 0.50)]:
        zone = wafer & (norm_r >= lo) & (norm_r < hi)
        rings.append(float(bad[zone].mean()) if zone.any() else 0.0)
    radial_non_uniformity = float(np.std(rings))

    cluster_intensity = _cluster_intensity(bad & wafer)
    weights = normalize_risk_weights(risk_weights)
    risk = 100.0 * (
        defect_density * weights["defect_density"]
        + max(edge_concentration, center_concentration) * weights["zone_concentration"]
        + scratch_score * weights["scratch_score"]
        + radial_non_uniformity * weights["radial_non_uniformity"]
        + cluster_intensity * weights["cluster_intensity"]
        + (1.0 - float(np.clip(confidence, 0.0, 1.0))) * weights["uncertainty"]
    )
    risk_score = float(np.clip(risk, 0.0, 100.0))
    return SpatialMetrics(
        defect_density=defect_density,
        yield_estimate=yield_estimate,
        edge_concentration=edge_concentration,
        center_concentration=center_concentration,
        scratch_score=scratch_score,
        radial_non_uniformity=radial_non_uniformity,
        risk_score=risk_score,
        edge_center_delta=float(edge_center_delta),
        cluster_intensity=float(cluster_intensity),
    )


def _cluster_intensity(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    values = mask.astype(np.float32)
    if uniform_filter is not None:
        local_density = uniform_filter(values, size=3, mode="constant", cval=0.0)
    else:  # pragma: no cover - fallback for minimal environments
        padded = np.pad(values, 1, mode="constant")
        local_density = np.zeros_like(values, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                local_density += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        local_density /= 9.0
    return float(np.clip(local_density[mask].mean(), 0.0, 1.0))


def choose_metadata(label: str, index: int, rng: np.random.Generator, tools: Sequence[str] | None = None, chambers: Sequence[str] | None = None, steps: Sequence[str] | None = None) -> tuple[str, str, str]:
    tools = list(tools or TOOLS)
    chambers = list(chambers or CHAMBERS)
    steps = list(steps or PROCESS_STEPS)
    hinted = PATTERN_STEP_HINTS.get(label)
    process = hinted if hinted in steps and rng.random() < 0.68 else str(rng.choice(steps))
    if label in {"Edge-Ring", "Edge-Loc"}:
        tool = str(rng.choice([x for x in tools if "ETCH" in x] or tools))
    elif label == "Scratch":
        tool = str(rng.choice([x for x in tools if "CLEAN" in x or "CMP" in x] or tools))
    else:
        tool = str(rng.choice(tools))
    chamber = str(rng.choice(chambers))
    return tool, chamber, process


def severity_band(risk_score: float, thresholds: Mapping[str, float] | None = None) -> str:
    bands = normalize_severity_thresholds(thresholds)
    if risk_score >= bands["critical"]:
        return "Critical"
    if risk_score >= bands["warning"]:
        return "Warning"
    if risk_score >= bands["monitor"]:
        return "Monitor"
    return "Normal"


def root_cause_hint(label: str, tool_id: str | None = None, chamber_id: str | None = None) -> tuple[str, str]:
    title, action = ROOT_CAUSE_HINTS.get(label, ROOT_CAUSE_HINTS["Random"])
    if tool_id and chamber_id:
        title = f"{title} @ {tool_id}-{chamber_id}"
    return title, action


def downsample_matrix(matrix: np.ndarray, target_size: int | None, method: str = "nearest") -> np.ndarray:
    """Downsample a categorical wafer matrix for API/UI payload control.

    `nearest` is fastest and preserves die classes for trellis overviews. `area`
    performs block-majority pooling, which gives cleaner high-resolution preview
    snapshots when the requested target size is much smaller than the source.
    """
    arr = np.asarray(matrix)
    if not target_size or target_size <= 0:
        return arr
    target_size = int(target_size)
    if arr.shape[0] <= target_size and arr.shape[1] <= target_size:
        return arr
    method = (method or "nearest").lower()
    if method == "area":
        y_edges = np.linspace(0, arr.shape[0], target_size + 1).astype(int)
        x_edges = np.linspace(0, arr.shape[1], target_size + 1).astype(int)
        out = np.zeros((target_size, target_size), dtype=arr.dtype)
        for y in range(target_size):
            y0, y1 = y_edges[y], max(y_edges[y + 1], y_edges[y] + 1)
            for x in range(target_size):
                x0, x1 = x_edges[x], max(x_edges[x + 1], x_edges[x] + 1)
                block = arr[y0:y1, x0:x1].reshape(-1)
                if block.size == 0:
                    continue
                values, counts = np.unique(block, return_counts=True)
                out[y, x] = values[int(np.argmax(counts))]
        return out
    y_idx = np.linspace(0, arr.shape[0] - 1, target_size).round().astype(int)
    x_idx = np.linspace(0, arr.shape[1] - 1, target_size).round().astype(int)
    return arr[np.ix_(y_idx, x_idx)]
