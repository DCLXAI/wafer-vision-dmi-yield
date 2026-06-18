from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def synthetic_edge_ring(size: int = 52) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    radius = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[radius <= size * 0.44] = 1
    ring = (radius >= size * 0.35) & (radius <= size * 0.43)
    wafer[ring] = 2
    return wafer


def synthetic_scratch(size: int = 52) -> np.ndarray:
    wafer = np.ones((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    radius = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    wafer[radius > size * 0.45] = 0
    for offset in range(-1, 2):
        y = (0.55 * xx + 8 + offset).astype(int)
        valid = (y >= 0) & (y < size) & (radius <= size * 0.43)
        wafer[y[valid], xx[valid]] = 2
    return wafer


def save_all(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = {
        "synthetic_edge_ring": synthetic_edge_ring(),
        "synthetic_scratch": synthetic_scratch(),
    }
    for name, arr in examples.items():
        np.savetxt(output_dir / f"{name}.csv", arr, delimiter=",", fmt="%d")
        np.save(output_dir / f"{name}.npy", arr)
        Image.fromarray((arr * 120).astype(np.uint8), mode="L").save(output_dir / f"{name}.png")
        print(f"Saved {name} -> {output_dir}")


if __name__ == "__main__":
    save_all(Path("samples"))
