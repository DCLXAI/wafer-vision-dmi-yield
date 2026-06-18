from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from wafer_vision.data import load_lswmd_dataframe, resize_wafer_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Save one sample wafer map per class.")
    parser.add_argument("--data-path", default="data/raw/LSWMD.pkl")
    parser.add_argument("--output", default="artifacts/reports/sample_wafer_maps.png")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--include-none", action="store_true", default=True)
    args = parser.parse_args()

    df, class_names = load_lswmd_dataframe(args.data_path, include_none=args.include_none)
    fig, axes = plt.subplots(1, len(class_names), figsize=(2 * len(class_names), 2.4))
    if len(class_names) == 1:
        axes = [axes]

    for ax, label in zip(axes, class_names):
        sample = df[df["label"] == label].iloc[0]
        image = resize_wafer_map(sample["wafer_map"], input_size=args.input_size)
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
