from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# Kaggle notebook order: mapping_type={Center:0, Donut:1, Edge-Loc:2, Edge-Ring:3, Loc:4, Random:5, Scratch:6, Near-full:7, none:8}
DEFECT_CLASSES_8 = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]

ALL_CLASSES_9 = DEFECT_CLASSES_8 + ["None"]


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    class_names: list[str]
    class_counts: dict[str, int]


def _squeeze_to_str(value: Any) -> str:
    """WM-811K labels are often nested numpy arrays. Convert them safely to strings."""
    try:
        squeezed = np.squeeze(value)
        if isinstance(squeezed, np.ndarray) and squeezed.size == 0:
            return ""
        text = str(squeezed)
    except Exception:
        text = str(value)

    text = text.strip()
    # Handles strings such as "['Center']", "[['none']]", "array(['Loc'])" lightly.
    text = text.replace("array", "")
    for ch in ["[", "]", "(", ")", "'", '"']:
        text = text.replace(ch, "")
    text = text.replace("dtype=<U6", "").replace("dtype=object", "")
    text = text.strip().strip(",")
    return text


def normalize_failure_type(value: Any) -> str:
    raw = _squeeze_to_str(value)
    aliases = {
        "center": "Center",
        "donut": "Donut",
        "edge-loc": "Edge-Loc",
        "edge_loc": "Edge-Loc",
        "edgeloc": "Edge-Loc",
        "edge-ring": "Edge-Ring",
        "edge_ring": "Edge-Ring",
        "edgering": "Edge-Ring",
        "loc": "Loc",
        "local": "Loc",
        "near-full": "Near-full",
        "near_full": "Near-full",
        "nearfull": "Near-full",
        "random": "Random",
        "scratch": "Scratch",
        "none": "None",
        "normal": "None",
        "": "",
        "nan": "",
        "null": "",
        "[]": "",
    }
    return aliases.get(raw.lower(), raw)


def normalize_train_test_label(value: Any) -> str:
    raw = _squeeze_to_str(value).lower()
    if raw in {"training", "train"}:
        return "Training"
    if raw in {"test", "testing"}:
        return "Test"
    return ""


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise KeyError(f"Could not find any of these columns in dataframe: {list(candidates)}")


def load_lswmd_dataframe(data_path: str | Path, include_none: bool = True, max_samples: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load and clean Kaggle WM-811K LSWMD.pkl.

    Expected dataframe columns are usually:
    waferMap, dieSize, lotName, waferIndex, trianTestLabel, failureType.
    Some converted versions use trainTestLabel instead of the misspelled trianTestLabel.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download WM-811K from Kaggle and place LSWMD.pkl at this path."
        )

    df = pd.read_pickle(path)
    wafer_col = find_column(df, ["waferMap", "WaferMap", "WaferImage"])
    label_col = find_column(df, ["failureType", "FailureType"])

    train_test_col = None
    for candidate in ["trianTestLabel", "trainTestLabel", "TrainingTestLabel"]:
        if candidate.lower() in {c.lower() for c in df.columns}:
            train_test_col = find_column(df, [candidate])
            break

    clean = pd.DataFrame(
        {
            "wafer_map": df[wafer_col],
            "label": df[label_col].map(normalize_failure_type),
        }
    )
    if train_test_col is not None:
        clean["source_split"] = df[train_test_col].map(normalize_train_test_label)
    else:
        clean["source_split"] = ""

    class_names = ALL_CLASSES_9 if include_none else DEFECT_CLASSES_8
    clean = clean[clean["label"].isin(class_names)].copy()
    clean = clean[clean["wafer_map"].notna()].copy()

    if max_samples is not None and max_samples > 0 and len(clean) > max_samples:
        # Stratified sampling keeps minority classes visible during quick experiments.
        clean, _ = train_test_split(
            clean,
            train_size=max_samples,
            stratify=clean["label"],
            random_state=42,
        )
        clean = clean.reset_index(drop=True)

    return clean.reset_index(drop=True), class_names


def resize_wafer_map(wafer_map: Any, input_size: int = 64) -> np.ndarray:
    """Resize an arbitrary wafer map to a fixed square array using nearest-neighbor.

    WM-811K wafer maps are categorical arrays with values usually in {0, 1, 2}:
    0 = background, 1 = passing die, 2 = defective die.
    Nearest-neighbor keeps these classes from being blurred.
    """
    arr = np.asarray(wafer_map)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D wafer map, got shape {arr.shape}")

    arr = arr.astype(np.uint8)
    image = Image.fromarray(arr, mode="L")
    resampling = getattr(Image, "Resampling", Image).NEAREST
    image = image.resize((input_size, input_size), resample=resampling)
    return np.asarray(image, dtype=np.float32)


def wafer_map_to_tensor(wafer_map: Any, input_size: int = 64) -> torch.Tensor:
    arr = resize_wafer_map(wafer_map, input_size=input_size)
    # Dataset values are 0, 1, 2. Scale to [0, 1] while preserving ordering.
    if arr.max() > 2:
        # Allows quick prediction from grayscale exported images; training data normally skips this branch.
        arr = np.rint((arr / max(float(arr.max()), 1.0)) * 2.0)
    arr = np.clip(arr, 0, 2) / 2.0
    return torch.from_numpy(arr).unsqueeze(0).float()


class WaferMapDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_names: list[str],
        input_size: int = 64,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_names = class_names
        self.label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.input_size = input_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        x = wafer_map_to_tensor(row["wafer_map"], input_size=self.input_size)
        if self.augment:
            x = self._augment(x)
        y = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)
        return x, y

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        # Wafer defect categories are mostly rotation/reflection invariant.
        # Keep augmentation simple and categorical-safe.
        k = int(torch.randint(low=0, high=4, size=(1,)).item())
        if k:
            x = torch.rot90(x, k=k, dims=(-2, -1))
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-1,))
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-2,))
        return x.contiguous()


def make_splits(
    df: pd.DataFrame,
    class_names: list[str],
    val_size: float = 0.15,
    test_size: float = 0.20,
    seed: int = 42,
) -> SplitFrames:
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("val_size and test_size must be fractions and sum to less than 1.")

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val["label"],
        random_state=seed,
    )
    counts = df["label"].value_counts().reindex(class_names).fillna(0).astype(int).to_dict()
    return SplitFrames(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        class_names=class_names,
        class_counts=counts,
    )


def compute_class_weights(frame: pd.DataFrame, class_names: list[str]) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex(class_names).fillna(0).astype(float).values
    counts = np.maximum(counts, 1.0)
    # Inverse sqrt is less aggressive than pure inverse frequency and works better on heavy imbalance.
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(
    data_path: str | Path,
    input_size: int = 64,
    include_none: bool = True,
    batch_size: int = 256,
    num_workers: int = 2,
    val_size: float = 0.15,
    test_size: float = 0.20,
    seed: int = 42,
    max_samples: int | None = None,
    augment_train: bool = True,
) -> tuple[dict[str, DataLoader], SplitFrames]:
    df, class_names = load_lswmd_dataframe(data_path, include_none=include_none, max_samples=max_samples)
    splits = make_splits(df, class_names, val_size=val_size, test_size=test_size, seed=seed)

    datasets = {
        "train": WaferMapDataset(splits.train, class_names, input_size=input_size, augment=augment_train),
        "val": WaferMapDataset(splits.val, class_names, input_size=input_size, augment=False),
        "test": WaferMapDataset(splits.test, class_names, input_size=input_size, augment=False),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )
        for name, dataset in datasets.items()
    }
    return loaders, splits
