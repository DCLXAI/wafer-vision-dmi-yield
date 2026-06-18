# DMI Feature Pipeline Integration

This document explains how WaferVision turns wafer-map pattern data into a deployable DMI feature pipeline for yield analysis.

## Feature Ideas Converted Into Product Modules

| Analysis Idea | Purpose | Production Module |
|---|---|---|
| Failure-type mapping | Keep defect labels stable across training, API responses, and UI explanations | `DEFECT_CLASSES_8` / `ALL_CLASSES_9` label contracts |
| Variable wafer dimensions | Accept wafer maps with different shapes without manual preprocessing | API parser, image resizing path, and feature extractor |
| Region density | Describe where defects concentrate spatially | `extract_region_density()` |
| Radial projection | Capture ring, edge, and center-weighted defect signatures | `extract_radon_features()` |
| Connected geometry | Quantify clusters, scratches, and large-area excursions | `extract_geometry_features()` |
| DMI feature vector | Compress wafer-map shape into tabular process features | feature-vector compatibility function |
| Classical feature classifier | Support interpretable feature-bundle inference alongside image checkpoints | `python -m wafer_vision.train_svm` |

## Production Hardening

The integrated version focuses on deployability and stable API behavior:

1. **No mutation during preprocessing** — feature extraction copies arrays safely before value normalization.
2. **Any 2D wafer shape accepted** — upload parsing and feature extraction work on varying map dimensions.
3. **Stable class order** — label order remains consistent across training, metadata, predictions, and UI copy.
4. **Feature API** — `/api/v1/features` exposes the DMI vector for frontend explainability.
5. **Model registry behavior** — the same FastAPI service loads either image checkpoints or tabular feature bundles.

## Why Keep Both Image And Feature Paths?

The image path is useful for end-to-end visual defect recognition. The DMI feature path is useful for engineering review because it exposes how region density, radial behavior, and connected geometry contribute to the result. Keeping both lets the project say:

- I can translate exploratory wafer-map analysis into a maintainable service.
- I can compare deep image inference with interpretable process features.
- I can expose model behavior in a UI that process engineers can inspect.
