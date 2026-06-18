# WaferVision Week 1 Model Card

## Model
Compact PyTorch CNN for WM-811K wafer map defect classification.

## Input
Single wafer map represented as a 2D categorical array. WM-811K maps generally use:
- `0`: background
- `1`: passing die
- `2`: defective die

The baseline resizes maps to `64x64` with nearest-neighbor interpolation and scales values to `[0, 1]`.

## Output
Softmax classification over either:
- 9 classes: `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`, `None`
- 8 defect-only classes when `include_none: false`

## Metrics to report
Do not report accuracy alone. WM-811K is imbalanced, especially if `None` is included. Report:
- Accuracy
- Macro F1
- Per-class precision / recall / F1
- Confusion matrix

## Known limitations
- Resizing to a fixed square can distort very non-square wafer maps.
- Rotation/flip augmentation assumes orientation is not semantically important. Turn it off if your analysis needs directional defect semantics.
- The model is a portfolio-grade baseline, not a fab-qualified inspection system.
- Production use would require process-specific calibration, drift monitoring, unknown-class detection, and human review.

## Week 2 bridge
The saved checkpoint already contains `class_names`, `input_size`, and config metadata, so FastAPI can load it directly for `/predict` serving.
