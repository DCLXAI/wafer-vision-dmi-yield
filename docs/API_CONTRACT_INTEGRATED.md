# Integrated API Contract

Base URL:

```text
http://localhost:8000/api/v1
```

## `GET /model`

Returns loaded model metadata. Works for both image checkpoints and DMI feature bundles.

```json
{
  "loaded": true,
  "model_version": "kaggle-59d-ovo-linear-svm-v1",
  "checkpoint_path": "artifacts/kaggle_svm_baseline/kaggle_svm_ovo.joblib",
  "input_size": null,
  "class_names": ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"],
  "device": null,
  "validation_macro_f1": 0.79,
  "model_kind": "kaggle_svm_ovo",
  "feature_dim": 59,
  "feature_schema": ["region_top_density", "..."],
  "load_error": null
}
```

## `POST /predict`

Multipart upload.

Supported formats:

```text
.csv, .npy, .png, .jpg, .jpeg, .bmp
```

Returns prediction, confidence, top-k, input metadata, model metadata, and stores the record in SQLite.

## `POST /predict/array`

JSON inline wafer map.

```json
{
  "wafer_map": [[0, 1, 2], [1, 2, 1], [0, 1, 0]],
  "filename": "toy-wafer",
  "note": "optional operator note"
}
```

## `POST /features`

Multipart upload. Returns the DMI wafer-map feature vector.

## `POST /features/array`

JSON inline wafer map. Returns grouped feature explainability.

```json
{
  "feature_dim": 59,
  "feature_schema": ["region_top_density", "..."],
  "groups": [
    {"name":"region_density", "values":[...], "labels":[...]},
    {"name":"radon_mean", "values":[...], "labels":[...]},
    {"name":"radon_std", "values":[...], "labels":[...]},
    {"name":"geometry", "values":[...], "labels":[...]}
  ],
  "vector": [...],
  "named_vector": {"region_top_density": 0.0},
  "input": {"filename":"toy-wafer", "rows":3, "cols":3}
}
```
