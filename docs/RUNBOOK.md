# WaferVision Integrated Runbook

## 1. Backend with synthetic DMI feature-bundle demo

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python scripts/create_demo_svm_bundle.py \
  --output artifacts/checkpoints/kaggle_svm_demo.joblib

WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/kaggle_svm_demo.joblib \
WAFERVISION_MODEL_KIND=kaggle_svm \
PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --host 0.0.0.0 --port 8000
```

## 2. Test feature endpoint

```bash
curl -X POST http://localhost:8000/api/v1/features/array \
  -H 'Content-Type: application/json' \
  -d '{"wafer_map": [[0,1,2],[1,2,1],[0,1,0]], "filename":"toy-wafer"}'
```

Expected:

```text
feature_dim = 59
groups = region_density, radon_mean, radon_std, geometry
```

## 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

For real backend:

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCKS=false
```

## 4. Real WM-811K benchmark training

```bash
cd backend
mkdir -p data/raw
# put LSWMD.pkl in data/raw/LSWMD.pkl

PYTHONPATH=src python -m wafer_vision.train_svm \
  --data-path data/raw/LSWMD.pkl \
  --output-dir artifacts/kaggle_svm_baseline
```

For faster demo training:

```bash
PYTHONPATH=src python -m wafer_vision.train_svm \
  --data-path data/raw/LSWMD.pkl \
  --output-dir artifacts/kaggle_svm_baseline_fast \
  --balanced-per-class 1000
```

## 5. Switch backend model

CNN:

```bash
WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/wafer_cnn_best.pt \
WAFERVISION_MODEL_KIND=cnn \
PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --port 8000
```

DMI feature model:

```bash
WAFERVISION_CHECKPOINT_PATH=artifacts/kaggle_svm_baseline/kaggle_svm_ovo.joblib \
WAFERVISION_MODEL_KIND=kaggle_svm \
PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --port 8000
```
