# WaferVision DMI Yield Intelligence — Build Report

## Scope

This report reflects the current public GitHub publication build for WaferVision as a DMI yield intelligence portfolio project.

The publication pass includes:

- DMI-focused README positioning
- Class imbalance and model metric tables
- Research-journal style commit convention
- React app copy cleanup for Korean and English UI
- Open Graph, manifest, and package metadata cleanup
- Privacy-preserving simulator run ledger implementation
- GitHub public repository publication

## Commands Run

Frontend:

```bash
cd frontend
npm run build
npm test
```

Backend syntax check:

```bash
python3 -m py_compile \
  backend/src/wafer_vision_api/app.py \
  backend/src/wafer_vision_api/services/model_service.py \
  backend/src/wafer_vision_api/db_models.py \
  backend/src/wafer_vision_api/routes/simulator.py \
  backend/src/wafer_vision_api/settings.py
```

Backend targeted pytest attempted:

```bash
cd backend
python3 -m pytest tests/test_simulation_run_logging.py
```

## Result

| Check | Result | Note |
|---|---|---|
| `npm run build` | Passed | Vite production bundle completed |
| `npm test` | Passed | Vitest: 1 file, 2 tests |
| Backend `py_compile` | Passed | Edited API/model/logging modules compile |
| Backend targeted pytest | Blocked by local dependency | Current system Python does not have `skimage` installed |

## Backend Test Dependency Note

The backend test suite imports `wafer_vision.features`, which requires `scikit-image`.

Install backend dependencies before running the full suite:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

## GitHub Publication

Repository:

```text
https://github.com/DCLXAI/wafer-vision-dmi-yield
```

Main publication commits use Conventional Commit style, for example:

```text
feat: publish DMI yield intelligence platform (06-18)
docs: update validation notes for GitHub publication (06-18)
```
