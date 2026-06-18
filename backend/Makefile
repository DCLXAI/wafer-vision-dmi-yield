.PHONY: install test train-smoke train evaluate samples demo-checkpoint api-dev rq-worker celery-worker temporal-worker api-smoke docker-build docker-run

install:
	python -m pip install -r requirements.txt

# Whole repo tests: preprocessing + FastAPI backend.
test:
	PYTHONPATH=src pytest -q

train-smoke:
	PYTHONPATH=src python -m wafer_vision.train --config configs/train.yaml --epochs 2 --max-samples 5000

train:
	PYTHONPATH=src python -m wafer_vision.train --config configs/train.yaml

evaluate:
	PYTHONPATH=src python -m wafer_vision.evaluate --config configs/train.yaml

samples:
	PYTHONPATH=src python scripts/create_sample_inputs.py

# This checkpoint is intentionally untrained. It is only for API wiring checks.
demo-checkpoint:
	PYTHONPATH=src python scripts/create_demo_checkpoint.py --output artifacts/checkpoints/wafer_cnn_demo.pt

api-dev:
	PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --host 0.0.0.0 --port 8000

api-smoke:
	PYTHONPATH=src WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/wafer_cnn_demo.pt python scripts/api_smoke_test.py

docker-build:
	docker build -t wafervision-api:v09 .

docker-run:
	docker run --rm -p 8000:8000 -e WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/wafer_cnn_best.pt wafervision-api:v09


rq-worker:
	PYTHONPATH=src python -m wafer_vision_api.workers.rq_worker

celery-worker:
	PYTHONPATH=src celery -A wafer_vision_api.jobs.celery_app.celery_app worker --loglevel=INFO --concurrency=1

temporal-worker:
	PYTHONPATH=src python -m wafer_vision_api.workers.temporal_worker
