from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from wafer_vision_api.app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the local FastAPI app without running uvicorn.")
    parser.add_argument("--sample", default="samples/synthetic_edge_ring.csv")
    args = parser.parse_args()

    sample = Path(args.sample)
    if not sample.exists():
        raise FileNotFoundError(f"Sample not found: {sample}. Run scripts/create_sample_inputs.py first.")

    with TestClient(app) as client:
        print("GET /api/v1/health")
        print(client.get("/api/v1/health").json())

        with open(sample, "rb") as f:
            response = client.post("/api/v1/predict", files={"file": (sample.name, f, "text/csv")})
        print("POST /api/v1/predict")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))

        print("GET /api/v1/stats/summary")
        print(json.dumps(client.get("/api/v1/stats/summary").json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
