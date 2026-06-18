from __future__ import annotations

from wafer_vision_api.database import get_sessionmaker, init_db, reset_database_state_for_tests
from wafer_vision_api.jobs.store import create_job_record, make_job_id, status_from_record, update_job_record


def test_durable_job_record_roundtrip(tmp_path):
    reset_database_state_for_tests()
    init_db(f"sqlite:///{tmp_path / 'jobs.db'}")
    job_id = make_job_id()
    record = create_job_record(
        job_id=job_id,
        backend="inline",
        queue_name="local-thread",
        request_payload={"wafer_count": 1, "size": 32, "noise_level": 0.01, "defect_density_scale": 1.0, "mixed_pattern_rate": 0.0},
        external_job_id=job_id,
    )
    status = status_from_record(record)
    assert status is not None
    assert status.status == "queued"
    assert status.backend == "inline"

    updated = update_job_record(job_id, status="succeeded", progress=1.0, session_id="sim-test", finished=True)
    status2 = status_from_record(updated)
    assert status2 is not None
    assert status2.status == "succeeded"
    assert status2.session_id == "sim-test"
