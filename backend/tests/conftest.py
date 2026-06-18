from __future__ import annotations

import os

os.environ.setdefault("WAFERVISION_DATABASE_URL", "sqlite:///./data/runtime/test_wafervision.db")
os.environ.setdefault("WAFERVISION_JOB_BACKEND", "inline")
os.environ.setdefault("WAFERVISION_RATE_LIMIT_BACKEND", "memory")
os.environ.setdefault("WAFERVISION_RATE_LIMIT_ENABLED", "false")
