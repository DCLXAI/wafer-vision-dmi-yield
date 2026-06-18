from __future__ import annotations

import numpy as np

from wafer_vision.features import FEATURE_SCHEMA, extract_feature_groups, extract_kaggle_feature_vector


def test_kaggle_feature_vector_has_59_values():
    wafer = np.ones((32, 32), dtype=np.uint8)
    wafer[10:14, 10:14] = 2
    vector = extract_kaggle_feature_vector(wafer)
    assert vector.shape == (59,)
    assert len(FEATURE_SCHEMA) == 59
    assert np.isfinite(vector).all()


def test_feature_groups_match_notebook_dimensions():
    wafer = np.zeros((64, 64), dtype=np.uint8)
    wafer[16:48, 16:48] = 1
    wafer[30:35, 30:35] = 2
    groups = extract_feature_groups(wafer)
    assert len(groups.region_density) == 13
    assert len(groups.radon_mean) == 20
    assert len(groups.radon_std) == 20
    assert len(groups.geometry) == 6
    assert len(groups.vector) == 59
    assert set(groups.named_vector) == set(FEATURE_SCHEMA)
