import numpy as np
import torch

from wafer_vision.data import normalize_failure_type, resize_wafer_map, wafer_map_to_tensor


def test_normalize_failure_type_nested_array():
    assert normalize_failure_type(np.array([["edge-loc"]])) == "Edge-Loc"
    assert normalize_failure_type(np.array([["none"]])) == "None"
    assert normalize_failure_type("Scratch") == "Scratch"


def test_resize_wafer_map_shape_and_values():
    wafer = np.zeros((10, 12), dtype=np.uint8)
    wafer[2:5, 4:7] = 2
    resized = resize_wafer_map(wafer, input_size=64)
    assert resized.shape == (64, 64)
    assert set(np.unique(resized)).issubset({0.0, 2.0})


def test_wafer_map_to_tensor_range():
    wafer = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    tensor = wafer_map_to_tensor(wafer, input_size=8)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 8, 8)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
