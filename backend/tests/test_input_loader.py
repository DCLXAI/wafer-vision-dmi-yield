from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from wafer_vision_api.input_loader import InputDecodeError, load_wafer_array_from_upload


def test_load_csv_upload():
    arr = load_wafer_array_from_upload(b"0,1,2\n2,1,0\n", "wafer.csv", "text/csv")
    assert arr.shape == (2, 3)
    assert arr.max() == 2


def test_load_npy_upload():
    source = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    buf = io.BytesIO()
    np.save(buf, source)
    arr = load_wafer_array_from_upload(buf.getvalue(), "wafer.npy", "application/octet-stream")
    assert arr.shape == (2, 2)


def test_load_png_upload():
    image = Image.fromarray(np.array([[0, 120], [240, 120]], dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    arr = load_wafer_array_from_upload(buf.getvalue(), "wafer.png", "image/png")
    assert arr.shape == (2, 2)


def test_reject_unsupported_upload():
    with pytest.raises(InputDecodeError):
        load_wafer_array_from_upload(b"hello", "wafer.txt", "text/plain")
