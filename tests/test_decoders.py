from __future__ import annotations

import json

import numpy as np
import pytest

from oracle_builder.data.decoders import decode_blob, encode_npy


def test_decode_basic_scalar_encodings():
    assert decode_blob(b"hello", "utf-8") == "hello"
    assert decode_blob(b'{"a": 1}', "json") == {"a": 1}
    assert decode_blob(b"3", "int") == 3
    assert decode_blob(b"1.5", "float") == 1.5


def test_decode_npy_with_shape():
    array = np.arange(4, dtype="float32").reshape(2, 2)
    decoded = decode_blob(encode_npy(array), "npy", json.dumps([4]))
    assert decoded.shape == (4,)
    assert decoded.dtype == np.float32


def test_zstd_stub_error():
    with pytest.raises(ValueError, match="Encoding 'zstd' is not available"):
        decode_blob(b"data", "zstd")

