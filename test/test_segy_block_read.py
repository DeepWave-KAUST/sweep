"""A SEG-Y read must not peak at several times its payload.

`segy_to_array` copied the whole file's traces contiguous in one go (the memmap
rows are strided, so dropping the 240-byte trace headers needs a copy) and then
handed the result to `_ibm_to_ieee`, which holds five temporaries the size of
what it is given -- sign, exponent, mantissa, the `ldexp` result and the
negation. Both scaled with the file instead of with a bound. It reads in blocks
now; these tests pin the values and that the block loop is what produces them.
"""
import struct

import numpy as np
import pytest

from sweep.datasets import _formats
from sweep.datasets._formats import segy_to_array

# IBM 32-bit float bit patterns, verified against the decoder's own output.
IBM = {0.0: 0x00000000, 1.0: 0x41100000, -1.0: 0xC1100000,
       0.5: 0x40800000, 2.0: 0x41200000, -0.5: 0xC0800000}


def _write_segy(path, traces, fmt):
    """Minimal SEG-Y rev-1: text header, binary header, then trace records."""
    n_traces, n_samples = traces.shape
    binhdr = bytearray(400)
    binhdr[20:22] = struct.pack(">H", n_samples)
    binhdr[24:26] = struct.pack(">H", fmt)
    with open(path, "wb") as f:
        f.write(b" " * 3200)
        f.write(bytes(binhdr))
        for row in traces:
            f.write(b"\0" * 240)
            f.write(row.tobytes())
    return path


def test_ieee_values_round_trip(tmp_path):
    # .astype LAST: arithmetic on a big-endian array returns a native-endian
    # one, and SEG-Y samples are big-endian on disk.
    data = (np.arange(12).reshape(3, 4) * 0.25).astype(">f4")
    p = _write_segy(tmp_path / "ieee.segy", data, fmt=5)
    got = segy_to_array(p)
    assert got.shape == (3, 4) and got.dtype == np.float32
    assert np.array_equal(got, np.asarray(data, np.float32))


def test_ibm_values_round_trip(tmp_path):
    values = [0.0, 1.0, -1.0, 0.5, 2.0, -0.5]
    words = np.array([IBM[v] for v in values], dtype=">u4").reshape(2, 3)
    p = _write_segy(tmp_path / "ibm.segy", words, fmt=1)
    got = segy_to_array(p)
    assert np.array_equal(got, np.asarray(values, np.float32).reshape(2, 3))


@pytest.mark.parametrize("fmt,dtype", [(5, ">f4"), (2, ">i4"), (3, ">i2")])
def test_every_format_still_reads(tmp_path, fmt, dtype):
    data = (np.arange(20).reshape(4, 5)).astype(dtype)
    p = _write_segy(tmp_path / f"f{fmt}.segy", data, fmt=fmt)
    assert np.array_equal(segy_to_array(p), np.asarray(data, np.float32))


def test_the_block_loop_is_what_produces_the_answer(tmp_path, monkeypatch):
    """Force many blocks and check the result is identical to one block."""
    data = (np.arange(40 * 7).reshape(40, 7) * 0.5).astype(">f4")
    p = _write_segy(tmp_path / "many.segy", data, fmt=5)
    whole = segy_to_array(p)
    monkeypatch.setattr(_formats, "_SEGY_BLOCK_SAMPLES", 7)   # one trace per block
    blocked = segy_to_array(p)
    assert np.array_equal(whole, blocked)
    assert np.array_equal(whole, np.asarray(data, np.float32))


def test_a_block_smaller_than_one_trace_still_works(tmp_path, monkeypatch):
    """The step is clamped to at least one trace, so a tiny bound cannot stall."""
    data = (np.arange(6).reshape(2, 3)).astype(">f4")
    p = _write_segy(tmp_path / "tiny.segy", data, fmt=5)
    monkeypatch.setattr(_formats, "_SEGY_BLOCK_SAMPLES", 1)
    assert np.array_equal(segy_to_array(p), np.asarray(data, np.float32))
