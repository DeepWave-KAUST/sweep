"""Downloads in sweep.datasets: mirrors tried in order, checksums enforced, and
the Overthrust 3-D sources (the SEG CD, cached or downloaded, else the HDF5
copy of the model).

Everything runs offline against a fake ``requests`` except the ``network``
tests, which download from Zenodo and run only with ``SWEEP_TEST_NETWORK=1``.
"""

import gzip
import hashlib
import io
import itertools
import os
import struct
import sys
import tarfile
import types

import numpy as np
import pytest

from sweep import datasets
from sweep.datasets import _benchmarks, _cache

network = pytest.mark.skipif(
    os.environ.get("SWEEP_TEST_NETWORK", "").lower() not in ("1", "true", "yes", "on"),
    reason="network test: set SWEEP_TEST_NETWORK=1",
)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ fakes
class _Response:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:  # requests.HTTPError is an OSError too
            raise OSError(f"{self.status_code} Client Error")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


class _Bar:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n):
        pass


@pytest.fixture
def fake_net(monkeypatch):
    """Serve ``routes[url] = (status, body)`` to ``_cache.download`` (any
    other URL is a 404) and record the URLs it fetches, in order."""
    routes, fetched = {}, []

    def get(url, stream=False, timeout=None):
        fetched.append(url)
        return _Response(*routes.get(url, (404, b"Not Found")))

    requests = types.ModuleType("requests")
    requests.get = get
    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = _Bar
    tqdm = types.ModuleType("tqdm")
    tqdm.auto = tqdm_auto
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm)
    monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto)
    return routes, fetched


GOOD = b"original model bytes " * 500


# ------------------------------------------------------- _cache.download
def test_download_falls_through_to_the_next_mirror(tmp_path, fake_net):
    routes, fetched = fake_net
    routes["https://a/m.bin"] = (403, b"<Error><Code>AccessDenied</Code></Error>")
    routes["https://b/m.bin"] = (200, GOOD)
    dest = tmp_path / "m.bin"
    assert _cache.download(["https://a/m.bin", "https://b/m.bin"], dest, sha256=_sha(GOOD)) == dest
    assert dest.read_bytes() == GOOD
    assert fetched == ["https://a/m.bin", "https://b/m.bin"]
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_a_200_that_is_not_the_file(tmp_path, fake_net):
    # e.g. a host answering 200 with a 189-byte JavaScript redirect page
    routes, _ = fake_net
    routes["https://a/m.bin"] = (200, b"<html><script>location.replace('/x')</script></html>")
    routes["https://b/m.bin"] = (200, GOOD)
    dest = tmp_path / "m.bin"
    _cache.download(["https://a/m.bin", "https://b/m.bin"], dest, sha256=_sha(GOOD))
    assert dest.read_bytes() == GOOD


def test_download_checks_md5_as_well(tmp_path, fake_net):
    routes, _ = fake_net
    routes["https://a/m.bin"] = (200, GOOD)
    dest = tmp_path / "m.bin"
    with pytest.raises(OSError, match="md5"):
        _cache.download("https://a/m.bin", dest, sha256=_sha(GOOD), md5="0" * 32)
    assert not dest.exists()
    _cache.download("https://a/m.bin", dest, sha256=_sha(GOOD), md5=hashlib.md5(GOOD).hexdigest())
    assert dest.read_bytes() == GOOD


def test_download_failure_names_every_source_and_where_to_put_the_file(tmp_path, fake_net):
    routes, _ = fake_net
    routes["https://a/m.bin"] = (403, b"denied")
    dest = tmp_path / "m.bin"
    with pytest.raises(OSError) as err:
        _cache.download(["https://a/m.bin", "https://b/m.bin"], dest, sha256="f" * 64)
    msg = str(err.value)
    assert "https://a/m.bin" in msg and "403" in msg
    assert "https://b/m.bin" in msg and "404" in msg
    assert str(dest) in msg and "f" * 64 in msg
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def test_a_verified_cache_is_not_fetched_again(tmp_path, fake_net):
    _, fetched = fake_net
    dest = tmp_path / "m.bin"
    dest.write_bytes(GOOD)
    assert _cache.download("https://a/m.bin", dest, sha256=_sha(GOOD)) == dest
    assert fetched == []


def test_a_cache_that_fails_its_checksum_is_fetched_again(tmp_path, fake_net):
    routes, fetched = fake_net
    routes["https://a/m.bin"] = (200, GOOD)
    dest = tmp_path / "m.bin"
    dest.write_bytes(GOOD[:100])  # an earlier download cut short
    with pytest.warns(UserWarning, match="does not match its checksum"):
        _cache.download("https://a/m.bin", dest, sha256=_sha(GOOD))
    assert dest.read_bytes() == GOOD and fetched == ["https://a/m.bin"]


def test_two_processes_fetching_the_same_file_both_get_it(tmp_path, fake_net, monkeypatch):
    # Every rank of a multi-GPU run calls load() at once. Simulate a second
    # process (another pid) that starts on the same file mid-transfer and
    # finishes first; with one shared temp file the first one then fails.
    pids = itertools.count(1001)
    monkeypatch.setattr(_cache.os, "getpid", lambda: next(pids))
    dest, other = tmp_path / "m.bin", []

    class Racing(_Response):
        def iter_content(self, chunk_size):
            chunks = super().iter_content(chunk_size)
            yield next(chunks)
            other.append(_cache.download("https://a/m.bin", dest, sha256=_sha(GOOD)))
            yield from chunks

    calls = []

    def get(url, stream=False, timeout=None):
        calls.append(url)
        return (Racing if len(calls) == 1 else _Response)(200, GOOD)

    monkeypatch.setattr(sys.modules["requests"], "get", get)
    assert _cache.download("https://a/m.bin", dest, sha256=_sha(GOOD), chunk_size=1024) == dest
    assert other == [dest] and len(calls) == 2
    assert dest.read_bytes() == GOOD and not list(tmp_path.glob("*.part"))


def test_cli_download_reports_a_failed_download_without_a_traceback(tmp_path, monkeypatch, fake_net, capsys):
    from sweep.datasets.cli import main

    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path))
    assert main(["download", "hess-vti"]) == 1  # the fake network answers 404 everywhere
    err = capsys.readouterr().err
    assert err.startswith("error: could not download timodel_vp.segy.gz from any source")
    assert "Traceback" not in err and str(tmp_path / "hessvti" / "timodel_vp.segy.gz") in err


class _Stop(Exception):
    pass


def test_every_download_entry_pins_a_sha256(tmp_path, monkeypatch):
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path))
    calls = {}

    def record(urls, dest, **kw):
        calls[dest.name] = (urls, kw)
        raise _Stop

    monkeypatch.setattr(_cache, "download", record)
    entries = [datasets.info(n, v) for n, v in datasets.available()]
    entries = [e for e in entries if e.loader.__module__ == _benchmarks.__name__]
    assert len(entries) == 8
    for e in entries:
        with pytest.raises(_Stop):
            e.loader()
    assert len(calls) == 7  # the two marmousi entries share one tarball
    for name, (urls, kw) in calls.items():
        assert urls and not isinstance(urls, str), name
        assert len(kw.get("sha256") or "") == 64, name
    # the five models the project re-hosts: its Zenodo record, then its Hugging Face copy, the bucket last
    for name in ("Overthrust_3D_CD1.tar.gz", "Salt_Model_3D.tar.gz", "elastic-marmousi-model.tar.gz",
                 "vel_z6.25m_x12.5m_exact.segy.gz", "ModelParams.tar.gz"):
        urls = calls[name][0]
        assert urls[0].startswith("https://zenodo.org/records/") and urls[0].endswith("/" + name), urls
        assert urls[1].startswith(_benchmarks._HF) and urls[1].endswith("/" + name), urls
        assert urls[-1].startswith(_benchmarks._S3), urls
    assert _benchmarks._OVERTHRUST_H5_MD5 == "ae4082cdcebbc7714eedd382d7925804"  # as Zenodo publishes it
    assert set(_benchmarks._HESS_SHA256) == set(_benchmarks._HESS_MEMBERS.values())
    # chevron: the Hugging Face copy only (its fallback is the original archive, see below)
    assert calls["SEG14.Vpsmoothstarting.segy"][0] == (_benchmarks._CHEVRON_URL + "SEG14.Vpsmoothstarting.segy",)
    assert all(len(s) == 64 for s in _benchmarks._CHEVRON_SHA256.values())
    assert all(len(s) == 64 for _, _, s in _benchmarks._CHEVRON_PARTS)


# ---------------------------------------------------------- overthrust 3d
@pytest.fixture
def small_overthrust(tmp_path, monkeypatch):
    """Shrink the grid to (z, y, x) = (3, 5, 4) -- y != x, so a swapped axis
    cannot pass -- and point the cache at a fresh directory. Returns the
    true vp and the overthrust cache dir."""
    shape = (3, 5, 4)
    monkeypatch.setattr(_benchmarks, "_OVERTHRUST_SHAPE", shape)
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path / "cache"))
    z, y, x = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    vp = (2000.0 + 500.0 * z + 37.0 * y + 3.0 * x).astype(np.float32)
    return vp, tmp_path / "cache" / "overthrust"


def _write_cd1(path, vp_zyx):
    """A CD1-style tarball: vp as raw big-endian float32, x fastest."""
    raw = vp_zyx.astype(">f4").tobytes()
    info = tarfile.TarInfo("Overthrust_3D_CD1/3D-Velocity-Grid/overthrust.vites")
    info.size = len(raw)
    with tarfile.open(path, "w:gz") as tf:
        tf.addfile(info, io.BytesIO(raw))


def _zenodo_h5_bytes(tmp_path, vp_zyx, water=20):
    """The Zenodo layout: squared slowness (s^2/km^2) on (z, x, y), water on top."""
    h5py = pytest.importorskip("h5py")
    v = vp_zyx.transpose(0, 2, 1).astype(np.float64) / 1000.0
    m = np.concatenate([np.full((water,) + v.shape[1:], 1.0 / 1.5 ** 2), 1.0 / v ** 2])
    path = tmp_path / "zenodo.h5"
    with h5py.File(path, "w") as f:
        f["m"] = m.astype(np.float32)
    return path.read_bytes()


def _serve(monkeypatch, fake_net, prefix, name, body, md5=False):
    """Serve ``body`` at a fake URL, made the only source of the loader's ``<prefix>`` file."""
    routes, fetched = fake_net
    url = f"https://zenodo.example/{name}"
    routes[url] = (200, body)
    monkeypatch.setattr(_benchmarks, f"{prefix}_URLS", (url,))
    monkeypatch.setattr(_benchmarks, f"{prefix}_SHA256", _sha(body))
    if md5:
        monkeypatch.setattr(_benchmarks, f"{prefix}_MD5", hashlib.md5(body).hexdigest())
    return url, fetched


def _cd1_bytes(tmp_path, vp_zyx):
    path = tmp_path / "cd1.tar.gz"
    _write_cd1(path, vp_zyx)
    return path.read_bytes()


def test_overthrust_reads_a_cached_cd1_without_network(small_overthrust, monkeypatch, fake_net):
    vp, d = small_overthrust
    d.mkdir(parents=True)
    cd1 = d / "Overthrust_3D_CD1.tar.gz"
    _write_cd1(cd1, vp)
    monkeypatch.setattr(_benchmarks, "_OVERTHRUST_CD1_SHA256", _sha(cd1.read_bytes()))
    out = datasets.load("overthrust", "3d-acoustic")
    assert fake_net[1] == []
    assert out["vp"].dtype == np.float32
    np.testing.assert_array_equal(out["vp"], vp)
    assert out["dh"] == (25.0, 25.0, 25.0)


def test_overthrust_downloads_the_cd1_when_none_is_cached(small_overthrust, tmp_path, monkeypatch, fake_net):
    vp, d = small_overthrust
    url, fetched = _serve(monkeypatch, fake_net, "_OVERTHRUST_CD1", "Overthrust_3D_CD1.tar.gz",
                          _cd1_bytes(tmp_path, vp))
    out = datasets.load("overthrust", "3d-acoustic")
    assert fetched == [url] and (d / "Overthrust_3D_CD1.tar.gz").exists()
    np.testing.assert_array_equal(out["vp"], vp)
    out = datasets.load("overthrust", "3d-acoustic", downsample=(1, 5, 2))
    assert out["vp"].shape == (3, 1, 2) and out["dh"] == (25.0, 125.0, 50.0)
    assert fetched == [url]  # second load: from the verified cache


def test_overthrust_falls_back_to_the_hdf5_copy(small_overthrust, tmp_path, monkeypatch, fake_net):
    vp, d = small_overthrust
    cd1_url = "https://cd.example/Overthrust_3D_CD1.tar.gz"  # not served: 404
    monkeypatch.setattr(_benchmarks, "_OVERTHRUST_CD1_URLS", (cd1_url,))
    url, fetched = _serve(monkeypatch, fake_net, "_OVERTHRUST_H5", "overthrust_3D_true_model.h5",
                          _zenodo_h5_bytes(tmp_path, vp), md5=True)
    with pytest.warns(UserWarning, match="HDF5 copy"):
        out = datasets.load("overthrust", "3d-acoustic")
    assert fetched == [cd1_url, url]
    assert out["vp"].dtype == np.float32 and out["vp"].shape == vp.shape
    np.testing.assert_allclose(out["vp"], vp, rtol=0, atol=1e-3)


def test_overthrust_refetches_a_cached_cd1_that_fails_its_checksum(
        small_overthrust, tmp_path, monkeypatch, fake_net):
    vp, d = small_overthrust
    d.mkdir(parents=True)
    (d / "Overthrust_3D_CD1.tar.gz").write_bytes(b"cut short")
    url, fetched = _serve(monkeypatch, fake_net, "_OVERTHRUST_CD1", "Overthrust_3D_CD1.tar.gz",
                          _cd1_bytes(tmp_path, vp))
    with pytest.warns(UserWarning, match="does not match its checksum"):
        out = datasets.load("overthrust", "3d-acoustic")
    assert fetched == [url]
    np.testing.assert_array_equal(out["vp"], vp)


def test_overthrust_h5_with_an_unexpected_grid_is_refused(small_overthrust, tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong.h5"
    with h5py.File(path, "w") as f:
        f["m"] = np.ones((23, 5, 4), dtype=np.float32)  # x and y swapped
    with pytest.raises(ValueError, match="expected"):
        _benchmarks._read_overthrust_h5(path)


def test_overthrust_h5_without_h5py_says_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match=r"sweep-solver\[datasets\]"):
        _benchmarks._read_overthrust_h5(tmp_path / "x.h5")


# ------------------------------------------------------- chevron fwi 2014
def _to_ibm(a) -> np.ndarray:
    """IEEE values (exactly representable in 24 bits) -> IBM float32 bit patterns."""
    a = np.asarray(a, dtype=np.float64).ravel()
    out = np.zeros(a.size, dtype=np.uint32)
    for k, v in enumerate(a):
        if v == 0:
            continue
        e4 = int(np.ceil(np.log2(abs(v)) / 4 + 1e-12))
        m = abs(v) / 16.0 ** e4
        if m >= 1:
            e4, m = e4 + 1, m / 16
        out[k] = (int(v < 0) << 31) | ((e4 + 64) << 24) | int(round(m * 2 ** 24))
    return out


def _segy(traces, headers, dt_us):
    """A SEG-Y rev-1 file: IBM float, ``traces`` (ntr, ns), ``headers`` a list of {byte offset: (fmt, value)}."""
    ntr, ns = traces.shape
    binhdr = bytearray(400)
    binhdr[16:18] = struct.pack(">h", dt_us)
    binhdr[20:22] = struct.pack(">h", ns)
    binhdr[24:26] = struct.pack(">h", 1)
    body = bytearray()
    for tr, hd in zip(traces, headers):
        h = bytearray(240)
        h[114:116] = struct.pack(">h", ns)
        h[116:118] = struct.pack(">h", dt_us)
        for off, (fmt, v) in hd.items():
            h[off:off + struct.calcsize(fmt)] = struct.pack(fmt, v)
        body += h + _to_ibm(tr).astype(">u4").tobytes()
    return b" " * 3200 + bytes(binhdr) + bytes(body)


@pytest.fixture
def small_chevron(monkeypatch, tmp_path):
    """A 3-shot, 4-receiver, 5-sample Chevron set with a 6x7 starting model; returns (members, truth)."""
    nshot, nrec, nt, (nz, nx) = 3, 4, 5, (6, 7)
    monkeypatch.setattr(_benchmarks, "_CHEVRON_NSHOT", nshot)
    monkeypatch.setattr(_benchmarks, "_CHEVRON_NREC", nrec)
    monkeypatch.setattr(_benchmarks, "_CHEVRON_NT", nt)
    monkeypatch.setattr(_benchmarks, "_CHEVRON_VP_SHAPE", (nz, nx))
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path / "cache"))
    s, r, k = np.meshgrid(np.arange(nshot), np.arange(nrec), np.arange(nt), indexing="ij")
    data = ((s * 64 + r * 8 + k) / 8.0 - 3.0).astype(np.float32)          # (shot, rec, t)
    hdrs = [{8: (">i", i + 1), 12: (">i", j + 1), 70: (">h", -10000),
             72: (">i", (1000 + 25 * i) * 10000), 80: (">i", (1000 + 25 * (i + j)) * 10000)}
            for i in range(nshot) for j in range(nrec)]
    seis = _segy(data.reshape(-1, nt), hdrs, 4000)
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    vp = (1500.0 + 100.0 * z + 4.0 * x).astype(np.float32)                 # (nz, nx)
    vp_segy = _segy(vp.T, [{}] * nx, 5000)
    members = {
        "SEG14.Pisoelastic.segy.gz": gzip.compress(seis),
        "SEG14.Vpsmoothstarting.segy": vp_segy,
        "SEG14.Vplog": b"Well Log: Depth in meters;  Vp in m/s;  X location = 39375 m\n"
                       b"Depth = 1000.0  Vp = 2414.7  \nDepth = 1005.0  Vp = 2242.4  \n",
        "Wavelet.txt": b"Source Wavelet. Sample Rate = 0.666666... milliseconds\n0.000000000\n0.900000000\n-1.5\n",
        "WellWavePic.JPG": b"\xff\xd8 picture",
        "README": b"readme",
        "DataLicenseAgreement": b"DATA LICENSE AGREEMENT",
    }
    monkeypatch.setattr(_benchmarks, "_CHEVRON_SHA256", {n: _sha(b) for n, b in members.items()})
    truth = {"data": data.transpose(0, 2, 1), "vp_init": vp, "wavelet": np.array([0, 0.9, -1.5], np.float32),
             "depth": np.array([1000.0, 1005.0]), "log_vp": np.array([2414.7, 2242.4], np.float32)}
    return members, truth


def _serve_chevron_hf(fake_net, members):
    routes, fetched = fake_net
    for n, b in members.items():
        routes[_benchmarks._CHEVRON_URL + n] = (200, b)
    return fetched


def _check_chevron(out, truth, shots):
    np.testing.assert_array_equal(out["data"], truth["data"][shots])
    assert out["data"].dtype == np.float32
    np.testing.assert_array_equal(out["vp_init"], truth["vp_init"])
    np.testing.assert_array_equal(out["wavelet"], truth["wavelet"])
    np.testing.assert_array_equal(out["well_log"]["depth"], truth["depth"])
    np.testing.assert_array_equal(out["well_log"]["vp"], truth["log_vp"])
    g = out["geometry"]
    np.testing.assert_array_equal(g["source_x"], 1000.0 + 25.0 * np.asarray(shots))
    np.testing.assert_array_equal(g["receiver_x"], g["source_x"][:, None] + 25.0 * np.arange(4))
    assert out["dh"] == (5.0, 12.5) and out["dt"] == 0.004 and out["nt"] == 5
    assert "vp" not in out


def test_chevron_loads_every_field_from_hugging_face(small_chevron, fake_net):
    members, truth = small_chevron
    fetched = _serve_chevron_hf(fake_net, members)
    out = datasets.load("chevron-fwi-2014")
    _check_chevron(out, truth, [0, 1, 2])
    assert all(u.startswith(_benchmarks._CHEVRON_URL) for u in fetched) and len(fetched) == 4
    out = datasets.load("chevron-fwi-2014", shots=[2, 0])
    _check_chevron(out, truth, [2, 0])
    assert len(fetched) == 4  # second load: from the verified cache


def test_chevron_fetches_the_gathers_only_when_asked(small_chevron, fake_net):
    members, truth = small_chevron
    fetched = _serve_chevron_hf(fake_net, members)
    out = datasets.load("chevron-fwi-2014", fields=("vp_init",), shots=slice(1, 3))
    np.testing.assert_array_equal(out["vp_init"], truth["vp_init"])
    assert "data" not in out and out["geometry"]["source_x"].tolist() == [1025.0, 1050.0]
    assert fetched == [_benchmarks._CHEVRON_URL + "SEG14.Vpsmoothstarting.segy"]
    with pytest.raises(ValueError, match="unknown fields"):
        datasets.load("chevron-fwi-2014", fields=("vp",))


def test_chevron_falls_back_to_the_original_archive(small_chevron, fake_net, monkeypatch, tmp_path):
    members, truth = small_chevron
    routes, fetched = fake_net
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for n, b in members.items():
            info = tarfile.TarInfo(n)
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    archive = buf.getvalue()
    cut = len(archive) // 3  # the bucket split the archive at an arbitrary byte
    parts = (("x.tar.gz.partaa", "20220726000001", archive[:cut]), ("x.tar.gz.partab", "20220726000002", archive[cut:]))
    monkeypatch.setattr(_benchmarks, "_CHEVRON_PARTS", tuple((n, ts, _sha(b)) for n, ts, b in parts))
    ia = [f"https://web.archive.org/web/{ts}id_/{_benchmarks._CHEVRON_BUCKET}{n}" for n, ts, _ in parts]
    routes[ia[0]] = (503, b"Service Unavailable")  # the Internet Archive is flaky: the bucket is next
    routes[_benchmarks._CHEVRON_BUCKET.replace("http://", "https://") + parts[0][0]] = (200, parts[0][2])
    routes[ia[1]] = (200, parts[1][2])
    with pytest.warns(UserWarning, match="original archive"):
        out = datasets.load("chevron-fwi-2014")
    _check_chevron(out, truth, [0, 1, 2])
    assert fetched[0] == _benchmarks._CHEVRON_URL + "SEG14.Vpsmoothstarting.segy" and ia[1] in fetched
    cache = tmp_path / "cache" / "chevron_fwi_2014"
    assert sorted(p.name for p in cache.iterdir()) == sorted(list(members) + ["SEG14.Pisoelastic.segy"])
    assert sum(u.startswith(_benchmarks._CHEVRON_URL) for u in fetched) == 1  # every member came out of the archive


def test_chevron_retries_a_part_the_internet_archive_did_not_serve(monkeypatch, tmp_path, fake_net):
    routes, fetched = fake_net
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path))
    monkeypatch.setattr(_benchmarks, "_CHEVRON_RETRY_WAIT", 0.0)
    body = b"part bytes" * 100
    url = f"https://web.archive.org/web/1id_/{_benchmarks._CHEVRON_BUCKET}p"
    get = sys.modules["requests"].get
    def flaky(u, stream=False, timeout=None):  # a read timeout on the first try only
        if u == url and fetched.count(u) == 0:
            fetched.append(u)
            raise OSError("Read timed out")
        return get(u, stream=stream, timeout=timeout)
    monkeypatch.setattr(sys.modules["requests"], "get", flaky)
    routes[url] = (200, body)
    assert _benchmarks._chevron_part("p", "1", _sha(body)).read_bytes() == body
    assert fetched.count(url) == 2
    monkeypatch.setattr(_benchmarks, "_CHEVRON_ATTEMPTS", 2)
    with pytest.raises(OSError, match="could not download q"):
        _benchmarks._chevron_part("q", "1", _sha(body))  # never served: gives up after two rounds


def test_chevron_refuses_gathers_whose_headers_disagree(small_chevron, fake_net, monkeypatch):
    members, _ = small_chevron
    seis = bytearray(gzip.decompress(members["SEG14.Pisoelastic.segy.gz"]))
    tb = 240 + 4 * 5
    off = 3600 + (1 * 4 + 2) * tb + 80  # gx of shot 1, receiver 2
    seis[off:off + 4] = struct.pack(">i", 999 * 10000)
    members["SEG14.Pisoelastic.segy.gz"] = gzip.compress(bytes(seis))
    monkeypatch.setitem(_benchmarks._CHEVRON_SHA256, "SEG14.Pisoelastic.segy.gz",
                        _sha(members["SEG14.Pisoelastic.segy.gz"]))
    _serve_chevron_hf(fake_net, members)
    datasets.load("chevron-fwi-2014", fields=("data",), shots=[0, 2])
    with pytest.raises(ValueError, match="shot 1"):
        datasets.load("chevron-fwi-2014", fields=("data",), shots=1)


# ---------------------------------------------------------------- network
# md5 of the original files, as the Zenodo records publish them
ZENODO_FILES = {
    "23469292": ("Overthrust_3D_CD1.tar.gz", "6592b1da46e5aa760d51436d766187c7"),
    "23469308": ("Salt_Model_3D.tar.gz", "bfa2d4a07400b80c928aa17e4524ae0a"),
    "23469300": ("elastic-marmousi-model.tar.gz", "6b1630b210c6d384b1e3b816f67d1a8e"),
    "23469290": ("vel_z6.25m_x12.5m_exact.segy.gz", "e2c4ad66a0553c9f108ffcf05506da82"),
    "23475636": ("ModelParams.tar.gz", "30ec0821ad7133caca1bb1eec4c645f3"),
    "4252588": ("overthrust_3D_true_model.h5", "ae4082cdcebbc7714eedd382d7925804"),
}


@network
@pytest.mark.parametrize("record", sorted(ZENODO_FILES))
def test_zenodo_records_serve_the_original_files(record):
    requests = pytest.importorskip("requests")
    name, md5 = ZENODO_FILES[record]
    r = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
    r.raise_for_status()
    f = {f["key"]: f for f in r.json()["files"]}[name]
    assert f["checksum"] == "md5:" + md5
    url = f"https://zenodo.org/records/{record}/files/{name}"
    lists = (_benchmarks._OVERTHRUST_CD1_URLS, _benchmarks._OVERTHRUST_H5_URLS, _benchmarks._SALT_URLS,
             _benchmarks._MARMOUSI2_URLS, _benchmarks._BP2004_URLS, _benchmarks._BP2007_URLS)
    assert any(url in urls for urls in lists)
    h = requests.head(url, allow_redirects=True, timeout=60)
    assert h.status_code == 200 and int(h.headers["Content-Length"]) == f["size"]


# the project's copies on Hugging Face: (folder, file, loader constant holding its sha256)
HF_FILES = [
    ("overthrust_3d", "Overthrust_3D_CD1.tar.gz", "_OVERTHRUST_CD1_SHA256"),
    ("seg_eage_salt_3d", "Salt_Model_3D.tar.gz", "_SALT_SHA256"),
    ("marmousi2_elastic", "elastic-marmousi-model.tar.gz", "_MARMOUSI2_SHA256"),
    ("bp2004_velocity", "vel_z6.25m_x12.5m_exact.segy.gz", "_BP2004_SHA256"),
    ("bp2007_tti", "ModelParams.tar.gz", "_BP2007_SHA256"),
]


@network
@pytest.mark.parametrize("folder, name, pin", HF_FILES)
def test_hugging_face_copy_serves_the_pinned_files(folder, name, pin):
    """Hugging Face reports the SHA-256 of a stored file (X-Linked-ETag) without sending it."""
    requests = pytest.importorskip("requests")
    url = _benchmarks._HF + f"{folder}/{name}"
    h = requests.head(url, allow_redirects=False, timeout=60)
    assert h.headers.get("X-Linked-ETag", "").strip('"') == getattr(_benchmarks, pin)
    assert requests.head(url, allow_redirects=True, timeout=60).status_code == 200


@network
def test_fresh_download_from_hugging_face_when_zenodo_is_unreachable(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path))
    monkeypatch.setattr(_benchmarks, "_BP2004_URLS", ("https://zenodo.org/records/0/files/missing",)
                        + _benchmarks._BP2004_URLS[1:])
    assert datasets.load("bp-2004")["vp"].shape == (1911, 5395)


@network
@pytest.mark.parametrize("name, variant, field, shape", [
    ("seg-eage-salt", "3d-acoustic", "vp", (210, 676, 676)),
    ("marmousi2", "2d-elastic", "rho", (2801, 13601)),
    ("bp-2004", "2d-acoustic", "vp", (1911, 5395)),
    ("bp-2007-tti", "2d-tti", "theta", (1801, 12596)),
])
def test_fresh_download_from_the_project_record(tmp_path, monkeypatch, name, variant, field, shape):
    """From an empty cache; the bucket answers 403, so the file comes from the Zenodo record."""
    pytest.importorskip("requests")
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path))
    assert datasets.load(name, variant)[field].shape == shape


@network
def test_overthrust_fresh_download_and_the_hdf5_fallback_agree(tmp_path, monkeypatch):
    """From an empty cache: the CD from the project's record (~175 MB); then, with no CD source
    reachable, the HDF5 copy (~530 MB). Both must give the same grid, cell for cell."""
    pytest.importorskip("h5py")
    pytest.importorskip("requests")
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path / "cd"))
    cd = datasets.load("overthrust", "3d-acoustic")["vp"]
    assert cd.shape == (187, 801, 801) and cd.dtype == np.float32
    assert abs(float(cd.min()) - 2178.834) < 1e-2 and float(cd.max()) == 6000.0
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path / "h5"))
    monkeypatch.setattr(_benchmarks, "_OVERTHRUST_CD1_URLS", ("https://zenodo.org/records/23469292/files/missing",))
    with pytest.warns(UserWarning, match="HDF5 copy"):
        h5 = datasets.load("overthrust", "3d-acoustic")["vp"]
    np.testing.assert_array_equal(h5, cd)


@network
@pytest.mark.parametrize("name", sorted(_benchmarks._CHEVRON_SHA256))
def test_hugging_face_serves_the_pinned_chevron_files(name):
    requests = pytest.importorskip("requests")
    url = _benchmarks._CHEVRON_URL + name
    h = requests.head(url, allow_redirects=False, timeout=60)
    linked = h.headers.get("X-Linked-ETag", "").strip('"')
    if len(linked) == 64:  # LFS file: Hugging Face reports its sha256
        assert linked == _benchmarks._CHEVRON_SHA256[name]
    else:  # small file stored in git (its ETag is the git blob id): fetch it
        assert _sha(requests.get(url, timeout=60).content) == _benchmarks._CHEVRON_SHA256[name]


@network
def test_chevron_fresh_download_without_the_gathers(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    monkeypatch.setenv("SWEEP_DATASETS_CACHE", str(tmp_path))
    out = datasets.load("chevron-fwi-2014", fields=("vp_init", "wavelet", "well_log"))
    assert out["vp_init"].shape == (1200, 3820) and out["dh"] == (5.0, 12.5)
    assert 1400 < float(out["vp_init"].min()) < 1600 < float(out["vp_init"].max())
    assert out["wavelet"].shape == (1025,)
    assert out["well_log"]["depth"][[0, -1]].tolist() == [1000.0, 2450.0]
