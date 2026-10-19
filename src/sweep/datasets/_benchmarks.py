"""Downloadable full-size benchmark velocity models.

Every entry here fetches its files on first use and caches them under
``sweep.datasets._cache.cache_root()``. Each file is pinned by the sha256 of
the original bytes (plus the md5 its host publishes, where there is one), so
any URL in a file's list, tried in order, must serve the original file byte
for byte. License / usage terms are attached to each
:class:`~sweep.datasets.registry.Entry` and printed on first load.

The SEG open-data S3 bucket (``open.source.geoscience``) that served all of
these files stopped serving anonymous downloads in 2026. The originals of
``overthrust``, ``seg-eage-salt``, ``marmousi2`` (and ``marmousi``),
``bp-2004`` and ``bp-2007-tti`` are re-hosted in the project's Zenodo records,
one per model, listed first, and in a Hugging Face dataset holding the same
files, listed second; the bucket stays last in every list in case it comes
back. ``overthrust`` falls back to the HDF5 copy of the model on Zenodo.
``hess-vti`` lists only the bucket, so it loads from an existing cache or from
the original files placed in the cache by hand.

Licensing summary (see CATALOG.md for the full audit):

* ``overthrust`` / ``seg-eage-salt``  — CC-BY-4.0 (SEG), redistributable with attribution.
* ``marmousi`` / ``marmousi2``        — CC-BY-4.0 (AGL, University of Houston), cite.
* ``bp-2004`` / ``bp-2007-tti`` / ``hess-vti`` — free download + attribution,
  "AS IS" terms; passed on only with their original documentation and terms,
  as the project's BP records do (``hess-vti`` is not re-hosted).

Not included:

* Sigsbee2A/2B — gated behind a signed Data Release Agreement, original host
  dead.
* SEAM Phase I — license ambiguous (CC-BY vs CC-BY-NC-SA), Cloudflare-gated.
* OpenFWI — CC BY-NC-SA 4.0 (non-commercial), Google-Drive hosted; incompatible
  with a permissive package and its download path could not be auto-verified.

Every loader here has been byte-verified against a real download (grid shape,
value range, and slice image confirmed).
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np

from . import _cache, _formats
from .registry import Entry, register

# The SEG open-data bucket (anonymous reads closed in 2026) and the project's
# copies of the original files (2026): Zenodo records, one per model, and a
# Hugging Face dataset with one folder per Zenodo record.
_S3 = "https://s3.amazonaws.com/open.source.geoscience/open_data/"
_ZENODO = "https://zenodo.org/records/"
_HF = "https://huggingface.co/datasets/shaowinw/seismic-benchmark-models/resolve/main/"


# ======================================================================
# Marmousi II (full acoustic vp) — CC-BY-4.0 (AGL), cite.
# vp is taken from the verified elastic-marmousi tarball (the standalone
# open.source.geoscience vp npy 403s), native 2801 (z) x 13601 (x) @ 1.25 m.
# Returns the native grid by default (like every other download entry); use
# downsample=N to coarsen, or the `2d-demo` variant for a small quick blob.
# ======================================================================
def _load_marmousi_full(*, downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_MARMOUSI2_URLS, _cache.cache_path("marmousi2", "elastic-marmousi-model.tar.gz"),
                          sha256=_MARMOUSI2_SHA256)
    workdir = _cache.cache_root() / "marmousi2"
    segy = _formats.extract_nested_tar_segy(tgz, _MARMOUSI2_MEMBERS["vp"], workdir)
    vp = _formats.read_model(segy, shape=_MARMOUSI2_SHAPE)  # (nz, nx) via transpose
    vp, fac = _formats.decimate(vp, downsample)
    nz, nx = vp.shape
    dh = (1.25 * fac[0], 1.25 * fac[1])
    dt, nt = 0.001, 4000
    sources = np.stack(
        [np.linspace(0, nx - 1, 8, dtype="int64"), np.zeros(8, dtype="int64")], axis=1
    )
    receivers = np.broadcast_to(
        np.stack([np.arange(nx, dtype="int64"), np.zeros(nx, dtype="int64")], axis=1),
        (sources.shape[0], nx, 2),
    ).copy()
    geometry = {"sources": sources, "receivers": receivers, "dt": dt, "nt": nt, "dh": dh}
    return {"vp": vp, "dh": dh, "dt": dt, "nt": nt, "geometry": geometry}


# ======================================================================
# SEG/EAGE Overthrust 3D — CC-BY-4.0. The official SEG/EAGE 3-D Modeling
# Series CD1 (re-hosted, doi:10.5281/zenodo.23469292) holds
# 3D-Velocity-Grid/overthrust.vites, a raw big-endian IEEE float32 grid; the
# ``.vites.h`` sidecar gives n1=801 n2=801 n3=187, d=25 m, esize=4, m/s (n1 is
# the fastest axis). If no source serves the CD, the HDF5 copy of the model is
# used (doi:10.5281/zenodo.4252588, CC-BY-4.0): dataset ``m`` is squared
# slowness in s^2/km^2 on axes (z, x, y), with 20 rows of 1500 m/s water on
# top that were added for its seismic data. 1000/sqrt(m) below the water, in
# float64 with x and y swapped and rounded to float32, equals the CD1 grid in
# every cell.
# ======================================================================
_OVERTHRUST_SHAPE = (187, 801, 801)  # (n3=z, n2=y, n1=x); n1 fastest -> C-order
_OVERTHRUST_CD1 = "Overthrust_3D_CD1.tar.gz"
_OVERTHRUST_CD1_URLS = (
    _ZENODO + "23469292/files/Overthrust_3D_CD1.tar.gz",
    _HF + "overthrust_3d/Overthrust_3D_CD1.tar.gz",
    _S3 + "seg_eage_models_cd/Overthrust_3D_CD1.tar.gz",
)
_OVERTHRUST_CD1_SHA256 = "7c10f9aa29beab25a40fc0cccede6631284bc2fa7e45a3511dbdceba8f5cf246"
_OVERTHRUST_H5 = "overthrust_3D_true_model.h5"
_OVERTHRUST_H5_URLS = ("https://zenodo.org/records/4252588/files/overthrust_3D_true_model.h5",)
_OVERTHRUST_H5_MD5 = "ae4082cdcebbc7714eedd382d7925804"  # as Zenodo publishes it
_OVERTHRUST_H5_SHA256 = "8957d1d012ea01a8c4fd72db23ea6cc6c72eeda383621d598f8d43a8225292b3"
_OVERTHRUST_WATER_ROWS = 20


def _read_overthrust_h5(path) -> np.ndarray:
    """vp in m/s, float32 ``(z, y, x)``, from the Zenodo HDF5 copy."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "No source served the Overthrust 3-D CD, and reading its HDF5 copy needs `h5py`. "
            "Install with `pip install sweep-solver[datasets]`."
        ) from e
    nz, ny, nx = _OVERTHRUST_SHAPE
    vp = np.empty(_OVERTHRUST_SHAPE, dtype=np.float32)
    with h5py.File(path, "r") as f:
        m = f["m"]
        if m.shape != (_OVERTHRUST_WATER_ROWS + nz, nx, ny):
            raise ValueError(
                f"{path}: dataset m has shape {m.shape}, expected "
                f"{(_OVERTHRUST_WATER_ROWS + nz, nx, ny)}"
            )
        for k in range(nz):  # a depth slice at a time, in float64
            vp[k] = (1000.0 / np.sqrt(m[_OVERTHRUST_WATER_ROWS + k].astype(np.float64))).T
    return vp


def _load_overthrust_3d(*, downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    workdir = _cache.cache_root() / "overthrust"
    try:
        cd1 = _cache.download(_OVERTHRUST_CD1_URLS, _cache.cache_path("overthrust", _OVERTHRUST_CD1),
                              sha256=_OVERTHRUST_CD1_SHA256)
    except OSError as e:  # no source serves the CD: the HDF5 copy holds the same grid
        warnings.warn(f"{e}\nUsing the HDF5 copy of the model instead.")
        h5 = _cache.download(_OVERTHRUST_H5_URLS, _cache.cache_path("overthrust", _OVERTHRUST_H5),
                             sha256=_OVERTHRUST_H5_SHA256, md5=_OVERTHRUST_H5_MD5)
        vp = _read_overthrust_h5(h5)
    else:
        vites = _formats.extract_tar_member(cd1, "3D-Velocity-Grid/overthrust.vites", workdir)
        vp = np.fromfile(vites, dtype=">f4").astype("float32").reshape(_OVERTHRUST_SHAPE)
    vp, fac = _formats.decimate(vp, downsample)
    dh = tuple(25.0 * f for f in fac)
    return {"vp": vp, "dh": dh, "dt": 0.002, "nt": 3000}


# ======================================================================
# SEG/EAGE Salt 3D — CC-BY-4.0. tar.gz -> VEL_GRIDS/SALTF.ZIP -> raw binary.
# SALTF is a raw big-endian float32 grid, x-fastest (like Overthrust's
# .vites): 676(x) x 676(y) x 210(z) @ 20 m -> read as C-order (nz, ny, nx).
# ======================================================================
_SALT_URLS = (
    _ZENODO + "23469308/files/Salt_Model_3D.tar.gz",
    _HF + "seg_eage_salt_3d/Salt_Model_3D.tar.gz",
    _S3 + "seg_eage_models_cd/Salt_Model_3D.tar.gz",
)
_SALT_SHA256 = "a6d9313d0de47a59c8903d93965c753e11794003cf109b10cafe01ce6ab6a7e5"
_SALT_SHAPE = (210, 676, 676)  # (nz, ny, nx); SALTF stored x-fastest


def _load_seg_eage_salt(*, dtype: str = ">f4", downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_SALT_URLS, _cache.cache_path("seg_eage_salt", "Salt_Model_3D.tar.gz"),
                          sha256=_SALT_SHA256)
    workdir = _cache.cache_root() / "seg_eage_salt"
    saltf_zip = _formats.extract_tar_member(tgz, "SALTF.ZIP", workdir)
    saltf = _formats.extract_zip_member(saltf_zip, "Saltf@@", workdir)
    vp = np.fromfile(saltf, dtype=np.dtype(dtype)).astype("float32")
    vp = vp.reshape(_SALT_SHAPE)  # (nz, ny, nx), x-fastest
    vp, fac = _formats.decimate(vp, downsample)
    dh = tuple(20.0 * f for f in fac)
    return {"vp": vp, "dh": dh, "dt": 0.002, "nt": 3000}


# ======================================================================
# Marmousi2 elastic (Martin et al. 2006) — public open data, cite.
# tar.gz of vp/vs/density SEG-Y. Native 2801 (z) x 13601 (x) @ 1.25 m.
# ======================================================================
_MARMOUSI2_URLS = (
    _ZENODO + "23469300/files/elastic-marmousi-model.tar.gz",
    _HF + "marmousi2_elastic/elastic-marmousi-model.tar.gz",
    _S3 + "elastic-marmousi/elastic-marmousi-model.tar.gz",
)
_MARMOUSI2_SHA256 = "b6fe8d10ce6a6dfadf203a8d0e1a040ce9f2c6cb52c1366193d35113d61797fe"
_MARMOUSI2_SHAPE = (2801, 13601)  # (nz, nx); SEG-Y is (nx, nz) traces, transposed on read
_MARMOUSI2_MEMBERS = {  # each is a nested .segy.tar.gz inside the outer tarball
    "vp": "MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz",
    "vs": "MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz",
    "rho": "MODEL_DENSITY_1.25m.segy.tar.gz",
}


def _load_marmousi2(*, fields=("vp", "vs", "rho"), downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_MARMOUSI2_URLS, _cache.cache_path("marmousi2", "elastic-marmousi-model.tar.gz"),
                          sha256=_MARMOUSI2_SHA256)
    workdir = _cache.cache_root() / "marmousi2"
    fac = _formats.normalize_downsample(downsample, 2)
    out: dict[str, Any] = {}
    for fld in fields:
        segy = _formats.extract_nested_tar_segy(tgz, _MARMOUSI2_MEMBERS[fld], workdir)
        arr = _formats.read_model(segy, shape=_MARMOUSI2_SHAPE)
        out[fld], _ = _formats.decimate(arr, fac)
    dh = (1.25 * fac[0], 1.25 * fac[1])
    out.update(dh=dh, dt=0.001, nt=5000)
    return out


# ======================================================================
# BP 2004 velocity benchmark — free download + attribution ("AS IS").
# vel_z6.25m_x12.5m_exact.segy.gz. Grid 1911 (z) x 5395 (x), dz=6.25 dx=12.5.
# ======================================================================
_BP2004_URLS = (
    _ZENODO + "23469290/files/vel_z6.25m_x12.5m_exact.segy.gz",
    _HF + "bp2004_velocity/vel_z6.25m_x12.5m_exact.segy.gz",
    _S3 + "bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz",
)
_BP2004_SHA256 = "01b3d65c6318491b1412465c904614a6c35bd73ea0ffc6a8d866917181749b9c"
_BP2004_SHAPE = (1911, 5395)  # (nz, nx)


def _load_bp2004(*, downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    gz = _cache.download(_BP2004_URLS, _cache.cache_path("bp2004", "vel_z6.25m_x12.5m_exact.segy.gz"),
                         sha256=_BP2004_SHA256)
    segy = _formats.gunzip(gz, _cache.cache_path("bp2004", "vel_z6.25m_x12.5m_exact.segy"))
    vp = _formats.read_model(segy, shape=_BP2004_SHAPE)
    vp, fac = _formats.decimate(vp, downsample)
    dh = (6.25 * fac[0], 12.5 * fac[1])
    return {"vp": vp, "dh": dh, "dt": 0.001, "nt": 6000}


# ======================================================================
# BP 2007 TTI anisotropic benchmark — free download, courtesy of BP.
# ModelParams.tar.gz of vp/epsilon/delta/theta SEG-Y. Grid 1801 x 12596 @ 6.25 m.
# ======================================================================
_BP2007_URLS = (
    _ZENODO + "23475636/files/ModelParams.tar.gz",
    _HF + "bp2007_tti/ModelParams.tar.gz",
    _S3 + "bptti2007/ModelParams.tar.gz",
)
_BP2007_SHA256 = "d023ef50634026afef3853e22395cc97b647339fa52826b21b23d1b2c090e80c"
_BP2007_SHAPE = (1801, 12596)  # (nz, nx); SEG-Y traces are (nx, nz), transposed on read
_BP2007_MEMBERS = {
    "vp": "Vp_Model.sgy",
    "epsilon": "Epsilon_Model.sgy",
    "delta": "Delta_Model.sgy",
    "theta": "Theta_Model.sgy",
}


def _load_bp2007_tti(*, fields=("vp", "epsilon", "delta", "theta"), downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_BP2007_URLS, _cache.cache_path("bp2007tti", "ModelParams.tar.gz"),
                          sha256=_BP2007_SHA256)
    workdir = _cache.cache_root() / "bp2007tti"
    fac = _formats.normalize_downsample(downsample, 2)
    out: dict[str, Any] = {}
    for fld in fields:
        segy = _formats.extract_tar_member(tgz, _BP2007_MEMBERS[fld], workdir)
        arr = _formats.read_model(segy, shape=_BP2007_SHAPE)
        out[fld], _ = _formats.decimate(arr, fac)
    dh = (6.25 * fac[0], 6.25 * fac[1])
    out.update(dh=dh, dt=0.001, nt=8000)
    return out


# ======================================================================
# Hess VTI — free download, Hess "AS IS" terms.
# timodel_{vp,epsilon,delta}.segy.gz. Decimated grid 1800 (x) x 750 (z) @ ~10 m.
# ======================================================================
_HESS_MEMBERS = {
    "vp": "timodel_vp.segy.gz",
    "epsilon": "timodel_epsilon.segy.gz",
    "delta": "timodel_delta.segy.gz",
}
_HESS_URLS = {m: (_S3 + "hessvti/" + m,) for m in _HESS_MEMBERS.values()}
_HESS_SHA256 = {
    "timodel_vp.segy.gz": "c28be0fd2ce06210a74bb99ce843a32102cb464f57566c837b391b85b0aaa1b8",
    "timodel_epsilon.segy.gz": "1524346e5dee9e3fb70a1a381b6ad2d78c66d136a940a776a14540760f994895",
    "timodel_delta.segy.gz": "c87eabf1936e43eec523eaaa0752906ed8c1497829146e6dd232706df6430f87",
}
_HESS_SHAPE = (1500, 3617)  # (nz, nx); real SEG-Y is 3617 traces x 1500 samples


_FT_TO_M = 0.3048


def _load_hess_vti(*, fields=("vp", "epsilon", "delta"), downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    fac = _formats.normalize_downsample(downsample, 2)
    out: dict[str, Any] = {}
    for fld in fields:
        member = _HESS_MEMBERS[fld]
        gz = _cache.download(_HESS_URLS[member], _cache.cache_path("hessvti", member),
                             sha256=_HESS_SHA256[member])
        segy = _formats.gunzip(gz, _cache.cache_path("hessvti", member[:-3]))
        arr = _formats.read_model(segy, shape=_HESS_SHAPE)
        if fld == "vp":
            arr = arr * _FT_TO_M  # Hess vp ships in ft/s -> convert to m/s
        out[fld], _ = _formats.decimate(arr, fac)
    # Grid spacing ships in feet; ~25 ft ≈ 7.62 m is the widely-used value.
    dh = (25.0 * _FT_TO_M * fac[0], 25.0 * _FT_TO_M * fac[1])
    out.update(dh=dh, dt=0.001, nt=6000, units="m/s")
    return out


# ======================================================================
# Registration
# ======================================================================
def _register_all() -> None:
    register(Entry(
        name="marmousi", variant="2d-acoustic", loader=_load_marmousi_full,
        description="Marmousi II 2-D acoustic vp (full size).",
        citation="Martin, Wiley, Marfurt (2006), DOI 10.1190/1.2172306; after Versteeg (1994), DOI 10.1190/1.1437051",
        license="CC-BY-4.0 (AGL/Univ. Houston) — cite the paper.",
        kind="download", redistributable=True, tags=["2d", "acoustic", "classic"],
    ))
    register(Entry(
        name="overthrust", variant="3d-acoustic", loader=_load_overthrust_3d,
        description="SEG/EAGE Overthrust 3-D acoustic vp (801x801x187 @ 25 m).",
        citation="Aminzadeh, Brac, Kunz (1997), SEG/EAGE 3-D Modeling Series No. 1",
        license="CC-BY-4.0",
        kind="download", redistributable=True, tags=["3d", "acoustic", "classic"],
    ))
    register(Entry(
        name="seg-eage-salt", variant="3d-acoustic", loader=_load_seg_eage_salt,
        description="SEG/EAGE Salt 3-D acoustic vp (SALTF grid, 676x676x210).",
        citation="Aminzadeh, Brac, Kunz (1997), SEG/EAGE 3-D Modeling Series No. 1",
        license="CC-BY-4.0",
        kind="download", redistributable=True, tags=["3d", "acoustic", "salt"],
    ))
    register(Entry(
        name="marmousi2", variant="2d-elastic", loader=_load_marmousi2,
        description="Marmousi2 elastic vp/vs/rho (AGL, 2801x13601 @ 1.25 m).",
        citation="Martin, Wiley, Marfurt (2006), DOI 10.1190/1.2172306",
        license="CC-BY-4.0 (AGL/Univ. Houston) — cite the paper.",
        kind="download", redistributable=True, tags=["2d", "elastic"],
    ))
    register(Entry(
        name="bp-2004", variant="2d-acoustic", loader=_load_bp2004,
        description="BP 2004 velocity benchmark vp (1911x5395, dz=6.25 dx=12.5).",
        citation="Billette & Brandsberg-Dahl (2005), 67th EAGE, B035",
        license="Free download + attribution; provided AS IS (acknowledge BP).",
        kind="download", redistributable=False, tags=["2d", "acoustic", "salt"],
    ))
    register(Entry(
        name="bp-2007-tti", variant="2d-tti", loader=_load_bp2007_tti,
        description="BP 2007 TTI benchmark vp/epsilon/delta/theta (1801x12596 @ 6.25 m).",
        citation="Hemang Shah, 2007 BP TTI benchmark (EAGE 2010 velocity workshop); courtesy of BP",
        license="Free download + attribution; courtesy of BP Exploration.",
        kind="download", redistributable=False, tags=["2d", "tti", "anisotropic"],
    ))
    register(Entry(
        name="hess-vti", variant="2d-vti", loader=_load_hess_vti,
        description="Hess VTI benchmark vp/epsilon/delta (1500x3617 @ ~7.62 m).",
        citation="Hess Corporation, SEG/Hess VTI benchmark (distributed via SEG)",
        license="Free download + attribution; Hess AS IS terms.",
        kind="download", redistributable=False, tags=["2d", "vti", "anisotropic"],
    ))


_register_all()
