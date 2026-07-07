"""Downloadable full-size benchmark velocity models.

Every entry here fetches from the *official* open-data source on first use
and caches under ``sweep.datasets._cache.cache_root()``. License / usage
terms are attached to each :class:`~sweep.datasets.registry.Entry` and printed
on first load.

Licensing summary (see the package README for the full audit):

* ``overthrust`` / ``seg-eage-salt``  — CC-BY-4.0 (freely redistributable).
* ``marmousi`` / ``marmousi2``        — public academic open data, cite.
* ``bp-2004`` / ``bp-2007-tti`` / ``hess-vti`` — free download + attribution,
  "AS IS" terms; fetched from the official source, never re-hosted.

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

from typing import Any, Sequence

import numpy as np

from . import _cache, _formats
from .registry import Entry, register


# ======================================================================
# Marmousi II (full acoustic vp) — public open data, cite.
# vp is taken from the verified elastic-marmousi tarball (the standalone
# open.source.geoscience vp npy 403s), native 2801 (z) x 13601 (x) @ 1.25 m.
# Returns the native grid by default (like every other download entry); use
# downsample=N to coarsen, or the `2d-demo` variant for a small quick blob.
# ======================================================================
def _load_marmousi_full(*, downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_MARMOUSI2_URL, _cache.cache_path("marmousi2", "elastic-marmousi-model.tar.gz"))
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
# SEG/EAGE Overthrust 3D — CC-BY-4.0. Official SEG/EAGE 3-D Modeling Series
# CD (open.source.geoscience). CD1 holds 3D-Velocity-Grid/overthrust.vites, a
# raw big-endian IEEE float32 grid; the ``.vites.h`` sidecar gives
# n1=801 n2=801 n3=187, d=25 m, esize=4, m/s (n1 is the fastest axis).
# ======================================================================
_OVERTHRUST_URL = (
    "https://s3.amazonaws.com/open.source.geoscience/open_data/"
    "seg_eage_models_cd/Overthrust_3D_CD1.tar.gz"
)
_OVERTHRUST_SHAPE = (187, 801, 801)  # (n3=z, n2=y, n1=x); n1 fastest -> C-order


def _load_overthrust_3d(*, downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_OVERTHRUST_URL, _cache.cache_path("overthrust", "Overthrust_3D_CD1.tar.gz"))
    workdir = _cache.cache_root() / "overthrust"
    vites = _formats.extract_tar_member(tgz, "3D-Velocity-Grid/overthrust.vites", workdir)
    vp = np.fromfile(vites, dtype=">f4").astype("float32").reshape(_OVERTHRUST_SHAPE)
    vp, fac = _formats.decimate(vp, downsample)
    dh = tuple(25.0 * f for f in fac)
    return {"vp": vp, "dh": dh, "dt": 0.002, "nt": 3000}


# ======================================================================
# SEG/EAGE Salt 3D — CC-BY-4.0. tar.gz -> VEL_GRIDS/SALTF.ZIP -> raw binary.
# SALTF is a raw big-endian float32 grid, x-fastest (like Overthrust's
# .vites): 676(x) x 676(y) x 210(z) @ 20 m -> read as C-order (nz, ny, nx).
# ======================================================================
_SALT_URL = (
    "https://s3.amazonaws.com/open.source.geoscience/open_data/"
    "seg_eage_models_cd/Salt_Model_3D.tar.gz"
)
_SALT_SHAPE = (210, 676, 676)  # (nz, ny, nx); SALTF stored x-fastest


def _load_seg_eage_salt(*, dtype: str = ">f4", downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_SALT_URL, _cache.cache_path("seg_eage_salt", "Salt_Model_3D.tar.gz"))
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
_MARMOUSI2_URL = (
    "https://s3.amazonaws.com/open.source.geoscience/open_data/"
    "elastic-marmousi/elastic-marmousi-model.tar.gz"
)
_MARMOUSI2_SHAPE = (2801, 13601)  # (nz, nx); SEG-Y is (nx, nz) traces, transposed on read
_MARMOUSI2_MEMBERS = {  # each is a nested .segy.tar.gz inside the outer tarball
    "vp": "MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz",
    "vs": "MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz",
    "rho": "MODEL_DENSITY_1.25m.segy.tar.gz",
}


def _load_marmousi2(*, fields=("vp", "vs", "rho"), downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_MARMOUSI2_URL, _cache.cache_path("marmousi2", "elastic-marmousi-model.tar.gz"))
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
_BP2004_URL = (
    "https://s3.amazonaws.com/open.source.geoscience/open_data/"
    "bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz"
)
_BP2004_SHAPE = (1911, 5395)  # (nz, nx)


def _load_bp2004(*, downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    gz = _cache.download(_BP2004_URL, _cache.cache_path("bp2004", "vel_z6.25m_x12.5m_exact.segy.gz"))
    segy = _formats.gunzip(gz, _cache.cache_path("bp2004", "vel_z6.25m_x12.5m_exact.segy"))
    vp = _formats.read_model(segy, shape=_BP2004_SHAPE)
    vp, fac = _formats.decimate(vp, downsample)
    dh = (6.25 * fac[0], 12.5 * fac[1])
    return {"vp": vp, "dh": dh, "dt": 0.001, "nt": 6000}


# ======================================================================
# BP 2007 TTI anisotropic benchmark — free download, courtesy of BP.
# ModelParams.tar.gz of vp/epsilon/delta/theta SEG-Y. Grid 1801 x 12596 @ 6.25 m.
# ======================================================================
_BP2007_URL = (
    "https://s3.amazonaws.com/open.source.geoscience/open_data/"
    "bptti2007/ModelParams.tar.gz"
)
_BP2007_SHAPE = (1801, 12596)  # (nz, nx); SEG-Y traces are (nx, nz), transposed on read
_BP2007_MEMBERS = {
    "vp": "Vp_Model.sgy",
    "epsilon": "Epsilon_Model.sgy",
    "delta": "Delta_Model.sgy",
    "theta": "Theta_Model.sgy",
}


def _load_bp2007_tti(*, fields=("vp", "epsilon", "delta", "theta"), downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    tgz = _cache.download(_BP2007_URL, _cache.cache_path("bp2007tti", "ModelParams.tar.gz"))
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
_HESS_BASE = "https://s3.amazonaws.com/open.source.geoscience/open_data/hessvti/"
_HESS_MEMBERS = {
    "vp": "timodel_vp.segy.gz",
    "epsilon": "timodel_epsilon.segy.gz",
    "delta": "timodel_delta.segy.gz",
}
_HESS_SHAPE = (1500, 3617)  # (nz, nx); real SEG-Y is 3617 traces x 1500 samples


_FT_TO_M = 0.3048


def _load_hess_vti(*, fields=("vp", "epsilon", "delta"), downsample: "int | Sequence[int]" = 1) -> dict[str, Any]:
    fac = _formats.normalize_downsample(downsample, 2)
    out: dict[str, Any] = {}
    for fld in fields:
        member = _HESS_MEMBERS[fld]
        gz = _cache.download(_HESS_BASE + member, _cache.cache_path("hessvti", member))
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
        citation="Versteeg (1994), DOI 10.1190/1.1437051",
        license="Public / open (SEG open data) — cite the paper.",
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
        license="Public academic open data (AGL/Univ. Houston) — cite the paper.",
        kind="download", redistributable=False, tags=["2d", "elastic"],
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
        citation="Shah (2007), 70th EAGE workshop; courtesy of BP",
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
