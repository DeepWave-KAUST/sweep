"""Curvilinear-grid Elastic 2-D — Stage 3 tests.

Mirror of ``test_acoustic_curvilinear.py`` for the elastic case. The
chief Stage-3 win is *long-time stability* under irregular topography
without the staircase exponential blow-up that limits Stage-2 elastic.
"""

import numpy as np
import pytest
import torch

from sweep.equations import Elastic, ElasticCurvilinear
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


NZ, NX = 64, 96
DH = 10.0
DT = 8.0e-4
NT = 200
ABCN = 30
SO = 4
FREQ = 8.0
DELAY = 0.12

# Auto-select CUDA when available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _wavelet(nt=NT):
    t = np.arange(nt, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32)).to(DEVICE)


def _geometry(src_xz=None):
    sx, sz = (NX // 2, 3) if src_xz is None else src_xz
    sources = torch.from_numpy(np.array([[sx, sz]], dtype=np.int64)).to(DEVICE)
    rec_x = np.arange(4, NX - 4, 6, dtype=np.int64)
    rec_z = np.full_like(rec_x, 1)
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...]).to(DEVICE)
    return sources, receivers


def _make_prop(topography, *, free_surface=True, equation_cls=ElasticCurvilinear):
    eq = equation_cls(spatial_order=SO, device=DEVICE, backend="torch")
    return PropTorch(
        eq, shape=(NZ, NX),
        free_surface=free_surface,
        topography=topography,
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl="eager",
        eager_options={"use_compile": False},
    )


def _models():
    vp = torch.full((NZ, NX), 2000.0).to(DEVICE)
    vs = torch.full((NZ, NX), 1200.0).to(DEVICE)
    rho = torch.full((NZ, NX), 1800.0).to(DEVICE)
    return [vp, vs, rho]


def test_flat_topography_runs_and_is_stable_elastic():
    sources, receivers = _geometry()
    prop = _make_prop(topography=None)
    syn = prop(_wavelet(), sources, receivers, models=_models())
    assert torch.isfinite(syn).all(), "curvilinear elastic flat run produced NaN/Inf"
    assert syn.abs().max().item() > 0, "record is identically zero"


def test_hill_long_time_stable_elastic():
    """Curvilinear elastic remains bounded for a moderate-NT run on a
    gentle hill — staircase Stage-2 elastic blows up by ~100× over the
    same window. The MVP cell-centred-metric curvilinear has a slow
    surface mode that appears for NT ≳ 1200 (see CURVILINEAR_PLAN §7
    and the follow-up Hestholm-Ruud / APM task), so we stay below that
    here. Demo / production users get a longer stable window with the
    APM implementation (Cao & Chen 2018; spec'd separately)."""
    nt_long = 800
    x = np.arange(NX, dtype=np.float32)
    hill = (
        6.0 * np.exp(-((x - NX * 0.4) ** 2) / (2.0 * 12.0**2))
        + 4.0 * np.exp(-((x - NX * 0.7) ** 2) / (2.0 * 10.0**2))
    ).round().astype(np.int64)

    src_x = NX // 4
    sources = torch.from_numpy(np.array([[src_x, 3]], dtype=np.int64)).to(DEVICE)
    rec_x = np.arange(4, NX - 4, 6, dtype=np.int64)
    receivers = torch.from_numpy(
        np.stack([rec_x, np.full_like(rec_x, 1)], axis=-1)[None, ...]
    ).to(DEVICE)

    prop = _make_prop(topography=hill)
    snap_times = list(range(0, nt_long, max(1, nt_long // 10)))
    syn, snaps = prop(
        _wavelet(nt_long), sources, receivers, models=_models(),
        return_wavefield=True, snapshot_times=snap_times,
    )
    # |v| = sqrt(vx^2 + vz^2) per snap
    vx_snaps = snaps[:, 0, 0, 0].detach().cpu().numpy()
    vz_snaps = snaps[:, 1, 0, 0].detach().cpu().numpy()
    vmag = np.sqrt(vx_snaps**2 + vz_snaps**2)
    per_frame_max = vmag.max(axis=(-2, -1))
    n = len(per_frame_max)
    # Compare AFTER-source-ringup peak against late-time peak. With source
    # delay ~150 timesteps and snap stride nt_long/10, the source is fully
    # firing by snap index 2-3; before that the wave is still ramping up
    # and "early" comparison would be misleading. We instead require the
    # late peak not exceed 10× the MIDDLE peak (when the wave is fully
    # developed but instability hasn't had time to grow). Stage-2 staircase
    # elastic blows up by 100×+ over the same window; this catches that.
    mid_peak = per_frame_max[n // 3 : 2 * n // 3].max()
    late_peak = per_frame_max[2 * n // 3 :].max()
    assert late_peak < 10.0 * max(mid_peak, 1e-30), (
        f"curvilinear elastic hill grew exponentially: mid {mid_peak:.3e}, "
        f"late {late_peak:.3e}"
    )
    assert torch.isfinite(syn).all()


def test_surface_stresses_zero_under_hill_elastic():
    """σ_zz and σ_xz at η = 0 (computational row 0 after PML strip) must
    be exactly zero — the standard flat-free-surface ``zero_top_row``
    is doing its job."""
    x = np.arange(NX, dtype=np.float32)
    hill = (5.0 * np.exp(-((x - NX / 2) ** 2) / (2.0 * 12.0**2))).round().astype(np.int64)
    sources, receivers = _geometry(src_xz=(NX // 2, 3))
    prop = _make_prop(topography=hill)
    snap_idx = list(range(0, NT, NT // 5))
    syn, snaps = prop(
        _wavelet(), sources, receivers, models=_models(),
        return_wavefield=True, snapshot_times=snap_idx,
    )
    # Field indices: 0=vx, 1=vz, 2=sxx, 3=szz, 4=sxz
    szz_surface = snaps[:, 3, 0, 0, 0, :]
    sxz_surface = snaps[:, 4, 0, 0, 0, :]
    assert szz_surface.abs().max().item() == 0.0, (
        f"σ_zz not zero on surface row (curvilinear): "
        f"{szz_surface.abs().max().item()}"
    )
    assert sxz_surface.abs().max().item() == 0.0, (
        f"σ_xz not zero on surface row (curvilinear): "
        f"{sxz_surface.abs().max().item()}"
    )
    assert syn.abs().max().item() > 0


def test_impl_c_rejects_curvilinear_elastic():
    eq = ElasticCurvilinear(spatial_order=SO, device=DEVICE, backend="torch")
    with pytest.raises(NotImplementedError):
        eq._C()
