"""3-D multi-tile domain decomposition vs single domain — bit-exact (M5).

Single GPU, manual halo copies. Covers x-split (1x2), y-split (2x1) and
the 2x2 rank grid. The acoustic Laplacian has no mixed derivatives, so
corner (diagonal) halos are never read — axis-by-axis exchange suffices;
bit-equality against the single-domain run is the proof.

Wavefield layout (B, C, nz, ny, nx): x is dim -1, y is dim -2.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import Acoustic3D  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBindingRunner,
    acoustic_psi_pairs,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NZ, NY, NX = 24, 20, 24
NT = 100
DT = 0.0015
SO = 4
M = SO // 2
ABCN = 8
PAD = ABCN + M


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def vp_global():
    vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None, None]
    vp = np.broadcast_to(vp, (NZ, NY, NX)).copy()
    vp[8:14, 6:14, 8:18] += 180.0  # anomaly straddling both cuts
    return vp


def make_prop(shape, topo=None):
    equation = Acoustic3D(spatial_order=SO, device=DEV, backend="torch")
    kwargs = dict(
        backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0, dt=DT,
        source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
        free_surface=False, pml_type="cpmlr", nt=NT, B=1,
        use_ckpt=False, boundary_saving_config={"enabled": False},
    )
    if topo is not None:
        kwargs["model_parallel"] = topo
    return PropTorch(equation, **kwargs)


def capture(prop):
    cap = {}
    impl = prop._backend_impl
    orig = impl.forward_func

    def wrapper(params):
        out = orig(params)
        cap["params"] = params
        cap["raw_out"] = out
        return out

    impl.forward_func = wrapper
    cap["func"] = orig
    return cap


def make_runner(prop, wavelet, sources, receivers, vp_np):
    cap = capture(prop)
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=[torch.tensor(vp_np, device=DEV)])
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(12)]
    assert len(L) == 12
    for t in L:
        t.zero_()
    record = torch.zeros_like(cap["raw_out"][2])
    p.record_out = record
    return SteppedBindingRunner(func, p, L, acoustic_psi_pairs(3)), record


def _exchange_axis(u_lo_tile, u_hi_tile, dim, n_phys):
    """Copy M-wide u_now halos across one cut along `dim` (full extent on
    the other axes, so ghost-of-ghost propagates for the next axis)."""
    lo, hi = PAD, PAD + n_phys

    def sl(t, a, b):
        idx = [slice(None)] * t.ndim
        idx[dim] = slice(a, b)
        return t[tuple(idx)]

    sl(u_lo_tile, hi, hi + M).copy_(sl(u_hi_tile, lo, lo + M))
    sl(u_hi_tile, lo - M, lo).copy_(sl(u_lo_tile, hi - M, hi))


@cuda_only
@pytest.mark.parametrize("py,px", [(1, 2), (2, 1), (2, 2)])
def test_tiles_3d_bitexact(py, px):
    nyp, nxp = NY // py, NX // px
    vp = vp_global()
    wavelet = ricker(NT, DT)
    src_g = (NX // 2, NY // 2, NZ // 4)  # on the cut(s) for even splits
    rec_z = 2
    rec_g = [(gx, NY // 2, rec_z) for gx in range(2, NX - 2, 5)]

    # ---------------- reference ----------------
    full_src = np.array([[list(src_g)]], dtype=np.int32)
    full_rec = np.array([[list(r) for r in rec_g]], dtype=np.int32)
    prop_full = make_prop((NZ, NY, NX))
    runner_full, record_full = make_runner(prop_full, wavelet, full_src, full_rec, vp)
    with torch.no_grad():
        runner_full.run_to(NT)

    # ---------------- tiles ----------------
    world = py * px
    tiles = {}
    rec_split = {}
    for rank in range(world):
        topo = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
        yi, xi = topo.yi, topo.xi
        y0, x0 = yi * nyp, xi * nxp
        vp_t = vp[:, y0:y0 + nyp, x0:x0 + nxp].copy()

        sx, sy, sz = src_g
        owns_src = (x0 <= sx < x0 + nxp) and (y0 <= sy < y0 + nyp)
        if owns_src:
            t_src = np.array([[[sx - x0, sy - y0, sz]]], dtype=np.int32)
            t_wav = wavelet
        else:
            t_src = np.array([[[1, 1, 1]]], dtype=np.int32)
            t_wav = np.zeros_like(wavelet)

        own = [(i, r) for i, r in enumerate(rec_g)
               if x0 <= r[0] < x0 + nxp and y0 <= r[1] < y0 + nyp]
        rec_split[rank] = [i for i, _ in own]
        if own:
            t_rec = np.array(
                [[[r[0] - x0, r[1] - y0, r[2]] for _, r in own]], dtype=np.int32
            )
        else:
            # dummy receiver: keeps nrec >= 1; its samples are discarded
            t_rec = np.array([[[1, 1, 1]]], dtype=np.int32)

        prop = make_prop((NZ, nyp, nxp), topo=topo)
        tiles[rank] = make_runner(prop, t_wav, t_src, t_rec, vp_t)

    def rank_at(yi, xi):
        return yi * px + xi

    with torch.no_grad():
        for it in range(NT):
            for r, (runner, _) in tiles.items():
                runner.run_to(it + 1)
            # x exchanges first (full y extent incl. y halos), then y
            for yi in range(py):
                for xi in range(px - 1):
                    a = tiles[rank_at(yi, xi)][0].u_now
                    b = tiles[rank_at(yi, xi + 1)][0].u_now
                    _exchange_axis(a, b, -1, nxp)
            for xi in range(px):
                for yi in range(py - 1):
                    a = tiles[rank_at(yi, xi)][0].u_now
                    b = tiles[rank_at(yi + 1, xi)][0].u_now
                    _exchange_axis(a, b, -2, nyp)

    # ---------------- compare ----------------
    rec_tiles = torch.zeros_like(record_full)
    for rank, (_, record) in tiles.items():
        for j, gi in enumerate(rec_split[rank]):
            rec_tiles[:, gi] = record[:, j]
    assert torch.equal(rec_tiles, record_full), "record differs"

    u_full = runner_full.u_now
    for rank, (runner, _) in tiles.items():
        topo_yi, topo_xi = rank // px, rank % px
        ref = u_full[
            ...,
            PAD + topo_yi * nyp: PAD + topo_yi * nyp + nyp,
            PAD + topo_xi * nxp: PAD + topo_xi * nxp + nxp,
        ]
        got = runner.u_now[..., PAD:PAD + nyp, PAD:PAD + nxp]
        assert torch.equal(got, ref), f"tile {rank} final u_now differs"
