"""Two-tile domain decomposition vs single domain — bit-exact (M2).

Single GPU, single process: two impl='c' propagators each own half of the
x-axis (MeshTopology px=2 supplies rank-local PML widths through
``PropBase.model_parallel``), advanced one step at a time through the
stepped forward API with a manual u_now halo copy between steps. This is
numerically identical to the NCCL path (same copy semantics) and is the
correctness harness for machines with one GPU.

Layout per tile (x axis, run grid): ``pad | physical nxp | pad`` with
``pad = abcn + M`` on BOTH sides (v1 keeps symmetric model padding; the
interior-facing pad has zero PML coefficients, so it behaves as plain
medium and is overwritten by the halo exchange / blocked beyond M).

Exchange after each completed step, on u_now:
    tile0.u[..., pad+nxp : pad+nxp+M] = tile1.u[..., pad : pad+M]
    tile1.u[..., pad-M  : pad       ] = tile0.u[..., pad+nxp-M : pad+nxp]
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

from sweep.equations import Acoustic  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBindingRunner,
    acoustic_psi_pairs,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NZ, NX = 48, 56
NT = 120
DT = 0.0015
SO = 4
M = SO // 2


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def vp_global():
    vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    vp = np.broadcast_to(vp, (NZ, NX)).copy()
    vp[20:30, 20:36] += 180.0  # box anomaly straddling the cut at x=28
    return vp


def make_prop(shape, abcn, free_surface, topo=None):
    equation = Acoustic(spatial_order=SO, device=DEV, backend="torch")
    kwargs = dict(
        backend="torch",
        impl="c",
        shape=shape,
        dev=DEV,
        dh=10.0,
        dt=DT,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=abcn,
        free_surface=free_surface,
        pml_type="cpmlr",
        nt=NT,
        B=1,
        use_ckpt=False,
        boundary_saving_config={"enabled": False},
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
    """Run the public prop once to capture params, then return a zeroed
    SteppedBindingRunner plus the bound record tensor."""
    cap = capture(prop)
    models = [torch.tensor(vp_np, device=DEV)]
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=models)
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(9)]
    assert len(L) == 9
    for t in L:
        t.zero_()
    record = torch.zeros_like(cap["raw_out"][2])
    p.record_out = record
    return SteppedBindingRunner(func, p, L, acoustic_psi_pairs(2)), record


@cuda_only
@pytest.mark.parametrize("abcn", [8, 0])
@pytest.mark.parametrize("src_gx", [14, 28])  # interior of tile0 / on the cut
@pytest.mark.parametrize("free_surface", [False, True])
def test_two_tile_bitexact(abcn, src_gx, free_surface):
    if free_surface and (abcn == 0 or src_gx == 28):
        pytest.skip("free-surface variant only run for the main config")

    pad = abcn + M
    nxp = NX // 2
    vp = vp_global()
    wavelet = ricker(NT, DT)
    src_z = 12

    rec_gx = list(range(2, NX - 2, 6))
    rec_z = 2

    # ---------------- reference: single domain ----------------
    full_sources = np.array([[[src_gx, src_z]]], dtype=np.int32)
    full_receivers = np.array([[[gx, rec_z] for gx in rec_gx]], dtype=np.int32)
    prop_full = make_prop((NZ, NX), abcn, free_surface)
    runner_full, record_full = make_runner(
        prop_full, wavelet, full_sources, full_receivers, vp
    )
    runner_full.run_to(NT)

    # ---------------- two tiles ----------------
    runners, records, rec_split, los = [], [], [], []
    for xi in range(2):
        topo = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * nxp
        vp_tile = vp[:, x0:x0 + nxp].copy()

        owns_src = (x0 <= src_gx < x0 + nxp)
        if owns_src:
            tile_sources = np.array([[[src_gx - x0, src_z]]], dtype=np.int32)
            tile_wavelet = wavelet
        else:
            # zero-amplitude corner source: keeps nsrc >= 1 without effect
            tile_sources = np.array([[[1, 1]]], dtype=np.int32)
            tile_wavelet = np.zeros_like(wavelet)

        own_rec = [gx for gx in rec_gx if x0 <= gx < x0 + nxp]
        rec_split.append([rec_gx.index(gx) for gx in own_rec])
        tile_receivers = np.array(
            [[[gx - x0, rec_z] for gx in own_rec]], dtype=np.int32
        )

        prop = make_prop((NZ, nxp), abcn, free_surface, topo=topo)
        # cut-aware low pad: edge side keeps abcn+M, cut side carries only M
        # (prop.padding is F.pad order: x_lo is index 0).
        los.append(prop.padding[0] + M)
        runner, record = make_runner(
            prop, tile_wavelet, tile_sources, tile_receivers, vp_tile
        )
        # Tell the forward which x-faces are cuts so its in_pml split uses the
        # cut-aware phys bounds that match the asymmetric pad. Without this the
        # kernel falls back to the symmetric abcn+M split and treats cut-side
        # interior cells as PML (algebraically zero, but FMA-reordered → drift).
        cm = 0
        if topo.neighbour_rank("x", -1) is not None:
            cm |= 1
        if topo.neighbour_rank("x", +1) is not None:
            cm |= 2
        runner.p.cut_face_mask = cm
        runners.append(runner)
        records.append(record)

    r0, r1 = runners
    lo0, lo1 = los
    hi0, hi1 = lo0 + nxp, lo1 + nxp
    for it in range(NT):
        r0.run_to(it + 1)
        r1.run_to(it + 1)
        u0, u1 = r0.u_now, r1.u_now
        # tile0 high halo <- tile1 low interior; tile1 low halo <- tile0 high interior
        u0[..., hi0:hi0 + M] = u1[..., lo1:lo1 + M]
        u1[..., lo1 - M:lo1] = u0[..., hi0 - M:hi0]

    # ---------------- compare ----------------
    # records, mapped back to global receiver order
    rec_tiles = torch.zeros_like(record_full)
    for rec, idxs in zip(records, rec_split):
        for j, gi in enumerate(idxs):
            rec_tiles[:, gi] = rec[:, j]
    assert torch.equal(rec_tiles, record_full), "record differs from single domain"

    # final u_now over each tile's physical region (per-tile interior offset)
    u_full = runner_full.u_now
    for xi, (r, lo_i) in enumerate(zip(runners, los)):
        ref = u_full[..., pad + xi * nxp: pad + xi * nxp + nxp]
        got = r.u_now[..., lo_i:lo_i + nxp]
        assert torch.equal(got, ref), f"tile {xi} final u_now differs"
