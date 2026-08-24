"""3-D elastic multi-tile domain decomposition vs single domain — bit-exact.

Single GPU, manual halo copies.  Covers x-split (1x2) and the 2x2 rank
grid, each with free surface on/off.  Per-step exchange of ALL NINE
physical fields (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz) with halo width
W = 2*M, axis-by-axis with full extent on the other axes (ghost-of-ghost,
so the second pass overwrites the corner halos with two-hop values).

Why W = 2M and all fields: one elastic step is v += dt*D(s) then
s += dt*D(v) — two stencil applications deep — so the per-step
error-propagation depth from a stale halo is 2M; refreshing every
physical field 2M-wide once per step closes the recursion.  CPML memory
variables m_* are NEVER exchanged: cut-axis memories have zero
coefficients at the cut (per-side PML widths), and transverse memories
at cut-adjacent planes depend only on that plane's field history, which
the halo refresh keeps correct.  Every individual derivative in the
staggered elastic update is axis-aligned, so no diagonal-neighbour reads
exist and axis-by-axis exchange suffices; bit-equality against the
single-domain run is the proof.

Static model halo: tile pad cells on cut sides must hold the TRUE
neighbour model values (the velocity update at a halo cell reads rho at
that cell and feeds interior stress within the same step) — done by a
one-time overwrite from the reference run's padded models, mirroring a
production driver that partitions the globally padded model.

NOTE: W = 2M must fit in the symmetric pad abcn+M, i.e. abcn >= M
(here abcn=10, so=4 -> M=2, W=4).

Wavefield layout (B, C, nz, ny, nx): x is dim -1, y is dim -2.
elastic.h::bind 3-D order (36 slots, no rotation):
    vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
    m_vxx, m_vxy, m_vxz, m_vyx, m_vyy, m_vyz, m_vzx, m_vzy, m_vzz,
    m_sxxx, m_sxxy, m_sxxz, m_syyx, m_syyy, m_syyz,
    m_szzx, m_szzy, m_szzz, m_sxyx, m_sxyy, m_sxyz,
    m_sxzx, m_sxzy, m_sxzz, m_syzx, m_syzy, m_syzz
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

from sweep.equations import Elastic3D  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import SteppedBindingRunner  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NZ, NY, NX = 24, 20, 32
NT = 60
DT = 0.0015
SO = 4
M = SO // 2
ABCN = 10          # abcn >= M so the 2M halo fits inside the pad
PAD = ABCN + M
W = 2 * M
NWF = 36
NPHYS = 9          # vx, vy, vz, sxx, syy, szz, sxy, sxz, syz = slots 0..8
WAVELET_SCALE = 1.0e6


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def global_models():
    grid = np.linspace(0.0, 1.0, num=NZ * NY * NX, dtype=np.float32)
    grid = grid.reshape(NZ, NY, NX)
    vp = 2200.0 + 40.0 * grid
    vs = 1200.0 + 20.0 * grid
    rho = 2000.0 + 10.0 * grid
    vp[8:14, 6:14, 10:24] += 100.0   # anomaly straddling both cuts
    vs[8:14, 6:14, 10:24] += 50.0
    rho[8:14, 6:14, 10:24] += 25.0
    return [vp, vs, rho]


def make_prop(shape, free_surface, topo=None):
    equation = Elastic3D(spatial_order=SO, device=DEV, backend="torch")
    kwargs = dict(
        backend="torch",
        impl="c",
        shape=shape,
        dev=DEV,
        dh=10.0,
        dt=DT,
        source_type=["sxx", "syy", "szz"],
        receiver_type=["vx", "vy", "vz"],
        abcn=ABCN,
        free_surface=free_surface,
        pml_type="cpmls",
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


def make_runner(prop, wavelet, sources, receivers, model_arrays):
    cap = capture(prop)
    models = [torch.tensor(a, device=DEV) for a in model_arrays]
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=models)
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(NWF)]
    assert len(L) == NWF
    for t in L:
        t.zero_()
    record = torch.zeros_like(cap["raw_out"][2])
    p.record_out = record
    return SteppedBindingRunner(func, p, L, psi_pairs=(), u_blocks=()), record


def fix_tile_models(tile_p, full_p, y0_run, x0_run):
    """One-time static model-halo fill from the reference's padded models
    (tile runtime (iy, ix) maps to global runtime (y0_run+iy, x0_run+ix);
    z is never split)."""
    tms = list(tile_p.models)
    fms = list(full_p.models)
    for mt, mf in zip(tms, fms):
        src = mf[...,
                 y0_run:y0_run + mt.size(-2),
                 x0_run:x0_run + mt.size(-1)]
        assert src.shape == mt.shape
        mt.copy_(src)


def _exchange_axis(L_lo, L_hi, dim, n_phys, slots, lo_a, lo_b):
    """Copy M-wide halos of the given field slots across one cut along
    `dim` (full extent on the other axes, so ghost-of-ghost propagates for
    the next axis). Used per HALF step: v slots (0..2) between the velocity
    and stress phases, s slots (3..8) after the stress phase — owned stress
    cells then always read exchanged velocities, which keeps the transverse
    CPML memory divergence in halo cells from reaching owned cells.

    Cut-aware: with the asymmetric pad the two tiles' low-side offsets along
    `dim` differ (edge=abcn+M, cut=M), so ``lo_a``/``lo_b`` (=
    ``prop.padding[axis_lo]+M`` of the low/high tile) are passed explicitly."""
    hi_a = lo_a + n_phys

    def sl(t, a, b):
        idx = [slice(None)] * t.ndim
        idx[dim] = slice(a, b)
        return t[tuple(idx)]

    for f in slots:
        a, b = L_lo[f], L_hi[f]
        sl(a, hi_a, hi_a + M).copy_(sl(b, lo_b, lo_b + M))
        sl(b, lo_b - M, lo_b).copy_(sl(a, hi_a - M, hi_a))


@cuda_only
@pytest.mark.parametrize("py,px,free_surface", [
    (1, 2, False),
    (1, 2, True),
    (2, 2, False),
    (2, 2, True),
])
def test_tiles_3d_elastic_bitexact(py, px, free_surface):
    nyp, nxp = NY // py, NX // px
    models_np = global_models()
    wavelet = ricker(NT, DT, scale=WAVELET_SCALE)
    src_g = (NX // 2, NY // 2, NZ // 4)  # on the cut(s) for even splits
    rec_z = 2
    rec_g = [(gx, NY // 2, rec_z) for gx in range(2, NX - 2, 5)]

    # ---------------- reference ----------------
    full_src = np.array([[list(src_g)]], dtype=np.int32)
    full_rec = np.array([[list(r) for r in rec_g]], dtype=np.int32)
    prop_full = make_prop((NZ, NY, NX), free_surface)
    runner_full, record_full = make_runner(
        prop_full, wavelet, full_src, full_rec, models_np
    )
    with torch.no_grad():
        runner_full.run_to(NT)
    assert record_full.abs().max() > 0, "reference record all zero"

    # ---------------- tiles ----------------
    world = py * px
    tiles = {}
    rec_split = {}
    for rank in range(world):
        topo = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
        yi, xi = topo.yi, topo.xi
        y0, x0 = yi * nyp, xi * nxp
        tile_models = [a[:, y0:y0 + nyp, x0:x0 + nxp].copy() for a in models_np]

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

        prop = make_prop((NZ, nyp, nxp), free_surface, topo=topo)
        runner, record = make_runner(prop, t_wav, t_src, t_rec, tile_models)
        # cut-aware per-tile low offsets (cut face = M, edge face = abcn+M)
        pad3 = prop.padding   # (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        lo_y, lo_x = pad3[2] + M, pad3[0] + M
        # Static model-halo fill from the reference's RUNTIME-padded models: the
        # tile runtime origin (0,0) maps to full runtime (PAD+y0-lo_y, ..) — the
        # symmetric-pad call passed the physical (y0,x0), which is only correct
        # when lo==PAD (now corrected for the cut-aware asymmetric pad).
        fix_tile_models(runner.p, runner_full.p, PAD + y0 - lo_y, PAD + x0 - lo_x)
        # Cut-face mask (bit0=x_lo, bit1=x_hi, bit4=y_lo, bit5=y_hi): the
        # kernels move the cut-side zero-coefficient PML band onto the
        # interior branch so cut-adjacent owned cells match the un-split
        # run bitwise (the two branches contract FMAs differently).
        runner.p.cut_face_mask = (
            (1 if xi > 0 else 0) | (2 if xi < px - 1 else 0)
            | (16 if yi > 0 else 0) | (32 if yi < py - 1 else 0)
        )
        tiles[rank] = (runner, record, lo_y, lo_x)

    def rank_at(yi, xi):
        return yi * px + xi

    def _exchange_all_cuts(slots):
        # x exchanges first (full y extent incl. y halos), then y
        for yi in range(py):
            for xi in range(px - 1):
                ta = tiles[rank_at(yi, xi)]
                tb = tiles[rank_at(yi, xi + 1)]
                _exchange_axis(ta[0].L, tb[0].L, -1, nxp, slots, ta[3], tb[3])
        for xi in range(px):
            for yi in range(py - 1):
                ta = tiles[rank_at(yi, xi)]
                tb = tiles[rank_at(yi + 1, xi)]
                _exchange_axis(ta[0].L, tb[0].L, -2, nyp, slots, ta[2], tb[2])

    with torch.no_grad():
        for it in range(NT):
            for r, (runner, _, _, _) in tiles.items():
                runner.run_phase(it + 1, 1)
            _exchange_all_cuts(range(3))           # vx, vy, vz
            for r, (runner, _, _, _) in tiles.items():
                runner.run_phase(it + 1, 2)
            _exchange_all_cuts(range(3, NPHYS))    # 6 stress fields

    # ---------------- compare ----------------
    rec_tiles = torch.zeros_like(record_full)
    for rank, (_, record, _, _) in tiles.items():
        for j, gi in enumerate(rec_split[rank]):
            rec_tiles[:, :, gi] = record[:, :, j]
    assert torch.equal(rec_tiles, record_full), "record differs"

    # final state over each tile's physical region — every wavefield slot
    # (9 physical + 27 CPML memories) must be bitwise identical.
    for rank, (runner, _, lo_y, lo_x) in tiles.items():
        topo_yi, topo_xi = rank // px, rank % px
        for f in range(NWF):
            ref = runner_full.L[f][
                ...,
                PAD + topo_yi * nyp: PAD + topo_yi * nyp + nyp,
                PAD + topo_xi * nxp: PAD + topo_xi * nxp + nxp,
            ]
            got = runner.L[f][..., lo_y:lo_y + nyp, lo_x:lo_x + nxp]
            assert torch.equal(got, ref), (
                f"tile {rank} wavefield slot {f} differs over physical region"
            )
