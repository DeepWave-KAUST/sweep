"""Two-tile elastic2d domain decomposition vs single domain — bit-exact.

Single GPU, single process: two impl='c' elastic propagators each own half
of the x-axis (MeshTopology px=2 supplies rank-local per-side PML widths),
advanced one step at a time through the stepped forward API with a manual
halo copy of ALL physical fields between steps.

Halo protocol (one exchange per step, AFTER the step), width W = 2*M:

    exchange ALL velocity AND stress fields (vx, vz, sxx, szz, sxz).

Rationale: one elastic step is v += dt*D(s) followed by s += dt*D(v) — two
stencil applications deep, so the per-step error-propagation depth from a
stale halo is 2M; refreshing all physical fields 2M-wide once per step
closes the recursion (the same induction as the acoustic M-wide u_now
case).  CPML memory variables m_* are NEVER exchanged: their update
``m_new = b*m_old + a*d(field)`` reads only same-cell m and along-axis
derivatives.  X-direction memories never cross an x-cut because the
per-side PML widths zero the coefficients there (both in the tile and at
the matching global-interior cells), and transverse(z)-direction m at the
cut-adjacent columns depends only on that column's field history, which
the per-step halo refresh keeps correct (m_s*z to depth 2M, m_v*z to depth
M — exactly the depths whose values feed the interior).

Static model halo: the velocity update at a halo cell of depth <= M feeds
interior stress through the SECOND stencil application of the same step,
and that update reads rho AT the halo cell — so the tile's pad columns on
the cut side must hold the TRUE neighbour model values, not the propagator's
edge-replication.  A production DD driver partitions the globally padded
model; the test reproduces that with a one-time overwrite of each tile's
runtime-padded models from the reference run's padded models (fields are
then exchanged per step as above).

NOTE: symmetric pad is abcn+M per side (model padding unchanged), so
W = 2M requires abcn >= M (here abcn=10, so=4 -> M=2, W=4).

Free surface: elastic FS is z-top handling inside the kernels (image
method); the cut is along x, so FS is orthogonal — per-side PML widths keep
z_low=0 on every rank (z is never split).  KNOWN PRE-EXISTING BUG (not
fixed here): elastic image-method FS inflates vx at the surface row z=0
~20x; it affects reference and DD identically, so bitwise comparisons
still pass.

Wavefield list layout (elastic.h::bind, 15 slots, no rotation):
    vx, vz, sxx, szz, sxz,
    m_vxx, m_vxz, m_vzx, m_vzz,
    m_sxxx, m_sxxz, m_szzx, m_szzz, m_sxzx, m_sxzz
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

from sweep.equations import Elastic  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import SteppedBindingRunner  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NZ, NX = 48, 56
NT = 120
DT = 0.0015
SO = 4
M = SO // 2
ABCN = 10          # abcn >= M so the 2M halo fits inside the pad
PAD = ABCN + M
W = 2 * M          # elastic halo width (two stencil applications per step)
NWF = 15
NPHYS = 5          # vx, vz, sxx, szz, sxz = wavefield slots 0..4
WAVELET_SCALE = 1.0e6


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def global_models():
    """Smoke recipe ramps (x-varying!) + box anomaly straddling the cut."""
    grid = np.linspace(0.0, 1.0, num=NZ * NX, dtype=np.float32).reshape(NZ, NX)
    vp = 2200.0 + 40.0 * grid
    vs = 1200.0 + 20.0 * grid
    rho = 2000.0 + 10.0 * grid
    vp[20:30, 20:36] += 100.0   # straddles the cut at x=28
    vs[20:30, 20:36] += 50.0
    rho[20:30, 20:36] += 25.0
    return [vp, vs, rho]


def make_prop(shape, free_surface, topo=None):
    equation = Elastic(spatial_order=SO, device=DEV, backend="torch")
    kwargs = dict(
        backend="torch",
        impl="c",
        shape=shape,
        dev=DEV,
        dh=10.0,
        dt=DT,
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vz"],
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
    """Run the public prop once to capture params, then return a zeroed
    SteppedBindingRunner (no rotation: elastic fields live at fixed slots)
    plus the bound record tensor."""
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


def fix_tile_models(tile_p, full_p, x0_run):
    """One-time static model-halo fill: overwrite the tile's runtime-padded
    models with the matching x-slice of the reference's padded models.

    Tile runtime column ix maps to global runtime column ``x0_run + ix``
    (z is never split).  This replaces edge-replication values in the
    cut-side pad with the TRUE neighbour model — what a production DD
    driver gets by partitioning the globally padded model."""
    tms = list(tile_p.models)
    fms = list(full_p.models)
    for mt, mf in zip(tms, fms):
        src = mf[..., x0_run:x0_run + mt.size(-1)]
        assert src.shape == mt.shape
        mt.copy_(src)


@cuda_only
@pytest.mark.parametrize("src_gx", [14, 28])  # interior of tile0 / on the cut
@pytest.mark.parametrize("free_surface", [False, True])
def test_two_tile_elastic_bitexact(src_gx, free_surface):
    nxp = NX // 2
    models_np = global_models()
    wavelet = ricker(NT, DT, scale=WAVELET_SCALE)
    src_z = 12

    rec_gx = list(range(2, NX - 2, 6))
    rec_z = 2

    # ---------------- reference: single domain ----------------
    full_sources = np.array([[[src_gx, src_z]]], dtype=np.int32)
    full_receivers = np.array([[[gx, rec_z] for gx in rec_gx]], dtype=np.int32)
    prop_full = make_prop((NZ, NX), free_surface)
    runner_full, record_full = make_runner(
        prop_full, wavelet, full_sources, full_receivers, models_np
    )
    with torch.no_grad():
        runner_full.run_to(NT)
    assert record_full.abs().max() > 0, "reference record all zero"

    # ---------------- two tiles ----------------
    runners, records, rec_split, los = [], [], [], []
    for xi in range(2):
        topo = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * nxp
        tile_models = [a[:, x0:x0 + nxp].copy() for a in models_np]

        owns_src = (x0 <= src_gx < x0 + nxp)
        if owns_src:
            tile_sources = np.array([[[src_gx - x0, src_z]]], dtype=np.int32)
            tile_wavelet = wavelet
        else:
            # zero-amplitude dummy source: keeps nsrc >= 1 without effect
            tile_sources = np.array([[[1, 1]]], dtype=np.int32)
            tile_wavelet = np.zeros_like(wavelet)

        own_rec = [gx for gx in rec_gx if x0 <= gx < x0 + nxp]
        rec_split.append([rec_gx.index(gx) for gx in own_rec])
        tile_receivers = np.array(
            [[[gx - x0, rec_z] for gx in own_rec]], dtype=np.int32
        )

        prop = make_prop((NZ, nxp), free_surface, topo=topo)
        runner, record = make_runner(
            prop, tile_wavelet, tile_sources, tile_receivers, tile_models
        )
        # cut faces so the forward in_pml split uses the cut-aware phys bounds
        # matching the asymmetric pad.
        cm = 0
        if topo.neighbour_rank("x", -1) is not None:
            cm |= 1
        if topo.neighbour_rank("x", +1) is not None:
            cm |= 2
        runner.p.cut_face_mask = cm
        lo_x = prop.padding[0] + M
        los.append(lo_x)
        # Static halo: true neighbour model values in the pad columns. The
        # tile's runtime column 0 maps to global runtime column PAD + x0 - lo_x.
        fix_tile_models(runner.p, runner_full.p, PAD + x0 - lo_x)
        runners.append(runner)
        records.append(record)

    r0, r1 = runners
    lo0, hi0 = los[0], los[0] + nxp
    lo1, hi1 = los[1], los[1] + nxp

    def _swap_fields(slots, width):
        for f in slots:
            a, b = r0.L[f], r1.L[f]
            a[..., hi0:hi0 + width] = b[..., lo1:lo1 + width]
            b[..., lo1 - width:lo1] = a[..., hi0 - width:hi0]

    # Half-step protocol: the per-step error path "stale transverse CPML
    # memory in the halo columns -> locally recomputed halo v -> owned s"
    # is cut by exchanging v BETWEEN the velocity and stress phases, so
    # owned stress columns always read exchanged (true) velocities. Each
    # exchange is the single-stencil width M.
    with torch.no_grad():
        for it in range(NT):
            r0.run_phase(it + 1, 1)
            r1.run_phase(it + 1, 1)
            _swap_fields(range(2), M)          # vx, vz
            r0.run_phase(it + 1, 2)
            r1.run_phase(it + 1, 2)
            _swap_fields(range(2, NPHYS), M)   # sxx, szz, sxz

    # ---------------- compare ----------------
    # records, mapped back to global receiver order
    rec_tiles = torch.zeros_like(record_full)
    for rec, idxs in zip(records, rec_split):
        for j, gi in enumerate(idxs):
            rec_tiles[:, :, gi] = rec[:, :, j]
    assert torch.equal(rec_tiles, record_full), "record differs from single domain"

    # final state over each tile's physical region — every wavefield slot
    # (5 physical + 10 CPML memories) must be bitwise identical.
    for xi, r in enumerate(runners):
        lo_x = los[xi]
        for f in range(NWF):
            ref = runner_full.L[f][..., PAD + xi * nxp: PAD + xi * nxp + nxp]
            got = r.L[f][..., lo_x:lo_x + nxp]
            assert torch.equal(got, ref), (
                f"tile {xi} wavefield slot {f} differs over physical region"
            )
