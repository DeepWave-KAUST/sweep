"""Two-tile domain-decomposed BACKWARD vs single domain — bit-exact (M4 B1).

Full pipeline on one GPU, single process:

  reference   single-domain forward (boundary saving, gpu-direct ring,
              requires_grad models) captured once via the public propagator,
              then a monolithic raw backward_bs replay with Python-bound
              grads_out/illum_out (the test_stepped_backward.py pattern)
              -> grad_vp, grad_wavelet, both illuminations.

  DD          two x-tiles (MeshTopology px=2 -> rank-local PML widths, cut
              PML coefficients zero), each replaying its DD forward through
              the stepped API with the per-step u_now halo exchange of
              test_dd_two_tile.py — WITH boundary saving enabled, so each
              tile's boundary_gpu ring + u_last_two hold DD-consistent
              values — followed by the Design-B DD backward: per global it
              from nt-1 down to 0 each tile runs one backward_bs segment
              (it+1, it) with cut_face_mask set (tile0: x_hi bit, tile1:
              x_lo bit), then the tiles exchange the adjoint lambda u_now
              halo and the reconstruction u_now halo (recon skipped after
              the it==0 adjoint-only tail).

Per-tile grads are assembled by cropping each tile's OWNED x-columns and
compared bitwise (torch.equal) against the single-domain reference.

Residual choice: instead of deriving the adjoint source from a loss, BOTH
pipelines are fed the SAME fixed synthetic adjoint source — the reference
public run's raw record (per-receiver traces, split by receiver ownership
for the tiles).  This removes any dependence on loss bookkeeping while
keeping a non-trivial per-receiver residual.

Notes mapping to the M4 spec, section 4.2:

* item 4 (one-time post-seed recon exchange): the seeding runs INSIDE the
  first backward segment, so a separate post-seed exchange cannot be
  inserted before the first reverse step.  It is also unnecessary here:
  the DD forward driver exchanges u_now after EVERY step including the
  last, so u_last_two[...,0] (the seed's recon u_now = u(nt-1)) already
  carries a current halo.  u_last_two[...,1] (recon u_prev = u(nt)) has a
  stale M-halo, but u_prev is only ever read at the cell centre; the only
  cells that consume the stale values are the cut-halo cells themselves,
  which are overwritten by the per-step exchange before anything owned
  reads them — owned cells are unaffected (verified by the bitwise bar).
* tile models must carry the GLOBAL model in their cut-side halo (spec
  sharp edge 12: the fused adjoint reads vp at stencil taps).  The public
  propagator pads each tile model by edge replication, so the test
  overwrites each tile's padded model with the matching slice of the
  single-domain padded model before replaying.

abcn=10 keeps the mid-x cut >= 2M clear of the global PML band (spec 4.1
cut-placement guard), unlike the abcn=30 canonical suite default.
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
    SteppedBackwardRunner,
    SteppedBindingRunner,
    acoustic_adj_pairs,
    acoustic_psi_pairs,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NZ, NX = 48, 56
NT = 120
DT = 0.0015
SO = 4
M = SO // 2
ABCN = 10            # cut at nx/2 clears the PML band by >= 2M
PAD = ABCN + M       # symmetric per-side model padding on the run grid
NXP = NX // 2        # physical columns per tile

SRC_GX, SRC_GZ = NX // 2, NZ // 4          # owned by tile1 (x >= 28)
REC_GX = list(range(2, NX - 2, 6))
REC_GZ = 2

X_LO_BIT, X_HI_BIT = 1, 2


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def vp_global():
    vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    vp = np.broadcast_to(vp, (NZ, NX)).copy()
    vp[20:30, 20:36] += 180.0  # box anomaly straddling the cut at x=28
    return vp


def make_prop(shape, topo=None):
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
        abcn=ABCN,
        free_surface=False,
        pml_type="cpmlr",
        nt=NT,
        B=1,
        use_ckpt=False,
        boundary_saving_config={
            "enabled": True, "storage": "gpu",
            "transfer_interval": 1, "pinned_memory": False,
        },
    )
    if topo is not None:
        kwargs["model_parallel"] = topo
    return PropTorch(equation, **kwargs)


def capture_both(prop):
    """Wrap forward_func + backward_bs_func so the populated raw inputs
    survive the public forward+backward run."""
    cap = {}
    impl = prop._backend_impl

    fwd_orig = impl.forward_func

    def fwd_wrapper(params):
        out = fwd_orig(params)
        cap["fp"] = params
        cap["fwd_raw_out"] = out
        cap["fwd_func"] = fwd_orig
        return out

    impl.forward_func = fwd_wrapper

    bwd_orig = impl.backward_bs_func

    def bwd_wrapper(params):
        out = bwd_orig(params)
        cap["bp"] = params
        cap["bwd_func"] = bwd_orig
        return out

    impl.backward_bs_func = bwd_wrapper
    return cap


def run_public_once(prop, wavelet, sources, receivers, vp_np):
    models = [torch.tensor(vp_np, device=DEV, requires_grad=True)]
    syn = prop(wavelet, sources, receivers, models=models)
    rec = syn[0] if isinstance(syn, (tuple, list)) else syn
    rec.sum().backward()


class TileState:
    """Replay state for one captured propagator (reference or tile)."""

    def __init__(self, cap):
        self.fp = cap["fp"]
        self.fwd_func = cap["fwd_func"]
        self.bp = cap["bp"]
        self.bwd_func = cap["bwd_func"]
        self.fwd_record_raw = cap["fwd_raw_out"][2]

        # forward wavefield list (9 slots) for the stepped forward replay
        L = list(self.fp.wavefields)
        if not L:
            L = [torch.zeros_like(self.fp.models[0]) for _ in range(9)]
        assert len(L) == 9
        self.L_fwd = L
        record = torch.zeros_like(self.fwd_record_raw)
        self.fp.record_out = record
        self.record = record

        # backward state: adjoint list (11), fresh recon (3), bound outputs
        self.L_adj = list(self.bp.adjoint_wavefields)
        assert len(self.L_adj) == 11
        self.recon = [torch.zeros_like(self.bp.models[0]) for _ in range(3)]
        self.gbufs = [torch.zeros_like(self.bp.forward_source)] + [
            torch.zeros_like(m) for m in self.bp.models
        ]
        self.ibufs = [torch.zeros_like(self.bp.models[0]) for _ in range(2)]
        self.bp.grads_out = self.gbufs
        self.bp.illum_out = self.ibufs

    def fwd_runner(self):
        for t in self.L_fwd:
            t.zero_()
        return SteppedBindingRunner(
            self.fwd_func, self.fp, self.L_fwd, acoustic_psi_pairs(2)
        )

    def zero_backward_state(self):
        for t in self.L_adj + self.recon + self.gbufs + self.ibufs:
            t.zero_()

    def bwd_runner(self):
        return SteppedBackwardRunner(
            self.bwd_func, self.bp, self.L_adj, self.recon,
            acoustic_adj_pairs(2),
        )

    def backward_snapshot(self):
        return {
            "grads": [t.clone() for t in self.gbufs],
            "illum": [t.clone() for t in self.ibufs],
        }


def build_reference():
    sources = np.array([[[SRC_GX, SRC_GZ]]], dtype=np.int32)
    receivers = np.array([[[gx, REC_GZ] for gx in REC_GX]], dtype=np.int32)
    prop = make_prop((NZ, NX))
    cap = capture_both(prop)
    run_public_once(prop, ricker(NT, DT), sources, receivers, vp_global())
    return TileState(cap)


def build_tiles(residual_raw):
    """Two tile TileStates; per-tile adjoint_source filled from the shared
    synthetic residual (reference raw record), split by receiver ownership."""
    vp = vp_global()
    wavelet = ricker(NT, DT)
    tiles, rec_split = [], []
    for xi in range(2):
        topo = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * NXP
        vp_tile = vp[:, x0:x0 + NXP].copy()

        owns_src = (x0 <= SRC_GX < x0 + NXP)
        if owns_src:
            tile_sources = np.array([[[SRC_GX - x0, SRC_GZ]]], dtype=np.int32)
            tile_wavelet = wavelet
        else:
            # zero-amplitude corner source: keeps nsrc >= 1 without effect
            tile_sources = np.array([[[1, 1]]], dtype=np.int32)
            tile_wavelet = np.zeros_like(wavelet)

        own_rec = [gx for gx in REC_GX if x0 <= gx < x0 + NXP]
        idxs = [REC_GX.index(gx) for gx in own_rec]
        rec_split.append(idxs)
        tile_receivers = np.array(
            [[[gx - x0, REC_GZ] for gx in own_rec]], dtype=np.int32
        )

        prop = make_prop((NZ, NXP), topo=topo)
        cap = capture_both(prop)
        run_public_once(prop, tile_wavelet, tile_sources, tile_receivers, vp_tile)
        t = TileState(cap)
        t.owns_src = owns_src
        t.cut_face_mask = X_HI_BIT if xi == 0 else X_LO_BIT
        t.x_off = xi * NXP  # run-grid offset of the tile in the global grid

        # this tile's receivers' traces of the shared synthetic residual
        adj = torch.zeros_like(t.bp.adjoint_source)
        assert adj.shape == (1, len(idxs), NT)
        for j, gi in enumerate(idxs):
            adj[:, j] = residual_raw[:, gi]
        t.bp.adjoint_source = adj
        tiles.append(t)
    return tiles


def replay_dd_forward(tiles, global_model):
    """Stepped DD forward with per-step u_now halo exchange (the bit-exact
    test_dd_two_tile.py protocol) — boundary ring + last_two repopulated."""
    with torch.no_grad():
        # spec sharp edge 12: tile models need the GLOBAL vp in the cut-side
        # halo (the fused adjoint reads vp at stencil taps); overwrite the
        # edge-replicated padding with the global padded slice.
        for t in tiles:
            sl = global_model[..., :, t.x_off:t.x_off + NXP + 2 * PAD]
            t.fp.models[0].copy_(sl)
            t.bp.models[0].copy_(sl)

        r0, r1 = tiles[0].fwd_runner(), tiles[1].fwd_runner()
        lo, hi = PAD, PAD + NXP
        for it in range(NT):
            r0.run_to(it + 1)
            r1.run_to(it + 1)
            u0, u1 = r0.u_now, r1.u_now
            u0[..., hi:hi + M] = u1[..., lo:lo + M]
            u1[..., lo - M:lo] = u0[..., hi - M:hi]

        # hand the DD-consistent forward products to the backward inputs
        # (normally the same storages already, but rebind to be explicit)
        for t in tiles:
            t.bp.boundary_gpu = list(t.fp.boundary_gpu)
            t.bp.u_last_two = t.fp.last_two


def replay_dd_backward(tiles):
    """Spec 4.2: per-step descending segments + lambda/recon halo exchange."""
    for t in tiles:
        t.zero_backward_state()
        t.bp.cut_face_mask = t.cut_face_mask
    r0, r1 = tiles[0].bwd_runner(), tiles[1].bwd_runner()
    lo, hi = PAD, PAD + NXP
    with torch.no_grad():
        for it in range(NT - 1, -1, -1):
            r0.run_segment(it + 1, it)
            r1.run_segment(it + 1, it)
            if it == 0:
                break  # adjoint-only tail: nothing left to consume halos
            l0, l1 = r0.lambda_now, r1.lambda_now
            l0[..., hi:hi + M] = l1[..., lo:lo + M]
            l1[..., lo - M:lo] = l0[..., hi - M:hi]
            f0, f1 = r0.recon_u_now, r1.recon_u_now
            f0[..., hi:hi + M] = f1[..., lo:lo + M]
            f1[..., lo - M:lo] = f0[..., hi - M:hi]
    assert r0.k_adj == NT and r0.k_f == NT - 1
    assert r1.k_adj == NT and r1.k_f == NT - 1


# --------------------------------------------------------------------- B1
@cuda_only
def test_dd_backward_two_tile_bitexact():
    ref = build_reference()

    # fixed synthetic residual shared by reference and DD
    residual_raw = ref.fwd_record_raw.clone()
    ref.bp.adjoint_source = residual_raw

    # determinism baseline: monolithic replay twice, bitwise
    ref.zero_backward_state()
    ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
    ref.bp.adjoint_wavefields = ref.L_adj
    ref.bp.forward_wavefields = ref.recon
    ref.bwd_func(ref.bp)
    snap_a = ref.backward_snapshot()
    ref.zero_backward_state()
    ref.bwd_func(ref.bp)
    snap_b = ref.backward_snapshot()
    for key in snap_a:
        for i, (a, b) in enumerate(zip(snap_a[key], snap_b[key])):
            assert torch.equal(a, b), f"determinism baseline: {key}[{i}]"

    # ---------------- DD pipeline ----------------
    tiles = build_tiles(residual_raw)
    replay_dd_forward(tiles, ref.fp.models[0])

    # DD forward sanity (same bar as test_dd_two_tile): records bitwise
    rec_tiles = torch.zeros_like(ref.record)
    rec_split = [[REC_GX.index(gx) for gx in REC_GX if x0 <= gx < x0 + NXP]
                 for x0 in (0, NXP)]
    # reference stepped-forward replay to fill ref.record deterministically
    rr = ref.fwd_runner()
    rr.run_to(NT)
    for t, idxs in zip(tiles, rec_split):
        for j, gi in enumerate(idxs):
            rec_tiles[:, gi] = t.record[:, j]
    assert torch.equal(rec_tiles, ref.record), "DD forward record differs"

    # the reference forward replay rewrote the reference ring/last_two with
    # values bitwise-identical to the public run; backward state untouched.
    replay_dd_backward(tiles)

    # ---------------- assemble + compare (owned slices) ----------------
    t0, t1 = tiles
    own = slice(PAD, PAD + NXP)            # owned columns, tile-local
    g_ref = ref.gbufs[1]

    # grad_vp
    assert torch.equal(t0.gbufs[1][..., own], g_ref[..., PAD:PAD + NXP]), \
        "tile0 grad_vp differs (bitwise)"
    assert torch.equal(t1.gbufs[1][..., own], g_ref[..., PAD + NXP:PAD + 2 * NXP]), \
        "tile1 grad_vp differs (bitwise)"

    # illuminations
    for k, name in enumerate(["source_illum", "receiver_illum"]):
        assert torch.equal(t0.ibufs[k][..., own], ref.ibufs[k][..., PAD:PAD + NXP]), \
            f"tile0 {name} differs (bitwise)"
        assert torch.equal(
            t1.ibufs[k][..., own], ref.ibufs[k][..., PAD + NXP:PAD + 2 * NXP]
        ), f"tile1 {name} differs (bitwise)"

    # grad_wavelet from the owning tile (tile1)
    assert t1.owns_src and not t0.owns_src
    assert torch.equal(t1.gbufs[0], ref.gbufs[0]), \
        "grad_wavelet differs (bitwise)"


# ------------------------------------------------------- negative guards
@cuda_only
def test_dd_backward_guards():
    ref = build_reference()
    bp, func = ref.bp, ref.bwd_func

    # 2D rejects y-face bits
    bp.cut_face_mask = 16
    with pytest.raises(RuntimeError, match="bits 0..3"):
        func(bp)

    # cut_face_mask requires gpu-direct boundary storage (v1)
    bp.cut_face_mask = X_HI_BIT
    bp.boundary_on_cpu = True
    with pytest.raises(RuntimeError, match="gpu-direct boundary storage only"):
        func(bp)
    bp.boundary_on_cpu = False
    bp.boundary_on_disk = True
    with pytest.raises(RuntimeError, match="gpu-direct boundary storage only"):
        func(bp)
    bp.boundary_on_disk = False
    bp.cut_face_mask = 0
