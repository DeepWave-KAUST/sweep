"""Two-tile elastic DD BACKWARD (gradients) vs single domain — bit-exact.

Full pipeline on one GPU, single process, for elastic2d (x-cut, FS on/off,
source interior or on the cut) and elastic3d (x-split, FS on/off):

  reference   single-domain forward (boundary saving, gpu-direct ring,
              requires_grad models) captured once via the public
              propagator, then a monolithic raw backward_bs replay with
              Python-bound grads_out (the test_stepped_backward_elastic.py
              pattern) -> grad_vp / grad_vs / grad_rho.

  DD          two x-tiles (MeshTopology px=2 -> rank-local PML widths, cut
              PML coefficients zero), each replaying its DD forward through
              the stepped API with the E1 half-step exchange protocol
              (phase 1 = velocity, exchange v M-wide; phase 2 = stress +
              tail incl. boundary save, exchange s M-wide) — WITH boundary
              saving enabled, so each tile's boundary_gpu ring + u_last_two
              hold DD-consistent values — followed by the phased DD
              backward below, per global it from nt-1 down to 1 with
              cut_face_mask set (tile0: x_hi bit, tile1: x_lo bit).

BACKWARD EXCHANGE PROTOCOL — derivation from the per-iteration kernel
read/write sets of elastic backward_bs (invariant: owned cells must never
read halo-cell values the halo tile computed locally):

  phase 1 kernels
    * adjoint receiver injection      point writes (velocity fields here)
    * recon -src injection            point writes into recon stresses
    * STRESS_NOPML                    reads recon v at +-M, writes recon s
                                      same-cell
    * restore s (cut faces skipped)   same-cell writes from the ring
    * CALCULATE_GRAD_ELASTIC_BS       reads recon v at +-M and same-cell,
                                      fv*_prev same-cell, adjoint v/s
                                      same-cell; writes grads SAME-CELL
                                      "+=" -> tile gradients assemble by
                                      owned slices
    * stress-adjoint prepare/apply    prepare reads adjoint s SAME-CELL
                                      (incl. halo cells -> their q feeds
                                      owned cells), writes workspace q;
                                      apply reads q at +-M, writes
                                      adjoint v same-cell

    Owned-cell halo inputs: recon v and adjoint s — both exchanged after
    the PREVIOUS iteration's phase 2.  Phase 1 locally recomputes recon s
    and adjoint v in the halo with stale deep taps (a depth-d halo cell
    needs taps to depth d+M > M), so the driver must refresh them before
    anything owned reads them in phase 2:

        after phase 1: exchange ADJOINT v (slots 0..1 / 0..2) and
                       RECON s (slots 2..4 / 3..8), each M wide.

  phase 2 kernels
    * velocity-adjoint prepare/apply  prepare reads adjoint v SAME-CELL
                                      (halo p feeds owned apply taps);
                                      apply reads p at +-M, writes
                                      adjoint s same-cell
    * fv*_prev capture                same-cell copies of recon v
    * VELOCITY_NOPML                  reads recon s at +-M, writes recon v
                                      same-cell
    * restore v (cut faces skipped)   same-cell

        after phase 2: exchange ADJOINT s and RECON v — consumed by the
                       next (lower-it) iteration's phase 1.

  No CPML memory (m_*) exchange: the adjoint m_* recurrence is same-cell
  in its field input, the cut-axis coefficients are zero on both sides of
  the cut (per-rank PML widths / global mid-domain), and the per-iteration
  refresh of the adjoint field halos keeps the transverse-memory history
  at halo cells in bitwise lockstep — the same induction as the E1
  forward.  fv*_prev needs no exchange (read same-cell at owned cells
  only).  v1 limitation: STRESS-type adjoint receivers within M of a cut
  would inject after the adjoint-s exchange that phase 1 consumes —
  unsupported (velocity receivers, the standard elastic record, are
  covered).

The first backward iteration needs no extra seeding exchange: u_last_two
velocity halos are current (captured after the forward's inter-phase v
exchange), the stale stress halos are refreshed by the post-phase-1
exchange before phase 2 reads them, and the adjoint state starts at zero.

abcn=10 keeps the mid-x cut >= 2M clear of the global PML band.
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

from sweep.equations import Elastic, Elastic3D  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBackwardRunner,
    SteppedBindingRunner,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

DT = 0.0015
SO = 4
M = SO // 2
ABCN = 10
PAD = ABCN + M
WAVELET_SCALE = 1.0e6
X_LO_BIT, X_HI_BIT = 1, 2

# 2D geometry (test_dd_elastic_two_tile.py)
NZ2, NX2 = 48, 56
NT2 = 120
NXP2 = NX2 // 2
SRC_Z2 = 12
REC_GX2 = list(range(2, NX2 - 2, 6))
REC_Z2 = 2

# 3D geometry (test_dd_elastic_tiles_3d.py, x-split)
NZ3, NY3, NX3 = 24, 20, 32
NT3 = 60
NXP3 = NX3 // 2
SRC3 = (NX3 // 2, NY3 // 2, NZ3 // 4)     # on the cut -> owned by tile1
REC_GX3 = list(range(2, NX3 - 2, 5))
REC_Y3, REC_Z3 = NY3 // 2, 2

NWF = {2: 15, 3: 36}
NPHYS = {2: 5, 3: 9}
NV = {2: 2, 3: 3}                          # velocity slots 0..NV-1
NRECON = {2: 7, 3: 12}                     # phys + fv*_prev carries


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def global_models(ndim):
    if ndim == 2:
        grid = np.linspace(0.0, 1.0, num=NZ2 * NX2, dtype=np.float32)
        grid = grid.reshape(NZ2, NX2)
        vp = 2200.0 + 40.0 * grid
        vs = 1200.0 + 20.0 * grid
        rho = 2000.0 + 10.0 * grid
        vp[20:30, 20:36] += 100.0   # straddles the cut at x=28
        vs[20:30, 20:36] += 50.0
        rho[20:30, 20:36] += 25.0
    else:
        grid = np.linspace(0.0, 1.0, num=NZ3 * NY3 * NX3, dtype=np.float32)
        grid = grid.reshape(NZ3, NY3, NX3)
        vp = 2200.0 + 40.0 * grid
        vs = 1200.0 + 20.0 * grid
        rho = 2000.0 + 10.0 * grid
        vp[8:14, 6:14, 10:24] += 100.0   # straddles the cut at x=16
        vs[8:14, 6:14, 10:24] += 50.0
        rho[8:14, 6:14, 10:24] += 25.0
    return [vp, vs, rho]


def make_prop(ndim, shape, free_surface, topo=None):
    eq_cls = Elastic if ndim == 2 else Elastic3D
    equation = eq_cls(spatial_order=SO, device=DEV, backend="torch")
    kwargs = dict(
        backend="torch",
        impl="c",
        shape=shape,
        dev=DEV,
        dh=10.0,
        dt=DT,
        source_type=["sxx", "szz"] if ndim == 2 else ["sxx", "syy", "szz"],
        receiver_type=["vx", "vz"] if ndim == 2 else ["vx", "vy", "vz"],
        abcn=ABCN,
        free_surface=free_surface,
        pml_type="cpmls",
        nt=NT2 if ndim == 2 else NT3,
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


def run_public_once(prop, wavelet, sources, receivers, model_arrays):
    models = [
        torch.tensor(a, device=DEV, requires_grad=True) for a in model_arrays
    ]
    syn = prop(wavelet, sources, receivers, models=models)
    rec = syn[0] if isinstance(syn, (tuple, list)) else syn
    rec.sum().backward()


class TileState:
    """Replay state for one captured elastic propagator (reference/tile)."""

    def __init__(self, cap, ndim):
        self.ndim = ndim
        self.fp = cap["fp"]
        self.fwd_func = cap["fwd_func"]
        self.bp = cap["bp"]
        self.bwd_func = cap["bwd_func"]
        self.fwd_record_raw = cap["fwd_raw_out"][2]

        L = list(self.fp.wavefields)
        if not L:
            L = [torch.zeros_like(self.fp.models[0]) for _ in range(NWF[ndim])]
        assert len(L) == NWF[ndim]
        self.L_fwd = L
        record = torch.zeros_like(self.fwd_record_raw)
        self.fp.record_out = record
        self.record = record

        self.L_adj = list(self.bp.adjoint_wavefields)
        assert len(self.L_adj) == NWF[ndim]
        self.recon = [
            torch.zeros_like(self.bp.models[0]) for _ in range(NRECON[ndim])
        ]
        self.gbufs = [torch.zeros_like(m) for m in self.bp.models]
        self.bp.grads_out = self.gbufs
        self.bp.illum_out = []

    def fwd_runner(self):
        for t in self.L_fwd:
            t.zero_()
        return SteppedBindingRunner(
            self.fwd_func, self.fp, self.L_fwd, psi_pairs=(), u_blocks=()
        )

    def zero_backward_state(self):
        for t in self.L_adj + self.recon + self.gbufs:
            t.zero_()

    def bwd_runner(self):
        return SteppedBackwardRunner(
            self.bwd_func, self.bp, self.L_adj, self.recon,
            adj_pairs=(), adj_u_blocks=(), recon_u_blocks=(),
        )

    def backward_snapshot(self):
        return {"grads": [t.clone() for t in self.gbufs]}


# =========================================================== 2D pipeline


def build_reference_2d(src_gx, free_surface):
    sources = np.array([[[src_gx, SRC_Z2]]], dtype=np.int32)
    receivers = np.array([[[gx, REC_Z2] for gx in REC_GX2]], dtype=np.int32)
    prop = make_prop(2, (NZ2, NX2), free_surface)
    cap = capture_both(prop)
    run_public_once(prop, ricker(NT2, DT, scale=WAVELET_SCALE),
                    sources, receivers, global_models(2))
    return TileState(cap, 2)


def build_tiles_2d(src_gx, free_surface, residual_raw):
    models_np = global_models(2)
    wavelet = ricker(NT2, DT, scale=WAVELET_SCALE)
    tiles, rec_split = [], []
    for xi in range(2):
        topo = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * NXP2
        tile_models = [a[:, x0:x0 + NXP2].copy() for a in models_np]

        owns_src = (x0 <= src_gx < x0 + NXP2)
        if owns_src:
            tile_sources = np.array([[[src_gx - x0, SRC_Z2]]], dtype=np.int32)
            tile_wavelet = wavelet
        else:
            tile_sources = np.array([[[1, 1]]], dtype=np.int32)
            tile_wavelet = np.zeros_like(wavelet)

        own_rec = [gx for gx in REC_GX2 if x0 <= gx < x0 + NXP2]
        idxs = [REC_GX2.index(gx) for gx in own_rec]
        rec_split.append(idxs)
        tile_receivers = np.array(
            [[[gx - x0, REC_Z2] for gx in own_rec]], dtype=np.int32
        )

        prop = make_prop(2, (NZ2, NXP2), free_surface, topo=topo)
        cap = capture_both(prop)
        run_public_once(prop, tile_wavelet, tile_sources, tile_receivers,
                        tile_models)
        t = TileState(cap, 2)
        t.owns_src = owns_src
        t.cut_face_mask = X_HI_BIT if xi == 0 else X_LO_BIT
        t.x_off = x0

        # this tile's receivers' traces of the shared synthetic residual
        adj = torch.zeros_like(t.bp.adjoint_source)
        assert adj.shape[2] == len(idxs)
        for j, gi in enumerate(idxs):
            adj[:, :, j] = residual_raw[:, :, gi]
        t.bp.adjoint_source = adj
        tiles.append(t)
    return tiles, rec_split


def replay_dd_forward_2d(tiles, global_models_t):
    """E1 half-step protocol with boundary saving enabled: phase 1 (v) ->
    exchange v M-wide -> phase 2 (s + save tail) -> exchange s M-wide."""
    with torch.no_grad():
        for t in tiles:
            for mt, mf in zip(t.fp.models, global_models_t):
                sl = mf[..., :, t.x_off:t.x_off + NXP2 + 2 * PAD]
                mt.copy_(sl)
            for mt, mf in zip(t.bp.models, global_models_t):
                sl = mf[..., :, t.x_off:t.x_off + NXP2 + 2 * PAD]
                mt.copy_(sl)

        r0, r1 = tiles[0].fwd_runner(), tiles[1].fwd_runner()
        lo, hi = PAD, PAD + NXP2

        def _swap(L0, L1, slots):
            for f in slots:
                a, b = L0[f], L1[f]
                a[..., hi:hi + M] = b[..., lo:lo + M]
                b[..., lo - M:lo] = a[..., hi - M:hi]

        for it in range(NT2):
            r0.run_phase(it + 1, 1)
            r1.run_phase(it + 1, 1)
            _swap(r0.L, r1.L, range(NV[2]))             # vx, vz
            r0.run_phase(it + 1, 2)
            r1.run_phase(it + 1, 2)
            _swap(r0.L, r1.L, range(NV[2], NPHYS[2]))   # sxx, szz, sxz

        for t in tiles:
            t.bp.boundary_gpu = list(t.fp.boundary_gpu)
            t.bp.u_last_two = t.fp.last_two


def replay_dd_backward_2d(tiles):
    """Phased descending per-step segments + the 4-group halo exchange."""
    for t in tiles:
        t.zero_backward_state()
        t.bp.cut_face_mask = t.cut_face_mask
    b0, b1 = tiles[0].bwd_runner(), tiles[1].bwd_runner()
    lo, hi = PAD, PAD + NXP2

    def _swap(pairs):
        for a, b in pairs:
            a[..., hi:hi + M] = b[..., lo:lo + M]
            b[..., lo - M:lo] = a[..., hi - M:hi]

    adj_v = list(zip(b0.L_adj[:NV[2]], b1.L_adj[:NV[2]]))
    adj_s = list(zip(b0.L_adj[NV[2]:NPHYS[2]], b1.L_adj[NV[2]:NPHYS[2]]))
    rec_v = list(zip(b0.L_recon[:NV[2]], b1.L_recon[:NV[2]]))
    rec_s = list(zip(b0.L_recon[NV[2]:NPHYS[2]], b1.L_recon[NV[2]:NPHYS[2]]))

    with torch.no_grad():
        for it in range(NT2 - 1, 0, -1):   # elastic BS floor: it == 1
            b0.run_phase(it + 1, it, 1)
            b1.run_phase(it + 1, it, 1)
            _swap(adj_v)
            _swap(rec_s)
            b0.run_phase(it + 1, it, 2)
            b1.run_phase(it + 1, it, 2)
            _swap(adj_s)
            _swap(rec_v)
    assert b0.k_adj == NT2 - 1 and b1.k_adj == NT2 - 1


@cuda_only
@pytest.mark.parametrize("src_gx", [14, NX2 // 2])  # tile0 interior / on cut
@pytest.mark.parametrize("free_surface", [False, True])
def test_dd_elastic_backward_two_tile_bitexact(src_gx, free_surface):
    ref = build_reference_2d(src_gx, free_surface)

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
    assert any(t.abs().max() > 0 for t in snap_a["grads"]), "all grads zero"
    ref.zero_backward_state()
    ref.bwd_func(ref.bp)
    snap_b = ref.backward_snapshot()
    for i, (a, b) in enumerate(zip(snap_a["grads"], snap_b["grads"])):
        assert torch.equal(a, b), f"determinism baseline: grads[{i}]"

    # ---------------- DD pipeline ----------------
    tiles, rec_split = build_tiles_2d(src_gx, free_surface, residual_raw)
    replay_dd_forward_2d(tiles, list(ref.fp.models))

    # DD forward sanity: records bitwise vs the reference stepped replay
    rr = ref.fwd_runner()
    with torch.no_grad():
        rr.run_to(NT2)
    rec_tiles = torch.zeros_like(ref.record)
    for t, idxs in zip(tiles, rec_split):
        for j, gi in enumerate(idxs):
            rec_tiles[:, :, gi] = t.record[:, :, j]
    assert torch.equal(rec_tiles, ref.record), "DD forward record differs"

    # the reference forward replay rewrote the reference ring/last_two with
    # values bitwise-identical to the public run; backward state untouched.
    replay_dd_backward_2d(tiles)

    # ---------------- assemble + compare (owned slices) ----------------
    t0, t1 = tiles
    own = slice(PAD, PAD + NXP2)
    for k, name in enumerate(["grad_vp", "grad_vs", "grad_rho"]):
        g_ref = ref.gbufs[k]
        assert torch.equal(t0.gbufs[k][..., own], g_ref[..., PAD:PAD + NXP2]), \
            f"tile0 {name} differs (bitwise)"
        assert torch.equal(
            t1.gbufs[k][..., own], g_ref[..., PAD + NXP2:PAD + 2 * NXP2]
        ), f"tile1 {name} differs (bitwise)"


# =========================================================== 3D pipeline


def build_reference_3d(free_surface):
    sources = np.array([[list(SRC3)]], dtype=np.int32)
    receivers = np.array(
        [[[gx, REC_Y3, REC_Z3] for gx in REC_GX3]], dtype=np.int32
    )
    prop = make_prop(3, (NZ3, NY3, NX3), free_surface)
    cap = capture_both(prop)
    run_public_once(prop, ricker(NT3, DT, scale=WAVELET_SCALE),
                    sources, receivers, global_models(3))
    return TileState(cap, 3)


def build_tiles_3d(free_surface, residual_raw):
    models_np = global_models(3)
    wavelet = ricker(NT3, DT, scale=WAVELET_SCALE)
    tiles, rec_split = [], []
    sx, sy, sz = SRC3
    for xi in range(2):
        topo = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * NXP3
        tile_models = [a[:, :, x0:x0 + NXP3].copy() for a in models_np]

        owns_src = (x0 <= sx < x0 + NXP3)
        if owns_src:
            tile_sources = np.array([[[sx - x0, sy, sz]]], dtype=np.int32)
            tile_wavelet = wavelet
        else:
            tile_sources = np.array([[[1, 1, 1]]], dtype=np.int32)
            tile_wavelet = np.zeros_like(wavelet)

        own_rec = [gx for gx in REC_GX3 if x0 <= gx < x0 + NXP3]
        idxs = [REC_GX3.index(gx) for gx in own_rec]
        rec_split.append(idxs)
        tile_receivers = np.array(
            [[[gx - x0, REC_Y3, REC_Z3] for gx in own_rec]], dtype=np.int32
        )

        prop = make_prop(3, (NZ3, NY3, NXP3), free_surface, topo=topo)
        cap = capture_both(prop)
        run_public_once(prop, tile_wavelet, tile_sources, tile_receivers,
                        tile_models)
        t = TileState(cap, 3)
        t.owns_src = owns_src
        t.cut_face_mask = X_HI_BIT if xi == 0 else X_LO_BIT
        t.x_off = x0
        # elastic3d FORWARD kernels are cut-aware (E1): the forward replay
        # needs the mask too, or cut-adjacent owned cells take the
        # zero-coefficient PML branch and diverge in the last ulp.
        t.fp.cut_face_mask = t.cut_face_mask

        adj = torch.zeros_like(t.bp.adjoint_source)
        assert adj.shape[2] == len(idxs)
        for j, gi in enumerate(idxs):
            adj[:, :, j] = residual_raw[:, :, gi]
        t.bp.adjoint_source = adj
        tiles.append(t)
    return tiles, rec_split


def replay_dd_forward_3d(tiles, global_models_t):
    with torch.no_grad():
        for t in tiles:
            for mt, mf in zip(t.fp.models, global_models_t):
                sl = mf[..., :, :, t.x_off:t.x_off + NXP3 + 2 * PAD]
                mt.copy_(sl)
            for mt, mf in zip(t.bp.models, global_models_t):
                sl = mf[..., :, :, t.x_off:t.x_off + NXP3 + 2 * PAD]
                mt.copy_(sl)

        r0, r1 = tiles[0].fwd_runner(), tiles[1].fwd_runner()
        lo, hi = PAD, PAD + NXP3

        def _swap(L0, L1, slots):
            for f in slots:
                a, b = L0[f], L1[f]
                a[..., hi:hi + M] = b[..., lo:lo + M]
                b[..., lo - M:lo] = a[..., hi - M:hi]

        for it in range(NT3):
            r0.run_phase(it + 1, 1)
            r1.run_phase(it + 1, 1)
            _swap(r0.L, r1.L, range(NV[3]))             # vx, vy, vz
            r0.run_phase(it + 1, 2)
            r1.run_phase(it + 1, 2)
            _swap(r0.L, r1.L, range(NV[3], NPHYS[3]))   # 6 stress fields

        for t in tiles:
            t.bp.boundary_gpu = list(t.fp.boundary_gpu)
            t.bp.u_last_two = t.fp.last_two


def replay_dd_backward_3d(tiles):
    for t in tiles:
        t.zero_backward_state()
        t.bp.cut_face_mask = t.cut_face_mask
    b0, b1 = tiles[0].bwd_runner(), tiles[1].bwd_runner()
    lo, hi = PAD, PAD + NXP3

    def _swap(pairs):
        for a, b in pairs:
            a[..., hi:hi + M] = b[..., lo:lo + M]
            b[..., lo - M:lo] = a[..., hi - M:hi]

    adj_v = list(zip(b0.L_adj[:NV[3]], b1.L_adj[:NV[3]]))
    adj_s = list(zip(b0.L_adj[NV[3]:NPHYS[3]], b1.L_adj[NV[3]:NPHYS[3]]))
    rec_v = list(zip(b0.L_recon[:NV[3]], b1.L_recon[:NV[3]]))
    rec_s = list(zip(b0.L_recon[NV[3]:NPHYS[3]], b1.L_recon[NV[3]:NPHYS[3]]))

    with torch.no_grad():
        for it in range(NT3 - 1, 0, -1):
            b0.run_phase(it + 1, it, 1)
            b1.run_phase(it + 1, it, 1)
            _swap(adj_v)
            _swap(rec_s)
            b0.run_phase(it + 1, it, 2)
            b1.run_phase(it + 1, it, 2)
            _swap(adj_s)
            _swap(rec_v)
    assert b0.k_adj == NT3 - 1 and b1.k_adj == NT3 - 1


@cuda_only
@pytest.mark.parametrize("free_surface", [False, True])
def test_dd_elastic_backward_two_tile_3d_bitexact(free_surface):
    ref = build_reference_3d(free_surface)

    residual_raw = ref.fwd_record_raw.clone()
    ref.bp.adjoint_source = residual_raw

    ref.zero_backward_state()
    ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
    ref.bp.adjoint_wavefields = ref.L_adj
    ref.bp.forward_wavefields = ref.recon
    ref.bwd_func(ref.bp)
    snap_a = ref.backward_snapshot()
    assert any(t.abs().max() > 0 for t in snap_a["grads"]), "all grads zero"
    ref.zero_backward_state()
    ref.bwd_func(ref.bp)
    snap_b = ref.backward_snapshot()
    for i, (a, b) in enumerate(zip(snap_a["grads"], snap_b["grads"])):
        assert torch.equal(a, b), f"determinism baseline: grads[{i}]"

    tiles, rec_split = build_tiles_3d(free_surface, residual_raw)
    replay_dd_forward_3d(tiles, list(ref.fp.models))

    rr = ref.fwd_runner()
    with torch.no_grad():
        rr.run_to(NT3)
    rec_tiles = torch.zeros_like(ref.record)
    for t, idxs in zip(tiles, rec_split):
        for j, gi in enumerate(idxs):
            rec_tiles[:, :, gi] = t.record[:, :, j]
    assert torch.equal(rec_tiles, ref.record), "DD forward record differs"

    replay_dd_backward_3d(tiles)

    t0, t1 = tiles
    own = slice(PAD, PAD + NXP3)
    for k, name in enumerate(["grad_vp", "grad_vs", "grad_rho"]):
        g_ref = ref.gbufs[k]
        assert torch.equal(t0.gbufs[k][..., own], g_ref[..., PAD:PAD + NXP3]), \
            f"tile0 {name} differs (bitwise)"
        assert torch.equal(
            t1.gbufs[k][..., own], g_ref[..., PAD + NXP3:PAD + 2 * NXP3]
        ), f"tile1 {name} differs (bitwise)"


# ------------------------------------------------------- negative guards
@cuda_only
def test_dd_elastic_backward_guards():
    ref = build_reference_2d(NX2 // 2, False)
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
