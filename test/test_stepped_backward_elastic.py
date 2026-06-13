"""Bit-exactness of the stepped backward API for elastic2d / elastic3d.

Strategy (mirrors ``test_stepped_backward.py``, acoustic): run the public
propagator forward+backward once only to capture a fully-populated
``BackwardInput`` (and force buffer allocation), then replay through the
raw binding from re-zeroed state — once as a single full-range call
(reference), once split into descending segments — and require
``torch.equal`` on every bound output (grad_vp, grad_vs, grad_rho) and
every piece of persistent state (adjoint wavefields, reconstruction
wavefields).  Same kernels, same launch order, same absolute time indices:
any mismatch is a bug, not tolerance.

Elastic specifics (vs the acoustic harness):

* NO buffer-role rotation anywhere: every adjoint/recon field updates in
  place at a fixed slot, so the SAME lists are bound for every segment
  (``adj_pairs=()``, ``adj_u_blocks=()``, ``recon_u_blocks=()``).

* Adjoint wavefield list = the full elastic layout (15 slots 2D / 36
  slots 3D: physical fields + CPML memories m_*).

* Reconstruction list (boundary saving only) = the physical fields plus
  the carried fv*_prev velocities consumed by the gradient kernel:
  7 slots 2D ``[vx, vz, sxx, szz, sxz, fvx_prev, fvz_prev]``,
  12 slots 3D ``[vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
  fvx_prev, fvy_prev, fvz_prev]``.

* grads_out = ``{grad_vp, grad_vs, grad_rho}`` (one per model — elastic
  computes no grad_wavelet and no illuminations, so illum_out stays
  empty).

* Loop tails differ per mode: the plain (full-storage) backward runs
  it == 0 as a gradient-only step (no adjoint update); the boundary-saving
  backward stops at it == 1, so the segment containing it == 0 runs
  nothing extra — both are position-based and segment-safe.

A determinism baseline (monolithic replay twice, bitwise) runs first in
every case.  Single source point (multi-source same-cell atomicAdd would
make even two identical runs bitwise-nondeterministic).
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
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import SteppedBackwardRunner  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NT2D = 120
NT3D = 60
DT = 0.0015
WAVELET_SCALE = 1.0e6

N_ADJ = {2: 15, 3: 36}
N_RECON = {2: 7, 3: 12}


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def build(ndim, *, free_surface, bs=None, nt=NT2D):
    shape = (48, 56) if ndim == 2 else (24, 20, 24)
    eq_cls = Elastic if ndim == 2 else Elastic3D
    equation = eq_cls(spatial_order=4, device=DEV, backend="torch")

    boundary_saving_config = {"enabled": False}
    if bs == "gpu":
        boundary_saving_config = {
            "enabled": True, "storage": "gpu",
            "transfer_interval": 1, "pinned_memory": False,
        }

    prop = PropTorch(
        equation,
        backend="torch",
        impl="c",
        shape=shape,
        dev=DEV,
        dh=10.0,
        dt=DT,
        source_type=["sxx", "szz"] if ndim == 2 else ["sxx", "syy", "szz"],
        receiver_type=["vx", "vz"] if ndim == 2 else ["vx", "vy", "vz"],
        abcn=10,
        free_surface=free_surface,
        pml_type="cpmls",
        nt=nt,
        B=1,
        use_ckpt=False,
        boundary_saving_config=boundary_saving_config,
    )

    if ndim == 2:
        nz, nx = shape
        grid = np.linspace(0.0, 1.0, num=nz * nx, dtype=np.float32).reshape(nz, nx)
        vp = 2200.0 + 40.0 * grid
        vs = 1200.0 + 20.0 * grid
        rho = 2000.0 + 10.0 * grid
        vp[20:30, 20:36] += 100.0
        vs[20:30, 20:36] += 50.0
        rho[20:30, 20:36] += 25.0
        sources = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array(
            [[[ix, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32
        )
    else:
        nz, ny, nx = shape
        grid = np.linspace(0.0, 1.0, num=nz * ny * nx, dtype=np.float32)
        grid = grid.reshape(nz, ny, nx)
        vp = 2200.0 + 40.0 * grid
        vs = 1200.0 + 20.0 * grid
        rho = 2000.0 + 10.0 * grid
        vp[8:14, 6:14, 10:20] += 100.0
        vs[8:14, 6:14, 10:20] += 50.0
        rho[8:14, 6:14, 10:20] += 25.0
        sources = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array(
            [[[ix, ny // 2, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32
        )

    models = [
        torch.tensor(a, device=DEV, requires_grad=True) for a in (vp, vs, rho)
    ]
    wavelet = ricker(nt, DT, scale=WAVELET_SCALE)
    return prop, wavelet, sources, receivers, models


def capture_backward(prop):
    cap = {}
    impl = prop._backend_impl
    for name in ("backward_func", "backward_bs_func", "backward_ckpt_func"):
        orig = getattr(impl, name, None)
        if orig is None:
            continue

        def make(orig, name):
            def wrapper(params):
                out = orig(params)
                cap["params"] = params
                cap["raw_out"] = out
                cap["func"] = orig
                cap["mode"] = name
                return out
            return wrapper

        setattr(impl, name, make(orig, name))
    return cap


def run_public_once(prop, wavelet, sources, receivers, models):
    syn = prop(wavelet, sources, receivers, models=models)
    rec = syn[0] if isinstance(syn, (tuple, list)) else syn
    rec.sum().backward()


def _assert_equal(snap_ref, snap_step, tag):
    for key in snap_ref:
        ref, got = snap_ref[key], snap_step[key]
        for i, (r, g) in enumerate(zip(ref, got)):
            assert torch.equal(g, r), f"[{tag}] {key}[{i}] differs (bitwise)"


class Harness:
    """Replay-state owner for one captured elastic BackwardInput."""

    def __init__(self, cap, ndim, mode):
        assert cap["mode"] == (
            "backward_bs_func" if mode == "bs" else "backward_func"
        ), f"captured {cap['mode']} but expected mode {mode}"
        self.p = cap["params"]
        self.func = cap["func"]
        self.ndim = ndim
        self.mode = mode
        self.nt = int(self.p.nt)

        self.L_adj = list(self.p.adjoint_wavefields)
        assert len(self.L_adj) == N_ADJ[ndim], (
            f"expected the full elastic adjoint layout ({N_ADJ[ndim]}), "
            f"got {len(self.L_adj)}"
        )
        self.recon = (
            [torch.zeros_like(self.p.models[0]) for _ in range(N_RECON[ndim])]
            if mode == "bs" else None
        )
        self.gbufs = [torch.zeros_like(m) for m in self.p.models]
        self.p.grads_out = self.gbufs
        self.p.illum_out = []

    def zero_state(self):
        for t in self.L_adj:
            t.zero_()
        if self.recon is not None:
            for t in self.recon:
                t.zero_()
        for t in self.gbufs:
            t.zero_()

    def snapshot(self):
        snap = {
            "grads": [t.clone() for t in self.gbufs],
            "adj": [t.clone() for t in self.L_adj],
        }
        if self.recon is not None:
            snap["recon"] = [t.clone() for t in self.recon]
        return snap

    def _restore_defaults(self):
        self.p.bw_it_begin, self.p.bw_it_end = -1, 0
        self.p.adjoint_wavefields = self.L_adj
        if self.recon is not None:
            self.p.forward_wavefields = self.recon

    def replay_monolithic(self):
        self._restore_defaults()
        self.func(self.p)

    def replay_stepped(self, cuts):
        assert cuts[0] == 0 and cuts[-1] == self.nt
        runner = SteppedBackwardRunner(
            self.func, self.p, self.L_adj, self.recon,
            adj_pairs=(), adj_u_blocks=(), recon_u_blocks=(),
        )
        for i in range(len(cuts) - 2, -1, -1):
            runner.run_segment(cuts[i + 1], cuts[i])
        assert runner.k_adj == self.nt
        assert runner.k_f == self.nt - 1
        self._restore_defaults()


def partitions_for(nt):
    return {
        "full": [0, nt],
        "halves": [0, nt // 2, nt],
        "per_step": list(range(nt + 1)),
        "ragged": [0, 7, 20, nt],
    }


def run_case(ndim, mode, nt, partition_names, free_surface):
    prop, wavelet, sources, receivers, models = build(
        ndim, free_surface=free_surface, bs=("gpu" if mode == "bs" else None),
        nt=nt,
    )
    cap = capture_backward(prop)
    run_public_once(prop, wavelet, sources, receivers, models)
    h = Harness(cap, ndim, mode)

    h.zero_state()
    h.replay_monolithic()
    ref = h.snapshot()
    for key in ("grads",):
        assert any(t.abs().max() > 0 for t in ref[key]), "all grads zero"
    h.zero_state()
    h.replay_monolithic()
    _assert_equal(ref, h.snapshot(), "determinism-baseline")

    parts = partitions_for(nt)
    for name in partition_names:
        cuts = parts[name] if isinstance(name, str) else name
        h.zero_state()
        h.replay_stepped(cuts)
        _assert_equal(ref, h.snapshot(), f"partition={name}")


# ----------------------------------------- 2D: plain backward (full storage)
@cuda_only
@pytest.mark.parametrize("free_surface", [False, True])
def test_stepped_backward_full_2d(free_surface):
    run_case(2, "full", NT2D, ["full", "halves", "per_step", "ragged"],
             free_surface)


# ------------------------------------ 2D: backward_bs (gpu-direct boundary)
@cuda_only
@pytest.mark.parametrize("free_surface", [False, True])
def test_stepped_backward_bs_2d(free_surface):
    run_case(2, "bs", NT2D, ["full", "halves", "per_step", "ragged"],
             free_surface)


# ------ tail-only split: segment [0, 1) alone (grad-only tail for "full",
# legal no-op for "bs")
@cuda_only
@pytest.mark.parametrize("mode", ["full", "bs"])
def test_stepped_backward_tail_only(mode):
    run_case(2, mode, NT2D, [[0, 1, NT2D]], free_surface=False)


# ----------------------------------------------------------------- 3D mini
@cuda_only
@pytest.mark.parametrize("mode", ["full", "bs"])
@pytest.mark.parametrize("free_surface", [False, True])
def test_stepped_backward_3d(mode, free_surface):
    run_case(3, mode, NT3D, ["full", "halves", "per_step", "ragged"],
             free_surface)


# ------------------------------------------------------------ guard checks
@cuda_only
def test_stepped_backward_elastic_guards():
    prop, wavelet, sources, receivers, models = build(
        2, free_surface=False, bs="gpu", nt=16
    )
    cap = capture_backward(prop)
    run_public_once(prop, wavelet, sources, receivers, models)
    p, func = cap["params"], cap["func"]
    nt = int(p.nt)
    recon = [torch.zeros_like(p.models[0]) for _ in range(7)]
    gbufs = [torch.zeros_like(m) for m in p.models]

    # segment range out of order / out of bounds
    p.bw_it_begin, p.bw_it_end = 4, 8
    with pytest.raises(RuntimeError, match="bw_it_end < bw_it_begin"):
        func(p)
    p.bw_it_begin, p.bw_it_end = nt + 1, 0
    with pytest.raises(RuntimeError, match="bw_it_end < bw_it_begin"):
        func(p)

    # stepped without bound grads_out
    p.bw_it_begin, p.bw_it_end = nt, nt // 2
    p.grads_out = []
    p.illum_out = []
    p.forward_wavefields = recon
    with pytest.raises(RuntimeError, match="grads_out"):
        func(p)

    # stepped with illum_out bound (elastic computes none)
    p.grads_out = gbufs
    p.illum_out = [torch.zeros_like(p.models[0]) for _ in range(2)]
    with pytest.raises(RuntimeError, match="illum_out must"):
        func(p)
    p.illum_out = []

    # stepped without the 7-tensor reconstruction list
    p.forward_wavefields = []
    with pytest.raises(RuntimeError, match="7-tensor"):
        func(p)
    p.forward_wavefields = recon

    # phases: only single-step segments
    p.step_phase = 1
    with pytest.raises(RuntimeError, match="single-step"):
        func(p)
    p.step_phase = 0

    # stepped + staged (cpu/disk) boundary storage is v1-unsupported
    p.boundary_on_cpu = True
    with pytest.raises(RuntimeError, match="gpu-direct boundary storage only"):
        func(p)
    p.boundary_on_cpu = False
    p.boundary_on_disk = True
    with pytest.raises(RuntimeError, match="gpu-direct boundary storage only"):
        func(p)
    p.boundary_on_disk = False

    # 2D rejects y-face cut bits
    p.bw_it_begin, p.bw_it_end = -1, 0
    p.cut_face_mask = 16
    with pytest.raises(RuntimeError, match="bits 0..3"):
        func(p)
    p.cut_face_mask = 0


@cuda_only
def test_stepped_backward_elastic_guards_ckpt():
    prop, wavelet, sources, receivers, models = build(
        2, free_surface=False, nt=16
    )
    prop_ck = PropTorch(
        Elastic(spatial_order=4, device=DEV, backend="torch"),
        backend="torch", impl="c", shape=(48, 56), dev=DEV, dh=10.0, dt=DT,
        source_type=["sxx", "szz"], receiver_type=["vx", "vz"], abcn=10,
        free_surface=False, pml_type="cpmls", nt=16, B=1, use_ckpt=True,
        boundary_saving_config={"enabled": False},
    )
    cap = capture_backward(prop_ck)
    run_public_once(prop_ck, wavelet, sources, receivers, models)
    assert cap["mode"] == "backward_ckpt_func"
    p, func = cap["params"], cap["func"]
    p.bw_it_begin, p.bw_it_end = int(p.nt), int(p.nt) // 2
    with pytest.raises(RuntimeError,
                       match="checkpoint backward does not support"):
        func(p)
