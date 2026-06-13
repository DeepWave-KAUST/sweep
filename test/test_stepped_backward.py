"""Bit-exactness of the stepped backward API (``bw_it_begin``/``bw_it_end``).

Strategy (mirrors ``test_stepped_forward.py``): run the public propagator
forward+backward once only to capture a fully-populated ``BackwardInput``
(and force buffer allocation), then replay through the raw binding from
re-zeroed adjoint state — once as a single full-range call (reference), once
split into descending segments with the adjoint/reconstruction wavefield
lists rotated per :mod:`sweep.propagator._stepped` — and require
``torch.equal`` on every bound output (grad_wavelet, grad_vp, both
illuminations) and every piece of persistent state (adjoint wavefields,
reconstruction wavefields).  Same kernels, same launch order, same absolute
time indices: any mismatch is a bug, not tolerance.

A determinism baseline (monolithic replay twice, bitwise) runs first in
every case; if the baseline itself is nondeterministic the stepped
comparison is meaningless.

All cases use a single source: multi-source same-cell atomicAdd would make
even two identical full runs bitwise-nondeterministic.
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

from sweep.equations import Acoustic, Acoustic3D  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBackwardRunner,
    acoustic_adj_pairs,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NT2D = 120
NT3D = 60


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def build(ndim, *, abcn=8, bs=None, use_ckpt=False, nt=NT2D):
    shape = (48, 56) if ndim == 2 else (24, 20, 24)
    eq_cls = Acoustic if ndim == 2 else Acoustic3D
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
        dt=0.0015,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=abcn,
        free_surface=False,
        pml_type="cpmlr",
        nt=nt,
        B=1,
        use_ckpt=use_ckpt,
        boundary_saving_config=boundary_saving_config,
    )

    if ndim == 2:
        nz, nx = shape
        vp = 1800.0 + 600.0 * np.linspace(0, 1, nz, dtype=np.float32)[:, None]
        vp = np.broadcast_to(vp, shape).copy()
        sources = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array(
            [[[ix, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32
        )
    else:
        nz, ny, nx = shape
        vp = 1800.0 + 600.0 * np.linspace(0, 1, nz, dtype=np.float32)[:, None, None]
        vp = np.broadcast_to(vp, shape).copy()
        sources = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array(
            [[[ix, ny // 2, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32
        )

    models = [torch.tensor(vp, device=DEV, requires_grad=True)]
    wavelet = ricker(nt, 0.0015)
    return prop, wavelet, sources, receivers, models


def capture_backward(prop):
    """Wrap the compiled propagator's backward funcs so the populated
    BackwardInput is kept (Warpper.apply reads the attrs at forward time)."""
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
    """Replay-state owner for one captured BackwardInput."""

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
        want = 11 if ndim == 2 else 15
        assert len(self.L_adj) == want, (
            f"expected the psi+zeta double-buffer adjoint layout "
            f"({want}), got {len(self.L_adj)}"
        )
        self.recon = (
            [torch.zeros_like(self.p.models[0]) for _ in range(3)]
            if mode == "bs" else None
        )
        self.gbufs = [torch.zeros_like(self.p.forward_source)] + [
            torch.zeros_like(m) for m in self.p.models
        ]
        self.ibufs = [torch.zeros_like(self.p.models[0]) for _ in range(2)]
        self.p.grads_out = self.gbufs
        self.p.illum_out = self.ibufs

    def zero_state(self):
        for t in self.L_adj:
            t.zero_()
        if self.recon is not None:
            for t in self.recon:
                t.zero_()
        for t in self.gbufs:
            t.zero_()
        for t in self.ibufs:
            t.zero_()

    def snapshot(self):
        snap = {
            "grads": [t.clone() for t in self.gbufs],
            "illum": [t.clone() for t in self.ibufs],
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
        """``cuts`` are ascending boundaries [0, c1, ..., nt]; segments are
        issued in descending order as the API requires."""
        assert cuts[0] == 0 and cuts[-1] == self.nt
        runner = SteppedBackwardRunner(
            self.func, self.p, self.L_adj, self.recon,
            acoustic_adj_pairs(self.ndim),
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


def run_case(ndim, mode, nt, partition_names):
    prop, wavelet, sources, receivers, models = build(
        ndim, bs=("gpu" if mode == "bs" else None), nt=nt
    )
    cap = capture_backward(prop)
    run_public_once(prop, wavelet, sources, receivers, models)
    h = Harness(cap, ndim, mode)

    # Determinism baseline: the monolithic replay must be bitwise repeatable
    # or the stepped comparison below is meaningless.
    h.zero_state()
    h.replay_monolithic()
    ref = h.snapshot()
    h.zero_state()
    h.replay_monolithic()
    _assert_equal(ref, h.snapshot(), "determinism-baseline")

    parts = partitions_for(nt)
    for name in partition_names:
        cuts = parts[name] if isinstance(name, str) else name
        h.zero_state()
        h.replay_stepped(cuts)
        _assert_equal(ref, h.snapshot(), f"partition={name}")


# ------------------------------------------- A1: backward (full storage), 2D
@cuda_only
def test_stepped_backward_full_2d():
    run_case(2, "full", NT2D, ["full", "halves", "per_step", "ragged"])


# ------------------------------------- A2: backward_bs (gpu-direct ring), 2D
@cuda_only
def test_stepped_backward_bs_2d():
    run_case(2, "bs", NT2D, ["full", "halves", "per_step", "ragged"])


# ------------------------- A3: tail-only split — it==0 tail in own segment,
# exercising the k_adj/k_f divergence (adjoint rotates, recon does not)
@cuda_only
@pytest.mark.parametrize("mode", ["full", "bs"])
def test_stepped_backward_tail_only(mode):
    run_case(2, mode, NT2D, [[0, 1, NT2D]])


# ----------------------------------------------------- A5: 3D mini, A1 + A2
@cuda_only
@pytest.mark.parametrize("mode", ["full", "bs"])
def test_stepped_backward_3d(mode):
    run_case(3, mode, NT3D, ["full", "halves", "per_step", "ragged"])


# --------------------------------------------------------- A4: guard checks
@cuda_only
def test_stepped_backward_guards():
    prop, wavelet, sources, receivers, models = build(2, bs="gpu", nt=16)
    cap = capture_backward(prop)
    run_public_once(prop, wavelet, sources, receivers, models)
    p, func = cap["params"], cap["func"]
    nt = int(p.nt)
    recon = [torch.zeros_like(p.models[0]) for _ in range(3)]
    gbufs = [torch.zeros_like(p.forward_source)] + [
        torch.zeros_like(m) for m in p.models
    ]
    ibufs = [torch.zeros_like(p.models[0]) for _ in range(2)]

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

    # stepped without bound illum_out
    p.grads_out = gbufs
    with pytest.raises(RuntimeError, match="illum_out"):
        func(p)

    # stepped without the 3-tensor reconstruction list
    p.illum_out = ibufs
    p.forward_wavefields = []
    with pytest.raises(RuntimeError, match="3-tensor reconstruction"):
        func(p)

    # stepped + staged (cpu/disk) boundary storage is v1-unsupported
    p.forward_wavefields = recon
    p.boundary_on_cpu = True
    with pytest.raises(RuntimeError, match="gpu-direct boundary storage only"):
        func(p)
    p.boundary_on_cpu = False
    p.boundary_on_disk = True
    with pytest.raises(RuntimeError, match="gpu-direct boundary storage only"):
        func(p)
    p.boundary_on_disk = False


@cuda_only
def test_stepped_backward_guards_ckpt():
    prop, wavelet, sources, receivers, models = build(2, use_ckpt=True, nt=16)
    cap = capture_backward(prop)
    run_public_once(prop, wavelet, sources, receivers, models)
    assert cap["mode"] == "backward_ckpt_func"
    p, func = cap["params"], cap["func"]
    p.bw_it_begin, p.bw_it_end = int(p.nt), int(p.nt) // 2
    with pytest.raises(RuntimeError,
                       match="checkpoint backward does not support"):
        func(p)


@cuda_only
def test_stepped_backward_guards_rtm():
    # full-storage params carry u_forward, which is what rtm() needs; the
    # stepped TORCH_CHECK must fire before anything else.
    prop, wavelet, sources, receivers, models = build(2, nt=16)
    cap = capture_backward(prop)
    run_public_once(prop, wavelet, sources, receivers, models)
    p = cap["params"]
    from sweep.propagator._c import _get_C
    _C = _get_C()
    p.bw_it_begin, p.bw_it_end = int(p.nt), 1
    with pytest.raises(RuntimeError, match="stepped RTM not supported"):
        _C.acoustic2d_rtm(p)
