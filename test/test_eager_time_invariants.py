"""An uncompiled eager rollout makes the step's time-invariant coefficients once.

``dt / rho``, ``vp**2 * dt**2``, ``dt * mu`` and the like depend on the models
and ``dt`` alone; recomputed every step, each was a new field that the product
reading it kept for the backward.  The uncompiled full-tape and checkpointed
rollouts hand the step a dict (``sweep.equations._invariants``) in which they
are made once.  These tests pin: the record is bitwise the same and the
gradients move by rounding only, on every family that takes the dict; a
rollout keeps fewer fields per step; compiled steps never get the dict; and a
subclass that overrides ``func`` without declaring ``takes_invariants`` is not
handed it.  ``sweep.propagator._torch_eager.HOIST_INVARIANTS = False`` is the
per-step computation.
"""
import numpy as np
import pytest
import torch

import sweep.propagator._torch_eager as TE
from sweep.equations import (Acoustic, Acoustic1st, Acoustic3D, AcousticTariq, AcousticVRZ, AcousticVTI1st, DASMu,
                             DASZhao, Elastic, ElasticBorn, ElasticVRR, ViscoAcoustic, ViscoElastic)
from sweep.equations.elastic3d import Elastic as Elastic3D
from sweep.propagator.options import EagerOptions, Full
from sweep.propagator.torch import PropTorch

VALUES = {"vp": 2500.0, "vs": 1400.0, "rho": 2000.0, "z": 5.0e6, "Qp": 80.0, "Qs": 60.0, "Q": 80.0,
          "omega": 2 * np.pi * 15.0, "epsilon": 0.1, "delta": 0.05, "eta": 0.05, "vv": 2500.0, "v": 2700.0,
          "Rp_x": 0.01, "Rp_z": 0.01, "Rs_x": 0.01, "Rs_z": 0.01, "dvp": 50.0, "dvs": 30.0, "drho": 20.0}


@pytest.fixture(autouse=True)
def _deterministic_cudnn(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)


def _prop(cls, shape, nt, memory="full", device="cpu", **kw):
    eq = cls(spatial_order=4, device=device, backend="torch")
    return PropTorch(eq, shape=shape, dh=10.0, dt=1e-3, nt=nt, dev=device, impl="eager", abcn=4, B=1,
                     memory=Full() if memory == "full" else None,
                     eager_options=EagerOptions(use_compile=False), **kw), eq


def _shoot(prop, names, shape, nt, device="cpu"):
    t = np.arange(nt) * 1e-3 - 0.02
    a = (np.pi * 25.0 * t) ** 2
    wavelet = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=device, requires_grad=True)
    nx = shape[-1]
    if len(shape) == 2:
        src, rec = np.array([[nx // 2, 3]]), np.array([[[ix, 3] for ix in range(2, nx - 2, 3)]])
    else:
        src = np.array([[nx // 2, shape[1] // 2, 3]])
        rec = np.array([[[ix, shape[1] // 2, 3] for ix in range(2, nx - 2, 3)]])
    ramp = torch.linspace(0, 1, shape[0]).view(-1, *([1] * (len(shape) - 1)))
    models = [(VALUES[n] * (1 + 0.05 * ramp)).expand(shape).contiguous().to(device).requires_grad_(True)
              for n in names]
    return prop(wavelet, src, rec, models=models), models + [wavelet]


CASES = [
    pytest.param(Acoustic, (20, 24), {}, id="acoustic2d"),
    pytest.param(Acoustic3D, (10, 9, 8), {}, id="acoustic3d"),
    pytest.param(AcousticVRZ, (20, 24), {}, id="acoustic_vrz"),
    pytest.param(AcousticTariq, (20, 24), {}, id="acoustic_tariq"),
    pytest.param(Acoustic1st, (20, 24), {}, id="acoustic1st"),
    pytest.param(AcousticVTI1st, (20, 24), {}, id="acoustic_vti_1st"),
    pytest.param(ViscoAcoustic, (20, 24), {}, id="visco_acoustic"),
    pytest.param(Elastic, (20, 24), {"free_surface": True}, id="elastic2d_free_surface"),
    pytest.param(Elastic, (20, 24), {"memory": "ckpt"}, id="elastic2d_ckpt"),
    pytest.param(Elastic3D, (10, 9, 8), {"free_surface": True}, id="elastic3d_free_surface"),
    pytest.param(ElasticBorn, (20, 24), {}, id="elastic_born"),
    pytest.param(ElasticVRR, (20, 24), {}, id="elastic_vrr"),
    pytest.param(DASZhao, (20, 24), {}, id="das_zhao"),
    pytest.param(DASMu, (20, 24), {"free_surface": True}, id="das_mu_free_surface"),
    pytest.param(ViscoElastic, (20, 24), {"free_surface": True}, id="visco_elastic_free_surface"),
]


@pytest.mark.parametrize("cls,shape,kw", CASES)
def test_same_record_gradient_moved_by_rounding(cls, shape, kw, monkeypatch):
    kw = dict(kw)
    memory = kw.pop("memory", "full")
    nt = 30 if len(shape) == 2 else 16
    prop, eq = _prop(cls, shape, nt, memory, **kw)
    assert TE._takes_invariants(eq)

    def run():
        d, leaves = _shoot(prop, eq.models, shape, nt)
        grads = torch.autograd.grad(d.square().sum(), leaves, allow_unused=True)
        return d.detach(), [g for g in grads if g is not None]

    d_once, g_once = run()
    monkeypatch.setattr(TE, "HOIST_INVARIANTS", False)
    d_each, g_each = run()
    assert torch.equal(d_once, d_each), "the record must not change: the coefficients are the same values"
    for a, b in zip(g_once, g_each):
        scale = b.abs().max().item()
        assert (a - b).abs().max().item() <= 1e-5 * scale, "gradient moved by more than rounding"


def _fields_per_step(cls, values_names, shape):
    """Field-sized storages autograd saves per step (16- minus 8-step rollout)."""
    per = []
    for nt in (8, 16):
        prop, _ = _prop(cls, shape, nt)
        storages = {}

        def pack(x):
            s = x.untyped_storage()
            storages[s.data_ptr()] = s.nbytes()
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            _shoot(prop, values_names, shape, nt)
        field = 4 * int(np.prod(prop._runtime_shape()))
        per.append(sum(b for b in storages.values() if b >= field // 2))
    return (per[1] - per[0]) / 8 / field


@pytest.mark.parametrize("cls,names,once,each", [
    # Acoustic: vp**2 * dt**2 next to the field the product needs.
    pytest.param(Acoustic, ["vp"], 1, 2, id="acoustic2d"),
    # Elastic: dt/rho twice and dt*mu next to the five derivative sums.
    pytest.param(Elastic, ["vp", "vs", "rho"], 5, 8, id="elastic2d"),
])
def test_fewer_fields_per_step(cls, names, once, each, monkeypatch):
    shape = (20, 24)
    assert _fields_per_step(cls, names, shape) == once
    monkeypatch.setattr(TE, "HOIST_INVARIANTS", False)
    assert _fields_per_step(cls, names, shape) == each


def test_compiled_steps_get_no_dict():
    eq = Acoustic(spatial_order=4, device="cpu", backend="torch")
    prop = PropTorch(eq, shape=(20, 24), dh=10.0, dt=1e-3, nt=8, dev="cpu", impl="eager", abcn=4,
                     eager_options=EagerOptions(use_compile=True))
    impl = prop._backend_impl
    assert impl.use_compile
    assert not impl._hoists_invariants()
    impl.use_compile = False
    assert impl._hoists_invariants()


def test_a_subclass_overriding_func_is_not_handed_the_dict():
    class Custom(Acoustic):
        def func(self, wavefields, models, dt, h, b):      # no ``invariants``
            return super().func(wavefields, models, dt, h, b)

    class Declared(Acoustic):
        takes_invariants = True

        def func(self, wavefields, models, dt, h, b, invariants=None, **kwargs):
            return super().func(wavefields, models, dt, h, b, invariants=invariants)

    assert TE._takes_invariants(Acoustic(spatial_order=4, device="cpu", backend="torch"))
    assert not TE._takes_invariants(Custom(spatial_order=4, device="cpu", backend="torch"))
    assert TE._takes_invariants(Declared(spatial_order=4, device="cpu", backend="torch"))
    # and the undeclared one still runs (computing in place)
    prop, eq = _prop(Custom, (20, 24), 8)
    d, _ = _shoot(prop, eq.models, (20, 24), 8)
    assert torch.isfinite(d).all()


def test_a_failed_compile_midway_keeps_the_rollouts_choice(monkeypatch):
    """Compiling decides at the start of the rollout that the coefficients are
    computed in place; when the compile fails and the steps fall back to the
    uncompiled ones, every checkpointed chunk -- forward and recompute -- keeps
    that choice (a recompute that made them once would save other tensors than
    its forward did, which torch.utils.checkpoint rejects)."""
    import warnings

    def cannot_build(fn, **kwargs):
        def compiled(*args, **kw):
            raise RuntimeError("compiler unavailable")
        return compiled

    monkeypatch.setattr(TE, "_compiler_failures", lambda: (RuntimeError,))
    monkeypatch.setattr(TE, "INDUCTOR_23", False)
    monkeypatch.setattr(torch, "compile", cannot_build)
    eq = Acoustic(spatial_order=4, device="cpu", backend="torch")
    prop = PropTorch(eq, shape=(20, 24), dh=10.0, dt=1e-3, nt=24, dev="cpu", impl="eager", abcn=4,
                     eager_options=EagerOptions(use_compile=True))   # default memory: checkpointed chunks
    prop._backend_impl.ckpt_chunks = 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d, leaves = _shoot(prop, eq.models, (20, 24), 24)
        (g,) = torch.autograd.grad(d.square().sum(), leaves[:1])
    assert prop._backend_impl.use_compile is False
    monkeypatch.setattr(TE, "HOIST_INVARIANTS", False)
    ref, _ = _prop(Acoustic, (20, 24), 24, memory="ckpt")
    ref._backend_impl.ckpt_chunks = 8
    d0, leaves0 = _shoot(ref, eq.models, (20, 24), 24)
    (g0,) = torch.autograd.grad(d0.square().sum(), leaves0[:1])
    assert torch.equal(d, d0.detach()) and torch.equal(g, g0)
