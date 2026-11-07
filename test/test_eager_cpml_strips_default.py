"""CPML strips are the eager default in 3-D, the full grid in 2-D and with topography.

``EagerOptions(cpml_strips=None)`` (the default) picks the eager state's layout when
the propagator is built. The tests check the layout itself, that the default runs the
same code as the explicit switch (bit-identical record and gradients), and that what a
run hands back -- snapshots, ``last_wavefields``, a custom-gradient record -- does not
depend on the layout. Uncompiled, the strips give the full-grid numbers bit for bit, so
the comparisons with ``cpml_strips=False`` are exact as well -- for every 2-D equation with
strip-held fields too, which takes the strips when asked.
"""
import numpy as np
import pytest
import torch

from conftest import ricker
from sweep.equations import Acoustic, Acoustic3D, Elastic, Elastic3D, get_equation, list_equations
from sweep.equations._cpml import StripLayout
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

NT, DT, ABCN = 24, 1e-3, 6
CASES = {
    "acoustic3d": (Acoustic3D, (12, 13, 14), {}),
    "elastic3d": (Elastic3D, (10, 11, 12), {"source_type": ["sxx", "syy", "szz"], "receiver_type": ["vz"]}),
    "acoustic2d": (Acoustic, (24, 28), {}),
    "elastic2d": (Elastic, (20, 22), {"source_type": ["sxx", "szz"], "receiver_type": ["vz"]}),
}


def _prop(name, strips, eq=None, store_last=False, **kw):
    cls, shape, extra = CASES[name]
    opts = EagerOptions(use_compile=False, store_last_wavefield=store_last)
    if strips != "default":
        opts.cpml_strips = strips
    return PropTorch(eq or cls(4, device="cpu"), impl="eager", shape=shape, dh=10.0, dt=DT, nt=NT,
                     abcn=ABCN, dev="cpu", eager_options=opts, **extra, **kw)


def _layout(prop):
    return prop._backend_impl._strips


def _models(name, n=1):
    shape = CASES[name][1]
    g = torch.Generator().manual_seed(0)
    vp = 2000 + 300 * torch.rand(shape, generator=g)
    if name.startswith("elastic"):
        return [vp, vp / 1.8, 1800 + 200 * torch.rand(shape, generator=g)]
    return [vp] + [0.1 * torch.randn(shape, generator=g) for _ in range(n - 1)]


def _geometry(name):
    shape = CASES[name][1]
    if len(shape) == 2:
        return (np.array([[shape[1] // 2, 3]]),
                np.array([[[ix, 3] for ix in range(1, shape[1] - 1, 3)]]))
    return (np.array([[shape[2] // 2, shape[1] // 2, 3]]),
            np.array([[[ix, iy, 3] for ix in range(1, shape[2] - 1, 3) for iy in range(1, shape[1] - 1, 3)]]))


def _shoot(prop, name, models, **kw):
    """Record (and snapshots, when asked for) and the gradient of every model."""
    src, rec = _geometry(name)
    ms = [m.clone().requires_grad_(True) for m in models]
    out = prop(torch.from_numpy(ricker(NT, DT, fm=25.0, delay=0.03)), src, rec, models=ms, **kw)
    record = out[0] if isinstance(out, tuple) else out
    record.square().sum().backward()
    return out, [m.grad for m in ms]


def _equal(a, b):
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


@pytest.mark.parametrize("memory", [{}, {"use_ckpt": False}], ids=["ckpt", "full"])
@pytest.mark.parametrize("name", ["acoustic3d", "elastic3d"])
def test_3d_default_runs_the_strips(name, memory):
    default, strips, full = (_prop(name, s, **memory) for s in ("default", True, False))
    assert _layout(default) is not None and _layout(strips) is not None
    assert _layout(full) is None
    assert len(_layout(default)) > len(default._backend_impl.wavefield_names)
    models = _models(name)
    out = _shoot(default, name, models)
    assert _equal(out, _shoot(strips, name, models))
    assert _equal(out, _shoot(full, name, models))


@pytest.mark.parametrize("name", ["acoustic2d", "elastic2d"])
def test_2d_default_keeps_the_full_grid(name):
    default = _prop(name, "default")
    assert _layout(default) is None
    assert _layout(_prop(name, True)) is not None
    models = _models(name)
    assert _equal(_shoot(default, name, models), _shoot(_prop(name, False), name, models))


# Every 2-D equation with strip-held fields, on the strips when asked
# (EagerOptions(cpml_strips=True)).  Model values per name (a 2% random variation on each).
STRIP_EQUATIONS_2D = ("Acoustic", "Acoustic1st", "AcousticBorn", "AcousticVTI1st", "Biot", "DASMu", "DASZhao",
                      "Elastic", "ElasticBorn", "ElasticCurvilinear", "ElasticTTISG", "ElasticVRR", "ViscoElastic")
VALUES = {"vp": 2200, "vs": 1200, "rho": 2100, "vp0": 2200, "vs0": 1200, "epsilon": 0.08, "delta": 0.03,
          "gamma": 0.05, "theta": 0.25, "phi": 0.15, "Qp": 60, "Qs": 40, "mp": 0.01, "dvp": 20, "dvs": 10,
          "drho": 10, "Rp_x": 0.01, "Rp_z": 0.01, "Rs_x": 0.01, "Rs_z": 0.01}
# Dai, Vafidis & Kanasewich (1995), Table 1, layer 1, as in test_biot.py.
DAI1 = dict(P=2.0332e10, Q=0.0953e10, R=0.0331e10, N=0.684e10, rho11=2167.0, rho12=-83.0, rho22=191.0)


def test_strip_equations_2d_are_every_2d_equation_with_strip_held_fields():
    found = [n for n in list_equations("fd")
             if (eq := get_equation(n)(4, device="cpu", backend="torch")).ndim == 2
             and len(StripLayout(eq)) > len(eq.wavefields)]
    assert sorted(found) == sorted(STRIP_EQUATIONS_2D)


@pytest.mark.parametrize("name", STRIP_EQUATIONS_2D)
def test_strips_give_the_full_grid_numbers_2d(name):
    """Record, every model's gradient and the last fields, strips against the full grid,
    over 60 steps that carry the wave well into the absorbing layers."""
    eq_cls = get_equation(name)
    shape, nt, dt = (20, 24), 60, 1e-3
    g = torch.Generator().manual_seed(0)
    if name == "Biot":
        from sweep.equations.biot import biot_from_biot_constants
        coef = biot_from_biot_constants(**DAI1, b=0.0, phi=0.3)
        models = [float(c) * (1 + 0.02 * torch.rand(shape, generator=g)) for c in coef]
    else:
        models = [VALUES[s.name] * (1 + 0.02 * torch.rand(shape, generator=g)) for s in eq_cls.MODEL_SPECS]
    src = np.array([[shape[1] // 2, 4]])
    rec = np.array([[[ix, 4] for ix in range(1, shape[1] - 1, 3)]])
    runs = []
    for strips in (True, False):
        prop = PropTorch(eq_cls(4, device="cpu", backend="torch"), impl="eager", shape=shape, dh=10.0, dt=dt,
                         nt=nt, abcn=6, dev="cpu", use_ckpt=False,
                         eager_options=EagerOptions(use_compile=False, cpml_strips=strips, store_last_wavefield=True))
        assert (_layout(prop) is not None) == strips
        ms = [m.clone().requires_grad_(True) for m in models]
        record = prop(torch.from_numpy(ricker(nt, dt, fm=30.0, delay=0.03)), src, rec, models=ms)
        record.square().sum().backward()
        runs.append((record.detach(), [m.grad for m in ms],
                     [f.detach() for f in prop._backend_impl.last_wavefields], prop._backend_impl))
    (record, grads, last, impl), (record_f, grads_f, last_f, _) = runs
    assert torch.isfinite(record).all() and record.abs().max() > 0
    assert all(gr.abs().max() > 0 for gr in grads)
    assert torch.equal(record, record_f) and _equal(grads, grads_f) and _equal(last, last_f)
    # the strip-held memory fields carried something, so the strips did the work
    memory = [k for k, n in enumerate(impl.wavefield_names)
              if getattr(impl._wavefield_spec_index.get(n), "cpml_axis", "")]
    assert max(last[k].abs().max() for k in memory) > 0


def test_topography_keeps_the_full_grid_by_default():
    ny, nx = CASES["acoustic3d"][1][1:]
    hill = np.full((ny, nx), 3, dtype=np.int64)
    hill[ny // 3: 2 * ny // 3, nx // 3: 2 * nx // 3] = 1
    default = _prop("acoustic3d", "default", topography=hill)
    assert _layout(default) is None
    (record, grads) = _shoot(default, "acoustic3d", _models("acoustic3d"))
    assert record.abs().max() > 0 and torch.isfinite(grads[0]).all()
    with pytest.raises(NotImplementedError, match="topography"):
        _prop("acoustic3d", True, topography=hill)


@pytest.mark.parametrize("strips", ["default", True])
@pytest.mark.parametrize("name", ["acoustic3d", "elastic3d"])
def test_snapshots_hold_full_grid_fields(name, strips):
    """The strips are put back in place: every field of every snapshot, the CPML
    memories included, is the full-grid layout's."""
    prop = _prop(name, strips, use_ckpt=False)
    assert _layout(prop) is not None
    models = _models(name)
    (record, snaps), grads = _shoot(prop, name, models, return_wavefield=True, snapshot_interval=5)
    (record_f, snaps_f), grads_f = _shoot(_prop(name, False, use_ckpt=False), name, models,
                                          return_wavefield=True, snapshot_interval=5)
    assert snaps.shape == snaps_f.shape
    assert snaps.shape[1] == len(prop._backend_impl.wavefield_names)
    assert _equal((record, snaps, grads), (record_f, snaps_f, grads_f))
    # the memory fields are not all zero, so the strips were really put back
    memory = [k for k, n in enumerate(prop._backend_impl.wavefield_names)
              if getattr(prop._backend_impl._wavefield_spec_index.get(n), "cpml_axis", "")]
    assert snaps[:, memory].abs().max() > 0


@pytest.mark.parametrize("strips", ["default", True])
@pytest.mark.parametrize("name", ["acoustic3d", "elastic3d"])
def test_last_wavefields_hold_full_grid_fields(name, strips):
    prop = _prop(name, strips, store_last=True, use_ckpt=False)
    assert _layout(prop) is not None
    full = _prop(name, False, store_last=True, use_ckpt=False)
    models = _models(name)
    _shoot(prop, name, models)
    _shoot(full, name, models)
    last, last_f = prop._backend_impl.last_wavefields, full._backend_impl.last_wavefields
    assert len(last) == len(prop._backend_impl.wavefield_names)
    assert _equal(tuple(w.detach() for w in last), tuple(w.detach() for w in last_f))


@pytest.mark.parametrize("name, strips", [("AcousticBorn3D", "default"), ("AcousticBorn", True)])
def test_custom_gradient_reads_the_receiver_field(name, strips):
    """The custom-gradient forward steps full-grid fields: it must find the receiver
    field by name. AcousticBorn records sh1, which comes after the background CPML
    fields, so its position in the strip-held eager state is not its field index."""
    case = "acoustic3d" if name.endswith("3D") else "acoustic2d"
    runs = []
    for s in (strips, False):
        prop = _prop(case, s, eq=get_equation(name)(4, device="cpu"), use_ckpt=False)
        assert (_layout(prop) is not None) == (s is not False)
        prop._backend_impl.register_gradient("mp", lambda f, a, m, dt, dh: f["h1"] * a["h1"])
        runs.append(_shoot(prop, case, _models(case, n=2)))
    (record, grads), (record_f, grads_f) = runs
    assert record_f.abs().max() > 0
    assert torch.equal(record, record_f)
    assert torch.equal(grads[1], grads_f[1])
