"""Options that used to be accepted and then quietly ignored.

Three of them, all the same shape of bug: the call succeeds, the numbers look
plausible, and the thing you asked for never happened.

* ``memory='boundary'`` reached six of the twelve ``impl='c'`` forward paths.
  On the other six it was stored and never read, so a caller believed the run
  was reconstructing the wavefield from a boundary tape -- and was quietly
  paying full-tape memory instead.
* ``memory=Ckpt(...)`` (the old ``checkpoint_chunk``) only reached
  ``_run_batched``. The ``nshots == 1`` branch returned before that, so a
  single-shot call dropped it; it was refused there, and now runs the
  checkpointed rollout (``sweep.sem.ckpt_rollout``).
* ``SEMGraphGradRunner`` never looked at ``receiver_type``; it returned the
  equation's own field order, so a selection or a re-ordering was ignored.

None of these should be "fixed" by silently doing something else. The point is
that asking for what the backend cannot do must SAY so.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import Ckpt
from sweep.sem import (PropSEM, AcousticSEM3D, ElasticSEM2DDeformed,
                       ElasticSEM3D, build_axis_aligned_quad9)
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled SEM binding")

DH, NO = 100.0, 4


def _mesh2():
    return build_axis_aligned_quad9(nex=4, nez=4, x_max=400.0, z_max=400.0,
                                    n_order=NO, device="cuda")


def _mesh3():
    return build_hex8_mesh(nex=2, ney=2, nez=2, x_max=200.0, y_max=200.0,
                           z_max=200.0, n_order=NO, device="cuda")


def _full(mesh, v, ndim=2):
    shape = (mesh.n_elements,) + (mesh.ngll,) * ndim
    return torch.full(shape, v, device="cuda")


def _prop(eq, ndim=2, elastic=False):
    xz = [(100.0, 50.0, 50.0)] if ndim == 3 else [(200.0, 50.0)]
    rc = [(150.0, 50.0, 50.0)] if ndim == 3 else [(250.0, 50.0)]
    kw = {}
    if elastic:
        kw = dict(source_kinds=["force"],
                  source_payloads=[(0.0, 0.0, 1.0) if ndim == 3 else (0.0, 1.0)])
    return SEMCudaPropagator.build(eq, dt=1e-4, nt=4, source_xz=xz,
                                   receiver_xz=rc, pml_guard="off", **kw)


# ------------------- memory='boundary' on impl='c' ------------------------- #
@cuda
def test_boundary_saving_is_rejected_where_it_is_not_implemented():
    """The six paths that never wire the ``bs`` flag must refuse, not ignore."""
    m2, m3 = _mesh2(), _mesh3()
    bare_ac = AcousticPotentialSEM2D.build(m2, _full(m2, 2000.0), _full(m2, 1000.0),
                                           dirichlet=("top",))
    bare_el = ElasticSEM2DDeformed.build(m2, _full(m2, 2400.0), _full(m2, 1250.0),
                                         _full(m2, 2000.0))
    el3 = ElasticSEM3D.build(m3, _full(m3, 2400.0, 3), _full(m3, 1250.0, 3),
                             _full(m3, 2000.0, 3))
    for eq, elastic, ndim in ((bare_ac, False, 2), (bare_el, True, 2),
                              (el3, True, 3)):
        with pytest.raises(NotImplementedError, match="memory='boundary'"):
            _prop(eq, ndim=ndim, elastic=elastic)._assert_memory_supported("boundary")


@cuda
def test_boundary_saving_is_allowed_where_it_is_implemented():
    """The guard must not over-reject: the CPML 2-D pair and 3-D acoustic
    genuinely honour it, and the existing BS tests rely on that."""
    m2, m3 = _mesh2(), _mesh3()
    cpa = build_cpml(m2, dt=1e-4, sides=("left", "right", "bottom"),
                     pml_thickness_elements=2, c_max=2000.0, f_dominant=10.0)
    cpe = build_elastic_cpml(m2, dt=1e-4, sides=("left", "right", "bottom"),
                             pml_thickness_elements=2, c_max=2880.0, f_dominant=10.0)
    cases = [
        (AcousticPotentialSEM2D.build(m2, _full(m2, 2000.0), _full(m2, 1000.0),
                                      dirichlet=("top",), cpml=cpa), False, 2),
        (ElasticSEM2DDeformed.build(m2, _full(m2, 2400.0), _full(m2, 1250.0),
                                    _full(m2, 2000.0), cpml=cpe), True, 2),
        (AcousticSEM3D.build(m3, _full(m3, 2000.0, 3), dirichlet=("top",)), False, 3),
    ]
    for eq, elastic, ndim in cases:
        _prop(eq, ndim=ndim, elastic=elastic)._assert_memory_supported("boundary")


@cuda
def test_memory_none_is_never_rejected():
    m2 = _mesh2()
    eq = ElasticSEM2DDeformed.build(m2, _full(m2, 2400.0), _full(m2, 1250.0),
                                    _full(m2, 2000.0))
    p = _prop(eq, elastic=True)
    p._assert_memory_supported(None)
    p._assert_memory_supported("")


# ------------------------------- Ckpt ------------------------------------- #
def _shot(nz, nx, nt, dt, f0=8.0):
    t = np.arange(nt) * dt
    a = (np.pi * f0 * (t - 1.5 / f0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))
    rix = np.arange(4, nx - 4)
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    return w, rec


def _elastic(dt, **kw):
    return PropSEM(ElasticSEM2DDeformed, shape=(41, 41), dh=12.5, dt=dt, nt=60,
                   n_order=NO, free_surface=("top",), abcn=3 * NO,
                   f_dominant=8.0, dev="cpu", impl="eager", **kw)


def test_ckpt_reaches_a_single_shot(monkeypatch):
    """A single shot used to drop it silently (it only reached the vmapped
    path); now it runs the checkpointed rollout -- and does run it."""
    import sweep.sem.ckpt_rollout as CR
    calls = []
    real = CR.run_checkpointed
    monkeypatch.setattr(CR, "run_checkpointed", lambda *a, **k: calls.append(1) or real(*a, **k))
    nz = nx = 41
    dt = 0.30 * (0.172 * NO * 12.5) / 2880.0
    w, rec = _shot(nz, nx, 60, dt)
    models = [torch.full((nz, nx), v) for v in (2400.0, 1250.0, 2000.0)]
    with torch.no_grad():
        a = _elastic(dt, memory=Ckpt(chunks=20))(w, np.array([[nx // 2, 3]]), rec, models=models)
        b = _elastic(dt)(w, np.array([[nx // 2, 3]]), rec, models=models)
    assert calls == [1]
    rel = float((a - b).abs().max() / b.abs().max())
    assert rel < 1e-6, f"the checkpointed single shot differs by {rel:.2e}"


def test_ckpt_still_works_for_multiple_shots():
    """The rejection must be scoped to nshots == 1, not to the option."""
    nz = nx = 41
    dt = 0.30 * (0.172 * NO * 12.5) / 2880.0
    w, rec = _shot(nz, nx, 60, dt)
    models = [torch.full((nz, nx), v) for v in (2400.0, 1250.0, 2000.0)]
    src = np.array([[12, 3], [28, 3]])
    with torch.no_grad():
        a = _elastic(dt, memory=Ckpt(chunks=20))(w, src, rec, models=models)
        b = _elastic(dt)(w, src, rec, models=models)
    assert a.shape == b.shape
    # chunking is a memory strategy, not a different computation
    rel = float((a - b).abs().max() / b.abs().max())
    assert rel < 1e-6, f"chunked multi-shot differs by {rel:.2e}"


# ------------- an absorbing layer that was never actually built ------------ #
def test_abcn_on_an_equation_without_cpml_is_rejected():
    """``PropSEM(ElasticSEM3D, abcn=8)`` used to construct happily,
    pad the mesh for the layer, and then build the equation with NO C-PML at
    all -- a reflecting box wearing an absorbing layer's padding. The
    equation's ``build`` takes no ``cpml`` argument, so there was nowhere for
    it to go.
    """
    from sweep.sem import ElasticSEM3D
    with pytest.raises(NotImplementedError, match="no `cpml` argument"):
        PropSEM(ElasticSEM3D, shape=(17, 17, 17), dh=20.0, dt=1e-3, nt=4,
                n_order=4, free_surface=("top",), abcn=8, dev="cpu")
    # ... and abcn=0 is the honest way to ask for that box
    PropSEM(ElasticSEM3D, shape=(17, 17, 17), dh=20.0, dt=1e-3, nt=4,
            n_order=4, free_surface=("top",), abcn=0, dev="cpu")


def test_the_fluid_solid_form_is_drivable_through_propsem():
    """It used to be refused at construction ("takes keyword argument(s)
    ['cpml_ac', 'cpml_el'] that PropSEM does not supply") although it has a
    perfectly good absorbing layer -- just split into an acoustic and an
    elastic profile. The equation now declares cpml_builders and PropSEM
    constructs both from one abcn."""
    from sweep.sem import AcousticElasticSEM2D
    for abcn in (0, 16):
        PropSEM(AcousticElasticSEM2D, shape=(25, 25), dh=25.0, dt=1e-4,
                nt=4, n_order=4, free_surface=("top",), abcn=abcn, dev="cpu")


def test_fluid_solid_forward_and_gradients_through_propsem():
    """The whole point of the wiring: a marine water-over-rock model runs
    end-to-end and every model gets a gradient. Models are GLL-NATIVE because
    a sharp fluid-solid interface must fall on element boundaries -- grid
    interpolation smears the vs step into the interface element and
    DomainMask.from_vs rightly refuses the mixed element."""
    import sweep.sem as S
    NO_, DH_, F0_ = 4, 25.0, 8.0
    nz = nx = 41
    dt = 0.30 * (0.172 * NO_ * DH_) / (2400.0 * 1.2)
    nt = 200
    t = np.arange(nt) * dt
    a = (np.pi * F0_ * (t - 1.5 / F0_)) ** 2
    w = torch.tensor((((1 - 2 * a) * np.exp(-a)) * 1e6).astype(np.float32))
    p = S.PropSEM(S.AcousticElasticSEM2D, shape=(nz, nx), dh=DH_, dt=dt,
                  nt=nt, n_order=NO_, free_surface=("top",), abcn=3 * NO_,
                  f_dominant=F0_, dev="cpu", impl="eager")
    assert p.source_type == ["chi"]          # equation-declared marine default
    m = p.mesh
    sh = (m.n_elements, m.ngll, m.ngll)
    water = (m.z_e.amax(dim=(1, 2)) <= 300.0)[:, None, None]
    models = [torch.where(water, a_, b_).expand(sh).clone().requires_grad_(True)
              for a_, b_ in ((1500., 2400.), (0., 1250.), (1000., 2000.))]
    src = np.array([[nx // 2, 6]])
    rix = np.arange(14, nx - 14, 2)
    rec = np.stack([rix, np.full_like(rix, 24)], -1)[None]
    seis = p(w, src, rec, models=models)
    assert seis.shape[-1] == 3 and torch.isfinite(seis).all()
    assert float(seis.abs().max()) > 0, "zero record: the source was not injected"
    (seis ** 2).sum().backward()
    for nm, mm in zip(("vp", "vs", "rho"), models):
        assert mm.grad is not None and torch.isfinite(mm.grad).all(), nm
        assert float(mm.grad.abs().max()) > 0, f"zero gradient for {nm}"


def test_fluid_solid_non_default_source_type_is_refused():
    """The stamp components zip onto source fields FROM FIELD 0, so a force
    selection would land on (chi, u_x) -- a silently wrong source. Before this
    guard the default itself was mis-mapped and PropSEM injected a ZERO source:
    the run succeeded and every seismogram was exactly 0.0."""
    import sweep.sem as S
    with pytest.raises(NotImplementedError, match="mis-mapped"):
        PropSEM(S.AcousticElasticSEM2D, shape=(41, 41), dh=25.0, dt=1e-4,
                nt=8, n_order=4, free_surface=("top",), abcn=12,
                f_dominant=8.0, dev="cpu", source_type=["u_z"])


# ------------- options swallowed by the **kwargs catch-all ----------------- #
def _tiny(impl="eager", dev="cpu", **kw):
    return PropSEM(AcousticPotentialSEM2D, shape=(25, 25), dh=25.0,
                   dt=2e-4, nt=20, n_order=4, free_surface=("top",), abcn=8,
                   f_dominant=8.0, dev=dev, impl=impl, **kw)


def _tiny_shot():
    w = torch.zeros(20); w[3] = 1.0
    src = np.array([[12, 3]])
    rec = np.stack([np.arange(5, 20), np.full(15, 2)], -1)[None]
    return w, src, rec, [torch.full((25, 25), 2000.0)]


def test_unknown_forward_options_are_rejected():
    """**kwargs accepted ANYTHING -- including a misspelt memory=, which then
    ran the full tape while the caller believed boundary saving was on."""
    w, src, rec, models = _tiny_shot()
    p = _tiny()
    with pytest.raises(TypeError, match="unknown option"):
        p(w, src, rec, models=models, memroy="boundary")
    # the low-level name for snapshots is pointed at PropSEM's own
    with pytest.raises(TypeError, match="return_wavefield"):
        p(w, src, rec, models=models, record_field_every=10)
    # and the per-call memory knobs at the constructor, where they live now
    with pytest.raises(TypeError, match="constructor option"):
        p(w, src, rec, models=models, memory="boundary")
    with pytest.raises(TypeError, match="Ckpt"):
        p(w, src, rec, models=models, checkpoint_chunk=8)


@cuda
def test_eager_only_options_are_rejected_on_impl_c():
    """compile_step and Ckpt act on the eager tape; the CUDA backend has
    neither, and used to accept both silently."""
    w, src, rec, models = _tiny_shot()
    p = _tiny(impl="c", dev="cuda")
    w, models = w.cuda(), [models[0].cuda()]
    with pytest.raises(NotImplementedError, match="compile_step"):
        p(w, src, rec, models=models, compile_step=True)
    with pytest.raises(NotImplementedError, match="Ckpt"):
        _tiny(impl="c", dev="cuda", memory=Ckpt(chunks=8))
    assert torch.isfinite(p(w, src, rec, models=models)).all()
