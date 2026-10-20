"""``PropSEM(source_type=...)`` takes PropTorch's names on the elastics.

PropTorch drives a source by adding the wavelet to the named fields every
step: ``["vz"]`` is a vertical force, ``["sxx", "szz"]`` an explosion,
``["sxz"]`` a shear moment. A SEM elastic equation has none of those as a
source field -- its loads act on ``u`` / ``v`` -- so PropSEM translates them:
a velocity name into the force along its axis, a stress name into that
moment-tensor entry. Because FD's stress injection makes the wavelet the
moment RATE, the moment-tensor load is driven by its time integral; without
that the SEM record is one time derivative off FD's, the same source type
giving a different waveform.

Pinned here: the names map to the loads they claim (bitwise, against the
low-level propagator), the refusals, and -- against PropTorch itself -- that
the same ``source_type`` and wavelet give the same waveform.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import (PropSEM, SEMPropagator, ElasticSEM2DDeformed,
                       ElasticSEM3D, ElasticVSSEM2DDeformed, ElasticVSSEM3D)
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D

NZ = NX = 41
NO, DH, F0 = 4, 12.5, 8.0
VP, VS, RHO = 2400.0, 1250.0, 2000.0
DT = 0.30 * (0.172 * NO * DH) / (VP * 1.2)
NT = 160


def _prop(equation=ElasticVSSEM2DDeformed, **kw):
    return PropSEM(equation, shape=(NZ, NX), dh=DH, dt=DT, nt=NT, n_order=NO,
                   free_surface=("top",), abcn=4 * NO,
                   f_dominant=F0, dev="cpu", dtype=torch.float32,
                   impl="eager", **kw)


def _shot():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))
    src = np.array([[NX // 2, 12]], dtype=np.int64)
    rix = np.arange(6, NX - 6, 3)
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    models = [torch.full((NZ, NX), v) for v in (VP, VS, RHO)]
    return w, src, rec, models


def _low_level(p, models, src, rec, kind, payload, wavelet):
    """The same run through SEMPropagator, with the load spelt out."""
    eq = p._build_eq(models)
    sp = SEMPropagator.build(
        eq, dt=p.dt, nt=p.nt, source_xz=[p._coords(src[0])],
        receiver_xz=[p._coords(rec[0, j]) for j in range(rec.shape[1])],
        source_kinds=[kind], source_payloads=[payload],
        receiver_type=p._recv_base)
    return sp.forward(wavelet)


# --------------------------- the names map ---------------------------------- #
@pytest.mark.parametrize("equation,own", [(ElasticVSSEM2DDeformed, "v"),
                                          (ElasticSEM2DDeformed, "u")])
@pytest.mark.parametrize("axis", ["x", "z"])
def test_fd_velocity_name_is_the_force_on_that_axis(equation, own, axis):
    w, src, rec, models = _shot()
    with torch.no_grad():
        fd = _prop(equation, source_type=[f"v{axis}"])(w, src, rec, models=models)
        native = _prop(equation, source_type=[f"{own}_{axis}"])(
            w, src, rec, models=models)
    assert torch.equal(fd, native)


@pytest.mark.parametrize("names,payload", [(["sxx", "szz"], (1.0, 1.0, 0.0)),
                                           (["sxz"], (0.0, 0.0, 1.0)),
                                           (["szz"], (0.0, 1.0, 0.0))])
def test_stress_names_are_the_moment_tensor_driven_by_the_moment(names, payload):
    """The load is the moment tensor with a unit entry per name, and its time
    function is the running integral of the wavelet -- spelt out by hand
    through the low-level propagator, the two must agree bit for bit."""
    w, src, rec, models = _shot()
    p = _prop(source_type=names)
    with torch.no_grad():
        got = p(w, src, rec, models=models)
        ref = _low_level(p, models, src, rec, "moment_tensor", payload,
                         torch.cumsum(w, 0) * p.dt)
    assert torch.equal(got[0], ref)
    assert got.abs().max() > 0


@pytest.mark.parametrize("names,even", [(["sxx", "szz"], True),
                                        (["vz"], True),
                                        (["vx", "vz"], False)])
def test_explosion_is_symmetric_and_the_diagonal_force_is_not(names, even):
    """Under a flat surface a source on the centre line radiates v_z even and
    v_x odd about it -- an explosion and a vertical force do, the 45-deg force
    ``["vx", "vz"]`` (what the field-list reading used to make of a stress
    pair) does not.

    The source sits at an element CENTRE; the element-corner case, where the
    moment tensor needs every element at the point, is
    test_sem_moment_edge.py::test_explosion_on_a_corner_is_symmetric.
    """
    nx = 45                                    # 11 elements: centre 22 is one's middle
    p = PropSEM(ElasticVSSEM2DDeformed, shape=(NZ, nx), dh=DH, dt=DT, nt=NT,
                n_order=NO, free_surface=("top",),
                abcn=4 * NO, f_dominant=F0, dev="cpu", impl="eager",
                dtype=torch.float32, source_type=names)
    w, _, _, _ = _shot()
    src = np.array([[nx // 2, 14]])           # x 22, z 14: both element middles
    rix = nx // 2 + np.array([-12, -9, -6, -3, 3, 6, 9, 12])
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    models = [torch.full((NZ, nx), v) for v in (VP, VS, RHO)]
    with torch.no_grad():
        s = p(w, src, rec, models=models)
    vx, vz = s[0, :, :, 0], s[0, :, :, 1]
    vz_defect = float((vz - vz.flip(-1)).norm() / vz.norm())
    vx_defect = float((vx + vx.flip(-1)).norm() / vx.norm())
    if even:
        assert vz_defect < 1e-4 and vx_defect < 1e-4, (vz_defect, vx_defect)
    else:
        assert vz_defect > 0.3 and vx_defect > 0.3, (vz_defect, vx_defect)


def test_wavelet_gradient_flows_through_the_integral():
    w, src, rec, models = _shot()
    w = w.clone().requires_grad_(True)
    s = _prop(source_type=["sxx", "szz"])(w, src, rec, models=models)
    (s ** 2).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert float(w.grad.abs().max()) > 0


# ------------------------------ refusals ------------------------------------ #
@pytest.mark.parametrize("names,match", [
    (["vz", "sxx"], "mixes a force with a stress"),
    (["sxx", "sxx"], "duplicate"),
    (["v_z", "vz"], "same force component"),
    (["txx"], "Unknown"),
    (["vy"], "Unknown"),            # a 3-D name on a 2-D equation
    (["syy"], "Unknown"),
])
def test_bad_source_types_are_refused(names, match):
    with pytest.raises(ValueError, match=match):
        _prop(source_type=names)


def test_unknown_name_error_lists_the_fd_names():
    with pytest.raises(ValueError) as e:
        _prop(source_type=["txx"])
    msg = str(e.value)
    assert "v_x" in msg and "v_z" in msg
    assert "vx, vz" in msg and "sxx, szz, sxz" in msg


@pytest.mark.parametrize("equation,abcn", [(ElasticSEM3D, 0),
                                           (ElasticVSSEM3D, NO)])
def test_stress_names_are_2d_only(equation, abcn):
    with pytest.raises(NotImplementedError, match="2-D only"):
        PropSEM(equation, shape=(9, 9, 9), dh=DH, dt=DT, nt=NT,
                n_order=NO, free_surface=("top",),
                abcn=abcn, f_dominant=F0, dev="cpu", impl="eager",
                source_type=["sxx", "syy", "szz"])
    # ... while the FD force names work there too
    p = PropSEM(equation, shape=(9, 9, 9), dh=DH, dt=DT, nt=NT, n_order=NO,
                free_surface=("top",),
                abcn=abcn, f_dominant=F0, dev="cpu", impl="eager",
                source_type=["vy"])
    assert p._source_vector() == (0.0, 1.0, 0.0)


def test_source_vector_is_refused_on_a_stress_source():
    w, src, rec, models = _shot()
    p = _prop(source_type=["sxx", "szz"])
    with pytest.raises(ValueError, match="source_vector"):
        p(w, src, rec, models=models, source_vector=(0.0, 1.0))


def test_acoustic_equations_keep_their_own_vocabulary():
    with pytest.raises(ValueError, match="Unknown"):
        PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                n_order=NO, free_surface=("top",),
                abcn=4 * NO, f_dominant=F0, dev="cpu", impl="eager",
                source_type=["sxx"])


def test_dd_refuses_a_node_source_that_is_not_the_default():
    """The DD driver's node-id source is the equation's default unit force;
    a PropSEM built with another source_type must not be silently run as it.
    """
    from sweep.sem.dd import SEMModelParallel, StripTopology
    p = PropSEM(ElasticSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                n_order=NO, free_surface=("top",),
                abcn=4 * NO, f_dominant=F0, dev="cpu", impl="eager",
                source_type=["sxx", "szz"])
    assert not p._default_source()
    assert PropSEM(ElasticSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                   n_order=NO, free_surface=("top",),
                   abcn=4 * NO, f_dominant=F0, dev="cpu", impl="eager",
                   source_type=["vz"])._default_source()
    dd = SEMModelParallel.__new__(SEMModelParallel)   # no CUDA needed for this
    dd.prop = p
    from sweep.sem.dd import _FAMILIES
    dd.fam = _FAMILIES[ElasticSEM2DDeformed]
    eq = p._build_eq([torch.full((NZ, NX), v) for v in (VP, VS, RHO)])
    with pytest.raises(NotImplementedError, match="source_type"):
        dd._source_patterns(eq, 7, torch.device("cpu"))


# ------------------------ the same waveform as FD --------------------------- #
FZ = FX = 81
FDH, FDT, FNT, FF0 = 10.0, 1e-3, 600, 10.0
FVP, FVS, FRHO = 2000.0, 1155.0, 2000.0


def _fd_wavelet():
    t = np.arange(FNT) * FDT
    a = (np.pi * FF0 * (t - 0.12)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))


def _fd_geometry():
    src = np.array([[FX // 2, FZ // 2]], dtype=np.int64)
    rix = np.arange(10, FX - 10, 4)
    rec = np.stack([rix, np.full_like(rix, 10)], -1)[None]
    return src, rec


def _fd_vz(source_type, wav):
    from sweep.equations import Elastic
    from sweep.propagator.torch import PropTorch
    eq = Elastic(spatial_order=4, device="cpu", backend="torch")
    prop = PropTorch(eq, shape=(FZ, FX), abcn=40, dh=FDH, dt=FDT,
                     free_surface=False, impl="eager", use_ckpt=False,
                     eager_options={"use_compile": False},
                     source_type=source_type, receiver_type=["vz"])
    src, rec = _fd_geometry()
    models = [torch.full((FZ, FX), v) for v in (FVP, FVS, FRHO)]
    with torch.no_grad():
        out = prop(wav, torch.tensor(src), torch.tensor(rec), models=models)
    return out[0, :, :, 0].double()


def _sem_vz(source_type, wav):
    p = PropSEM(ElasticVSSEM2DDeformed, shape=(FZ, FX), dh=FDH, dt=FDT,
                nt=FNT, n_order=NO, free_surface=False,
                abcn=6 * NO, f_dominant=FF0, dev="cpu", impl="eager",
                dtype=torch.float32, source_type=source_type,
                receiver_type=["v_z"])
    src, rec = _fd_geometry()
    models = [torch.full((FZ, FX), v) for v in (FVP, FVS, FRHO)]
    with torch.no_grad():
        out = p(wav, src, rec, models=models)
    return out[0, :, :, 0].double()


def _corr(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b) / float(a.norm() * b.norm())


@pytest.mark.parametrize("source_type", [["sxx", "szz"], ["vz"], ["sxz"]])
def test_same_source_type_same_waveform_as_proptorch(source_type):
    w = _fd_wavelet()
    c = _corr(_fd_vz(source_type, w), _sem_vz(source_type, w))
    assert c > 0.9, f"{source_type}: FD vs SEM v_z correlate {c:.3f}"


def test_the_waveform_check_can_tell_a_derivative():
    """Without the integral the SEM explosion would be driven by the wavelet
    itself -- reproduce that by handing it the wavelet's derivative, and the
    check above must fail on it."""
    w = _fd_wavelet()
    dw = torch.diff(w, prepend=w[:1] * 0) / FDT
    c = _corr(_fd_vz(["sxx", "szz"], w), _sem_vz(["sxx", "szz"], dw))
    assert abs(c) < 0.6, f"a derivative-off record still correlates {c:.3f}"


# ------------------------------ impl='c' ------------------------------------ #
def _c_ready():
    from sweep.sem.prop_c import sem_c_available
    return torch.cuda.is_available() and sem_c_available()


@pytest.mark.skipif(not _c_ready(), reason="needs CUDA + the SEM core")
@pytest.mark.parametrize("names", [["sxx", "szz"], ["sxz"], ["vx"]])
def test_impl_c_carries_the_same_source(names):
    """The CUDA kernels read the assembled per-field source patterns, so a
    moment tensor reaches them like a force does; the record matches eager to
    the atomics' noise floor."""
    w, src, rec, models = _shot()
    out = {}
    for impl in ("eager", "c"):
        p = PropSEM(ElasticSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                    n_order=NO, free_surface=("top",),
                    abcn=4 * NO, f_dominant=F0, dev="cuda",
                    dtype=torch.float32, impl=impl, source_type=names)
        with torch.no_grad():
            out[impl] = p(w.cuda(), src, rec,
                          models=[m.cuda() for m in models]).double()
    scale = float(out["eager"].abs().max())
    err = float((out["c"] - out["eager"]).abs().max())
    assert scale > 0 and err <= 1e-4 * scale, f"{names}: c vs eager {err / scale:.2e}"
