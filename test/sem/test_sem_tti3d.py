"""ElasticVTISEM3D / ElasticTTISEM3D (+ Deformed): the 3-D velocity-stress SEM in a
transversely isotropic medium.

Zero anisotropy is the isotropic class (Hex8 and Hex27); a TTI medium with no tilt is the VTI
one bit for bit; an axis tilted by 90 degrees toward x or toward y is the VTI problem turned by
that rotation, which the cubic mesh maps onto itself -- the records are the turned VTI records
to round-off, a check of the Bond rotation and of every one of the 21 stiffnesses in the
step; the stiffnesses are the FD ``ElasticTTISG3D``'s; the gradient in the tilt at zero tilt is
the derivative of the records; boundary saving (a bare box) gives the full tape's gradients; a
tilted medium in the C-PML dies away instead of growing; a flat Hex27 mesh gives the Hex8
records; all four classes are on the CUDA dispatch (impl='c' against eager:
test_sem_c_tti3d.py).
"""
import math

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, EagerOptions, Full
from sweep.sem import (ElasticTTISEM3D, ElasticTTISEM3DDeformed, ElasticVSSEM3D,
                       ElasticVSSEM3DDeformed, ElasticVTISEM3D, ElasticVTISEM3DDeformed, PropSEM)
from sweep.sem.constitutive import thomsen_to_cij3d

F64 = torch.float64
SHAPE, DT, NT = (9, 13, 17), 5e-4, 120
VP, VS, RHO = 2400.0, 1300.0, 2000.0
ANISO = (0.2, 0.1, 0.15)                       # epsilon, delta, gamma


def _prop(equation, shape=SHAPE, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), "abcn": 8,
          "free_surface": True, "dh": 10.0, "f_dominant": 20.0, "source_type": ["vz"],
          "receiver_type": ["vx", "vy", "vz"], **kw}
    return PropSEM(equation, shape=shape, dt=DT, n_order=4, dev="cpu", dtype=F64,
                   impl="eager", **kw)


def _wavelet(nt=NT, f0=20.0, delay=0.05):
    a = (math.pi * f0 * (torch.arange(nt, dtype=F64) * DT - delay)) ** 2
    return (1 - 2 * a) * torch.exp(-a)


def _models(shape=SHAPE, aniso=ANISO, tilt=None, grad=False):
    full = lambda v: torch.full(shape, float(v), dtype=F64)
    z = torch.linspace(0.0, 1.0, shape[0], dtype=F64)[:, None, None].expand(shape)
    ms = [VP + 200.0 * z, VS + 100.0 * z, full(RHO)] + [full(a) for a in aniso]
    if tilt is not None:
        ms += [full(t) for t in tilt]
    return [m.clone().requires_grad_(grad) for m in ms]


def _call(prop, models, src=((8.0, 6.0, 4.0),), rec=None):
    if rec is None:
        rx = np.arange(2.0, 15.0, 4.0)
        rec = np.stack([rx, np.full_like(rx, 6.0), np.full_like(rx, 2.0)], 1)
    return prop(_wavelet(), np.array(src), np.asarray(rec)[None], models=models)


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("aniso, iso", [(ElasticVTISEM3D, ElasticVSSEM3D),
                                        (ElasticVTISEM3DDeformed, ElasticVSSEM3DDeformed),
                                        (ElasticTTISEM3D, ElasticVSSEM3D)],
                         ids=["vti_hex8", "vti_hex27", "tti_hex8"])
def test_zero_anisotropy_is_the_isotropic_class(aniso, iso):
    ms = _models(aniso=(0.0, 0.0, 0.0), tilt=(0.3, 0.7) if aniso is ElasticTTISEM3D else None)
    ref = _call(_prop(iso), ms[:3])
    out = _call(_prop(aniso), ms)
    assert _rel(out, ref) < 1e-12


def test_no_tilt_is_the_vti_class_bit_for_bit():
    ms = _models(tilt=(0.0, 0.7))
    ref = _call(_prop(ElasticVTISEM3D), ms[:6])
    assert torch.equal(_call(_prop(ElasticTTISEM3D), ms), ref)


@pytest.mark.parametrize("phi", [0.0, math.pi / 2], ids=["axis_x", "axis_y"])
def test_a_90_degree_tilt_is_the_turned_vti_problem(phi):
    """The rotation R(pi/2, phi) of the symmetry axis turns the cube (all six sides free) onto
    itself: the TTI records at the turned stations are the turned VTI records."""
    n = 17
    L = n - 1.0
    shape = (n, n, n)
    full = lambda v: torch.full(shape, float(v), dtype=F64)
    vti = [full(VP), full(VS), full(RHO)] + [full(a) for a in ANISO]
    s = np.array([6.3, 7.9, 5.2])
    rec = np.array([[3.1, 9.4, 11.6], [12.2, 4.4, 8.0], [8.0, 8.0, 2.5], [14.3, 1.7, 13.9]])
    if phi == 0.0:      # R p = (z, y, -x): the axis along x
        turn = lambda p: np.stack([p[..., 2], p[..., 1], L - p[..., 0]], -1)
        force, comp = "vx", lambda v: torch.stack([v[..., 2], v[..., 1], -v[..., 0]], -1)
    else:               # R p = (-y, z, -x): the axis along y
        turn = lambda p: np.stack([L - p[..., 1], p[..., 2], L - p[..., 0]], -1)
        force, comp = "vy", lambda v: torch.stack([-v[..., 1], v[..., 2], -v[..., 0]], -1)
    kw = dict(shape=shape, abcn=0, free_surface=False)
    ref = _call(_prop(ElasticVTISEM3D, **kw), vti, src=s[None], rec=rec)
    tti = vti + [full(math.pi / 2), full(phi)]
    out = _call(_prop(ElasticTTISEM3D, source_type=[force], **kw), tti, src=turn(s)[None],
                rec=turn(rec))
    assert float(ref.abs().max()) > 0.0
    assert _rel(out, comp(ref)) < 1e-11


def test_the_stiffnesses_are_the_fd_classes():
    from sweep.equations.elastic_tti import ElasticTTI
    from sweep.equations.elastic_tti_sg3d import STIFFNESS_KEYS_3D, ElasticTTISG3D
    t = lambda v: torch.tensor([v], dtype=F64)
    vp, vs, rho, eps, dl, gm, th, ph = (t(v) for v in (3000.0, 1500.0, 2400.0, 0.25, 0.08,
                                                       0.12, 0.4, 1.1))
    ours = thomsen_to_cij3d(vp, vs, eps, dl, gm, rho, th, ph)
    c33, c44 = rho * vp ** 2, rho * vs ** 2
    fd = ElasticTTISG3D._bond_rotate_3d_torch(
        c33 * (1 + 2 * eps), ElasticTTI._compute_C13_from_delta(c33, c44, dl), c33, c44,
        c44 * (1 + 2 * gm), th, ph)
    for k, c in zip(STIFFNESS_KEYS_3D, ours):
        torch.testing.assert_close(c, fd[k], rtol=1e-14, atol=1e-14 * float(c33))


def test_the_gradient_in_the_tilt_at_zero_tilt_is_the_derivative():
    """At theta = 0 the stiffnesses are the VTI ones exactly, but d/d theta is the
    rotation's: a directional derivative against central differences."""
    ms = _models(tilt=(0.0, 0.6), grad=True)
    p = _prop(ElasticTTISEM3D)
    # off every mirror plane of the source: a symmetric spread makes the loss even in the
    # tilt, its derivative at zero tilt zero
    src = ((7.3, 5.2, 4.1),)
    rec = ((2.2, 6.7, 2.0), (10.9, 3.1, 3.3), (13.4, 9.6, 1.7))
    loss = lambda models: _call(p, models, src=src, rec=rec).square().sum()
    loss(ms).backward()
    d = torch.zeros(SHAPE, dtype=F64)
    d[2:-2, 2:-2, 2:-2] = 1.0                  # off the outermost layers
    analytic = float((ms[6].grad * d).sum())
    assert analytic != 0.0
    errs = []
    for h in (1e-3, 1e-4, 1e-5):
        plus = [m.detach().clone() for m in ms]
        minus = [m.detach().clone() for m in ms]
        plus[6] += h * d
        minus[6] -= h * d
        with torch.no_grad():
            fd = float(loss(plus) - loss(minus)) / (2 * h)
        errs.append(abs(fd - analytic) / abs(analytic))
    assert min(errs) < 1e-6, errs


def test_boundary_saving_gives_the_full_tape():
    """The reverse step runs the stiffnesses as the model leaves (a bare box: with the C-PML
    the parent's own boundary saving is not yet exact)."""
    def run(memory):
        ms = _models(tilt=(0.5, 0.8), grad=True)
        out = _call(_prop(ElasticTTISEM3D, memory=memory, abcn=0, free_surface=False), ms)
        out.square().sum().backward()
        return out.detach(), [m.grad for m in ms]
    r0, g0 = run(Full())
    r1, g1 = run(BoundarySaving())
    assert _rel(r1, r0) < 1e-12
    for a, b in zip(g1, g0):
        assert _rel(a, b) < 1e-10


def test_a_flat_hex27_mesh_gives_the_hex8_records():
    ms = _models(tilt=(0.5, 0.8))
    ref = _call(_prop(ElasticTTISEM3D), ms)
    out = _call(_prop(ElasticTTISEM3DDeformed), ms)
    assert _rel(out, ref) < 1e-12


@pytest.mark.slow                  # ~1 min on 8 CPU threads
def test_a_tilted_medium_decays_in_the_cpml():
    """A tilted medium in the C-PML with a free top, over 2 s: the late records are smaller
    than those a second earlier (in a box this small the 20 Hz waves outlast the layer's
    absorption, as the isotropic ones do, so it is the trend that is checked)."""
    nt = 4000
    ms = _models(aniso=(0.25, 0.05, 0.1), tilt=(0.5, 0.8))
    with torch.no_grad():
        p = _prop(ElasticTTISEM3D)
        out = p(torch.cat([_wavelet(), torch.zeros(nt - NT, dtype=F64)]),
                np.array([[8.0, 6.0, 4.0]]), np.array([[[4.0, 6.0, 2.0], [12.0, 3.0, 6.0]]]),
                models=ms)[0]
    mid = float(out[1000:2000].abs().max())
    late = float(out[-1000:].abs().max())
    assert torch.isfinite(out).all() and late < 0.7 * mid, late / mid


@pytest.mark.parametrize("equation", [ElasticVTISEM3D, ElasticTTISEM3D,
                                      ElasticVTISEM3DDeformed, ElasticTTISEM3DDeformed])
def test_on_the_cuda_dispatch(equation):
    """Each class has its own row (its stiffnesses, not the parent's Lame set):
    an exact entry of C_SUPPORTED, and the row it takes is its own."""
    from sweep.sem.prop_c import C_SUPPORTED, _C_ROUTES
    assert equation in C_SUPPORTED
    first = next(kind for kind, _, _ in _C_ROUTES if issubclass(equation, kind))
    assert first is equation
