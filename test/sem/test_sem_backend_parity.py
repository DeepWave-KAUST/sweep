"""Same solver call, same arguments -> same numbers on eager and impl='c'.

Both the requirement and its trap: neither backend is bit-reproducible on CUDA
(both scatter with atomics), so running the SAME configuration twice already
differs. A fixed tolerance would therefore be a guess. Every assertion here is
against the run-to-run FLOOR measured in the same session, so the bar tracks
the hardware instead of a number someone picked once.

Two things this pins that were broken:

* ``PropSEM(impl='c')`` used to route only three equations. Five more HAD a
  CUDA kernel that the solver could not reach -- callers had to drop to
  ``SEMCudaPropagator`` and assemble the call by hand, so the same ``PropSEM``
  call meant different things on the two backends.
* ``_forward_c`` hard-coded a scalar source, which is why it could only serve
  acoustic equations. Both backends now derive (kind, payload) the same way.

Harness note, learned the hard way: the source amplitude is a parameter of the
TEST, not physics. With a unit source the elastic seismograms come out ~1e-13,
so a squared-error loss is ~1e-26 and its gradient ~1e-27 -- fp32 round-off
near underflow, where a first version of this comparison reported a 60%
"gradient disagreement" that was entirely an artefact. Scale the source so the
fields are O(1), and compute the metrics in float64.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.sem.prop_c import C_SUPPORTED, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NT = 4, 25.0, 8.0, 100
VP, VS, RHO = 2400.0, 1250.0, 2000.0
DT = 0.30 * (0.172 * NO * DH) / (VP * 1.2)      # GLL-limited, not the FD CFL
SRC_AMP = 1e12                                   # see the module docstring
# How far above the measured run-to-run floor a backend difference may sit
# before it counts as a real disagreement. Fused CUDA kernels re-associate the
# sums, so a few times the floor is expected; an order of magnitude is not.
FLOOR_FACTOR = 25.0
ABS_CEILING = 1e-4          # a backstop for the case where the floor is ~0


def _grid(ndim):
    return (25, 25) if ndim == 2 else (17, 17, 17)


def _acq(ndim, shape):
    if ndim == 2:
        _, nx = shape
        src = np.array([[nx // 2, 3]], dtype=np.int64)
        rix = np.arange(5, nx - 5)
        rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    else:
        _, ny, nx = shape
        src = np.array([[nx // 2, ny // 2, 3]], dtype=np.int64)
        rix = np.arange(4, nx - 4)
        rec = np.stack([rix, np.full_like(rix, ny // 2),
                        np.full_like(rix, 2)], -1)[None]
    return src, rec


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    w = ((1 - 2 * a) * np.exp(-a)) * SRC_AMP
    return torch.tensor(w.astype(np.float32), device="cuda")


def _is_acoustic(cls):
    n = cls.__name__
    return "Acoustic" in n and "Elastic" not in n


def _models(cls, shape, grad):
    vals = (VP,) if _is_acoustic(cls) else (VP, VS, RHO)
    out = [torch.full(shape, v, device="cuda") for v in vals]
    return [m.clone().requires_grad_(True) for m in out] if grad else out


_ABCN: dict = {}          # equation -> the absorbing elements it can carry


def _prop(cls, ndim, impl, abcn=None):
    if abcn is None:
        abcn = _ABCN.get(cls, 3)
    return S.PropSEM(cls, shape=_grid(ndim), dh=DH, dt=DT, nt=NT, n_order=NO,
                     free_surface=("top",),
                     abcn=abcn * NO, f_dominant=F0, dev="cuda", impl=impl)


def _run(cls, ndim, impl, grad):
    shape = _grid(ndim)
    src, rec = _acq(ndim, shape)
    models = _models(cls, shape, grad)
    p = _prop(cls, ndim, impl)
    if not grad:
        with torch.no_grad():
            return p(_wav(), src, rec, models=models), None
    seis = p(_wav(), src, rec, models=models)
    # scale-free loss, so the gradient magnitude does not depend on the equation
    (seis ** 2).sum().div(seis.detach().pow(2).sum().clamp_min(1e-30)).backward()
    return seis.detach(), [m.grad.clone() for m in models]


def _rel(a, b, scale=None):
    """Relative difference, normalised by ``scale`` (default: a's own peak).

    Pass the FULL record's peak when comparing one component: a vertical force
    with receivers along x leaves u_y at ~3e-08 of u_z by symmetry, and
    dividing by that structural zero turns 1-ulp noise into "1.9e+00". A
    component that is 1e-8 of the record contributes 1e-8 to any misfit, so the
    record scale is the meaningful denominator.
    """
    a, b = a.double(), b.double()
    d = float(a.abs().max()) if scale is None else float(scale)
    return float((a - b).abs().max() / d) if d else 0.0


# Only the equations PropSEM can actually route to impl='c'. Derived from
# C_SUPPORTED so a newly wired equation is covered without editing this list.
def _routable():
    """Equations PropSEM can drive on impl='c'.

    Tries a C-PML first, then a bare box: the 3-D displacement elastic
    equations model no C-PML at all, so demanding one would drop them from the
    parity matrix for a reason unrelated to whether the backends agree.
    """
    out = []
    for cls in C_SUPPORTED:
        ndim = 3 if "3D" in cls.__name__ else 2
        for abcn in (3, 0):
            try:
                _prop(cls, ndim, "c", abcn=abcn)
            except Exception:
                continue
            _ABCN[cls] = abcn
            out.append(cls)
            break
    return out


ROUTABLE = _routable()
IDS = [c.__name__ for c in ROUTABLE]


def test_the_cuda_gate_matches_the_cuda_dispatch():
    """C_SUPPORTED is read by PropSEM to decide impl='c'; it must name exactly
    the equations the dispatch routes.

    It used to be a second hand-written tuple, checked here by regex-scraping
    `_forward_components` for `isinstance(eq, X)`. Both are gone: the dispatch
    is `_C_ROUTES` and C_SUPPORTED is derived from it, so this now states the
    derivation rather than policing two lists."""
    from sweep.sem.prop_c import _C_ROUTES
    assert tuple(kind for kind, _, _ in _C_ROUTES) == tuple(C_SUPPORTED)


def test_every_cuda_route_names_a_method_that_exists():
    """What a dispatch TABLE can get wrong that an isinstance chain could not:
    a method name that is a string, so a typo or a rename survives import and
    fails only when that one equation is run. Both entries of every row, and
    the order (most derived first) that keeps a deformed equation off its flat
    twin's row."""
    from sweep.sem.prop_c import SEMCudaPropagator, _C_ROUTES
    for kind, with_cpml, bare in _C_ROUTES:
        for name in (with_cpml, bare):
            if name is not None:
                assert callable(getattr(SEMCudaPropagator, name, None)), (
                    f"{kind.__name__} routes to {name!r}, which is not a "
                    "method of SEMCudaPropagator")
    seen = []
    for kind, _, _ in _C_ROUTES:
        assert not any(issubclass(kind, e) for e in seen), (
            f"{kind.__name__} is a subclass of an EARLIER row, so it can "
            "never be reached; move it up")
        seen.append(kind)


def test_more_than_the_three_acoustic_equations_are_routable():
    """The regression this file exists for: five equations had a CUDA kernel
    the solver refused to use."""
    assert len(ROUTABLE) >= 7, [c.__name__ for c in ROUTABLE]
    names = {c.__name__ for c in ROUTABLE}
    assert "ElasticSEM2DDeformed" in names
    assert "ElasticSEM3D" in names


@pytest.mark.parametrize("cls", ROUTABLE, ids=IDS)
def test_forward_agrees_with_eager(cls):
    ndim = 3 if "3D" in cls.__name__ else 2
    c1, _ = _run(cls, ndim, "c", grad=False)
    c2, _ = _run(cls, ndim, "c", grad=False)      # run-to-run floor
    e, _ = _run(cls, ndim, "eager", grad=False)
    floor = _rel(c1, c2)
    got = _rel(c1, e)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"{cls.__name__}: eager vs c forward {got:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("cls", ROUTABLE, ids=IDS)
def test_gradients_agree_with_eager(cls):
    ndim = 3 if "3D" in cls.__name__ else 2
    _, g1 = _run(cls, ndim, "c", grad=True)
    _, g2 = _run(cls, ndim, "c", grad=True)       # run-to-run floor
    _, ge = _run(cls, ndim, "eager", grad=True)
    names = ("vp",) if _is_acoustic(cls) else ("vp", "vs", "rho")
    for nm, a, b, e in zip(names, g1, g2, ge):
        floor, got = _rel(a, b), _rel(a, e)
        assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"{cls.__name__} d/d{nm}: eager vs c {got:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("cls", ROUTABLE, ids=IDS)
def test_both_backends_produce_the_same_shape(cls):
    """Shape parity is separate from value parity, and cheaper to break."""
    ndim = 3 if "3D" in cls.__name__ else 2
    c, _ = _run(cls, ndim, "c", grad=False)
    e, _ = _run(cls, ndim, "eager", grad=False)
    assert c.shape == e.shape, f"{cls.__name__}: c {c.shape} vs eager {e.shape}"
    # ncomp is the equation's receiver-capable field count, NOT ndim: the
    # fluid-solid form records (chi, u_x, u_z) = 3 components in 2-D. This
    # assert used ndim and broke the day that equation became routable.
    ncomp = sum(f.supports_receiver for f in cls.field_specs)
    assert c.shape[-1] == ncomp


_MULTI = [c for c in ROUTABLE if not _is_acoustic(c)]


@pytest.mark.parametrize("cls", _MULTI, ids=[c.__name__ for c in _MULTI])
def test_receiver_type_selection_agrees_between_backends(cls):
    """The DEFAULT receiver_type is "every component in field order", so a
    backend that ignores the argument entirely still matches on the default --
    which is exactly how this went unnoticed. Ask for a non-default order.
    """
    ndim = 3 if "3D" in cls.__name__ else 2
    shape = _grid(ndim)
    src, rec = _acq(ndim, shape)
    names = [s.name for s in cls.field_specs if s.supports_receiver]
    reversed_names = list(reversed(names))
    assert reversed_names != names, cls.__name__

    def run(impl, rtype):
        abcn = _ABCN.get(cls, 3)      # 0 for the equations that model no C-PML
        p = S.PropSEM(cls, shape=shape, dh=DH, dt=DT, nt=NT, n_order=NO,
                      free_surface=("top",),
                      abcn=abcn * NO, f_dominant=F0, dev="cuda", impl=impl,
                      receiver_type=rtype)
        with torch.no_grad():
            return p(_wav(), src, rec, models=_models(cls, shape, False))

    c_fwd, c_rev = run("c", names), run("c", reversed_names)
    e_rev = run("eager", reversed_names)
    scale = float(c_fwd.abs().max())          # see _rel: components can be ~0
    floor = _rel(c_fwd, run("c", names), scale)
    # the c backend must honour the order ...
    for k, nm in enumerate(reversed_names):
        j = names.index(nm)
        got = _rel(c_fwd[..., j], c_rev[..., k], scale)
        assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"{cls.__name__}: c ignored receiver_type for {nm} ({got:.2e})")
    # ... and agree with eager under the same non-default order
    got = _rel(c_rev, e_rev, scale)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"{cls.__name__}: reversed receiver_type, eager vs c {got:.2e}")


def test_the_floor_is_actually_nonzero():
    """If two identical CUDA runs ever became bit-identical, every assertion
    above would silently fall back to ABS_CEILING -- a much weaker bar. Say so
    rather than let the suite quietly loosen."""
    cls = ROUTABLE[0]
    ndim = 3 if "3D" in cls.__name__ else 2
    a, _ = _run(cls, ndim, "c", grad=False)
    b, _ = _run(cls, ndim, "c", grad=False)
    if _rel(a, b) == 0.0:
        pytest.skip("CUDA became run-to-run deterministic; the floor-based "
                    "bar has degenerated to the absolute ceiling")
