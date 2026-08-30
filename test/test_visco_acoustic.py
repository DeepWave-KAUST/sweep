"""ViscoAcoustic: revival regression tests (torch backend).

The equation sat un-exercised for a long time and every test here pins down a
bug that was actually hit while bringing it back:

* the class was not exported from ``sweep.equations`` at all (the import was
  commented out because the module used to ``import jax, torch`` at the top);
* the module had to import in a torch-only *and* a jax-only environment;
* ``self.k`` was sized for the un-padded grid and mismatched the CPML-padded
  wavefield; rebuilding it inside ``func`` behind a side cache then broke the
  *second* ``forward()`` on one instance, because ``init_abc`` -- which runs
  per forward -- reset ``self.k`` while the side cache still claimed a hit;
* ``free_surface=True`` was silently ignored (``func`` never applied it).

Both toggles off must reduce bit-exactly to ``Acoustic``, which is the anchor
the rest of the suite leans on.  The jax path lives in
``test_visco_acoustic_jax.py``.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "sweep").exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

NZ, NX = 40, 48
DH, DT, NT, ABCN = 10.0, 1e-3, 80, 15
VP, Q_VAL, F0 = 2000.0, 30.0, 15.0

# skipif rather than a module-level importorskip: the two import-surface tests
# below are backend-agnostic and must still run in a jax-only environment.
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch-free env
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


def _ricker(nt, dt, f0):
    t = np.arange(nt) * dt - 1.0 / f0
    a = (np.pi * f0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def _geom():
    # sweep coordinates are (x, z)
    w = _ricker(NT, DT, F0)
    src = np.array([[NX // 2, 4]], dtype=np.int64)
    rx = np.arange(4, NX - 4, 4, dtype=np.int64)
    rec = np.stack([rx, np.full(rx.size, 2, np.int64)], -1)[None]
    return w, src, rec


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# --------------------------------------------------------------------------
# import surface (backend agnostic)
# --------------------------------------------------------------------------

def test_exported_from_sweep_equations():
    """Regression: the export in equations/__init__.py was commented out, so a
    working equation was unreachable through the public API."""
    from sweep.equations import ViscoAcoustic

    assert ViscoAcoustic.__name__ == "ViscoAcoustic"


def test_no_top_level_backend_imports():
    """Regression: the module used to ``import jax, torch`` at the top, which
    crashed ``import sweep.equations`` in a single-backend environment."""
    import sweep.equations.visco_acoustic as m

    for line in Path(m.__file__).read_text().splitlines():
        if line.startswith(("import ", "from ")):
            assert "torch" not in line, f"top-level torch import: {line!r}"
            assert "jax" not in line, f"top-level jax import: {line!r}"


# --------------------------------------------------------------------------
# torch backend
# --------------------------------------------------------------------------

def _build(phase_shift=True, amplitude_damping=True, free_surface=False):
    from sweep.equations import ViscoAcoustic
    from sweep.propagator.torch import PropTorch

    eq = ViscoAcoustic(spatial_order=4, backend="torch", device="cpu",
                       phase_shift=phase_shift, amplitude_damping=amplitude_damping)
    prop = PropTorch(eq, shape=(NZ, NX), dh=DH, dt=DT, dev=torch.device("cpu"),
                     source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
                     impl="eager", use_ckpt=False, free_surface=free_surface)
    return eq, prop


def _models(vp=VP):
    return [torch.full((NZ, NX), float(vp)),
            torch.full((NZ, NX), Q_VAL),
            torch.full((NZ, NX), 2 * np.pi * F0)]


def _run(prop, models=None):
    w, src, rec = _geom()
    out = prop.forward(w, src, rec, models=models if models is not None else _models())
    d = out[0] if isinstance(out, (tuple, list)) else out
    return d.detach().cpu().numpy()


def _acoustic_run(free_surface=False):
    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch

    ea = Acoustic(spatial_order=4, backend="torch", device="cpu")
    pa = PropTorch(ea, shape=(NZ, NX), dh=DH, dt=DT, dev=torch.device("cpu"),
                   source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
                   impl="eager", use_ckpt=False, free_surface=free_surface)
    return _run(pa, models=[torch.full((NZ, NX), VP)])


@requires_torch
def test_forward_runs_and_is_finite():
    _, prop = _build()
    d = _run(prop)
    assert d.shape == (1, NT, _geom()[2].shape[1], 1)
    assert np.isfinite(d).all()


@requires_torch
def test_repeated_forward_on_one_instance():
    """Regression: forward #2 crashed with a wavenumber/wavefield shape
    mismatch.  ``init_abc`` runs inside every ``forward`` and reset ``self.k``
    to the un-haloed shape, while a side cache still reported a hit and skipped
    the rebuild.  An FWI loop died on its second forward."""
    _, prop = _build()
    models = _models()
    first = _run(prop, models)
    for i in (2, 3):
        d = _run(prop, models)
        assert np.isfinite(d).all(), f"forward #{i} produced non-finite output"
        assert _rel_l2(d, first) == 0.0, f"forward #{i} drifted from #1"


@requires_torch
def test_wavenumber_grid_matches_padded_wavefield():
    """``k`` must match what the FFT sees: PML pad plus stencil halo."""
    eq, prop = _build()
    _run(prop)
    expected = tuple(s + 2 * ABCN + eq.so for s in (NZ, NX))
    assert tuple(eq.k.shape[-2:]) == expected


@requires_torch
def test_both_toggles_off_matches_acoustic():
    """Both terms off must reduce to the plain CPML acoustic step, bit-exactly."""
    _, prop = _build(phase_shift=False, amplitude_damping=False)
    assert _rel_l2(_run(prop), _acoustic_run()) == 0.0


@requires_torch
def test_toggles_are_independent():
    """All four combinations run and give four distinct wavefields."""
    out = {}
    for ps in (False, True):
        for ad in (False, True):
            _, prop = _build(phase_shift=ps, amplitude_damping=ad)
            out[(ps, ad)] = _run(prop)
            assert np.isfinite(out[(ps, ad)]).all()

    keys = list(out)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert _rel_l2(out[a], out[b]) > 1e-6, f"{a} and {b} are identical"


@requires_torch
def test_amplitude_damping_removes_energy():
    """The dissipative term must lower the recorded energy; the dispersion term
    alone must not (it moves the wavefront, it does not absorb)."""
    def rms(ps, ad):
        _, prop = _build(phase_shift=ps, amplitude_damping=ad)
        return float(np.sqrt((_run(prop) ** 2).mean()))

    assert rms(False, True) < rms(False, False)
    assert rms(True, True) < rms(True, False)


@requires_torch
def test_free_surface_is_applied():
    """Regression: ``func`` returned the step output unchanged, so
    ``free_surface=True`` was silently ignored."""
    _, p_fs = _build(free_surface=True)
    _, p_no = _build(free_surface=False)
    assert _rel_l2(_run(p_fs), _run(p_no)) > 1e-6


@requires_torch
def test_free_surface_matches_acoustic_with_toggles_off():
    """With both terms off the free-surface handling must agree with Acoustic."""
    _, prop = _build(phase_shift=False, amplitude_damping=False, free_surface=True)
    assert _rel_l2(_run(prop), _acoustic_run(free_surface=True)) == 0.0


# --------------------------------------------------------------------------
# gradients (structural)
#
# These are the sharp, cheap checks. The quantitative finite-difference
# validation is expensive and cancellation-limited (the solver runs float32),
# so it lives in ``visco_acoustic_grad_fdcheck.py`` -- a standalone script, the
# same place the other FD gradient checks in this directory live.
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _target():
    """Fixed 'observed' data: one perturbed-model forward, shared by every
    gradient test (it is the reference, not a function of the toggles)."""
    _, prop = _build()
    w, src, rec = _geom()
    with torch.no_grad():
        t = prop.forward(w, src, rec, models=_models(vp=VP + 40.0))
    return (t[0] if isinstance(t, (tuple, list)) else t).detach()


def _grads(ps=True, ad=True):
    _, prop = _build(phase_shift=ps, amplitude_damping=ad)
    w, src, rec = _geom()
    ms = [m.clone().requires_grad_(True) for m in _models()]
    syn = prop.forward(w, src, rec, models=ms)
    syn = syn[0] if isinstance(syn, (tuple, list)) else syn
    (syn - _target()).pow(2).sum().backward()
    return [m.grad for m in ms]


@requires_torch
def test_vp_and_q_gradients_exist():
    g = _grads()
    for name, gi in zip(("vp", "Q"), g[:2]):
        assert gi is not None, f"no gradient for {name}"
        assert torch.isfinite(gi).all(), f"non-finite gradient for {name}"
        assert float(gi.abs().max()) > 0.0, f"all-zero gradient for {name}"


@requires_torch
def test_q_gradient_vanishes_when_both_terms_are_off():
    """Structural: with both toggles off, Q does not enter the computation at
    all, so it must receive no gradient -- while vp still does."""
    g = _grads(ps=False, ad=False)
    assert g[1] is None or float(g[1].abs().max()) == 0.0
    assert g[0] is not None and float(g[0].abs().max()) > 0.0


@requires_torch
@pytest.mark.parametrize("ps,ad", [(True, False), (False, True)])
def test_q_gradient_flows_through_each_term(ps, ad):
    """Q reaches the misfit through both terms: the dispersion term via
    vp_step = vp*sqrt(1-c(Q)), and the attenuation term via tt(Q, omega)."""
    g = _grads(ps=ps, ad=ad)
    assert g[1] is not None and float(g[1].norm()) > 0.0


# --------------------------------------------------------------------------
# per-edge free surface (eager)
#
# The zeroing is the shared ``SecondOrderEquation._apply_free_surface``; these
# pin (a) the top-only path staying bit-exact, (b) genuine per-edge physics:
# four-face closed-box mirror symmetry WITH the FFT damping active (the |k|
# filter is an even periodic convolution, so it commutes with the mirror), and
# (c) discrimination — a non-top face must actually change the record.
# --------------------------------------------------------------------------

def _build_fs(fs, N=None):
    from sweep.equations import ViscoAcoustic
    from sweep.propagator.torch import PropTorch

    eq = ViscoAcoustic(spatial_order=4, backend="torch", device="cpu")
    shape = (N, N) if N is not None else (NZ, NX)
    prop = PropTorch(eq, shape=shape, dh=DH, dt=DT, dev=torch.device("cpu"),
                     source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
                     impl="eager", use_ckpt=False, free_surface=fs)
    return prop


@requires_torch
def test_per_edge_top_only_bit_exact():
    a = _run(_build_fs(True))
    b = _run(_build_fs(["top"]))
    c = _run(_build_fs([True, False, False, False]))
    assert np.array_equal(a, b) and np.array_equal(a, c)


@requires_torch
def test_per_edge_faces_discriminate():
    """A non-top free surface must genuinely change the record (a top-only
    collapse would pass any symmetry test trivially)."""
    top = _run(_build_fs(True))
    for fs in (["left"], ["right"], ["bottom"]):
        assert _rel_l2(_run(_build_fs(fs)), top) > 0.05, fs


@requires_torch
def test_per_edge_closed_box_mirror_symmetry():
    """Odd N + exactly centred source: a 4-face free-surface box must give
    machine-precision z- and x-mirror symmetric records, damping included."""
    from sweep.equations import ViscoAcoustic
    from sweep.propagator.torch import PropTorch

    M, nt = 49, 160
    c = M // 2
    eq = ViscoAcoustic(spatial_order=4, backend="torch", device="cpu")
    prop = PropTorch(eq, shape=(M, M), dh=DH, dt=DT, dev=torch.device("cpu"),
                     source_type=["h1"], receiver_type=["h1"], abcn=0,
                     impl="eager", use_ckpt=False,
                     free_surface=["top", "bottom", "left", "right"])
    w = _ricker(nt, DT, F0)
    src = np.array([[c, c]], dtype=np.int64)
    rec = np.array([[[c, c - 10], [c, c + 10], [c - 10, c], [c + 10, c]]],
                   dtype=np.int64)  # (x, z): x-pair then z-pair
    vp = torch.full((M, M), VP)
    Q = torch.full((M, M), Q_VAL)
    om = torch.full((M, M), 2 * np.pi * F0)
    out = prop.forward(w, src, rec, models=[vp, Q, om])
    tr = (out[0] if isinstance(out, (tuple, list)) else out)[0, :, :, 0].detach().numpy()
    peak = np.abs(tr).max()
    assert peak > 0
    assert np.abs(tr[:, 0] - tr[:, 1]).max() / peak < 1e-4   # x-mirror
    assert np.abs(tr[:, 2] - tr[:, 3]).max() / peak < 1e-4   # z-mirror
