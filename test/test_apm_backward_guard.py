"""``topo_method='apm'`` must refuse to produce a gradient on impl='c'.

The CUDA APM path (Cao & Chen 2018 parameter-modified surface) is a forward-only
implementation — the dispatch comment in ``_c.py`` has always said gradients
belong on eager autograd — but nothing enforced it, so it quietly returned a
wrong one.  Finite-difference arbitration, elastic 2-D, source ``['vz']``,
d(loss)/d(rho) at the source cell:

    FD (truth)  +2.606052e-05
    eager       +2.606044e-05   rel 3.2e-06   correct
    impl='c'    +2.243012e-06   rel 9.1e-01   off by 11.6x

Confined to that one cell (masking it drops whole-field rel_l2 from 8.9e-01 to
2.4e-04), needs a body-force source (rho cos vs eager: 0.54 for vz, 0.29 for vx,
1.0000000 for pure-stress loadings), does not need a hill, fires in all four
backward modes, and reproduces bit-identically on dev.

These tests pin the guard: it fires for any APM backward on impl='c', and it
leaves alone the things that work — APM forward-only on impl='c', APM on eager,
and image-method topography on impl='c'.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

SO, ABCN, NT, DT, DH = 4, 16, 40, 1e-3, 10.0
NZ, NX = 40, 48


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _binding_ready(),
                                reason="CUDA + compiled sweep._C required")


def _hill(nx=NX, height=5.0, width=10.0):
    x = np.arange(nx, dtype=np.float32)
    return (height * np.exp(-((x - nx / 2) ** 2) / (2.0 * width ** 2))).round().astype(np.int64)


def _prop(topo, *, topo_method="apm", impl="c", boundary=False):
    from sweep.equations import Elastic
    from sweep.propagator.torch import PropTorch

    eq = Elastic(spatial_order=SO, device="cuda", backend="torch")
    kw = dict(backend="torch", shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT,
              dev="cuda", pml_type="cpmls", nt=NT, B=1, use_ckpt=False,
              topography=topo, topo_method=topo_method)
    if impl == "c":
        kw["boundary_saving_config"] = {"enabled": bool(boundary), "storage": "gpu"}
    else:
        from sweep.propagator.options import EagerOptions
        kw["eager_options"] = EagerOptions(use_compile=False)
    return PropTorch(eq, impl=impl, **kw)


def _run(prop, topo, requires_grad=True):
    t = np.arange(NT, dtype=np.float32) * DT - 0.02
    x = np.pi * 15.0 * t
    wav = torch.tensor(((1 - 2 * x * x) * np.exp(-x * x)).astype(np.float32), device="cuda")
    src = np.array([[NX // 2, int(topo[NX // 2]) + 4]], np.int64)
    rx = np.arange(6, NX - 6, 6, dtype=np.int64)
    rec = np.stack([rx, (topo[rx] + 1).astype(np.int64)], -1)[None]
    models = [torch.tensor(np.full((NZ, NX), v, np.float32), device="cuda",
                           requires_grad=requires_grad)
              for v in (2000.0, 1150.0, 2000.0)]
    prop.equation._apm_cache_key = None
    prop.equation._apm_cache = None
    out = prop(wav, src, rec, models=models)
    if requires_grad:
        out.pow(2).sum().backward()
    return out


@pytest.mark.parametrize("boundary", [False, True])
def test_apm_backward_on_c_is_refused(boundary):
    topo = _hill()
    prop = _prop(topo, boundary=boundary)
    with pytest.raises(NotImplementedError, match="topo_method='apm' has no gradient"):
        _run(prop, topo)


def test_apm_backward_refused_on_flat_topography_too():
    """The defect does not need a hill — flat-zero APM reproduces it."""
    topo = np.zeros(NX, np.int64)
    prop = _prop(topo)
    with pytest.raises(NotImplementedError, match="topo_method='apm' has no gradient"):
        _run(prop, topo)


def test_apm_backward_refused_for_stress_sources_too():
    """Only body-force sources produce the wrong number, but the guard is not
    source-dependent — a user who later adds a velocity component must not
    silently start getting a wrong gradient."""
    from sweep.equations import Elastic
    from sweep.propagator.torch import PropTorch

    topo = _hill()
    eq = Elastic(spatial_order=SO, device="cuda", backend="torch")
    prop = PropTorch(eq, backend="torch", impl="c", shape=(NZ, NX), abcn=ABCN,
                     dh=DH, dt=DT, dev="cuda", pml_type="cpmls", nt=NT, B=1,
                     use_ckpt=False, topography=topo, topo_method="apm",
                     source_type=["sxx", "szz"], receiver_type=["vx", "vz"],
                     boundary_saving_config={"enabled": False})
    with pytest.raises(NotImplementedError, match="topo_method='apm' has no gradient"):
        _run(prop, topo)


def test_apm_message_points_at_the_working_paths():
    topo = _hill()
    prop = _prop(topo)
    with pytest.raises(NotImplementedError) as exc:
        _run(prop, topo)
    msg = str(exc.value)
    assert "impl='eager'" in msg
    assert "forward-only" in msg
    assert "topo_method='image'" in msg


def test_apm_forward_only_on_c_still_works():
    """The CUDA APM forward is the supported use and must stay open."""
    topo = _hill()
    prop = _prop(topo)
    with torch.no_grad():
        _run(prop, topo, requires_grad=False)


def test_apm_gradient_on_eager_still_works():
    topo = _hill()
    prop = _prop(topo, impl="eager")
    _run(prop, topo)


def test_image_method_topography_gradient_on_c_still_works():
    """image-method topography is gradient-consistent on impl='c' (full and
    checkpoint modes) and must not be caught by the APM guard."""
    topo = _hill()
    prop = _prop(topo, topo_method="image", boundary=False)
    _run(prop, topo)
