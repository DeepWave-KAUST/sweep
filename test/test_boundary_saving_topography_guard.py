"""impl='c' boundary saving must refuse to run under ``topography=``.

The boundary-saving backward rebuilds the forward wavefield by reverse
time-stepping from saved edge strips.  With a per-column surface that reverse
pass does not reproduce the surface treatment the forward applied, and the
gradient comes out wrong — not slightly off, but pointing elsewhere:

    elastic 2-D, Gaussian hill (5 cells), source vz, d(loss)/d(vp,vs,rho)
        full / chunk ckpt / recursive ckpt   cos  1.0000000   rel 1e-6
        boundary saving                      cos -0.028       rel 11

The forward record is unaffected (cos 1.0000000, rel 5e-7 against the full
path), so nothing warns you.  It scales with the surface's distance from row 0
— a one-cell hill already gives cos 0.52, a constant non-zero topography gives
0.55 — and it hits Acoustic too (cos 0.9935 on the same hill), so it is a
property of the reconstruction rather than of the elastic kernels.

These tests pin the guard: it fires for topography + boundary saving, and it
does NOT fire for the configurations that are fine (flat free surface, the full
and checkpoint paths, forward-only runs, eager boundary saving).
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


def _prop(*, topography=None, free_surface=False, boundary=False, impl="c",
          topo_method=None):
    from sweep.equations import Elastic
    from sweep.propagator.torch import PropTorch

    eq = Elastic(spatial_order=SO, device="cuda", backend="torch")
    kw = dict(backend="torch", shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT,
              dev="cuda", pml_type="cpmls", nt=NT, B=1,
              free_surface=free_surface, use_ckpt=False,
              boundary_saving_config={"enabled": bool(boundary), "storage": "gpu"})
    if topography is not None:
        kw["topography"] = topography
        if topo_method is not None:
            kw["topo_method"] = topo_method
    return PropTorch(eq, impl=impl, **kw)


def _inputs(topo=None):
    surf = np.zeros(NX, np.int64) if topo is None else topo
    t = np.arange(NT, dtype=np.float32) * DT - 0.02
    x = np.pi * 15.0 * t
    wav = torch.tensor(((1 - 2 * x * x) * np.exp(-x * x)).astype(np.float32), device="cuda")
    src = np.array([[NX // 2, int(surf[NX // 2]) + 4]], np.int64)
    rx = np.arange(6, NX - 6, 6, dtype=np.int64)
    rec = np.stack([rx, (surf[rx] + 1).astype(np.int64)], -1)[None]
    vp = np.full((NZ, NX), 2000.0, np.float32)
    vs = np.full((NZ, NX), 1150.0, np.float32)
    rho = np.full((NZ, NX), 2000.0, np.float32)
    return wav, src, rec, [vp, vs, rho]


def _run(prop, topo=None, requires_grad=True):
    wav, src, rec, arrays = _inputs(topo)
    models = [torch.tensor(a, device="cuda", requires_grad=requires_grad)
              for a in arrays]
    out = prop(wav, src, rec, models=models)
    if requires_grad:
        out.pow(2).sum().backward()
    return out


def test_boundary_saving_with_topography_is_refused():
    topo = _hill()
    prop = _prop(topography=topo, boundary=True, topo_method="image")
    with pytest.raises(NotImplementedError, match="boundary saving cannot be combined"):
        _run(prop, topo)


def test_guard_message_names_the_alternatives():
    topo = _hill()
    prop = _prop(topography=topo, boundary=True, topo_method="image")
    with pytest.raises(NotImplementedError) as exc:
        _run(prop, topo)
    msg = str(exc.value)
    # every escape hatch that is actually gradient-consistent must be named,
    # and the message must say that boundary saving is the impl='c' DEFAULT —
    # otherwise a user who never asked for it cannot tell why they got this.
    assert "ckpt" in msg
    assert "'enabled': False" in msg
    assert "eager" in msg
    assert "DEFAULT" in msg


def test_flat_free_surface_with_boundary_saving_still_works():
    """No topography= — this configuration is gradient-consistent and must not
    be caught by the guard."""
    prop = _prop(free_surface=True, boundary=True)
    _run(prop)                                   # must not raise


def test_no_free_surface_with_boundary_saving_still_works():
    prop = _prop(free_surface=False, boundary=True)
    _run(prop)


def test_topography_without_boundary_saving_still_works():
    """The full path is the recommended alternative — it must stay open."""
    topo = _hill()
    prop = _prop(topography=topo, boundary=False, topo_method="image")
    _run(prop, topo)


def test_topography_forward_only_with_boundary_saving_is_allowed():
    """Boundary saving only affects the backward; a forward-only run under
    topography is fine and must not be blocked."""
    topo = _hill()
    prop = _prop(topography=topo, boundary=True, topo_method="image")
    with torch.no_grad():
        _run(prop, topo, requires_grad=False)


def test_eager_boundary_saving_under_topography_is_not_guarded():
    """The eager boundary-saving path reconstructs correctly under topography
    (rel ~1e-5 vs the eager tape), so it stays available."""
    from sweep.equations import Elastic
    from sweep.propagator.options import BoundaryOptions, EagerOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    topo = _hill()
    eq = Elastic(spatial_order=SO, device="cuda", backend="torch")
    prop = PropTorch(
        eq, backend="torch", impl="eager", shape=(NZ, NX), abcn=ABCN, dh=DH,
        dt=DT, dev="cuda", pml_type="cpmls", nt=NT, B=1, use_ckpt=False,
        topography=topo, topo_method="image",
        eager_options=EagerOptions(use_compile=False),
        memory=MemoryOptions(strategy="boundary",
                             boundary=BoundaryOptions(storage="gpu")),
    )
    _run(prop, topo)
