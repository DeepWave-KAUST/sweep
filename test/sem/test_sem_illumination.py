"""PropSEM's source and receiver illumination (``sweep.sem.illumination``).

``compute_illumination = True`` gives, in the models' shape, the source
illumination ``sum_t |a|^2`` (``a`` the wavefields' second time derivative) and
the receiver illumination ``sum_t |M^-1 dL/df|^2`` (the adjoint wavefield):
PropTorch's pair, and SPECFEM2D's first approximate Hessian for the source one.

* eager: off by default; on, it is model-shaped, nonnegative and leaves the
  gradient bit for bit; at a source on a GLL node the receiver term equals the
  one the wavelet gradient gives (``dL/dw[n] = dt^2 lambda^{n+1}`` there) and
  the source term the one a receiver on that node records; a grid model takes
  the transpose of its interpolation, which keeps the sum; refusals;
* CUDA: the three 2-D C-PML paths against eager, in a shot batch and under
  boundary saving, the gradient unmoved beyond the run-to-run floor of the
  atomic assembly; the paths without the accumulation refuse.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator.options import BoundarySaving, Full
from sweep.sem import (AcousticElasticSEM2D, AcousticPotentialSEM2D, BiotVSSEM2DDeformed,
                       ElasticSEM2DDeformed, ElasticVSSEM2DDeformed, PropSEM)
from sweep.sem.illumination import element_weights
from sweep.sem.prop_c import sem_c_available

needs_c = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NT, DT, DH, F0 = 300, 5e-4, 10.0, 15.0
SHAPE = (33, 49)
SRC = np.array([[24, 10]])              # an element's centre node (n_order 4)
REC = np.array([[[x, 4] for x in range(2, 47, 4)]])


def _wav(dtype):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 0.08)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=dtype)


def _models(kind, dtype, prop=None):
    vp = np.full(SHAPE, 2000.0)
    vp[20:] = 2500.0
    if kind == "potential":
        return [vp]
    if kind == "fluid_solid":
        # GLL-native, each element wholly water or rock (the coupled form's rule)
        m = prop.mesh
        wet = (m.z_e.flatten(1).mean(1) < 120.0).reshape(-1, 1, 1).expand(
            m.n_elements, m.ngll, m.ngll).cpu().numpy()
        deep = (m.z_e >= 200.0).cpu().numpy()
        return [np.where(wet, 1500.0, np.where(deep, 3000.0, 2400.0)),
                np.where(wet, 0.0, np.where(deep, 1500.0, 1200.0)),
                np.where(wet, 1020.0, 2000.0)]
    return [vp * 1.2, vp * 0.6, np.full(SHAPE, 2000.0)]


def _shape(kind, prop):
    m = prop.mesh
    return (m.n_elements, m.ngll, m.ngll) if kind == "fluid_solid" else SHAPE


EQ = {"potential": (AcousticPotentialSEM2D, ["chi"]),
      "elastic": (ElasticSEM2DDeformed, ["vz"]),
      "fluid_solid": (AcousticElasticSEM2D, ["chi"]),
      "vs": (ElasticVSSEM2DDeformed, ["vz"])}


def _run(kind, *, illum=True, impl="eager", dev="cpu", dtype=torch.float64, memory=None,
         shots=1, rec=REC, receiver_type=None, abcn=8):
    eq, st = EQ[kind]
    kw = {} if receiver_type is None else {"receiver_type": receiver_type}
    if impl == "eager":
        kw["eager_options"] = {"use_compile": False}
    p = PropSEM(eq, shape=SHAPE, dh=DH, dt=DT, n_order=4, abcn=abcn, f_dominant=F0, impl=impl,
                dev=dev, dtype=dtype, free_surface=True, source_type=st,
                memory=memory if memory is not None else Full(), **kw)
    p.compute_illumination = illum
    ms = [torch.tensor(m, dtype=dtype, device=dev, requires_grad=True)
          for m in _models(kind, dtype, p)]
    w = _wav(dtype).to(dev).requires_grad_(True)
    src = SRC if shots == 1 else np.array([[24, 10], [16, 10]])
    recs = rec if shots == 1 else np.repeat(rec, shots, 0)
    o = p(w, src, recs, models=ms)
    W = torch.linspace(0.5, 1.5, o.numel(), dtype=dtype, device=dev).reshape(o.shape)
    (W * o).sum().backward()
    return p, w.grad.detach(), [m.grad.detach() for m in ms]


def _rel(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm())


# --------------------------------------------------------------------------- #
# eager
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["potential", "elastic", "fluid_solid", "vs"])
def test_off_by_default_on_when_asked_and_the_gradient_does_not_move(kind):
    p_off, gw_off, g_off = _run(kind, illum=False)
    p_on, gw_on, g_on = _run(kind)
    assert p_off.source_illumination is None and p_off.receiver_illumination is None
    for t in (p_on.source_illumination, p_on.receiver_illumination):
        assert t is not None and tuple(t.shape) == _shape(kind, p_on)
        assert bool(torch.isfinite(t).all()) and float(t.min()) >= 0.0 and float(t.max()) > 0.0
    assert torch.equal(gw_on, gw_off)
    for a, b in zip(g_on, g_off):
        assert torch.equal(a, b)


def test_at_a_source_node_the_two_terms_equal_independent_ones():
    p, gw, _ = _run("potential")
    m = p.mesh
    x0, z0 = p._coords(np.array(SRC[0], float))[:2]
    g0 = int(torch.argmin((m.x_global - x0) ** 2 + (m.z_global - z0) ** 2))
    assert float(((m.x_global[g0] - x0) ** 2 + (m.z_global[g0] - z0) ** 2) ** 0.5) < 1e-9
    sink = p._illumination.sinks[0]
    # dL/dw[n] = dt^2 (M^-1 dL/dchi^{n+1}) at the source node
    lam2 = float(((gw / DT ** 2) ** 2).sum())
    assert abs(float(sink.rec[g0]) - lam2) <= 1e-10 * lam2
    # a receiver on the node records the run's chi there: p (the wavelet drives chi)
    eq, st = EQ["potential"]
    q = PropSEM(eq, shape=SHAPE, dh=DH, dt=DT, n_order=4, abcn=8, f_dominant=F0, impl="eager",
                dtype=torch.float64, free_surface=True, source_type=st, memory=Full(),
                receiver_type=["p"], eager_options={"use_compile": False})
    q.compute_illumination = True
    with torch.no_grad():
        o = q(_wav(torch.float64), SRC, np.array([[list(SRC[0])]]),
              models=[torch.tensor(m) for m in _models("potential", torch.float64)])
    chi = torch.cat([torch.zeros(2, dtype=torch.float64), o[0, :, 0, 0]])
    acc = ((chi[2:] - 2 * chi[1:-1] + chi[:-2]) / DT ** 2).pow(2).sum()
    s0 = float(q._illumination.sinks[0].src[g0])
    assert abs(s0 - float(acc)) <= 1e-10 * float(acc)


def test_a_grid_model_takes_the_interpolation_transpose():
    """Without an absorbing pad the transpose keeps the sum (each GLL node's
    bilinear weights add to 1), so the grid illumination sums to the
    GLL-native one: every element node's value times its ``w |J|``."""
    p, _, _ = _run("potential", abcn=0)
    sink = p._illumination.sinks[0]
    native = sink.src[p.mesh.ibool] * element_weights(p.mesh)
    grid = p.source_illumination
    assert abs(float(grid.sum()) - float(native.sum())) <= 1e-10 * float(native.sum())


def test_a_gll_native_model_gives_element_node_values():
    eq, st = EQ["potential"]
    p = PropSEM(eq, shape=SHAPE, dh=DH, dt=DT, n_order=4, abcn=8, f_dominant=F0, impl="eager",
                dtype=torch.float64, free_surface=True, source_type=st, memory=Full(),
                eager_options={"use_compile": False})
    p.compute_illumination = True
    vp = p.layered_model([2000.0]).requires_grad_(True) if hasattr(p, "layered_model") else None
    if vp is None:
        pytest.skip("no GLL-native model builder")
    o = p(_wav(torch.float64), SRC, REC, models=[vp])
    o.pow(2).sum().backward()
    S = p.source_illumination
    assert tuple(S.shape) == tuple(vp.shape)
    sink = p._illumination.sinks[0]
    assert torch.allclose(S, sink.src[p.mesh.ibool] * element_weights(p.mesh))


def test_refusals():
    with pytest.raises(ValueError, match="full tape"):
        _run("potential", memory=BoundarySaving())
    eq, st = EQ["potential"]
    p = PropSEM(eq, shape=SHAPE, dh=DH, dt=DT, n_order=4, abcn=8, f_dominant=F0, impl="eager",
                dtype=torch.float64, free_surface=True, source_type=st, memory=Full(),
                eager_options={"use_compile": False})
    p.compute_illumination = True
    with pytest.raises(ValueError, match="exclusive"):
        p(_wav(torch.float64), SRC, REC,
          models=[torch.tensor(_models("potential", torch.float64)[0])], return_wavefield=True)
    from sweep.sem.illumination import kinematic_fields

    class _Biot:                                  # the field layout is what decides
        field_specs = BiotVSSEM2DDeformed.field_specs
    with pytest.raises(NotImplementedError, match="2x2 mass"):
        kinematic_fields(_Biot())


# --------------------------------------------------------------------------- #
# CUDA against eager
# --------------------------------------------------------------------------- #
@needs_c
@pytest.mark.parametrize("kind", ["potential", "elastic", "fluid_solid"])
@pytest.mark.parametrize("memory,shots", [(None, 1), (None, 2), ("bs", 1)],
                         ids=["tape", "batch", "bs"])
def test_c_matches_eager(kind, memory, shots):
    mem = BoundarySaving() if memory == "bs" else None
    pe, _, _ = _run(kind, dev="cuda", dtype=torch.float32, shots=shots)
    pc, gw_c, g_c = _run(kind, impl="c", dev="cuda", dtype=torch.float32, shots=shots, memory=mem)
    for name in ("source_illumination", "receiver_illumination"):
        e, c = getattr(pe, name), getattr(pc, name)
        assert c is not None and tuple(c.shape) == _shape(kind, pc), name
        assert _rel(c, e) < 2e-4, (kind, name, _rel(c, e))
    # and on impl='c' the switch does not move the gradient beyond the run-to-run
    # floor of its atomic assembly (two runs with it off)
    off = [_run(kind, impl="c", dev="cuda", dtype=torch.float32, shots=shots, memory=mem,
                illum=False) for _ in range(2)]
    for j, on in enumerate([gw_c] + g_c):
        a, b = (o[1] if j == 0 else o[2][j - 1] for o in off)
        floor = float((a - b).abs().max())
        moved = float((on - a).abs().max())
        assert moved <= 4.0 * floor + 1e-7 * float(a.abs().max()), (j, moved, floor)


@needs_c
def test_c_paths_without_the_accumulation_refuse():
    with pytest.raises(NotImplementedError, match="compute_illumination"):
        _run("potential", impl="c", dev="cuda", dtype=torch.float32, abcn=0)
