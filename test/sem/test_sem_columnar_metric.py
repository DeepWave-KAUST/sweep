"""The columnar-metric shortcut of the deformed 3-D kernels.

``build_topography_hex27`` maps x(xi), y(eta), z(xi, eta, gamma), so J is lower
triangular and inv_J has four identically-zero components and two mesh-wide
constants. ``prop_c._columnar_metric`` detects that from inv_J and the deformed
CUDA kernels then read four (E, m) streams instead of ten.

What has to hold, and what these tests pin:
  * the shortcut fires on a topography mesh, with the ANALYTIC constants;
  * it does NOT fire on a mesh that is genuinely curved in x, however small the
    curvature -- a wrong fast path here would silently mis-solve;
  * the flag reaches the C struct, and only on the deformed path.

The kernels' own agreement is covered by ``test_sem_c_acoustic3d_deformed``,
which runs the whole deformed path against eager; this file is about the
detection, which is the part that can be wrong without anything crashing.
"""
from __future__ import annotations

import math

import pytest
import torch

from sweep.sem.mesh.hex27 import (build_topography_hex27,
                                  hex27_from_control_nodes, _pack_control)
from sweep.sem.prop_c import _columnar_metric, sem_c_available as _c_ok

NEX, NEY, NEZ = 6, 5, 4
XMAX, YMAX, ZBOT = 2000.0, 1500.0, 2000.0


def _hill(x, y):
    return 300.0 * torch.sin(3 * math.pi * x / XMAX) \
         * torch.cos(2 * math.pi * y / YMAX) + 400.0


def _detect(mesh):
    """``_columnar_metric`` on a mesh, with inv_J in the ``eq._I`` order."""
    J = mesh.inv_J
    return _columnar_metric(mesh, tuple(J[..., i, j]
                                        for i in range(3) for j in range(3)))


def _bent(dx):
    """A Hex27 mesh whose x control positions bend with depth by ``dx`` metres,
    so the map is no longer separable and the shortcut must not fire."""
    nc = lambda n: 2 * n + 1
    xa = torch.linspace(0.0, XMAX, nc(NEX))
    ya = torch.linspace(0.0, YMAX, nc(NEY))
    YY, XX = torch.meshgrid(ya, xa, indexing="ij")
    ztop = _hill(XX, YY)
    Z = ztop[None] + torch.linspace(0.0, 1.0, nc(NEZ))[:, None, None] * (ZBOT - ztop[None])
    X = XX[None].expand_as(Z) + dx * torch.sin(2 * math.pi * Z / ZBOT)
    Y = YY[None].expand_as(Z).clone()
    ctl = _pack_control(X.contiguous(), Y.contiguous(), Z.contiguous(), NEZ, NEY, NEX)
    return hex27_from_control_nodes(ctl, NEX, NEY, NEZ, n_order=4)


def test_topography_is_columnar_with_the_analytic_constants():
    mesh = build_topography_hex27(NEX, NEY, NEZ, XMAX, YMAX, ZBOT, _hill,
                                  n_order=4)
    col = _detect(mesh)
    assert col["columnar"] == 1
    # d(xi)/dx is 2/dx_e on a uniformly spaced x axis, and likewise for eta
    assert col["col_ixx"] == pytest.approx(2.0 / (XMAX / NEX), rel=1e-5)
    assert col["col_iey"] == pytest.approx(2.0 / (YMAX / NEY), rel=1e-5)


@pytest.mark.parametrize("dx", [60.0, 0.6, 1e-3])
def test_a_mesh_that_bends_in_x_is_not_columnar(dx):
    """The structural test demands EXACTLY zero drift, so it refuses a bend of
    a millimetre over a 2 km domain -- 5e-10 of the span, which any relative
    tolerance would have accepted. That matters because a tolerance relative to
    the span accepts more real curvature the larger the domain gets."""
    assert _detect(_bent(dx)) == {}


def test_the_flag_reaches_the_struct_only_on_the_deformed_path():
    # ``sweep._C`` always imports (it is the lazy ctypes layer): what this
    # needs is a core that carries the SEM entries, and with them the structs.
    from sweep.sem.prop_c import sem_c_available
    if not sem_c_available():
        pytest.skip("needs a sweep core with the SEM entry points")
    import sweep._C as _C
    for name in ("Sem3dCpmlForwardInput", "Sem3dCpmlBackwardInput",
                 "Sem3dElasticForwardInput", "Sem3dElasticBackwardInput"):
        s = getattr(_C, name)()
        assert s.columnar == 0, "the safe full-3x3 path must be the default"
        s.columnar, s.col_ixx, s.col_iey = 1, 0.5, 0.25
        assert (s.columnar, s.col_ixx, s.col_iey) == (1, 0.5, 0.25)


@pytest.mark.skipif(not (torch.cuda.is_available() and _c_ok()),
                    reason="needs CUDA + the compiled sweep._C SEM binding")
def test_the_cpml_path_actually_sets_the_flag():
    """The metric has TWO construction sites -- ``prop_c.metric3d`` for the bare
    deformed entries and ``dd._geom_ac3d`` for the C-PML ones -- and the C-PML
    one is where the time is. Teaching only the first about the shortcut left
    it silently off on the path that matters: nothing crashed, nothing was
    wrong, the kernels just kept reading ten streams. Both go through
    ``prop_c.metric_consts`` now; this pins the C-PML end of it."""
    import sweep.sem as S
    import sweep.sem.drivers.acoustic3d as D
    from sweep.propagator import BoundarySaving
    from sweep.sem.equations import AcousticSEM3DDeformed

    seen = []
    real = D._ac3d_metric

    def spy(s, c, A):
        real(s, c, A)
        seen.append((s.deformed, s.columnar, s.col_ixx))

    nex, no, dh, dt, nt = 8, 4, 40.0, 6e-4, 20
    n = nex * no + 1
    D._ac3d_metric = spy
    try:
        span = dh * (n - 1)
        topo = lambda x, y: 120.0 * torch.sin(3 * math.pi * x / span) \
                                  * torch.cos(2 * math.pi * y / span) + 200.0
        Y, X = torch.meshgrid(torch.arange(n, dtype=torch.float64) * dh,
                              torch.arange(n, dtype=torch.float64) * dh,
                              indexing="ij")
        rows = (topo(X, Y) / dh).numpy()       # the surface row of each column
        p = S.PropSEM(AcousticSEM3DDeformed, shape=(n, n, n), dh=dh, dt=dt,
                      nt=nt, n_order=no, free_surface=("top",),
                      abcn=2 * no, f_dominant=15.0,
                      dev="cuda", impl="c", topography=rows,
                      memory=BoundarySaving())
        vp = torch.full((n, n, n), 2500.0, device="cuda")
        c0, zs = n // 2, int(0.75 * n)     # below the hill, which tops out at 320 m
        src = torch.tensor([[c0, c0, zs]], device="cuda")
        rec = torch.tensor([[[c0 - 2, c0, zs]]], device="cuda")
        t = torch.arange(nt, dtype=torch.float32, device="cuda") * dt - 0.08
        a = (math.pi * 15.0 * t) ** 2
        p.forward((1.0 - 2.0 * a) * torch.exp(-a), src, rec, [vp])
    finally:
        D._ac3d_metric = real
    assert seen, "the deformed driver was never reached"
    for deformed, columnar, col_ixx in seen:
        assert deformed == 1
        assert columnar == 1, "the C-PML path lost the columnar shortcut"
        assert col_ixx == pytest.approx(2.0 / (dh * no), rel=1e-5)


@pytest.mark.skipif(not (torch.cuda.is_available() and _c_ok()),
                    reason="needs CUDA + the compiled sweep._C SEM binding")
def test_the_elastic_deformed_path_actually_sets_the_flag():
    """The elastic twin of the C-PML check above. Its four launch sites are the
    reason: three are in the backward, and a launch left on the full path
    compiles clean and reads ~1.00x end to end -- the shape of the recorded
    incident. This pins the Python half; only a timing pins the CUDA half, so
    treat a flat A/B as a plumbing bug until proven otherwise."""
    import sweep.sem as S
    import sweep.sem.drivers.bare as B
    from sweep.sem.equations import ElasticSEM3DDeformed

    seen = []
    real = B._el3d_geometry

    def spy(st, c, A):
        real(st, c, A)
        seen.append((type(st).__name__, st.deformed, st.columnar, st.col_ixx))

    nex, no, dh, dt, nt = 8, 4, 40.0, 6e-4, 20
    n = nex * no + 1
    span = dh * (n - 1)
    B._el3d_geometry = spy
    try:
        topo = lambda x, y: 120.0 * torch.sin(3 * math.pi * x / span) \
                                  * torch.cos(2 * math.pi * y / span) + 200.0
        Y, X = torch.meshgrid(torch.arange(n, dtype=torch.float64) * dh,
                              torch.arange(n, dtype=torch.float64) * dh,
                              indexing="ij")
        rows = (topo(X, Y) / dh).numpy()       # the surface row of each column
        p = S.PropSEM(ElasticSEM3DDeformed, shape=(n, n, n), dh=dh, dt=dt,
                      nt=nt, n_order=no, free_surface=("top",),
                      abcn=0, f_dominant=15.0,
                      dev="cuda", impl="c", topography=rows)
        mk = lambda v: torch.full((n, n, n), v, device="cuda").requires_grad_(True)
        models = [mk(2500.0), mk(1400.0), mk(2200.0)]
        c0, zs = n // 2, int(0.75 * n)
        t = torch.arange(nt, dtype=torch.float32, device="cuda") * dt - 0.08
        a = (math.pi * 15.0 * t) ** 2
        out = p.forward((1.0 - 2.0 * a) * torch.exp(-a),
                        torch.tensor([[c0, c0, zs]], device="cuda"),
                        torch.tensor([[[c0 - 2, c0, zs]]], device="cuda"),
                        models, source_vector=(1e6, 0.7e6, 0.5e6))
        out.pow(2).sum().backward()      # the backward carries three of the four
    finally:
        B._el3d_geometry = real
    assert {nm for nm, *_ in seen} == {"Sem3dElasticForwardInput",
                                       "Sem3dElasticBackwardInput"}, seen
    for _, deformed, columnar, col_ixx in seen:
        assert deformed == 1
        assert columnar == 1, "the elastic deformed path lost the shortcut"
        assert col_ixx == pytest.approx(2.0 / (dh * no), rel=1e-5)


def test_the_detection_does_not_decay_with_mesh_size():
    """The reason the test lives on the control nodes and not on inv_J: the
    metric's structural zeros are pure fp32 round-off and that round-off grows
    with the domain (2e-6 of the matrix at nex=8, 2.5e-5 at nex=48). A fixed
    threshold on THEM would switch the shortcut off silently on a big enough
    mesh; on the control nodes the separation is bitwise at every size."""
    for nex in (4, 12, 24):
        span = 160.0 * nex
        hill = (lambda span: lambda x, y: 120.0 * torch.sin(3 * math.pi * x / span)
                * torch.cos(2 * math.pi * y / span) + 200.0)(span)
        mesh = build_topography_hex27(nex, nex, nex, span, span, span, hill,
                                      n_order=4)
        x, y = mesh.control_xyz[..., 0], mesh.control_xyz[..., 1]
        for t, dims in ((x, (1, 2)), (y, (1, 3))):
            for d in dims:
                assert float((t.amax(dim=d) - t.amin(dim=d)).abs().max()) == 0.0
        assert _detect(mesh)["columnar"] == 1, f"lost the shortcut at nex={nex}"
