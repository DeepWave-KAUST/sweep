"""The strip's C-PML block, built from 1-D axis values instead of gathered.

The nine C-PML arrays are stored per NODE but on an axis-aligned mesh each one
varies along a single axis: measured spread across the other two axes is
exactly 0.000e+00. That makes them 36.0 of the 158.5 B/node a rank holds for
the global equation while carrying only nx+ny+nz numbers -- 6.3 MiB of storage
for 4.6 KiB of information at 65^3. `_parts3d` now reduces them once and lets
each strip materialise its own block, so the global 3-D profile is never
gathered per strip.

The whole safety of that rests on two claims, and both are tested here rather
than argued:

* the generated block is **bit-identical** to what gathering the global array
  would have produced -- not close, equal, so no tolerance is involved;
* separability is **checked at run time**, not assumed. Under topography the
  node coordinates move and it is false (`b_z` varies across x and y), and
  that case must fall back to gathering. A silent
  wrong answer there would look like a physically plausible but wrong
  absorbing layer, which is exactly the kind of thing no downstream test
  reliably catches.
"""
from __future__ import annotations

import pytest
import torch

import sweep.sem as S
from sweep.sem.dd import (SEMModelParallel, StripTopology, _axis_cpml_ok,
                          _axis_profiles, _strip_cpml_from_axes)
from sweep.sem.equations import AcousticSEM3D
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0 = 4, 40.0, 8.0
NEX = 16
NAMES = ("a_x", "b_x", "dbx_dx", "a_y", "b_y", "dby_dy",
         "a_z", "b_z", "dbz_dz")


def _ztop(x, y):
    return 60.0 * torch.exp(-(((x - NEX * DH * 0.5) / 200.0) ** 2
                              + ((y - NEX * DH * 0.5) / 200.0) ** 2))


def _setup(deformed):
    n = NEX * NO + 1
    kw = dict(shape=(n, n, n), dh=DH, dt=6e-4, nt=10, n_order=NO,
              free_surface=("top",), abcn=4 * NO,
              f_dominant=F0, dev="cuda", impl="c")
    if deformed:
        # the surface row of every model column, (ny, nx)
        y, x = torch.meshgrid(torch.arange(n, dtype=torch.float64) * DH,
                              torch.arange(n, dtype=torch.float64) * DH,
                              indexing="ij")
        kw["topography"] = (_ztop(x, y) / DH).numpy()
    p = S.PropSEM(AcousticSEM3DDeformed if deformed else AcousticSEM3D, **kw)
    torch.manual_seed(0)
    vp = 2500.0 + 200.0 * torch.rand(n, n, n, device="cuda")
    eq, _ = p._resolve_model_batch([vp])
    m = eq.mesh
    return p, eq, m, (m.nex * NO + 1, m.ney * NO + 1, m.nez * NO + 1)


def test_flat_mesh_is_separable_and_topography_is_not():
    """The run-time check must SEPARATE the two cases -- if it answered True
    for both, the fallback would never run; if False for both, the fast path
    would never run and every other test here would be vacuous."""
    _, eq, m, (nxg, nyg, nzg) = _setup(False)
    assert _axis_cpml_ok(m, eq.cpml, nxg, nyg, nzg) is True
    _, eqd, md, (dx, dy, dz) = _setup(True)
    assert _axis_cpml_ok(md, eqd.cpml, dx, dy, dz) is False


@pytest.mark.parametrize("px", [1, 2, 4])
def test_strip_cpml_matches_the_sliced_global_one(px):
    """Bit-identical, all nine arrays, every strip. `== 0.0`, not a tolerance:
    the axis values ARE the gathered values, so anything else is a bug."""
    _, eq, m, (nxg, nyg, nzg) = _setup(False)
    axes = _axis_profiles(eq.cpml, nxg, nyg, nzg)
    rows = torch.arange(nzg * nyg, device="cuda")
    nex_l = m.nex // px
    for r in range(px):
        cols = torch.arange(r * nex_l * NO, r * nex_l * NO + nex_l * NO + 1,
                            device="cuda")
        gidx = (rows[:, None] * nxg + cols[None, :]).reshape(-1)
        got = _strip_cpml_from_axes(axes, rows, cols, nyg)
        for nm in NAMES:
            ref = getattr(eq.cpml, nm).reshape(-1)[gidx]
            assert got[nm].shape == ref.shape, f"px={px} r={r} {nm}: shape"
            d = float((got[nm] - ref).abs().max())
            assert d == 0.0, f"px={px} strip {r} {nm}: max |diff| = {d:.3e}"


@pytest.mark.parametrize("px", [1, 2])
def test_parts3d_consts_are_unchanged_by_the_axis_path(px):
    """End to end: the consts `_parts3d` hands the kernels must be exactly what
    the gathering path produced. This is what would catch a wrong `rows`/`cols`
    mapping, which the unit test above shares with the implementation."""
    p, eq, m, (nxg, nyg, nzg) = _setup(False)
    ng = m.n_global
    pat = torch.zeros(ng, device="cuda")
    pat[ng // 2] = 1.0
    rec = torch.tensor([ng // 3], device="cuda")
    d = SEMModelParallel(p, StripTopology(px))
    parts, _ = d._parts3d(eq, (pat,), rec)

    rows = torch.arange(nzg * nyg, device="cuda")
    nex_l = m.nex // px
    for r, part in enumerate(parts):
        cols = torch.arange(r * nex_l * NO, r * nex_l * NO + nex_l * NO + 1,
                            device="cuda")
        gidx = (rows[:, None] * nxg + cols[None, :]).reshape(-1)
        for nm in NAMES:
            ref = getattr(eq.cpml, nm).reshape(-1)[gidx].contiguous()
            got = part.c[nm]
            d_ = float((got - ref).abs().max())
            assert d_ == 0.0, f"px={px} strip {r} {nm}: consts differ {d_:.3e}"
