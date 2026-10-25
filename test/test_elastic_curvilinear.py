"""Curvilinear-grid elastic 2-D: asymmetric-operator scheme.

The equation implements Bartana, Xu & Rickett (SLB), *Solving the
elastodynamic equations in curvilinear coordinates using asymmetric
operators*: collocated fields, conservative momentum equation, and a free
surface imposed by rotating the stress-rate tensor into the frame of the
surface normal.  What these tests pin down is (a) the flat limit still
agrees with the flat solver's geometry, (b) the traction really is free on
a curved surface, and (c) the scheme stays bounded on a hill, which is
what the staggered chain-rule predecessor could not do.
"""

import numpy as np
import pytest
import torch

from sweep.equations import ElasticCurvilinear
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

NZ, NX = 80, 120
DH = 10.0
DT = 6.0e-4
NT = 300
ABCN = 25
SO = 8                    # L = 4
FREQ = 8.0
DELAY = 0.12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _wavelet(nt=NT):
    t = np.arange(nt, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32)).to(DEVICE)


def _geometry(topo=None, src_x=NX // 2, src_depth=3.0):
    """Physical grid indices: the source ``src_depth`` rows below the surface,
    the receivers on it. The propagator maps them onto its rows."""
    surf = np.zeros(NX) if topo is None else np.asarray(topo, dtype=np.float64)
    sources = torch.tensor([[src_x, surf[src_x] + src_depth]],
                           dtype=torch.float64).to(DEVICE)
    rec_x = np.arange(6, NX - 6, 5)
    receivers = torch.from_numpy(
        np.stack([rec_x, surf[rec_x]], axis=-1)[None, ...]).to(DEVICE)
    return sources, receivers


def _models():
    return [torch.full((NZ, NX), 2000.0).to(DEVICE),
            torch.full((NZ, NX), 1200.0).to(DEVICE),
            torch.full((NZ, NX), 1800.0).to(DEVICE)]


def _make_prop(topography, *, free_surface=True, so=SO, **eq_kwargs):
    eq = ElasticCurvilinear(spatial_order=so, device=DEVICE, backend="torch",
                            **eq_kwargs)
    prop = PropTorch(
        eq, shape=(NZ, NX),
        free_surface=free_surface,
        topography=topography,
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl="eager",
        eager_options={"use_compile": False},
    )
    return eq, prop


def _hill(height_cells=6.0, sigma_cells=12.0):
    x = np.arange(NX, dtype=np.float32)
    z = height_cells * np.exp(-((x - NX * 0.5) ** 2) / (2.0 * sigma_cells ** 2))
    return np.round(z.max() - z).astype(np.int64)


# --------------------------------------------------------------------------


def test_flat_topography_runs_and_is_stable():
    sources, receivers = _geometry()
    _, prop = _make_prop(topography=None)
    syn = prop(_wavelet(), sources, receivers, models=_models())
    assert torch.isfinite(syn).all(), "flat curvilinear run produced NaN/Inf"
    assert syn.abs().max().item() > 0, "record is identically zero"


def test_zero_topography_matches_no_topography():
    """A flat ``topography`` array must be the same problem as no array."""
    sources, receivers = _geometry()
    _, prop_none = _make_prop(topography=None)
    a = prop_none(_wavelet(), sources, receivers, models=_models())
    _, prop_zero = _make_prop(topography=np.zeros(NX, dtype=np.int64))
    b = prop_zero(_wavelet(), sources, receivers, models=_models())
    assert torch.equal(a, b)


def test_medium_above_the_surface_stays_void():
    """No field may leak into the rows above the topography."""
    sources, receivers = _geometry(_hill())
    eq, prop = _make_prop(topography=_hill())
    _, snaps = prop(_wavelet(), sources, receivers, models=_models(),
                    return_wavefield=True,
                    snapshot_times=list(range(0, NT, NT // 4)))
    # Snapshots come back at the runtime shape: the FD halo is stripped, so
    # row 0 is the free surface and there is nothing above it to check
    # inside the returned array -- probe the equation's own mask instead.
    geom = eq._geom_cache
    assert geom is not None
    below = geom["below"][0, 0, :, 0].cpu().numpy()
    assert below[:eq.surface_row].max() == 0.0
    assert below[eq.surface_row:].min() == 1.0
    assert torch.isfinite(snaps).all()


def test_traction_is_free_on_a_curved_surface():
    """sigma_ij n_j = 0 on the surface row, to a small fraction of |sigma|.

    On the boundary-fitted grid the surface normal is proportional to
    ``(-h', 1)``, so the traction is ``(-h' sxx + sxz, -h' sxz + szz)``.
    """
    from sweep.utils.curvilinear import CurvilinearGrid

    topo = _hill()
    sources, receivers = _geometry(topo)
    eq, prop = _make_prop(topography=topo)
    # the grid's own rows: the surface is row 0 of every column there
    _, snaps = prop(_wavelet(), sources, receivers, models=_models(),
                    return_wavefield=True, snapshot_grid="computational",
                    snapshot_times=[NT // 2, NT - 1])
    hp = CurvilinearGrid(topo, NZ, NX, DH).h_prime.numpy()
    snaps = snaps.detach().cpu().numpy()
    for k in range(snaps.shape[0]):
        sxx = snaps[k, 2, 0, 0][0, ABCN:ABCN + NX]
        szz = snaps[k, 3, 0, 0][0, ABCN:ABCN + NX]
        sxz = snaps[k, 4, 0, 0][0, ABCN:ABCN + NX]
        scale = max(np.abs(snaps[k, 2, 0, 0]).max(), 1e-30)
        assert np.abs(-hp * sxx + sxz).max() / scale < 5e-3
        assert np.abs(-hp * sxz + szz).max() / scale < 5e-3


def test_hill_long_time_bounded():
    """Stays bounded on a hill over a long window.

    The staggered chain-rule implementation this replaced diverged to
    ``inf`` on the same geometry.
    """
    nt_long = 2500
    topo = _hill()
    sources, receivers = _geometry(topo, src_x=NX // 4)
    _, prop = _make_prop(topography=topo)
    probes = list(np.linspace(0, nt_long - 1, 25).astype(int))
    syn, snaps = prop(_wavelet(nt_long), sources, receivers, models=_models(),
                      return_wavefield=True, snapshot_times=probes)
    assert torch.isfinite(syn).all()
    vmag = torch.sqrt(snaps[:, 0, 0, 0] ** 2 + snaps[:, 1, 0, 0] ** 2)
    peak = vmag.amax(dim=(-2, -1)).cpu().numpy()
    early = peak[:len(peak) // 3].max()
    late = peak[2 * len(peak) // 3:].max()
    assert late < 2.0 * max(early, 1e-30), (
        f"hill run grew: early {early:.3e}, late {late:.3e}")


def test_cfl_limit_reports_the_operator_bound():
    _, prop = _make_prop(topography=_hill())
    eq = prop.equation
    limit = eq.cfl_limit(2000.0)
    assert 0 < limit < 1.0
    assert DT < limit, "the test's own dt should be inside the stability bound"


def test_odd_spatial_order_is_rejected():
    with pytest.raises(ValueError):
        ElasticCurvilinear(spatial_order=7)


def test_unknown_surface_closure_is_rejected():
    with pytest.raises(ValueError):
        ElasticCurvilinear(spatial_order=SO, surface_closure="nope")


def test_impl_c_rejects_curvilinear_elastic():
    eq = ElasticCurvilinear(spatial_order=SO, device=DEVICE, backend="torch")
    with pytest.raises(NotImplementedError):
        eq._C()


def test_fractional_topography_is_kept_not_snapped_to_rows():
    """A boundary-fitted grid takes a smooth surface; snapping it to whole
    rows makes a staircase whose metric no longer matches the grid."""
    from sweep.utils.curvilinear import CurvilinearGrid

    x = np.arange(NX) * DH
    z = 60.0 * np.exp(-((x - NX*DH*0.5) ** 2) / (2.0 * 120.0 ** 2))
    smooth = (z.max() - z) / DH                      # fractional rows
    grid = CurvilinearGrid(smooth, NZ, NX, DH)
    h = grid.h_surface.numpy()
    assert np.allclose(h, smooth * DH, atol=1e-5), "fractional surface was rounded"
    # h' must be the derivative of the surface the mapping actually uses.
    assert np.allclose(grid.h_prime.numpy(), np.gradient(h, DH), atol=1e-5)
    # ... and the propagator must accept it without integer truncation.
    sources, receivers = _geometry(smooth)
    _, prop = _make_prop(topography=smooth)
    syn = prop(_wavelet(), sources, receivers, models=_models())
    assert torch.isfinite(syn).all()
    assert syn.abs().max().item() > 0


def test_integer_topography_metric_matches_its_own_grid():
    """Legacy integer input is smoothed once, and both the mapping and the
    slope come from that same smoothed surface."""
    from sweep.utils.curvilinear import CurvilinearGrid

    grid = CurvilinearGrid(_hill(), NZ, NX, DH)
    h = grid.h_surface.numpy()
    assert np.allclose(grid.h_prime.numpy(), np.gradient(h, DH), atol=1e-5)
    # D = z_bottom - h, so beta must be built from the same surface.
    z_bottom = (NZ - 1) * DH
    assert np.allclose(1.0 / grid.beta.numpy()[0], z_bottom - h, rtol=1e-5)


def test_source_footprint_is_band_limited_by_default():
    eq = ElasticCurvilinear(spatial_order=SO)
    k = eq.source_receiver_stencil
    assert k is not None and k.shape == (5, 5)
    assert k.sum() == pytest.approx(1.0, abs=1e-6)
    assert ElasticCurvilinear(spatial_order=SO,
                              source_spread=0).source_receiver_stencil is None


# --------------------------------------------------------------------------
# Geometry bookkeeping: one computational row is NOT dh
# --------------------------------------------------------------------------

def test_row_spacing_is_not_dh():
    """The mapping spreads eta over nz-1 rows but the physical column below
    the surface is only nz-1-topo rows deep, so a computational row spans
    D(x)/(nz-1) metres.  Assuming dh puts sources at the wrong depth."""
    import numpy as np
    from sweep.utils.curvilinear import CurvilinearGrid

    nz, nx, dh, surf = 192, 64, 5.0, 24
    grid = CurvilinearGrid(np.full(nx, float(surf)), nz_phys=nz, nx_phys=nx, dh=dh)
    spacing = grid.row_spacing()
    expected = ((nz - 1) - surf) * dh / (nz - 1)
    assert np.allclose(spacing, expected)
    assert abs(spacing[0] - dh) / dh > 0.10          # 12.6% short here


def test_physical_to_computational_round_trip():
    import numpy as np
    from sweep.utils.curvilinear import CurvilinearGrid

    nz, nx, dh = 192, 64, 5.0
    x = np.arange(nx, dtype=np.float64)
    topo = 24.0 - 10.0 * np.exp(-((x - nx / 2) ** 2) / (2.0 * 8.0 ** 2))
    grid = CurvilinearGrid(topo, nz_phys=nz, nx_phys=nx, dh=dh)
    for ix in (0, nx // 2, nx - 1):
        z_phys = topo[ix] + 6.0                       # six physical rows down
        _, i_eta = grid.physical_to_computational(ix, z_phys)
        back = grid.computational_to_physical(ix, i_eta)
        assert abs(float(back) - z_phys) < 1e-9
        # the naive index (6) lands somewhere else entirely
        assert abs(i_eta - 6.0) > 0.3
    # vectorised call must agree with the scalar one, element by element
    ixs = np.arange(nx)
    _, etas = grid.physical_to_computational(ixs, topo + 2.0)
    assert etas.shape == (nx,)
    for ix in (0, 3, nx // 2, nx - 1):
        _, one = grid.physical_to_computational(ix, topo[ix] + 2.0)
        assert abs(one - etas[ix]) < 1e-12
    # the helper must use the surface the MAPPING uses, not its rounded shadow
    assert np.allclose(np.asarray(grid.h_surface.detach().cpu()) / dh,
                       grid.computational_to_physical(ixs, 0.0))


def test_gradient_matches_finite_difference():
    """Adjoint gradient vs a central finite difference at the best-illuminated
    cell.  Small and cheap, but it is the only thing that actually validates
    the adjoint of the curvilinear step."""
    import numpy as np
    import torch
    from sweep.equations import ElasticCurvilinear
    from sweep.propagator.torch import PropTorch
    from sweep.signal import ricker

    dty = torch.float64
    nz, nx, dh, dt, nt, abcn = 56, 72, 10.0, 1.0e-3, 260, 20
    x = np.arange(nx) * dh
    topo = 8.0 - 3.0 * np.exp(-((x - nx * dh / 2) ** 2) / (2.0 * 120.0 ** 2))
    t = np.arange(nt, dtype=np.float64) * dt - 0.14
    wav = torch.tensor(1.0e3 * ricker(t, f=10.0), dtype=dty)
    # physical indices: three rows below the surface, receivers one below it
    src = torch.tensor([[nx // 2, topo[nx // 2] + 3.0]], dtype=torch.float64)
    rx = np.arange(6, nx - 6, 4)
    rec = torch.from_numpy(np.stack([rx, topo[rx] + 1.0], -1)[None])

    def run(vp, vs, rho):
        eq = ElasticCurvilinear(spatial_order=4, device="cpu", backend="torch")
        prop = PropTorch(eq, shape=(nz, nx), topography=topo, abcn=abcn,
                         dh=dh, dt=dt, use_ckpt=False, impl="eager",
                         eager_options={"use_compile": False})
        return prop(wav, src, rec, models=[vp, vs, rho])

    vp0 = torch.full((nz, nx), 2200.0, dtype=dty)
    vs0 = torch.full((nz, nx), 1100.0, dtype=dty)
    rho0 = torch.full((nz, nx), 2000.0, dtype=dty)
    vp_true = vp0.clone()
    vp_true[24:32, 30:42] += 150.0
    with torch.no_grad():
        obs = run(vp_true, vs0, rho0).detach()

    def misfit(vp):
        return 0.5 * ((run(vp, vs0, rho0) - obs) ** 2).sum()

    vp = vp0.clone().requires_grad_(True)
    misfit(vp).backward()
    g = vp.grad.detach()

    iz, ix = np.unravel_index(int(g.abs().argmax()), g.shape)
    h = 10.0
    js = []
    for sgn in (+1.0, -1.0):
        vpp = vp0.clone()
        vpp[iz, ix] += sgn * h
        with torch.no_grad():
            js.append(misfit(vpp).item())
    g_fd = (js[0] - js[1]) / (2 * h)
    assert abs(g_fd) > 0.0
    ratio = g[iz, ix].item() / g_fd
    assert abs(ratio - 1.0) < 0.05, f"adjoint/FD = {ratio:.4f} at ({iz}, {ix})"


# --------------------------------------------------------------------------
# Sources and receivers are PHYSICAL grid indices; the propagator maps them
# --------------------------------------------------------------------------

def _smooth_hill():
    x = np.arange(NX) * DH
    z = 60.0 * np.exp(-((x - NX * DH * 0.5) ** 2) / (2.0 * 120.0 ** 2))
    return (z.max() - z) / DH                        # fractional rows, 0 at the crest


def test_physical_indices_land_on_the_nearest_computational_row():
    topo = _smooth_hill()
    _, prop = _make_prop(topography=topo)
    ix = np.array([10, NX // 2, NX - 10])
    z = topo[ix] + 7.0                                # seven dh-rows below the surface
    got = prop._place_on_curvilinear(np.stack([ix, z], -1)[None], "source")
    _, eta = prop.to_computational(ix, z)
    assert got.dtype == torch.int64
    assert got[0, :, 0].tolist() == ix.tolist()
    assert got[0, :, 1].tolist() == np.rint(eta).astype(int).tolist()
    # on the flank a computational row is short of dh, so it is NOT row 7
    assert int(got[0, 0, 1]) != 7


def test_running_with_physical_indices_is_running_on_the_mapped_rows():
    """The mapping is the only change: physical indices give the very record
    the equivalent computational indices give with the mapping switched off."""
    topo = _smooth_hill()
    sources, receivers = _geometry(topo, src_x=NX // 3, src_depth=5.0)
    _, prop = _make_prop(topography=topo)
    a = prop(_wavelet(), sources, receivers, models=_models())
    src_c = prop._place_on_curvilinear(sources, "source")
    rec_c = prop._place_on_curvilinear(receivers, "receiver")
    impl = prop._backend_impl
    impl._place_on_curvilinear = lambda coords, role: coords
    try:
        b = prop(_wavelet(), src_c, rec_c, models=_models())
    finally:
        del impl._place_on_curvilinear
    assert torch.equal(a, b)


def test_a_flat_grid_maps_nothing():
    _, prop = _make_prop(topography=None)
    pts = np.array([[[5, 0], [20, 7], [NX - 1, NZ - 1]]])
    assert prop._place_on_curvilinear(pts, "receiver").tolist() == pts.tolist()


def test_a_point_in_the_air_is_refused():
    topo = _smooth_hill()
    _, prop = _make_prop(topography=topo)
    with pytest.raises(ValueError, match="above the surface"):
        prop._place_on_curvilinear(np.array([[10, topo[10] - 2.0]]), "source")


def test_a_receiver_on_an_integer_surface_row_is_on_the_surface():
    """An integer staircase is smoothed into the surface the grid uses, so a
    receiver on the integer row can sit a fraction of a row above it: it goes
    on the surface row, not into an error."""
    topo = _hill()
    _, prop = _make_prop(topography=topo)
    rec_x = np.arange(6, NX - 6, 5)
    got = prop._place_on_curvilinear(np.stack([rec_x, topo[rec_x]], -1)[None],
                                     "receiver")
    assert (got[0, :, 1] <= 1).all()


@pytest.mark.parametrize("pt,match", [([10.5, 20.0], "whole grid indices"),
                                      ([NX, 20.0], "outside"),
                                      ([10, NZ + 2.0], "outside")])
def test_off_grid_points_are_refused(pt, match):
    _, prop = _make_prop(topography=_smooth_hill())
    with pytest.raises(ValueError, match=match):
        prop._place_on_curvilinear(np.array([pt]), "source")


# --------------------------------------------------------------------------
# Snapshots come back on the Cartesian grid
# --------------------------------------------------------------------------

def _snap_run(prop, topo, **kw):
    sources, receivers = _geometry(topo, src_x=NX // 3, src_depth=5.0)
    _, snaps = prop(_wavelet(), sources, receivers, models=_models(),
                    return_wavefield=True, snapshot_times=[NT // 2, NT - 1], **kw)
    return snaps.detach().cpu()


def test_snapshots_are_the_grid_rows_interpolated_per_column():
    """The default snapshot is the computational one, each Cartesian row
    interpolated from the grid's rows around its depth -- checked against a
    plain per-column np.interp, and against a second run that asks for it."""
    topo = _smooth_hill()
    _, prop = _make_prop(topography=topo)
    comp = _snap_run(prop, topo, snapshot_grid="computational")
    phys = _snap_run(prop, topo)
    mapped = prop._curvilinear_snapshots(comp.clone())
    scale = float(comp.abs().max())
    assert float((phys - mapped).abs().max()) <= 1e-6 * scale
    # the reference: x = model column, rows below the surface by depth
    grid = prop.curvilinear_grid
    surf = grid.h_surface.numpy() / DH
    k, f = 0, 1                                     # first snapshot, vz
    for ix in (5, NX // 3, NX // 2, NX - 6):
        col = comp[k, f, 0, 0, :, ABCN + ix].numpy()
        rows = np.arange(NZ, dtype=np.float64)
        eta = (rows - surf[ix]) * (NZ - 1) / ((NZ - 1) - surf[ix])
        ref = np.interp(eta, np.arange(col.shape[0]), col)
        wet = rows >= surf[ix]
        got = mapped[k, f, 0, 0, :NZ, ABCN + ix].numpy()
        assert np.allclose(got[wet], ref[wet], rtol=1e-5, atol=1e-6 * scale)
        assert (got[~wet] == 0).all()


def test_air_cells_are_zero_in_every_field():
    topo = _smooth_hill()
    _, prop = _make_prop(topography=topo)
    phys = _snap_run(prop, topo)
    surf = prop.curvilinear_grid.h_surface.numpy() / DH
    air = np.arange(NZ)[:, None] < surf[None, :]
    assert air.any()
    assert (phys[..., :NZ, ABCN:ABCN + NX][..., torch.from_numpy(air)] == 0).all()
    assert float(phys.abs().max()) > 0


def test_flat_grid_snapshots_are_returned_as_they_are():
    _, prop = _make_prop(topography=None)
    snaps = torch.randn(2, 3, 1, 1, *prop.shape)
    assert prop._curvilinear_snapshots(snaps) is snaps


@pytest.mark.parametrize("kw,match", [(dict(return_wavefield=True, snapshot_grid="grid"), "snapshot_grid"),
                                      (dict(snapshot_grid="physical"), "return_wavefield")])
def test_snapshot_grid_is_checked(kw, match):
    topo = _smooth_hill()
    _, prop = _make_prop(topography=topo)
    sources, receivers = _geometry(topo)
    with pytest.raises(ValueError, match=match):
        prop(_wavelet(), sources, receivers, models=_models(), **kw)


# --------------------------------------------------------------------------
# Models are on the Cartesian grid too; the propagator samples them
# --------------------------------------------------------------------------

def _sampled(prop, m, topo):
    """Each column at the depth of each grid row (``computational_to_physical``)
    by np.interp, no higher than the first row at or below the surface the
    caller gave."""
    grid = prop.curvilinear_grid
    z = grid.computational_to_physical(np.arange(NX)[None, :], np.arange(NZ)[:, None])
    z = np.maximum(z, np.ceil(np.asarray(topo, dtype=np.float64) - 1e-4)[None, :])
    m = m.detach().cpu().double().numpy()
    return np.stack([np.interp(z[:, i], np.arange(NZ), m[:, i]) for i in range(NX)], -1)


def _layered(topo=None, air=None):
    """vp, vs, rho with an interface at row 30; ``air`` fills the rows above
    ``topo``."""
    z = np.arange(NZ)[:, None]
    out = []
    for a, b in ((2000.0, 2400.0), (1150.0, 1350.0), (1800.0, 2100.0)):
        m = np.where(np.broadcast_to(z >= 30, (NZ, NX)), b, a)
        if air is not None:
            m = np.where(z < np.asarray(topo)[None, :], air, m)
        out.append(torch.tensor(m, dtype=torch.float32).to(DEVICE))
    return out


@pytest.mark.parametrize("topo", [_smooth_hill(), _hill()], ids=["fractional", "integer"])
def test_models_are_sampled_at_the_depth_of_each_row(topo):
    _, prop = _make_prop(topography=topo)
    m = 1500.0 + 1000.0 * torch.rand(NZ, NX, generator=torch.Generator().manual_seed(0))
    (got,) = prop._curvilinear_models([m])
    np.testing.assert_allclose(got.double().numpy(), _sampled(prop, m, topo), rtol=1e-6)


def test_the_forward_runs_on_the_sampled_models():
    topo = _smooth_hill()
    _, prop = _make_prop(topography=topo)
    sources, receivers = _geometry(topo)
    models = _layered()
    prop(_wavelet(8), sources, receivers, models=models)
    xl, xh, zl, zh = prop._runtime_padding()
    for used, m in zip(prop.models_padded, models):
        got = used[zl:used.shape[-2] - zh, xl:used.shape[-1] - xh].cpu()
        want = torch.as_tensor(_sampled(prop, m, topo), dtype=torch.float32)
        assert torch.allclose(got, want, rtol=1e-6, atol=0)
        assert not torch.equal(got, m.cpu())


def test_a_constant_model_and_a_flat_grid_are_left_as_they_are():
    _, prop = _make_prop(topography=_smooth_hill())
    m = torch.full((NZ, NX), 2000.0)
    assert torch.equal(prop._curvilinear_models([m])[0], m)
    _, flat = _make_prop(topography=None)
    assert flat._curvilinear_models([m])[0] is m


def test_the_air_is_not_read_and_gets_no_gradient():
    """Rows above the surface are the air: what they hold changes nothing (the
    zeros here would be infinities if read), and their gradient is zero. An
    integer staircase is smoothed into the surface the grid uses, which then
    sits above it in places: still the air."""
    topo = _hill()
    _, prop = _make_prop(topography=topo)
    sources, receivers = _geometry(topo)
    rock = _layered()
    vp = rock[0].clone().requires_grad_(True)
    a = prop(_wavelet(), sources, receivers, models=[vp] + rock[1:])
    a.square().sum().backward()
    b = prop(_wavelet(), sources, receivers, models=_layered(topo, air=0.0))
    assert torch.equal(a.detach(), b)
    air = torch.as_tensor(np.arange(NZ)[:, None] < topo[None, :]).to(DEVICE)
    assert air.any() and (vp.grad[air] == 0).all()
    assert float(vp.grad[~air].abs().max()) > 0


def test_gradient_on_a_layered_model_matches_finite_difference():
    """The gradient is the physical grid's: central differences of the misfit
    at cells under the surface, either side of an interface, and in the air,
    where both are zero."""
    nz, nx, dh, dt, nt, abcn = 56, 72, 10.0, 1.0e-3, 260, 20
    x = np.arange(nx) * dh
    topo = 8.0 - 3.0 * np.exp(-((x - nx * dh / 2) ** 2) / (2.0 * 120.0 ** 2))
    t = np.arange(nt, dtype=np.float64) * dt - 0.14
    wav = torch.tensor(1.0e3 * ricker(t, f=10.0), dtype=torch.float32)
    src = torch.tensor([[nx // 2, topo[nx // 2] + 3.0]], dtype=torch.float64)
    rx = np.arange(6, nx - 6, 4)
    rec = torch.from_numpy(np.stack([rx, topo[rx] + 1.0], -1)[None])
    eq = ElasticCurvilinear(spatial_order=4, device="cpu", backend="torch")
    prop = PropTorch(eq, shape=(nz, nx), topography=topo, abcn=abcn, dh=dh,
                     dt=dt, use_ckpt=False, impl="eager",
                     eager_options={"use_compile": False})
    deep = np.broadcast_to(np.arange(nz)[:, None] >= 22, (nz, nx))

    def layered(a, b):
        return torch.tensor(np.where(deep, b, a), dtype=torch.float32)

    vp0, vs0, rho0 = layered(2000.0, 2300.0), layered(1000.0, 1200.0), layered(1900.0, 2100.0)
    vp_true = vp0.clone()
    vp_true[12:20, 28:44] += 120.0
    with torch.no_grad():
        obs = prop(wav, src, rec, models=[vp_true, vs0, rho0])

    def misfit(vp):
        with torch.no_grad():
            return float(0.5 * ((prop(wav, src, rec, models=[vp, vs0, rho0]) - obs).double() ** 2).sum())

    vp = vp0.clone().requires_grad_(True)
    (0.5 * ((prop(wav, src, rec, models=[vp, vs0, rho0]) - obs) ** 2).sum()).backward()
    g = vp.grad.double()
    xf = nx // 2 + 14                       # on the flank: the surface is row 6.48
    s = int(np.ceil(topo[xf]))
    h = 10.0
    for name, (iz, ix) in {"first row under the surface": (s, xf),
                           "the next": (s + 1, xf),
                           "above the interface": (21, nx // 2 - 6),
                           "below it": (22, nx // 2 - 6)}.items():
        js = []
        for sgn in (1.0, -1.0):
            vpp = vp0.clone()
            vpp[iz, ix] += sgn * h
            js.append(misfit(vpp))
        g_fd = (js[0] - js[1]) / (2 * h)
        assert abs(g[iz, ix].item()) > 1e-3 * float(g.abs().max()), name
        ratio = g[iz, ix].item() / g_fd
        assert abs(ratio - 1.0) < 0.05, f"{name}: adjoint/FD = {ratio:.4f} at ({iz}, {ix})"
    vpp = vp0.clone()
    vpp[s - 1, xf] += 500.0                 # in the air, under a row of the surface
    assert g[s - 1, xf] == 0 and misfit(vpp) == misfit(vp0)
