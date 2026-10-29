"""Quantised boundary-saving ring on the CUDA path
(``BoundarySaving(storage_dtype=...)``).

The ring is ``O(nt * nband)`` and is the dominant boundary-saving allocation, so
storing it as bf16 or int8 is the main memory lever after rim mode. int8 uses
per-(step, field) symmetric quantisation with an fp32 amax table, mirroring the
eager implementation in ``sweep.sem.boundary_saving``.

Note which assertion does the work. The ACCURACY tests do not discriminate on
their own: if ``band_kind`` were dropped somewhere in the plumbing the ring would
silently stay fp32 and every cosine would get BETTER. What pins the feature is
``test_quantised_ring_actually_shrinks_the_allocation`` -- ignoring the request
cannot make the memory go down -- plus the cross-impl test, which a wrong scale
factor would break.

Configuration is the discriminating one from test_sem_bs_rim (4-sided C-PML,
off-centre source, 1500/4500 contrast); dt comes from the smallest GLL spacing.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving
from sweep.sem import PropSEM
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NZ, NX, DH, F0, ABCN, N_ORDER = 49, 65, 20.0, 8.0, 6, 4
NT, VP_LO, VP_HI, SRC_X_FRAC = 900, 1500.0, 4500.0, 0.15
DT = 0.30 * (0.172 * N_ORDER * DH) / VP_HI
CROP = 8
STORES = {"bf16": torch.bfloat16, "int8": torch.int8}


def _bs(store):
    """Boundary saving with its ring stored as ``store`` (None: fp32)."""
    return BoundarySaving(storage_dtype={None: "fp32", torch.bfloat16: "bf16",
                                         torch.int8: "int8"}.get(store, store))


def _vp():
    z, x = np.arange(NZ)[:, None], np.arange(NX)[None, :]
    return (VP_LO + (VP_HI - VP_LO) * z / (NZ - 1)
            + 0.08 * (VP_HI - VP_LO) * np.sin(x / 9.0)).astype(np.float32)


def _wav():
    t = np.arange(NT, dtype=np.float32) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                        device="cuda")


def _run(impl, memory, mode, store, backward=True):
    p = PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                n_order=N_ORDER, free_surface=False,
                abcn=ABCN * N_ORDER, f_dominant=F0, dev="cuda",
                dtype=torch.float32, impl=impl,
                memory=None if memory is None else _bs(store),
                bs_band_mode=mode)
    src = np.array([[int(SRC_X_FRAC * NX), 4]], dtype=np.int64)
    rix = np.arange(3, NX - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 4)], -1)[None]
    vp = torch.tensor(_vp(), device="cuda", requires_grad=True)
    s = p(_wav(), src, rec, models=[vp])
    assert torch.isfinite(s).all(), "forward diverged (check the GLL CFL)"
    if not backward:
        del p, s
        return None
    (s ** 2).sum().backward()
    g = vp.grad.detach().cpu().numpy().astype(np.float64)
    del p, vp, s
    gc.collect()
    torch.cuda.empty_cache()
    return g


def _cos(a, b, crop=CROP):
    sl = (slice(crop, NZ - crop), slice(crop, NX - crop))
    u, v = a[sl].reshape(-1), b[sl].reshape(-1)
    return float(u @ v) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)))


@pytest.mark.parametrize("name", list(STORES))
@pytest.mark.parametrize("mode", ["full", "rim"])
def test_quantised_ring_matches_full_tape(name, mode):
    ref = _run("c", None, "full", None)
    g = _run("c", "boundary", mode, STORES[name])
    assert _cos(g, ref) >= 1 - 1e-4, f"{mode}/{name} cos={_cos(g, ref):.8f}"


@pytest.mark.parametrize("name", list(STORES))
def test_cuda_agrees_with_eager_at_the_same_precision(name):
    """Two independent implementations of the same quantisation.

    A wrong scale factor on either side shows up here even though both would
    still look plausible against the full tape on their own.
    """
    ge = _run("eager", "boundary", "rim", STORES[name])
    gc_ = _run("c", "boundary", "rim", STORES[name])
    assert _cos(gc_, ge) >= 1 - 1e-5, f"{name} c-vs-eager cos={_cos(gc_, ge):.8f}"


@pytest.mark.parametrize("name,itemsize", [("bf16", 2), ("int8", 1)])
def test_quantised_ring_actually_shrinks_the_allocation(name, itemsize):
    """The assertion that a silently-ignored band_kind cannot pass.

    Judged on the SAVING, not on the peak ratio: the peak also carries ~22 MB of
    fixed state that has nothing to do with the ring, which drags a 4x ring
    reduction down to a 2.9x peak reduction and would make a ratio bound either
    meaningless or wrong.
    """
    from sweep.sem.boundary_saving import cpml_node_masks, band_support_idx
    p = PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                n_order=N_ORDER, free_surface=False,
                abcn=ABCN * N_ORDER, f_dominant=F0, dev="cuda",
                dtype=torch.float32, impl="c")
    eq = p._build_eq(torch.tensor(_vp(), device="cuda"))
    band = cpml_node_masks(eq)[1].to(torch.int64)
    # chi rings the whole band; psi_ax / zeta_ax ring only the band nodes
    # their damping coefficient is non-zero on (the per-field ring split)
    n_ring = int(band.numel()) + 2 * sum(
        int(band_support_idx(band, getattr(eq.cpml, f"b_{ax}")).numel())
        for ax in "xz")
    del p, eq
    ring_fp32 = NT * n_ring * 4
    want = 0.9 * ring_fp32 * (1 - itemsize / 4)    # 10% slack on the accounting

    peaks = {}
    for tag, store in (("fp32", None), (name, STORES[name])):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _run("c", "boundary", "full", store, backward=False)
        peaks[tag] = torch.cuda.max_memory_allocated()
        gc.collect()
        torch.cuda.empty_cache()
    saved = peaks["fp32"] - peaks[name]
    assert saved >= want, (
        f"{name} saved {saved/1e6:.1f} MB of the {ring_fp32/1e6:.1f} MB ring "
        f"(peaks {peaks['fp32']/1e6:.1f} -> {peaks[name]/1e6:.1f} MB); "
        f"expected >= {want/1e6:.1f} MB")


def test_every_boundary_saving_path_honours_the_store_dtype():
    """The inverse of the test that used to live here.

    For three releases this slot held "the one path that still refuses" --
    3-D acoustic, then 2-D elastic, then coupled fluid-solid -- because the
    failure mode being guarded was a path SILENTLY storing fp32 while
    reporting success. Every SEM boundary-saving path takes the flag now, so
    the guard flips: ask each of them for int8 and assert the ring really is
    int8. A path that regrows the silent-fp32 behaviour fails here.
    """
    import numpy as np
    import sweep.sem as S
    from sweep.sem.dd import (SEMModelParallel, StripTopology,
                              _SEMAcousticCpmlDDC, _SEMAcoustic3DCpmlDDC,
                              _SEMElasticCpmlDDC, _SEMFluidSolidDDC)
    from sweep.sem.equations import (AcousticElasticSEM2D,
                                     AcousticPotentialSEM2D, AcousticSEM3D,
                                     ElasticSEM2DDeformed)
    no, dh, dt, nt, f0 = 4, 25.0, 1.0e-3, 24, 6.0

    def grids(eq, shape, **kw):
        prop = S.PropSEM(eq, shape=shape, dh=dh, dt=dt, nt=nt, n_order=no,
                         abcn=2 * no, f_dominant=f0, dev="cuda", impl="c",
                         memory=BoundarySaving(storage_dtype="int8"), **kw)
        return prop

    def node_rings(fn, prop, models, src, rec):
        """Run one DD step group through ``fn`` and return the ring tensors."""
        seen = {}
        real = fn.forward

        def spy(ctx, wav, parts, exchange, bs, *mats):
            out = real(ctx, wav, parts, exchange, bs, *mats)
            seen["rings"] = ([list(cr.values()) for cr in ctx.carry]
                             if hasattr(ctx, "carry") else
                             ctx.rings if hasattr(ctx, "rings") else ctx.saves)
            return out

        fn.forward = staticmethod(spy)
        try:
            SEMModelParallel(prop, StripTopology(2))(
                torch.zeros(nt, device="cuda"), src, rec, models)
        finally:
            fn.forward = staticmethod(real)
        return seen["rings"]

    def flat(x, out):
        if torch.is_tensor(x):
            out.append(x)
        elif isinstance(x, (tuple, list)):
            for y in x:
                flat(y, out)
        return out

    cases = []
    # 2-D acoustic potential
    p = grids(AcousticPotentialSEM2D, (25, 49), free_surface=("top",))
    vp = (2000.0 + 300.0 * torch.rand(25, 49)).cuda().requires_grad_(True)
    eq, _ = p._resolve_model_batch([vp])
    nxg = eq.mesh.nex * no + 1
    cases.append((_SEMAcousticCpmlDDC, p, [vp], 2 * no * nxg + 5 * no,
                  (torch.arange(4, nxg - 4, 4) + nxg).cuda()))
    # 2-D elastic
    p = grids(ElasticSEM2DDeformed, (25, 49), free_surface=False)
    z = torch.linspace(0, 1, 25)[:, None].expand(25, 49)
    ms = [(2400.0 + 600.0 * z).contiguous().float().cuda().requires_grad_(True),
          (1300.0 + 350.0 * z).contiguous().float().cuda().requires_grad_(True),
          (2100.0 + 200.0 * z).contiguous().float().cuda().requires_grad_(True)]
    eq, _ = p._resolve_model_batch([t.detach() for t in ms])
    nxg = eq.mesh.nex * no + 1
    cases.append((_SEMElasticCpmlDDC, p, ms, 3 * no * nxg + 5 * no,
                  (torch.arange(4, nxg - 4, 4) + 2 * nxg).cuda()))
    # coupled fluid-solid
    p = grids(AcousticElasticSEM2D, (25, 49), free_surface=("top",),
              receiver_type=["p", "vx", "vz"])
    m = p.mesh
    fl = (m.z_e.flatten(1).mean(1) < 8 * dh).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    ms = [torch.where(fl, 1500.0, 2300.0).float().cuda().requires_grad_(True),
          torch.where(fl, 0.0, 1250.0).float().cuda().requires_grad_(True),
          torch.where(fl, 1000.0, 2100.0).float().cuda().requires_grad_(True)]
    eq, _ = p._resolve_model_batch([t.detach() for t in ms])
    nxg = eq.mesh.nex * no + 1
    cases.append((_SEMFluidSolidDDC, p, ms, 2 * no * nxg + 5 * no,
                  (torch.arange(4, nxg - 4, 4) + 9 * nxg).cuda()))
    # 3-D acoustic
    p = grids(AcousticSEM3D, (17, 13, 25), free_surface=("top",))
    vp3 = (2500.0 + 200.0 * torch.rand(17, 13, 25)).cuda().requires_grad_(True)
    eq, _ = p._resolve_model_batch([vp3])
    nxg, nyg = eq.mesh.nex * no + 1, eq.mesh.ney * no + 1
    gid = lambda ix, iy, iz: (iz * nyg + iy) * nxg + ix
    cases.append((_SEMAcoustic3DCpmlDDC, p, [vp3],
                  gid(nxg // 2, nyg // 2, 2 * no),
                  torch.tensor([gid(x, nyg // 2, no)
                                for x in range(3, nxg - 3, 3)]).cuda()))

    for fn, prop, models, src, rec in cases:
        rings = node_rings(fn, prop, models, src, rec)
        quantised = [t for t in flat(rings, []) if t.dtype == torch.int8]
        assert quantised, (
            f"{prop.equation.__name__}: asked for int8 and got no int8 ring "
            "-- the flag is being dropped")


# --------------------------------------------------------------------------- #
# 3-D. Seven ring fields instead of five, so the ring dominates the run even
# more: measured 2.17 GB of a 2.45 GB peak at 48^3 grid points and nt = 300,
# which int8 cuts to 0.82 GB -- 3.0x less memory for the whole propagation.
# --------------------------------------------------------------------------- #
NE3, DH3, F0_3, VP3, NT3, ABCN3 = 12, 40.0, 6.0, 2500.0, 300, 3
N3 = NE3 * 4 + 1


def _vp3():
    """A contrast, not a smooth gradient: reflections have to reach the band."""
    zc, yc, xc = np.meshgrid(np.arange(N3), np.arange(N3), np.arange(N3),
                             indexing="ij")
    return (VP3 * (1 + 0.5 * zc / (N3 - 1))
            + 0.10 * VP3 * np.sin(xc / 7.0)
            + 0.05 * VP3 * np.cos(yc / 9.0)).astype(np.float32)


# dt from the model's ACTUAL maximum and the smallest GLL spacing. Taking it
# from the nominal VP3 instead left the run marginally over the CFL once the
# contrast was added, which inflated the int8 error 15x and failed the bar --
# the error was in the test setup, not in the quantisation.
DT3 = 0.30 * (0.172 * 4 * DH3) / float(_vp3().max())


def _run3(impl, store, backward=True, memory="boundary"):
    from sweep.sem.equations.acoustic3d import AcousticSEM3D
    p = PropSEM(AcousticSEM3D, shape=(N3, N3, N3), dh=DH3, dt=DT3, nt=NT3,
                n_order=4, free_surface=True,
                abcn=ABCN3 * 4, f_dominant=F0_3, dev="cuda",
                dtype=torch.float32, impl=impl,
                memory=None if memory is None else _bs(store))
    vp = _vp3()
    t = np.arange(NT3) * DT3
    a = (np.pi * F0_3 * (t - 1.5 / F0_3)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device="cuda")
    src = np.array([[int(0.2 * N3), int(0.3 * N3), 3]], dtype=np.int64)
    rix = np.arange(4, N3 - 4, 3)
    rec = np.stack([rix, np.full_like(rix, N3 // 2),
                    np.full_like(rix, 3)], -1)[None]
    m = torch.tensor(vp, device="cuda", requires_grad=True)
    s = p(wav, src, rec, models=[m])
    assert torch.isfinite(s).all()
    if not backward:
        del p, s
        return None
    (s ** 2).sum().backward()
    out = m.grad.cpu().numpy().astype(np.float64)
    del p, m, s
    gc.collect()
    torch.cuda.empty_cache()
    return out


@pytest.mark.parametrize("name", list(STORES))
def test_3d_quantised_ring_matches_eager(name):
    ge = _run3("eager", STORES[name])
    gc_ = _run3("c", STORES[name])
    assert _cos3(gc_, ge) >= 1 - 1e-6, f"3-D {name} cos={_cos3(gc_, ge):.10f}"


@pytest.mark.parametrize("name,itemsize", [("bf16", 2), ("int8", 1)])
def test_3d_quantised_ring_actually_shrinks_the_allocation(name, itemsize):
    """Again the assertion a silently-ignored band_kind cannot pass.

    The budget is the ring's own account, not a number written down once:
    u rings the whole band, and each psi_ax / zeta_ax only the band nodes
    where b_ax != 0 (the per-field ring split) -- so it is
    |band| + 2 (|band_x| + |band_y| + |band_z|) slots per step. (This used to
    be 700 / 1100 MB, taken from the union layout's 7 |band|; the split cut
    the fp32 ring to ~975 MB here and the saving with it.)"""
    from sweep.sem.boundary_saving import cpml_node_masks, band_support_idx
    from sweep.sem.equations.acoustic3d import AcousticSEM3D
    p = PropSEM(AcousticSEM3D, shape=(N3, N3, N3), dh=DH3, dt=DT3, nt=NT3,
                n_order=4, free_surface=True,
                abcn=ABCN3 * 4, f_dominant=F0_3, dev="cuda",
                dtype=torch.float32, impl="c")
    eq = p._build_eq(torch.tensor(_vp3(), device="cuda"))
    band = cpml_node_masks(eq)[1].to(torch.int64)
    n_ring = int(band.numel()) + 2 * sum(
        int(band_support_idx(band, getattr(eq.cpml, f"b_{ax}")).numel())
        for ax in "xyz")
    del p, eq
    ring_fp32 = NT3 * n_ring * 4 / 1e6
    least_mb = 0.9 * ring_fp32 * (1 - itemsize / 4)    # 10% slack
    peaks = {}
    for tag, store in (("fp32", None), (name, STORES[name])):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _run3("c", store, backward=False)
        peaks[tag] = torch.cuda.max_memory_allocated() / 1e6
        gc.collect()
        torch.cuda.empty_cache()
    saved = peaks["fp32"] - peaks[name]
    assert saved >= least_mb, (
        f"3-D {name} saved {saved:.0f} MB of the {ring_fp32:.0f} MB ring "
        f"(peaks {peaks['fp32']:.0f} -> {peaks[name]:.0f} MB); "
        f"expected >= {least_mb:.0f} MB")


def _cos3(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_fp16_ring_is_refused(impl):
    # fp16 is a storage_dtype BoundarySaving takes on PropTorch; PropSEM refuses it
    # on both backends instead of storing it as fp32 or as a plain fp16 cast,
    # which flushed the SEM's small field values to zero
    with pytest.raises(NotImplementedError, match="fp16"):
        _run(impl, "boundary", "full", "fp16")


# Per-precision bars, each about 10x above what is measured on this
# configuration (49^3, nt = 300, 5-sided band, off-centre source, vp 2500-4125):
#
#   fp32  1 - cos = 1.7e-13   at the run-to-run atomicAdd floor (1.3e-13)
#   bf16  1 - cos = 6.0e-08
#   int8  1 - cos = 9.6e-07
#
# A single bar for all three would be arbitrary: int8 IS less accurate, and
# pinning it at the fp32 bar only produces a flaky test. How much less depends
# on how much energy reaches the absorbing band -- a milder contrast measured
# 7.4e-08 for int8 on the same code, an order of magnitude better.
_FULL_TAPE_BAR = {"fp32": 1e-9, "bf16": 1e-6, "int8": 1e-5}


@pytest.mark.parametrize("name", ["fp32", "bf16", "int8"])
def test_3d_quantised_ring_matches_the_full_tape(name):
    """The bar that matters, and the one the eager-vs-c test above does NOT set.

    Comparing eager-BS-int8 against c-BS-int8 checks that two implementations of
    the same quantisation agree; it says nothing about whether the quantised
    ring reproduces the exact gradient. This compares against the save-all tape.
    """
    store = {"fp32": None, "bf16": torch.bfloat16, "int8": torch.int8}[name]
    ref = _run3("c", None, memory=None)
    g = _run3("c", store)
    c = _cos3(g, ref)
    bar = _FULL_TAPE_BAR[name]
    assert c >= 1 - bar, f"3-D {name} vs full tape: 1-cos={1-c:.2e} > {bar:.0e}"


def test_3d_full_tape_reference_is_reproducible():
    """Pins the floor the test above is judged against.

    The assembly uses atomicAdd, so two identical runs already differ; without
    this, a cosine of 1 - 7e-8 for int8 could not be called small.
    """
    a = _run3("c", None, memory=None)
    b = _run3("c", None, memory=None)
    assert _cos3(a, b) >= 1 - 1e-9, f"floor cos={_cos3(a, b):.12f}"


def test_every_FUSED_path_stamps_the_store_dtype_onto_its_consts():
    """The companion to the DD test above, on the single-domain paths.

    The DD drivers were guarded; the seven FUSED forward paths were not, and
    one of them -- fluid-solid -- set ``c["bs"]`` and dropped both precision
    flags, so ``_SEMFluidSolidC.forward``'s ``int(c.get("bs_kind", 0))``
    silently fell back to fp32 rings. The feature's own kernels had supported
    quantised rings there since they were written; only the Python flag was
    missing, and the suite could not see it because nothing asserted on the
    single-domain side. Measured once it was connected, on this file's own
    fluid-solid configuration: boundary-saved peak 75.8 -> 46.1 MB (bf16) ->
    33.0 MB (int8), with the gradients moving 1e-12 to 3e-11 in ``1 - cos``.

    Asserting on the consts rather than on the ring dtype is deliberate: it
    covers every fused path with one mechanism, including those whose rings
    live entirely inside the C driver.
    """
    import numpy as np
    import sweep.sem as S
    import sweep.sem.prop_c as PC
    from sweep.sem.equations import (AcousticElasticSEM2D,
                                     AcousticPotentialSEM2D, AcousticSEM3D)

    no, dh, dt, nt, f0 = 4, 25.0, 1.0e-3, 16, 6.0
    seen = {}

    def spy_on(fn_name):
        import sweep.sem.dd as DDm
        fn = getattr(PC, fn_name, None) or getattr(DDm, fn_name)
        real = fn.forward

        def spy(ctx, *a):
            c = next((x for x in reversed(a) if isinstance(x, dict)), None)
            if c is None:                    # the strip driver: parts carry the consts
                parts = next((x for x in a if isinstance(x, list) and x and hasattr(x[0], "c")), None)
                c = parts[0].c if parts else None
            if c is not None and "bs" in c:
                seen[fn_name] = (c.get("bs"), c.get("bs_kind"),
                                 c.get("bs_kind_e"))
            return real(ctx, *a)

        fn.forward = staticmethod(spy)
        return fn, real

    cases = [
        ("_SEMAcousticCpmlC", AcousticPotentialSEM2D, (33, 49), {}),
        ("_SEMAcoustic3DCpmlDDC", AcousticSEM3D, (17, 17, 17), {}),
        ("_SEMFluidSolidC", AcousticElasticSEM2D, (33, 49),
         dict(receiver_type=["vx", "vz"])),
    ]
    for fn_name, eq, shape, kw in cases:
        fn, real = spy_on(fn_name)
        try:
            p = S.PropSEM(eq, shape=shape, dh=dh, dt=dt, nt=nt, n_order=no,
                          free_surface=("top",), abcn=2 * no,
                          f_dominant=f0, dev="cuda", impl="c",
                          memory=BoundarySaving(storage_dtype="int8"), **kw)
            nd = len(shape)
            mid = [s // 2 for s in shape[::-1]]          # FD column order
            src = np.array([mid], dtype=np.int64)
            rec = np.array([[mid]], dtype=np.int64)
            ms = [torch.full(shape, v, device="cuda")
                  for v in ([1500.0, 0.0, 1000.0] if eq is AcousticElasticSEM2D
                            else ([1500.0, 1000.0]
                                  if eq is AcousticPotentialSEM2D
                                  else [1500.0]))]
            ms = [m.requires_grad_(True) for m in ms[:len(p.model_names)]]
            p(torch.zeros(nt, device="cuda"), src, rec, models=ms)
        finally:
            fn.forward = staticmethod(real)
        assert fn_name in seen, f"{fn_name} was never reached by this case"
        bs, kind, kind_e = seen[fn_name]
        assert bs is True, f"{fn_name}: memory=BoundarySaving() did not reach it"
        assert kind == 2, (
            f"{fn_name}: asked for int8 node rings, consts carry bs_kind="
            f"{kind} (0 = fp32, i.e. the flag was dropped)")
        assert kind_e == 2, (
            f"{fn_name}: asked for int8, consts carry bs_kind_e={kind_e}; "
            "element rings would silently stay fp32")
