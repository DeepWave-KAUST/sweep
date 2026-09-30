"""VRZ (variable-density) gradients under boundary saving: the rim and the
imaging time offset, 2-D and 3-D, impl='c'.

Three things pinned here, all found on field-data int8 gradients:

1. Imaging time offset.  The full/checkpoint backward imaged U_{it+1} (the
   deferred history capture ran after the buffer rotation) and the
   boundary-saving backward imaged U_{it-1} (post-swap u_now).  Every mode was
   one step off in a different direction, so bs vs full disagreed by rel 0.12
   while each still passed the loose eager gate.  Fixed: capture before the
   rotation, image U_it in every mode.  Guarded by full == ckpt == bs.
2. Shell width.  The VRZ vp gradient contains div(lambda*vp*grad p): a
   divergence of a gradient, reach 2M, while the reverse step only needs M.
   With the M+1 shell the outermost physical cells imaged cells that were
   neither restored nor reconstructed (zeros), so the outermost line of
   bs - full was ~250x the interior.  Fixed: 2M+1 shell at offset -2M.
   Guarded by the outermost-line ratio below.
3. p_tt imaging.  2*vp*lambda*lap(p) is replaced through the forward identity
   U_{it+1} - 2U_it + U_{it-1} = dt^2*kappa*(beta*lap(p) + grad b . grad p),
   discretely exact, so grad_vp takes no second spatial derivative of the
   stored/reconstructed field (the int8 rim shrank 3-3.5x on field data).  Exact
   means full == ckpt == bs to rounding, and c == eager as before: both are
   asserted.
4. sigma=0 boundary buffer.  The two divergence terms still reach 2M, so
   VRZ declares BOUNDARY_BUFFER_REACH=1 and the propagator pads M+1 replicate
   cells between the physical box and the damping ramp (every memory
   strategy and impl, so the modes share one grid); the shell went back to
   M+1 at offset -M and sits out of the imaging reach.  The rim is gone
   (outermost line ~0.3-0.8x the interior) and the boundary memory is back
   to the M+1 level.  Pinned by the bookkeeping tests at the end.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import ricker as _ricker  # the shared test wavelet (bit-identical after the float32 cast)

SO = 4
DH, DT = 10.0, 1.5e-3
# Measured on the fixed tree (order 4, the grids below): bs-full rel 3.5e-4 (2-D)
# / 9e-8 (3-D), edge ratio 0.4-0.9; c-vs-eager 8e-4.  Pre-fix: offset bug rel
# 0.12; M+1 shell rel 9.7e-3 and edge ratio 109 (2-D) / 1.2e5 (3-D).
REL_TOL_MODES = 5e-3     # bs / ckpt vs full (fp32)
EDGE_RATIO_TOL = 5.0     # outermost line of (bs - full) / interior
REL_TOL_EAGER = 1e-3     # c full vs eager (fused CPML form), 2-D interior; measured 1.4e-4
# int8 rim (bs int8 - full), outermost line / interior.  With the sigma=0
# buffer the shell is out of the imaging stencil's reach and the ratio is
# ~0.3-0.8 (2-D / 3-D, vp and z).  Without it: vp 1.5 / 2.3 (p_tt imaging),
# 7.8 / 12.3 (Laplacian imaging); z 3.9 / 9.1.
INT8_EDGE_TOL = {"vp": 2.0, "z": 2.0}


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


def _ramp(shape, top, bottom):
    nz = shape[0]
    ramp = np.linspace(top, bottom, nz, dtype=np.float32)
    return np.broadcast_to(ramp.reshape((nz,) + (1,) * (len(shape) - 1)), shape).copy()


def _setup(ndim):
    if ndim == 2:
        shape, abcn, nt = (64, 80), 20, 200
        src = np.array([[shape[1] // 2, shape[0] // 3]], np.int64)
        rx = np.arange(2, shape[1] - 2, 4, dtype=np.int64)
        rec = np.stack([rx, np.full(rx.size, 2, np.int64)], -1)[None]
    else:
        shape, abcn, nt = (32, 28, 32), 12, 100
        src = np.array([[shape[2] // 2, shape[1] // 2, shape[0] // 3]], np.int64)
        rx = np.arange(2, shape[2] - 2, 4, dtype=np.int64)
        ry = np.arange(2, shape[1] - 2, 4, dtype=np.int64)
        gy, gx = np.meshgrid(ry, rx, indexing="ij")
        rec = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2, np.int64)], -1)[None]
    vp = _ramp(shape, 1800.0, 2400.0)
    rho = _ramp(shape, 1000.0, 1300.0)
    z = (vp * rho).astype(np.float32)
    return shape, abcn, nt, src, rec, vp, z


def _grads(ndim, mode, impl="c"):
    from sweep.equations import AcousticVRZ, AcousticVRZ3D
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import (BoundaryOptions, CkptOptions, CUDAOptions,
                                          EagerOptions, MemoryOptions)

    dev = "cuda"
    shape, abcn, nt, src, rec, vp, z = _setup(ndim)
    cls = AcousticVRZ if ndim == 2 else AcousticVRZ3D
    eq = cls(spatial_order=SO, device=dev, backend="torch")
    common = dict(shape=shape, abcn=abcn, dh=DH, dt=DT, nt=nt)
    if impl == "eager":
        prop = PropTorch(eq, backend="torch", impl="eager", use_ckpt=False,
                         eager_options=EagerOptions(use_compile=False), **common)
    else:
        if mode == "full":
            mem = MemoryOptions(strategy="full")
        elif mode == "bs":
            mem = MemoryOptions(strategy="boundary",
                                boundary=BoundaryOptions(storage="gpu", storage_dtype="fp32"))
        elif mode == "bs_int8":
            mem = MemoryOptions(strategy="boundary",
                                boundary=BoundaryOptions(storage="gpu", storage_dtype="int8"))
        else:
            mem = MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=8))
        prop = PropTorch(eq, backend="torch", impl="c", cuda_options=CUDAOptions(memory=mem), **common)
    wav = torch.tensor(_ricker(nt, DT), device=dev)
    m = [torch.tensor(vp, device=dev, requires_grad=True),
         torch.tensor(z, device=dev, requires_grad=True)]
    out = prop(wav, src.copy(), rec.copy(), models=m)
    out.pow(2).sum().backward()
    return m[0].grad.detach().clone(), m[1].grad.detach().clone()


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _edge_ratio(d):
    """RMS of the outermost physical line/plane of every face over the RMS of
    the cells at least 4 in from every face."""
    d = d.double()
    faces = []
    for ax in range(d.dim()):
        faces.append(d.narrow(ax, 0, 1).flatten())
        faces.append(d.narrow(ax, d.shape[ax] - 1, 1).flatten())
    edge = torch.cat(faces)
    inner = d[tuple(slice(4, -4) for _ in range(d.dim()))]
    return float(edge.pow(2).mean().sqrt() / inner.pow(2).mean().sqrt().clamp_min(1e-300))


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
@pytest.mark.parametrize("ndim", [2, 3])
def test_vrz_bs_and_ckpt_match_full_including_the_rim(ndim):
    g_full = _grads(ndim, "full")
    g_bs = _grads(ndim, "bs")
    g_ck = _grads(ndim, "ckpt")
    for name, gf, gb, gc in (("vp", g_full[0], g_bs[0], g_ck[0]), ("z", g_full[1], g_bs[1], g_ck[1])):
        assert torch.isfinite(gb).all() and torch.isfinite(gc).all()
        rel_bs, rel_ck = _rel(gb, gf), _rel(gc, gf)
        ratio = _edge_ratio(gb - gf)
        print(f"{ndim}D {name}: bs-full rel {rel_bs:.3e}  ckpt-full rel {rel_ck:.3e}  bs-full edge/interior {ratio:.2f}")
        assert rel_ck < REL_TOL_MODES, f"{ndim}D {name}: ckpt vs full rel {rel_ck:.3e} (imaging time offset?)"
        assert rel_bs < REL_TOL_MODES, f"{ndim}D {name}: bs vs full rel {rel_bs:.3e} (imaging time offset?)"
        assert ratio < EDGE_RATIO_TOL, f"{ndim}D {name}: bs-full outermost line is {ratio:.1f}x the interior (shell narrower than the imaging reach?)"


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
@pytest.mark.parametrize("ndim", [2, 3])
def test_vrz_int8_bs_rim_is_bounded(ndim):
    """int8 storage noise is amplified by the spatial stencils the imaging takes
    of restored cells.  The p_tt imaging removed the Laplacian term; the two
    divergence terms remain, so the rim is bounded, not gone."""
    g_full = _grads(ndim, "full")
    g_i8 = _grads(ndim, "bs_int8")
    for name, gf, gi in (("vp", g_full[0], g_i8[0]), ("z", g_full[1], g_i8[1])):
        ratio = _edge_ratio(gi - gf)
        rel = _rel(gi, gf)
        print(f"{ndim}D {name}: int8-full rel {rel:.3e}  edge/interior {ratio:.2f}")
        assert rel < 5e-2, f"{ndim}D {name}: int8 bs vs full rel {rel:.3e}"
        assert ratio < INT8_EDGE_TOL[name], f"{ndim}D {name}: int8 rim {ratio:.1f}x the interior"


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_vrz2d_c_matches_eager():
    g_c = _grads(2, "full", impl="c")
    g_e = _grads(2, "full", impl="eager")
    for name, gc, ge in (("vp", g_c[0], g_e[0]), ("z", g_c[1], g_e[1])):
        rel = _rel(gc, ge)
        cos = float((gc * ge).sum() / (gc.norm() * ge.norm()).clamp_min(1e-30))
        print(f"2D {name}: c-vs-eager rel {rel:.3e} cos {cos:.5f}")
        assert rel < REL_TOL_EAGER and cos > 0.995, f"2D {name}: c vs eager rel {rel:.3e} cos {cos:.4f}"


# ---------------------------------------------------------------------------
# The sigma=0 boundary buffer: pad bookkeeping (CPU, eager construction) and
# the refusal of a boundary-saving run whose buffer is narrower than the
# imaging reach.
# ---------------------------------------------------------------------------

def _prop(cls, **kw):
    from sweep.propagator.torch import PropTorch
    args = dict(shape=(40, 50), abcn=10, dh=10.0, dt=1e-3, nt=10, backend="torch", impl="eager")
    args.update(kw)
    return PropTorch(cls(spatial_order=SO, device="cpu", backend="torch"), **args)


def test_vrz_boundary_buffer_defaults():
    from sweep.equations import Acoustic, AcousticVRZ
    from sweep.propagator.options import BoundarySaving, Ckpt, Full

    M = SO // 2
    # every strategy gets the equation's buffer, so the modes share one grid
    for mem in (BoundarySaving(), Full(), Ckpt(mode="chunk", chunks=2)):
        p = _prop(AcousticVRZ, memory=mem)
        assert p.boundary_buffer == M + 1
        assert p.pml_pad == (10, 10, 10, 10)
        assert p.pad == tuple(10 + M + 1 for _ in range(4))
        assert p.shape == (40 + 2 * (10 + M + 1), 50 + 2 * (10 + M + 1))

    p = _prop(AcousticVRZ, memory=Full(), free_surface=True)
    assert p.pad[0] == 0 and p.pml_pad[0] == 0          # no ramp, no buffer on the free surface
    assert p.pad[1] == 10 + M + 1

    p = _prop(AcousticVRZ, memory=Full(), boundary_buffer=7)   # explicit, honoured
    assert p.boundary_buffer == 7 and p.pad == (17, 17, 17, 17)
    p = _prop(AcousticVRZ, memory=Full(), boundary_buffer=0)   # explicit off is fine without bs
    assert p.boundary_buffer == 0 and p.pad == p.pml_pad

    p = _prop(Acoustic, memory=BoundarySaving())                # pointwise imaging: no buffer
    assert p.boundary_buffer == 0 and p.pad == p.pml_pad

    with pytest.raises(ValueError, match="non-negative"):
        _prop(AcousticVRZ, memory=Full(), boundary_buffer=-1)


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_vrz_compiled_bs_refuses_a_short_buffer():
    from sweep.equations import AcousticVRZ
    from sweep.propagator.options import BoundarySaving
    with pytest.raises(ValueError, match="buffer of at least"):
        _prop(AcousticVRZ, impl="c", dev="cuda", memory=BoundarySaving(), boundary_buffer=SO // 2)


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_vrz_compiled_bs_has_the_buffer():
    from sweep.equations import AcousticVRZ
    from sweep.propagator.options import BoundarySaving
    M = SO // 2
    p = _prop(AcousticVRZ, impl="c", dev="cuda", memory=BoundarySaving())
    assert p.boundary_buffer == M + 1 and p.pad == tuple(10 + M + 1 for _ in range(4))
