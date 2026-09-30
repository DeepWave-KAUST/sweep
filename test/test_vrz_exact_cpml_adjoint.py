"""The compiled VRZ adjoint is the exact discrete transpose of the compiled
forward, CPML band included (impl='c', 2-D).

Reference: eager autograd, which differentiates the eager forward exactly.
The eager forward differs from the CUDA one in two details: the CPML memory
update takes D(a*psi) fused where CUDA takes the product rule
(D a)*psi + a*(D psi), and eager applies the db*du profile term everywhere
while CUDA runs its CPML branch only inside the band (db = np.gradient(b)
leaks one nonzero cell past the band edge, ~1e-3 of the operator there).
With both patched to the CUDA form the two forwards agree to fp32 rounding
(the record rel below is the floor), so the gradients must too -- unless the
compiled adjoint is not the transpose of the compiled forward.

The previous CUDA adjoint applied the FORWARD CPML operator to lambda inside
the band (a discretisation of the continuous adjoint, not a transpose).  Its
error is invisible on the interior-source/receiver setups the older tests use
(rel 8e-4 there) and dominates when sources or receivers sit next to the band,
which is where every real survey puts them; on field data it moved the deep
single-shot gradient by 70-120% between two physically equivalent absorbing
geometries while eager moved 1e-4.  The edge geometry below is the sensitive
case: source and receivers two physical cells from the band on three sides.
Measured with the old adjoint (2-D): interior vp rel 5.1e-4, edge vp rel
3.0e-2 (cos 0.99989) against the patched eager; exact transpose: 2e-7 / 5e-7
(below the record floor).

TF32.  torch leaves ``cudnn.allow_tf32`` on by default; on Ampere/Hopper the
eager 3-D conv3d BACKWARD then runs in TF32 while the forward happens to take
a full-precision algorithm, so the records still match C to 1e-7 and the eager
3-D gradient is silently 0.8% (interior) to 6% (edge, suite grid) off its own
forward's finite differences.  The 2-D conv2d path is not affected.  The flag
is switched off here (both dims); with it the 3-D c-vs-eager rel is 8e-7 /
3e-7 (interior / edge), with it on 3.8e-3 / 2.9e-2.

A second, implementation-free arbiter: central finite differences of J along
a smooth random direction, on the compiled forward.  The direction vanishes
on the outermost physical layer: that layer is replicate-padded and both
gradients CROP the pad (EdgePadding.backward), so a direction touching it
would measure the fold convention instead.  Measured |FD/<g,d>-1| <= 2.6e-4;
the old 3-D adjoint gave 7.9e-2 on the edge geometry.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# The eager reference must not run its conv3d backward in TF32 (see module doc).
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_vrz_boundary_imaging import DH, DT, SO, _binding_ready, _ricker, _setup  # noqa: E402

# c (exact transpose) vs eager with the CUDA CPML form, TF32 off.  Measured
# 2-D 1.7e-7 / 5.1e-7, 3-D 8.3e-7 / 3.1e-7 (interior / edge) against record
# floors of 8.5e-7 / 2.4e-7 / 4.5e-7 / 2.1e-7; the old adjoint gave 2-D
# 5.5e-5 / 3.0e-2 and 3-D 3.7e-3 / 1.1e-2.
REL_TOL = 1e-5
# gradient rel may not exceed this many times the record rel (adjoint exact
# => gradient error inherits the forward error, nothing more)
FLOOR_FACTOR = 20.0
# 2-D and 3-D: directional derivative vs central FD of the compiled forward.
# Measured |FD/<g,d> - 1| <= 2.6e-4 over eps = 2..10 m/s on every geometry
# (fp32 J); the old 3-D adjoint gave 7.9e-2 on the edge geometry.
FD_TOL = 1.5e-3
FD_EPS = (2.0, 10.0)      # m/s peak perturbation, both must pass


def _edge_geometry(ndim):
    """Source and receivers two physical cells from the band on three sides."""
    shape, abcn, nt, _, _, vp, z = _setup(ndim)
    if ndim == 2:
        nz, nx = shape
        src = np.array([[2, 2]], np.int64)                       # (x, z)
        rx = np.arange(1, nx - 1, 3, dtype=np.int64)
        rec = np.stack([rx, np.full(rx.size, nz - 3, np.int64)], -1)   # bottom edge
        zs = np.arange(2, 2 + nz // 3, dtype=np.int64)
        rec_side = np.stack([np.full(zs.size, nx - 3, np.int64), zs], -1)   # right edge
    else:
        nz, ny, nx = shape
        src = np.array([[2, 2, 2]], np.int64)                    # (x, y, z)
        gy, gx = np.meshgrid(np.arange(1, ny - 1, 3), np.arange(1, nx - 1, 3), indexing="ij")
        rec = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, nz - 3)], -1).astype(np.int64)
        gz, gy = np.meshgrid(np.arange(2, nz - 2, 3), np.arange(1, ny - 1, 3), indexing="ij")
        rec_side = np.stack([np.full(gz.size, nx - 3), gy.ravel(), gz.ravel()], -1).astype(np.int64)
    rec = np.concatenate([rec, rec_side], 0)[None]
    return shape, abcn, nt, src, rec, vp, z


def _run(impl, ndim, geometry, product_rule_pml=False, vp_override=None, want_grad=True):
    from sweep.equations import AcousticVRZ, AcousticVRZ3D
    import sweep.equations.acoustic_vrz as vrz_mod
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import CUDAOptions, EagerOptions, MemoryOptions

    dev = "cuda"
    shape, abcn, nt, src, rec, vp, z = geometry(ndim)
    if vp_override is not None:
        vp = vp_override
    cls = AcousticVRZ if ndim == 2 else AcousticVRZ3D
    eq = cls(spatial_order=SO, device=dev, backend="torch")
    common = dict(shape=shape, abcn=abcn, dh=DH, dt=DT, nt=nt)
    if impl == "eager":
        prop = PropTorch(eq, backend="torch", impl="eager", use_ckpt=False,
                         eager_options=EagerOptions(use_compile=False), **common)
    else:
        prop = PropTorch(eq, backend="torch", impl="c",
                         cuda_options=CUDAOptions(memory=MemoryOptions(strategy="full")), **common)
    wav = torch.tensor(_ricker(nt, DT), device=dev)
    m = [torch.tensor(vp, device=dev, requires_grad=want_grad),
         torch.tensor(z, device=dev, requires_grad=want_grad)]

    orig = vrz_mod.cpml_axis_update

    def cuda_form(lap_axis, du, psi, zeta, a, b, dbd, h, axis, grad_op, kernels=None):
        # CUDA runs the CPML branch only where the cell is in SOME axis's band
        # (in_pml_2d/3d); b is exactly 0 off-band, so the band is b != 0.
        inband = None
        for i in range(0, len(eq.b), 3):
            m_ = eq.b[i + 1] != 0
            inband = m_ if inband is None else (inband | m_)
        dbd = dbd * inband.to(dbd.dtype)
        # (D a)*psi + a*(D psi): the product-rule form of the CUDA forward.
        # Differentiate the profile on the full grid: grad_op zeroes the
        # stencil halo of its output, which on a 1-row/1-column broadcast
        # profile is the whole profile (da == 0 -> a*D(psi) only, 1e-3 off).
        da = grad_op(a.expand_as(psi).contiguous(), h, axis, kernels=kernels)
        tmp = ((1 + b) * lap_axis + dbd * du) + (da * psi + a * grad_op(psi, h, axis, kernels=kernels))
        contrib = (1 + b) * tmp + a * zeta
        psi_next = b * du + a * psi
        zeta_next = b * tmp + a * zeta
        return contrib, psi_next, zeta_next

    if product_rule_pml:
        vrz_mod.cpml_axis_update = cuda_form
    try:
        out = prop(wav, src.copy(), rec.copy(), models=m)
        J = out.double().pow(2).sum()
        if want_grad:
            J.backward()
    finally:
        vrz_mod.cpml_axis_update = orig
    if not want_grad:
        return float(J)
    return (out.detach().double(), m[0].grad.detach().double(), m[1].grad.detach().double())


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def _cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-300))


def _fd_direction(shape, seed=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    d = gaussian_filter(rng.standard_normal(shape), sigma=2.0)
    d = (d / np.abs(d).max()).astype(np.float32)
    for ax in range(d.ndim):                 # keep the replicate-padded layer fixed
        sl = [slice(None)] * d.ndim
        sl[ax] = 0; d[tuple(sl)] = 0.0
        sl[ax] = d.shape[ax] - 1; d[tuple(sl)] = 0.0
    return d


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("geometry", [_setup, _edge_geometry], ids=["interior", "edge"])
def test_vrz_compiled_adjoint_is_the_transpose_of_the_compiled_forward(ndim, geometry):
    rec_c, gvp_c, gz_c = _run("c", ndim, geometry)
    rec_e, gvp_e, gz_e = _run("eager", ndim, geometry, product_rule_pml=True)
    floor = _rel(rec_c, rec_e)
    print(f"\n[{ndim}D {geometry.__name__}] record c-vs-eager(cuda form) rel {floor:.3e}")
    for name, gc, ge in (("vp", gvp_c, gvp_e), ("z", gz_c, gz_e)):
        rel, cos = _rel(gc, ge), _cos(gc, ge)
        print(f"  grad_{name}: c-vs-eager(cuda form) rel {rel:.3e} cos {cos:.6f}")
        assert rel < REL_TOL and cos > 0.9999, f"grad_{name}: rel {rel:.3e} cos {cos:.5f}"
        assert rel < FLOOR_FACTOR * max(floor, 1e-6), \
            f"grad_{name}: rel {rel:.3e} is {rel / max(floor, 1e-30):.0f}x the forward floor {floor:.2e}"


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("geometry", [_setup, _edge_geometry], ids=["interior", "edge"])
def test_vrz_compiled_gradient_matches_finite_differences(ndim, geometry):
    shape, abcn, nt, src, rec, vp0, z0 = geometry(ndim)
    d = _fd_direction(shape)
    _, gvp_c, _ = _run("c", ndim, geometry)
    dc = float((gvp_c.cpu().numpy().squeeze() * d).sum())
    line = f"\n[{ndim}D {geometry.__name__}] <g_c,d> {dc:+.6e}"
    worst = 0.0
    for eps in FD_EPS:
        Jp = _run("c", ndim, geometry, vp_override=(vp0 + eps * d).astype(np.float32), want_grad=False)
        Jm = _run("c", ndim, geometry, vp_override=(vp0 - eps * d).astype(np.float32), want_grad=False)
        fd = (Jp - Jm) / (2 * eps)
        err = abs(fd / dc - 1)
        worst = max(worst, err)
        line += f"\n    eps {eps:4.1f} m/s: FD {fd:+.6e}  |FD/<g_c,d>-1| {err:.2e}"
    print(line)
    assert worst < FD_TOL, f"{ndim}D {geometry.__name__}: compiled gradient vs FD worst {worst:.2e}"
