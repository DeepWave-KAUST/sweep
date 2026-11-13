"""A receiver above an image-method surface records nothing -- and moves nothing.

The acoustic image method clears the vacuum above the per-column surface every
step, before the record is taken, so a receiver placed there records an
identically-zero trace whatever the model.  Its trace adds nothing to a loss,
so the gradient must be exactly the gradient without it.  Eager has always
satisfied that (autograd through the air clear).  The compiled backward
injected the receiver's cotangent into the vacuum, where the adjoint kernels
neither update nor clear it, and the solid cells next to it read it back:
the gradient moved by 5-40% on a hill.  ``propagator/_c.py
_drop_air_receivers`` now zeroes that cotangent (the transpose of the air
clear) for equations that declare ``air_cells_are_vacuum``.

The elastic image method keeps image values in the air -- a receiver there
records a non-zero trace on both impls -- so it is not in scope here.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, Acoustic3D
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DH, DT, NT, ABCN, SO, RZ = 10.0, 1.5e-3, 160, 12, 4, 4


@pytest.fixture(autouse=True)
def _no_tf32():
    old = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32 = old


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - 0.06
    return torch.tensor((1e3 * ricker(t, f=10.0)).astype(np.float32), device=DEVICE)


def _grads(ndim, impl, keep):
    """vp gradient of sum(record * w), receivers at depth RZ under a hill --
    the ones above the surface included (``keep='all'``) or dropped; each
    weight follows its receiver.  Returns (grad, air record peak)."""
    if ndim == 2:
        nz, nx = 40, 48
        x = np.arange(nx, dtype=np.float32)
        topo = (2.0 + 5.0 * np.exp(-((x - nx / 2) ** 2) / (2.0 * 7.0 ** 2))).round().astype(np.int64)
        cols = [(c,) for c in range(2, nx - 2, 3)]
        surface = lambda c: topo[c[0]]
        eq, shape = Acoustic(spatial_order=SO, device=DEVICE, backend="torch"), (nz, nx)
        src = np.array([[nx // 3, nz // 2]], dtype=np.int64)
    else:
        nz, ny, nx = 24, 20, 26
        y, x = np.meshgrid(np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing="ij")
        topo = (2.0 + 5.0 * np.exp(-((x - nx / 2) ** 2 + (y - ny / 2) ** 2) / (2.0 * 5.0 ** 2))).round().astype(np.int64)
        cols = [(cx, cy) for cy in range(2, ny - 2, 3) for cx in range(2, nx - 2, 3)]
        surface = lambda c: topo[c[1], c[0]]
        eq, shape = Acoustic3D(spatial_order=SO, device=DEVICE, backend="torch"), (nz, ny, nx)
        src = np.array([[nx // 3, ny // 2, nz // 2]], dtype=np.int64)
    air = [RZ < surface(c) for c in cols]
    assert any(air) and not all(air)
    kept = [i for i in range(len(cols)) if keep == "all" or not air[i]]
    rec = np.array([[list(cols[i]) + [RZ] for i in kept]], dtype=np.int64)
    kw = (dict(use_ckpt=False, boundary_saving_config={"enabled": False}) if impl == "c"
          else dict(use_ckpt=False, use_compile=False))
    prop = PropTorch(eq, backend="torch", impl=impl, shape=shape, abcn=ABCN, dh=DH, dt=DT,
                     dev=DEVICE, nt=NT, B=1, topography=topo, **kw)
    if prop.impl != impl:
        pytest.skip(f"impl={impl!r} not available here (built {prop.impl!r})")
    depth = torch.linspace(0, 1, shape[0], device=DEVICE).view(-1, *([1] * (len(shape) - 1)))
    vp = (1800.0 + 600.0 * depth).expand(shape).contiguous().requires_grad_(True)
    d = prop(_wavelet(), src, rec, models=[vp])
    assert d.shape[1] == NT and d.shape[2] == len(kept), d.shape     # (B, nt, nrec, 1)
    w = torch.randn((1, NT, len(cols), 1), generator=torch.Generator().manual_seed(7)).to(DEVICE)
    (d * w[:, :, kept]).sum().backward()
    air_cols = [j for j, i in enumerate(kept) if air[i]]
    air_peak = float(d.detach()[:, :, air_cols].abs().max()) if air_cols else 0.0
    return vp.grad.detach().double(), air_peak


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("impl", ["eager", "c"])
def test_receivers_above_the_surface_do_not_move_the_gradient(ndim, impl):
    g_all, air_peak = _grads(ndim, impl, "all")
    g_below, _ = _grads(ndim, impl, "below")
    assert air_peak == 0.0, f"a receiver in the vacuum recorded {air_peak:.3e}"
    assert torch.count_nonzero(g_below) > 0
    rel = ((g_all - g_below).norm() / g_below.norm()).item()
    # eager on CUDA does not accumulate its gradient in a fixed order, so two
    # runs with different receiver sets differ at the 1e-6 level by themselves
    # (measured 1.7e-6 for 3-D); the defect this pins moved it by 5-40%.
    tol = 1e-5 if impl == "eager" and DEVICE == "cuda" else 1e-6
    assert rel < tol, f"{ndim}-D impl={impl}: the air receivers moved the gradient by {rel:.2e}"
