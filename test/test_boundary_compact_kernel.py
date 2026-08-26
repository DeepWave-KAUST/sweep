"""The band-only (compact) boundary kernels must be bit-identical to the scan.

``boundary_kernel2d`` walks the whole padded grid and lets ~96% of its threads
return after the prologue -- four ``SolverContext::phys_*()`` evaluations that,
since the per-edge / DD rework, are a branch chain rather than a constant.  That
prologue is essentially the kernel's entire cost, so the scan was replaced by
``boundary_kernel2d_compact``, which launches one thread per boundary-band cell
(the 3-D path has worked this way for a while).

The compact kernel changes only *which* threads do the work, never the
arithmetic, so every gradient must come back bit-for-bit identical.  These tests
pin that by running the same problem twice, once with ``SWEEP_BOUNDARY_NO_COMPACT``
forcing the scan kernels, and comparing the raw bytes.
"""

import os

import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, Acoustic3D, Elastic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="impl='c' requires CUDA."
)

_DT = 1.5e-3
_NT = 900          # long enough for the reflection to reach the receivers
_PERTURB_FRAC = 4  # perturbation at nz/4, not nz/2 -- see _assert_nonvacuous


def _wavelet(nt=_NT, dt=_DT, f=9.0):
    t = np.arange(nt, dtype=np.float32) * dt
    return ricker(t - 0.12, f=f)


def _acoustic_grad(force_scan, *, shape, free_surface, batch, storage_dtype="fp32"):
    """d(loss)/d(vp) for a small 2-D acoustic problem with boundary saving."""
    nz, nx = shape
    dev = torch.device("cuda")
    z = np.arange(nz, dtype=np.float32)[:, None]
    vp_init = (1500.0 + 8.0 * z).repeat(nx, 1).astype(np.float32)
    vp_true = vp_init.copy()
    vp_true[nz // _PERTURB_FRAC:, nx // 3: 2 * nx // 3] += 400.0

    src_x = np.linspace(nx * 0.2, nx * 0.8, batch).astype(np.int64)
    sources = np.stack([src_x, np.full(batch, 3, dtype=np.int64)], axis=1)
    rec_x = np.linspace(2, nx - 3, min(nx - 4, 48)).astype(np.int64)
    receivers = np.repeat(
        np.stack([rec_x, np.full(rec_x.size, 4, dtype=np.int64)], axis=1)[None, ...],
        batch, axis=0)

    with _forced(force_scan):
        solver = PropTorch(
            Acoustic(spatial_order=4, device=dev, backend="torch"),
            shape=shape, dh=15.0, dt=_DT, impl="c", free_surface=free_surface,
            boundary_saving_config={"enabled": True, "storage": "gpu",
                                    "storage_dtype": storage_dtype},
        )
        wavelet = _wavelet()
        with torch.no_grad():
            obs = solver(wavelet, sources, receivers,
                         models=[torch.tensor(vp_true, device=dev)]).detach()
        vp = torch.tensor(vp_init, device=dev, requires_grad=True)
        loss = (solver(wavelet, sources, receivers, models=[vp]) - obs).pow(2).mean()
        loss.backward()
    return vp.grad.detach().cpu().numpy(), float(loss.detach())


def _acoustic3d_grad(force_scan, *, shape=(48, 44, 52), nt=500):
    nz, ny, nx = shape
    dev = torch.device("cuda")
    z = np.arange(nz, dtype=np.float32)[:, None, None]
    vp_init = (1800.0 + 12.0 * z).repeat(ny, 1).repeat(nx, 2).astype(np.float32)
    vp_true = vp_init.copy()
    vp_true[nz // _PERTURB_FRAC:, :, :] += 300.0

    sources = np.array([[nx // 2, ny // 2, 3]], dtype=np.int64)
    rx = np.linspace(2, nx - 3, 6).astype(np.int64)
    ry = np.linspace(2, ny - 3, 6).astype(np.int64)
    rxg, ryg = np.meshgrid(rx, ry, indexing="ij")
    receivers = np.stack([rxg.ravel(), ryg.ravel(),
                          np.full(rxg.size, 3, dtype=np.int64)], axis=1)[None, ...]

    with _forced(force_scan):
        solver = PropTorch(
            Acoustic3D(spatial_order=4, device=dev, backend="torch"),
            shape=shape, dh=20.0, dt=_DT, impl="c",
            boundary_saving_config={"enabled": True, "storage": "gpu"},
        )
        wavelet = _wavelet(nt=nt)
        with torch.no_grad():
            obs = solver(wavelet, sources, receivers,
                         models=[torch.tensor(vp_true, device=dev)]).detach()
        vp = torch.tensor(vp_init, device=dev, requires_grad=True)
        loss = (solver(wavelet, sources, receivers, models=[vp]) - obs).pow(2).mean()
        loss.backward()
    return vp.grad.detach().cpu().numpy(), float(loss.detach())


def _elastic_grad(force_scan, *, shape=(70, 110), free_surface=False, nt=900):
    """Multi-field boundary saving -> the per-field save/restore launch sites."""
    nz, nx = shape
    dev = torch.device("cuda")
    z = np.arange(nz, dtype=np.float32)[:, None]
    vp = (2000.0 + 6.0 * z).repeat(nx, 1).astype(np.float32)
    vs = (vp / 1.8).astype(np.float32)
    rho = (1800.0 + 0.3 * z).repeat(nx, 1).astype(np.float32)
    vp_true = vp.copy()
    vp_true[nz // _PERTURB_FRAC:, :] += 300.0

    sources = np.array([[nx // 2, 4]], dtype=np.int64)
    rec_x = np.linspace(2, nx - 3, 32).astype(np.int64)
    receivers = np.stack([rec_x, np.full(rec_x.size, 3, dtype=np.int64)], axis=1)[None, ...]

    with _forced(force_scan):
        solver = PropTorch(
            Elastic(spatial_order=4, device=dev, backend="torch"),
            shape=shape, dh=15.0, dt=1e-3, impl="c", free_surface=free_surface,
            boundary_saving_config={"enabled": True, "storage": "gpu"},
        )
        wavelet = _wavelet(nt=nt, dt=1e-3, f=8.0)
        t = lambda a: torch.tensor(a, device=dev)
        with torch.no_grad():
            obs = solver(wavelet, sources, receivers,
                         models=[t(vp_true), t(vs), t(rho)]).detach()
        vpp = torch.tensor(vp, device=dev, requires_grad=True)
        loss = (solver(wavelet, sources, receivers,
                       models=[vpp, t(vs), t(rho)]) - obs).pow(2).mean()
        loss.backward()
    return vpp.grad.detach().cpu().numpy(), float(loss.detach())


class _forced:
    """Force the scan kernels for the lifetime of the block (env is read when
    the C++ BoundarySaver is constructed, i.e. inside PropTorch)."""

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.prev = os.environ.get("SWEEP_BOUNDARY_NO_COMPACT")
        if self.on:
            os.environ["SWEEP_BOUNDARY_NO_COMPACT"] = "1"
        else:
            os.environ.pop("SWEEP_BOUNDARY_NO_COMPACT", None)
        return self

    def __exit__(self, *exc):
        if self.prev is None:
            os.environ.pop("SWEEP_BOUNDARY_NO_COMPACT", None)
        else:
            os.environ["SWEEP_BOUNDARY_NO_COMPACT"] = self.prev


def _assert_nonvacuous(g, loss, what):
    """A zero loss or an all-zero gradient makes the byte comparison pass no
    matter what the kernels do.  That is the failure mode this whole file
    exists to catch, so check the check."""
    assert loss > 0.0, (
        f"{what}: loss is exactly 0 -- nothing reached the receivers, so the "
        "bit-comparison below would be vacuous. Lengthen nt or move the "
        "perturbation shallower.")
    assert np.abs(g).max() > 0.0, (
        f"{what}: gradient is identically zero -- the bit-comparison would be "
        "vacuous.")


def _assert_bit_identical(a, b, what):
    ga, la = a
    gb, lb = b
    _assert_nonvacuous(ga, la, what + " [compact]")
    _assert_nonvacuous(gb, lb, what + " [scan]")
    assert ga.shape == gb.shape, f"{what}: shape {ga.shape} vs {gb.shape}"
    assert la.hex() == lb.hex(), f"{what}: loss {la!r} vs {lb!r}"
    assert ga.tobytes() == gb.tobytes(), (
        f"{what}: gradient differs (max |d| = {np.abs(ga - gb).max():.3e}, "
        f"rel = {np.abs(ga - gb).max() / max(np.abs(ga).max(), 1e-30):.3e})")


@cuda_only
@pytest.mark.parametrize("free_surface", [False, True])
@pytest.mark.parametrize("shape,batch", [((40, 48), 2), ((110, 240), 3)])
def test_compact_matches_scan_acoustic2d(shape, batch, free_surface):
    """The headline path: 2-D acoustic, FP32 boundary saving on GPU."""
    kw = dict(shape=shape, free_surface=free_surface, batch=batch)
    _assert_bit_identical(_acoustic_grad(False, **kw), _acoustic_grad(True, **kw),
                          f"acoustic2d {shape} fs={free_surface}")


@cuda_only
@pytest.mark.parametrize("storage_dtype", ["bf16", "int8"])
def test_lossy_storage_keeps_scan_kernel(storage_dtype):
    """Lossy storage dtypes are excluded from the compact path -- their two
    copies of a corner cell need not round-trip identically -- so forcing the
    scan must be a no-op for them."""
    kw = dict(shape=(60, 96), free_surface=False, batch=2, storage_dtype=storage_dtype)
    _assert_bit_identical(_acoustic_grad(False, **kw), _acoustic_grad(True, **kw),
                          f"acoustic2d {storage_dtype}")


@cuda_only
@pytest.mark.parametrize("free_surface", [False, True])
def test_compact_matches_scan_elastic2d(free_surface):
    """Elastic saves several fields -> the per-field launch sites."""
    _assert_bit_identical(_elastic_grad(False, free_surface=free_surface),
                          _elastic_grad(True, free_surface=free_surface),
                          f"elastic2d fs={free_surface}")


@cuda_only
def test_compact_matches_scan_acoustic3d():
    """The 3-D compact path shares the gate; pin it against the scan too."""
    _assert_bit_identical(_acoustic3d_grad(False), _acoustic3d_grad(True),
                          "acoustic3d")
