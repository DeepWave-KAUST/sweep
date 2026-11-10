"""``Acoustic`` and ``ViscoAcoustic`` take their CPML first derivatives with the
equation's FD-order kernels on every device and backend.

Only a CUDA device built those kernels.  On a CPU and on JAX ``gradient`` fell
back to ``torch.gradient`` / ``jnp.gradient``: second order, one-sided at the
grid edges, whatever ``spatial_order``.  The interior scheme was the same, the
absorbing layers were not, and a CPU record was 8.6e-5 (``Acoustic``) and
8.2e-5 (``ViscoAcoustic``) off the same run on a GPU, against ~1e-6 of fp32
round-off.
"""
import numpy as np
import pytest
import torch

from conftest import ricker
from sweep.equations import Acoustic, ViscoAcoustic
from sweep.propagator.options import EagerOptions, Full
from sweep.propagator.torch import PropTorch

try:
    import jax.numpy as jnp
except ImportError:  # jax is optional
    jnp = None

SO = 4
EQUATIONS = [Acoustic, ViscoAcoustic]
BACKENDS = [
    ("torch", "cpu"),
    pytest.param(("torch", "cuda"),
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    pytest.param(("jax", "cpu"), marks=pytest.mark.skipif(jnp is None, reason="needs jax")),
]


@pytest.mark.parametrize("backend_device", BACKENDS, ids=lambda b: "-".join(b))
@pytest.mark.parametrize("cls", EQUATIONS, ids=lambda c: c.__name__)
def test_first_derivatives_have_the_fd_order(cls, backend_device):
    """A fourth-order first derivative is exact on a cubic; ``torch.gradient``
    and ``jnp.gradient`` are off by h**2 there."""
    backend, device = backend_device
    eq = cls(spatial_order=SO, device=device, backend=backend)
    h = 0.05
    z = np.arange(20, dtype=np.float32)[:, None] * h - 0.5
    x = np.arange(24, dtype=np.float32)[None, :] * h - 0.6
    u = (z ** 3 + x ** 3 + z * x ** 2)[None, None]
    m = SO // 2
    for axis, exact in ((-2, 3 * z ** 2 + x ** 2), (-1, 3 * x ** 2 + 2 * z * x)):
        if backend == "torch":
            d = eq.gradient(torch.tensor(u, device=device), h, axis, kernels=eq.grad_kernels).cpu().numpy()
        else:
            d = np.asarray(eq.gradient(jnp.asarray(u), h, axis, kernels=eq.grad_kernels))
        err = np.abs(d - exact)[..., m:-m, m:-m].max()
        assert err < 0.01 * h ** 2, (axis, err)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cls", EQUATIONS, ids=lambda c: c.__name__)
def test_cpu_and_cuda_records_agree(cls):
    """One run on both devices: the records agree to fp32 round-off (~1e-6);
    they were 8e-5 apart while the CPU took torch.gradient."""
    nz, nx, nt = 48, 56, 400
    vp = (2500.0 * (1 + 0.05 * np.linspace(0, 1, nz)[:, None]) * np.ones((1, nx))).astype(np.float32)
    models = [vp]
    if cls is ViscoAcoustic:
        models += [np.full((nz, nx), 30.0, np.float32), np.full((nz, nx), 2 * np.pi * 12.0, np.float32)]
    src = np.array([[nx // 2, nz // 4]])
    rx = np.arange(3, nx - 3, 3)
    rec = np.stack([rx, np.full(rx.size, 6)], -1)[None]
    w = ricker(nt, 1e-3, 12.0, 0.1)
    records = []
    for device in ("cpu", "cuda"):
        prop = PropTorch(cls(spatial_order=SO, device=device, backend="torch"), shape=(nz, nx), dh=10.0, dt=1e-3,
                         nt=nt, abcn=30, dev=device, impl="eager", memory=Full(),
                         eager_options=EagerOptions(use_compile=False))
        with torch.no_grad():
            d = prop(torch.tensor(w, device=device), src, rec,
                     models=[torch.tensor(m, device=device) for m in models])
        records.append(d.double().cpu())
    rel = float((records[0] - records[1]).norm() / records[1].norm())
    assert rel < 1e-5, rel
