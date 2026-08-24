"""An ``impl='c'`` run must give the same answer twice.

Equations whose ``kernels.cuh`` includes another family's kernels inherit that
family's CPML addressing, which reads the aux slabs off the ``SolverContext``.
If the borrowing equation's driver never installs them the descriptors stay
default constructed -- ``lo = hi = n = 0`` -- so ``AuxSlab::tot()`` is 0, the
aux row stride collapses to zero and every row aliases the first.  Threads then
race on those cells and the same input yields a different answer each run.

That failure is invisible to the usual guards: a single-run bitwise comparison
against a reference can pass by luck, and ``compute-sanitizer`` sees nothing
because the aliased cells *are* written -- by the wrong thread.  Only a repeat
catches it.  ``DASMu`` / ``DASMu3D`` (elastic kernels) shipped exactly this
defect; the controls below own their drivers and install their own slabs.
"""
import hashlib

import numpy as np
import pytest
import torch

from sweep import equations as E
from sweep.propagator.torch import PropTorch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="impl='c' needs CUDA")
DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
VALUE = {"vp": 2000.0, "vs": 1150.0, "rho": 2000.0, "z": 4.0e6, "mp": 0.01}


def _digest(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def _models(cls, shape, grad=False):
    out = []
    for spec in cls.MODEL_SPECS:
        name = getattr(spec, "name", spec)
        m = torch.full(shape, VALUE.get(name, 2000.0), device=DEV)
        m[tuple(slice(s // 3, s // 3 + max(2, s // 6)) for s in shape)] *= 1.08
        out.append(m.requires_grad_(grad))
    return out


def _geometry(shape, nt):
    t = np.arange(nt, dtype=np.float32) * 1e-3 - 0.04
    a = np.pi * 12.0 * t
    wav = torch.as_tensor(((1 - 2 * a**2) * np.exp(-a**2)).astype("float32"), device=DEV)
    ctr = [s // 2 for s in shape]
    rx = np.arange(6, shape[-1] - 6, 3, dtype=np.int64)
    if len(shape) == 2:
        src = np.array([[[ctr[1], 4]]], dtype=np.int64)
        rec = np.stack([rx, np.full_like(rx, 6)], -1)[None]
    else:
        src = np.array([[[ctr[2], ctr[1], 4]]], dtype=np.int64)
        rec = np.stack([rx, np.full_like(rx, ctr[1]), np.full_like(rx, 6)], -1)[None]
    return wav, src, rec


def _build(cls, shape, nt, free_surface):
    eq = cls(spatial_order=4, device=DEV, backend="torch")
    return PropTorch(eq, backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0,
                     dt=1e-3, nt=nt, abcn=12, free_surface=free_surface, B=1,
                     use_ckpt=False, boundary_saving_config={"enabled": False})


CASES = [
    # equations that borrow another family's kernels -- the ones at risk
    ("DASMu", 2), ("DASMu3D", 3), ("AcousticLSRTM", 2), ("AcousticVRZ3D", 3),
    # controls: own drivers, own slab installation
    ("Acoustic", 2), ("Acoustic3D", 3), ("Elastic", 2), ("Elastic3D", 3),
]


@needs_cuda
@pytest.mark.parametrize("name,ndim", CASES)
@pytest.mark.parametrize("free_surface", [False, True])
def test_forward_is_reproducible(name, ndim, free_surface):
    cls = getattr(E, name, None)
    if cls is None:
        pytest.skip(f"{name} not available in this build")
    shape = (40, 44) if ndim == 2 else (32, 36, 40)
    nt = 220
    wav, src, rec = _geometry(shape, nt)
    models = _models(cls, shape)
    digests = []
    for _ in range(2):
        prop = _build(cls, shape, nt, free_surface)
        with torch.no_grad():
            digests.append(_digest(prop(wav, src, rec, models=models)))
        del prop
    assert digests[0] == digests[1], (
        f"{name} forward is not reproducible (free_surface={free_surface}): "
        f"{digests[0][:16]} vs {digests[1][:16]}. A collapsed CPML aux row "
        f"stride aliases every row onto the first -- check that this "
        f"equation's driver calls its kernel family's *_init_aux_slabs.")


@needs_cuda
@pytest.mark.parametrize("name,ndim", [("DASMu", 2), ("DASMu3D", 3)])
def test_gradient_is_reproducible(name, ndim):
    cls = getattr(E, name, None)
    if cls is None:
        pytest.skip(f"{name} not available in this build")
    shape = (40, 44) if ndim == 2 else (32, 36, 40)
    nt = 220
    wav, src, rec = _geometry(shape, nt)
    digests = []
    for _ in range(2):
        models = _models(cls, shape, grad=True)
        prop = _build(cls, shape, nt, free_surface=True)
        (0.5 * prop(wav, src, rec, models=models).pow(2).sum()).backward()
        digests.append("|".join(_digest(m.grad) for m in models))
        del prop
    assert digests[0] == digests[1], f"{name} gradient is not reproducible"
