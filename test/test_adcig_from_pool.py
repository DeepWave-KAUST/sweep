"""The space-lag ADCIG cube is propagator-bound (BackwardInput.adcig_out).

The driver used to allocate the (nlag, N, C, nz, nx[, ny]) cube itself --
the last C++-side allocation on the acoustic skeleton.  Python now allocates
it per backward, exactly like the illumination pair, and the driver requires
it.  This pins the contract from the user's side: with ``compute_adcig`` on a
boundary-saving solver, ``prop.adcig`` comes back model-shaped, finite, with
energy in every lag, run-to-run bit-identical, and independent of whether
illumination was asked for on the same backward (the two used to be refused
together as a side effect of the old binding).  Without the toggle nothing is
bound and nothing comes back.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.equations import Acoustic, Acoustic3D
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import BoundarySaving

DEV = "cuda"
ORDER, DH, DT, NT, ABCN = 4, 10.0, 1e-3, 200, 16
CASES = [("acoustic2d", Acoustic, (48, 56), 2), ("acoustic3d", Acoustic3D, (24, 20, 28), 3)]


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _binding_ready(), reason="needs the CUDA binding")


def _heterogeneous(shape):
    idx = np.indices(shape).astype(np.float32)
    bump = sum((i - s / 2) ** 2 / s for i, s in zip(idx, shape))
    return (2500.0 + 4.0 * idx[0] + 60.0 * np.sin(bump)).astype(np.float32)


def _run(cls, shape, ndim, max_lag, adcig=True, illum=False):
    prop = PropTorch(cls(spatial_order=ORDER, device=DEV), backend="torch", impl="c",
                     shape=shape, dh=DH, dt=DT, nt=NT, abcn=ABCN, B=1, dev=DEV,
                     source_type=["h1"], receiver_type=["h1"], memory=BoundarySaving(storage="gpu"))
    prop.compute_adcig = adcig
    prop.adcig_max_lag = max_lag
    prop.compute_illumination = illum
    nz, nx = shape[0], shape[-1]
    src = np.array([[nx // 2, nz // 3] if ndim == 2 else [nx // 2, shape[1] // 2, nz // 3]], np.int64)
    rxx = np.arange(4, nx - 4, 4, np.int64)
    cols = ([rxx, np.full(rxx.size, 4)] if ndim == 2 else
            [rxx, np.full(rxx.size, shape[1] // 2), np.full(rxx.size, 4)])
    rec = np.stack(cols, -1)[None]
    t = np.arange(NT, dtype=np.float32) * DT - 0.012
    a = np.pi * 15.0 * t
    wav = torch.tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e3).astype(np.float32), device=DEV)
    vp = torch.tensor(_heterogeneous(shape), device=DEV, requires_grad=True)
    out = prop(wav, src, rec, models=[vp])
    out.pow(2).mean().backward()
    return prop, vp.grad.detach()


@pytest.mark.parametrize("name,cls,shape,ndim", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("max_lag", [0, 2])
def test_adcig_cube_is_written_model_shaped_and_deterministic(name, cls, shape, ndim, max_lag):
    prop, grad = _run(cls, shape, ndim, max_lag)
    cube = prop.adcig
    assert isinstance(cube, torch.Tensor)
    assert tuple(cube.shape) == (2 * max_lag + 1, *shape)
    assert torch.isfinite(cube).all()
    energy = cube.flatten(1).abs().sum(1)
    assert (energy > 0).all(), f"a lag came back empty: {energy.tolist()}"
    prop2, grad2 = _run(cls, shape, ndim, max_lag)
    assert torch.equal(prop2.adcig, cube)
    assert torch.equal(grad2, grad)


@pytest.mark.parametrize("name,cls,shape,ndim", CASES, ids=[c[0] for c in CASES])
def test_adcig_with_illumination_on_the_same_backward(name, cls, shape, ndim):
    """Both buffers are Python-bound now, so asking for both is just two
    bindings; the cube must not depend on the illumination toggle."""
    prop_a, grad_a = _run(cls, shape, ndim, 1, adcig=True, illum=False)
    prop_b, grad_b = _run(cls, shape, ndim, 1, adcig=True, illum=True)
    assert torch.equal(prop_b.adcig, prop_a.adcig)
    assert torch.equal(grad_b, grad_a)
    assert prop_b.source_illumination is not None and prop_b.source_illumination.abs().sum() > 0


def test_no_toggle_binds_nothing():
    prop, _ = _run(Acoustic, (48, 56), 2, 0, adcig=False)
    assert prop.adcig is None
