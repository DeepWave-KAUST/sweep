"""``impl='c'`` takes float32 models, and says so when it gets another dtype.

The compiled kernels read every model as float32, and nothing on the way there
checked. What a model of another dtype did depended on its width:

* float64 and int64 (8 bytes) were refused by the core's buffer check, as "a
  torch.float64 tensor is 8 bytes wide but the boundary storage tag implies 4"
  -- about boundary storage, in every memory mode, and not a word about the
  model. A float64 model from ``np.linspace(0, 1, n, np.float32)`` (the fourth
  positional argument is ``endpoint``) surfaced that way;
* float16 and bfloat16 (2 bytes) passed that check under the half-precision
  storage tags and tripped ``Buf::data_ptr<float>``'s assertion, which aborts
  the process (a build without assertions reads the 2-byte values as floats,
  past the end of the buffer);
* int32 (4 bytes) passed everything and was read as float bits: a record 20x
  off, and no error.

``_CompiledPropagator.forward`` now refuses any model that is not float32 with a
``TypeError`` that names it, before anything is built from it. Each case runs
float32 on the same propagator first and again after the refusal, so a check
that refused everything, or one that left state behind, fails here too.
``impl='eager'`` casts models to float32 and is not covered here.
"""
import numpy as np
import pytest
import torch

from conftest import requires_binding

from sweep.equations import Acoustic, Elastic
from sweep.propagator.options import BoundarySaving, Ckpt, Full
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

pytestmark = requires_binding("acoustic2d_forward")

NZ, NX, NT, DT, DH, ABCN = 48, 56, 200, 1e-3, 10.0, 10

MEMORY = {
    "full": Full(),
    "boundary": BoundarySaving(),
    "ckpt-chunk": Ckpt(mode="chunk", chunks=50),
    "ckpt-recursive": Ckpt(mode="recursive", count=4),
}


def _inputs(dev):
    t = np.arange(NT, dtype=np.float32) * DT - 0.1
    wav = torch.as_tensor(np.asarray(ricker(t, 12.0), dtype=np.float32), device=dev)
    src = np.array([[NX // 2, 4]], np.int32)
    rec = np.array([[[x, 2] for x in range(2, NX - 2, 2)]], np.int32)
    # 1500 -> 3000 m/s with depth; the float64 version is the same values.
    vp = 1500.0 + 1500.0 * np.linspace(0.0, 1.0, NZ)[:, None] * np.ones((1, NX))
    return wav, src, rec, vp


def _acoustic(memory, dev):
    return PropTorch(Acoustic(spatial_order=4, device=dev, backend="torch"), backend="torch",
                     impl="c", shape=(NZ, NX), dh=DH, dt=DT, nt=NT, abcn=ABCN, dev=dev,
                     memory=memory)


def _record(prop, wav, src, rec, vp, grad):
    if grad:
        vp = vp.detach().clone().requires_grad_(True)
        d = prop(wav, src, rec, models=[vp])
        (d.double() ** 2).sum().backward()
        assert torch.isfinite(vp.grad).all() and float(vp.grad.abs().max()) > 0
        return d.detach()
    with torch.no_grad():
        return prop(wav, src, rec, models=[vp])


@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no-grad"])
@pytest.mark.parametrize("memory", list(MEMORY.values()), ids=list(MEMORY))
def test_a_float64_model_is_refused_in_every_memory_mode(memory, grad):
    dev = torch.device("cuda:0")
    wav, src, rec, vp_np = _inputs(dev)
    assert vp_np.dtype == np.float64
    prop = _acoustic(memory, dev)
    vp32 = torch.tensor(vp_np, dtype=torch.float32, device=dev)
    before = _record(prop, wav, src, rec, vp32, grad)

    vp64 = torch.tensor(vp_np, device=dev, requires_grad=grad)
    with pytest.raises(TypeError) as exc:
        prop(wav, src, rec, models=[vp64])
    msg = str(exc.value)
    assert "models[0] ('vp') is torch.float64" in msg, msg
    assert ".float()" in msg, "the refusal should say how to fix it: " + msg

    after = _record(prop, wav, src, rec, vp32, grad)
    assert torch.equal(before, after), "the refused call changed what a float32 call computes"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32, torch.int64,
                                   torch.complex64], ids=lambda d: str(d).split(".")[-1])
def test_every_other_dtype_is_refused(dtype):
    """Forward only, as a model of these dtypes cannot ask for a gradient here:
    float16 / bfloat16 used to abort the process, int32 to give a wrong record."""
    dev = torch.device("cuda:0")
    wav, src, rec, vp_np = _inputs(dev)
    prop = _acoustic(Full(), dev)
    with pytest.raises(TypeError, match=rf"models\[0\] \('vp'\) is {dtype}"):
        with torch.no_grad():
            prop(wav, src, rec, models=[torch.tensor(vp_np, device=dev).to(dtype)])


@requires_binding("elastic2d_forward")
def test_the_refusal_names_which_model():
    dev = torch.device("cuda:0")
    wav, src, rec, vp_np = _inputs(dev)
    prop = PropTorch(Elastic(spatial_order=4, device=dev, backend="torch"), backend="torch",
                     impl="c", shape=(NZ, NX), dh=DH, dt=5e-4, nt=NT, abcn=ABCN, dev=dev,
                     memory=Full())
    vp = torch.tensor(vp_np, dtype=torch.float32, device=dev)
    vs = (vp / 1.8).double()
    rho = torch.full_like(vp, 2000.0)
    with pytest.raises(TypeError, match=r"models\[1\] \('vs'\) is torch\.float64"):
        with torch.no_grad():
            prop(wav, src, rec, models=[vp, vs, rho])


def test_model_parallel_refuses_a_float64_first_call_and_recovers():
    """The one-time capture builds its probe models in the first tile's dtype
    and runs them through the same forward, so a float64 model on the first
    call meets the same refusal. The instance must not keep that dtype: the
    next call, with a float32 model, captures afresh and runs.
    (world=1: the dd_propagator path without NCCL.)"""
    from sweep.parallel import MeshTopology, ModelParallel

    dev = torch.device("cuda:0")
    wav, src, rec, vp_np = _inputs(dev)
    ddp = ModelParallel(_acoustic(BoundarySaving(), dev),
                        MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0))
    src3 = src[None]
    with pytest.raises(TypeError, match=r"models\[0\] \('vp'\) is torch\.float64"):
        ddp(wav, src3, rec, models=[torch.tensor(vp_np, device=dev, requires_grad=True)])

    vp = torch.tensor(vp_np, dtype=torch.float32, device=dev, requires_grad=True)
    (ddp(wav, src3, rec, models=[vp]).double() ** 2).sum().backward()
    assert torch.isfinite(vp.grad).all() and float(vp.grad.abs().max()) > 0
