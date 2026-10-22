"""AcousticLSRTM / AcousticLSRTM3D (impl='c') pay for the RWI vp gradient only
when vp asks for it.

The RWI tomographic vp gradient needs a second adjoint and the scattered field
(both u_tt in the full-mode history; both fields boundary-saved and
reconstructed in bs mode): about twice the backward time and up to twice the
memory.  A call whose vp needs no gradient -- the classic LSRTM, reflectivity
only -- must get the reflectivity-only layout and run the mp-only path:

* the declared buffers shrink back (workspace slots, reconstruction grids);
* vp.grad stays None, mp.grad and the record are bit-identical to the RWI run
  (the mp path does not read anything the RWI part writes);
* switching back to vp.requires_grad=True on the same propagator reallocates
  and reproduces the first RWI run bit for bit.
"""
import numpy as np
import pytest
import torch

from conftest import requires_binding

from sweep.equations import AcousticLSRTM, AcousticLSRTM3D
from sweep.propagator import BoundarySaving, Full, PropTorch
from sweep.propagator.options import Ckpt

# workspace slots (rwi, mp only) per mode: 2-D full/bs, 3-D full/bs
SLOTS = {(2, "full"): (2, 1), (2, "bs"): (2, 1), (3, "full"): (2, 1), (3, "bs"): (3, 2)}


def _case(dim):
    if dim == 2:
        shape = (32, 40)
        src = np.array([[20, 6]], np.int64)
        rec = np.array([[[ix, 2] for ix in range(2, 38, 4)]], np.int64)
    else:
        shape = (16, 16, 16)
        src = np.array([[8, 8, 4]], np.int64)
        rec = np.array([[[ix, iy, 2] for iy in range(2, 14, 4) for ix in range(2, 14, 4)]], np.int64)
    g = np.linspace(0.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return shape, src, rec, 2200.0 + 60.0 * g, 0.03 + 0.01 * g


@requires_binding("acoustic_lsrtm2d_forward", "acoustic_lsrtm3d_forward")
@pytest.mark.parametrize("dim, mode", sorted(SLOTS))
def test_vp_gradient_only_when_asked(dim, mode):
    shape, src, rec, vp0, mp0 = _case(dim)
    eq = (AcousticLSRTM if dim == 2 else AcousticLSRTM3D)(spatial_order=4, device="cuda", backend="torch")
    prop = PropTorch(eq, impl="c", memory=Full() if mode == "full" else BoundarySaving(),
                     shape=shape, dh=10.0, dt=0.0015, nt=60, abcn=10, device="cuda",
                     pml_type="cpmlr", source_type=["h1"], receiver_type=["sh1"], backend="torch")
    t = np.arange(60, dtype=np.float32) * 0.0015 - 0.03
    a = (np.pi * 15.0 * t) ** 2
    wavelet = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32, device="cuda")

    def run(vp_grad):
        vp = torch.tensor(vp0, device="cuda", requires_grad=vp_grad)
        mp = torch.tensor(mp0, device="cuda", requires_grad=True)
        rec_out = prop(wavelet, src, rec, models=[vp, mp])
        (rec_out.double() ** 2).sum().backward()
        core = prop._backend_impl
        return (rec_out.detach().clone(), vp.grad, mp.grad.clone(),
                len(core.adjoint_workspace), core._cuda_layout().bs_reconstruction_nvar)

    r1, gv1, gm1, ws1, nrec1 = run(True)
    r0, gv0, gm0, ws0, nrec0 = run(False)
    r2, gv2, gm2, ws2, _ = run(True)

    assert gv1 is not None and torch.isfinite(gv1).all() and float(gv1.abs().max()) > 0
    assert float(gm1.abs().max()) > 0
    assert gv0 is None, "vp asked for no gradient but got one"
    assert (ws1, ws0) == SLOTS[(dim, mode)], "workspace slots do not follow vp.requires_grad"
    assert (nrec1, nrec0) == (6, 3)
    assert torch.equal(r0, r1), "the record depends on vp.requires_grad"
    assert torch.equal(gm0, gm1), "the mp gradient depends on vp.requires_grad"
    assert ws2 == ws1 and torch.equal(gv2, gv1) and torch.equal(gm2, gm1) and torch.equal(r2, r1)


@requires_binding("acoustic_lsrtm2d_forward", "acoustic_lsrtm3d_forward")
@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("what", ("wavelet", "ckpt_vp"))
def test_gradients_the_backward_does_not_return_are_refused(dim, what):
    """The compiled backward returns no wavelet gradient (the background adjoint
    at the source), and no vp gradient in the checkpoint modes (they replay the
    background field alone): both are refused before anything runs, instead of
    coming back as zeros (2-D) or as the scattered adjoint (3-D wavelet).  The
    checkpoint modes keep serving the mp gradient."""
    shape, src, rec, vp0, mp0 = _case(dim)
    eq = (AcousticLSRTM if dim == 2 else AcousticLSRTM3D)(spatial_order=4, device="cuda", backend="torch")
    prop = PropTorch(eq, impl="c", memory=Ckpt(mode="chunk", chunks=4) if what == "ckpt_vp" else Full(),
                     shape=shape, dh=10.0, dt=0.0015, nt=60, abcn=10, device="cuda",
                     pml_type="cpmlr", source_type=["h1"], receiver_type=["sh1"], backend="torch")
    t = np.arange(60, dtype=np.float32) * 0.0015 - 0.03
    a = (np.pi * 15.0 * t) ** 2
    w0 = (1 - 2 * a) * np.exp(-a)

    def run(wavelet_grad, vp_grad):
        wavelet = torch.tensor(w0, dtype=torch.float32, device="cuda", requires_grad=wavelet_grad)
        vp = torch.tensor(vp0, device="cuda", requires_grad=vp_grad)
        mp = torch.tensor(mp0, device="cuda", requires_grad=True)
        (prop(wavelet, src, rec, models=[vp, mp]).double() ** 2).sum().backward()
        return mp.grad

    with pytest.raises(NotImplementedError, match="wavelet gradient" if what == "wavelet" else "RWI vp gradient"):
        run(what == "wavelet", what == "ckpt_vp")
    gm = run(False, False)
    assert torch.isfinite(gm).all() and float(gm.abs().max()) > 0
