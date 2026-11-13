"""``impl='c'`` refuses a wavelet gradient its backward does not compute.

The elastic, DAS and first-order VTI backwards have no wavelet slot in
``grads_out`` (``cuda_layout.grads_out_has_wavelet=False``), so a wavelet that
required grad came back with ``wavelet.grad`` None without a word -- and an
optimizer stepping it never moved it.  ``_CompiledPropagator.forward`` now
refuses it before anything runs, as ``c_refuse_grads`` does for the Born and VRZ
families and ``ModelParallel`` for every equation.  A wavelet that asks for
nothing still gets the model gradients, under no_grad nothing is asked, and the
equations that do return the wavelet gradient keep returning it.  The host
(CPU) core runs the same ``forward``, so each case runs on both devices.
"""
import os

import numpy as np
import pytest
import torch

from gradient_cases import model_arrays
from sweep.equations import (Acoustic, AcousticVTI1st, AcousticVTI1st3D, DASElastic, DASElastic3D, DASMu,
                             DASMu3D, Elastic, Elastic3D, ElasticTTI2nd, ElasticTTISG, ElasticTTISG3D,
                             ElasticVRR, ViscoElastic)
from sweep.propagator.options import Full
from sweep.propagator.torch import PropTorch



def _host_core() -> bool:
    if os.environ.get("SWEEP_TEST_HOST_CORE", "1").lower() in ("0", "false", "no", "off"):
        return False
    from sweep.backend.c import hostcore
    return hostcore.available()


DEVICES = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
    pytest.param("cpu", marks=pytest.mark.skipif(not _host_core(), reason="no host (CPU) core at hand")),
]

DH, DT, NT, ABCN = 10.0, 1e-3, 60, 8
# Keyed by the registered name: Elastic3D's class is named Elastic, DASElastic's DASZhao.
NO_WAVELET_SLOT = {"Elastic": Elastic, "Elastic3D": Elastic3D, "ElasticVRR": ElasticVRR,
                   "ElasticTTI2nd": ElasticTTI2nd, "ElasticTTISG": ElasticTTISG, "ElasticTTISG3D": ElasticTTISG3D,
                   "ViscoElastic": ViscoElastic, "DASElastic": DASElastic, "DASElastic3D": DASElastic3D,
                   "DASMu": DASMu, "DASMu3D": DASMu3D, "AcousticVTI1st": AcousticVTI1st,
                   "AcousticVTI1st3D": AcousticVTI1st3D}
# model_arrays has no recipe for these (values only need to be plausible).
EXTRA = {"Rp_x": 0.0, "Rp_z": 0.01, "Rs_x": 0.0, "Rs_z": 0.01, "Qp": 100.0, "Qs": 50.0}


def _solver(cls, dev):
    eq = cls(spatial_order=4, device=dev, backend="torch")
    shape = (24, 26) if eq.ndim == 2 else (16, 14, 18)
    prop = PropTorch(eq, backend="torch", impl="c", memory=Full(), shape=shape, abcn=ABCN, dh=DH, dt=DT,
                     nt=NT, dev=dev)
    assert prop.impl == "c"
    return prop, shape


def _inputs(prop, shape, wavelet_grad):
    dev = prop.equation.device
    t = np.arange(NT, dtype=np.float32) * DT - 0.03
    a = (np.pi * 25.0 * t) ** 2
    wav = torch.tensor((1 - 2 * a) * np.exp(-a), device=dev, requires_grad=wavelet_grad)
    names = list(prop.equation.models)
    known = model_arrays([n for n in names if n not in EXTRA], shape)
    arrays = [np.full(shape, EXTRA[n], np.float32) if n in EXTRA else known.pop(0) for n in names]
    models = [torch.tensor(a, device=dev, requires_grad=True) for a in arrays]
    if len(shape) == 2:
        src = np.array([[shape[1] // 2, 4]], np.int32)
        rec = np.array([[[x, 4] for x in range(2, shape[1] - 2, 4)]], np.int32)
    else:
        src = np.array([[shape[2] // 2, shape[1] // 2, 4]], np.int32)
        rec = np.array([[[x, shape[1] // 2, 4] for x in range(2, shape[2] - 2, 4)]], np.int32)
    return wav, src, rec, models


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("cls", NO_WAVELET_SLOT.values(), ids=NO_WAVELET_SLOT.keys())
def test_c_refuses_the_wavelet_gradient(cls, dev):
    prop, shape = _solver(cls, dev)
    assert not prop._cuda_layout().grads_out_has_wavelet
    wav, src, rec, models = _inputs(prop, shape, wavelet_grad=True)
    with pytest.raises(NotImplementedError, match=f"impl='c' {cls.__name__} does not return the wavelet gradient"):
        prop(wav, src, rec, models=models)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("cls", (Elastic, AcousticVTI1st), ids=lambda c: c.__name__)
def test_model_gradient_without_the_wavelet(cls, dev):
    """The control: a wavelet that asks for nothing still gets the model
    gradients, and under no_grad a requires_grad wavelet is not refused --
    nothing is asked of the backward there."""
    prop, shape = _solver(cls, dev)
    wav, src, rec, models = _inputs(prop, shape, wavelet_grad=False)
    (prop(wav, src, rec, models=models).double() ** 2).sum().backward()
    assert wav.grad is None
    assert torch.isfinite(models[0].grad).all() and float(models[0].grad.abs().max()) > 0
    wav, _, _, models = _inputs(prop, shape, wavelet_grad=True)
    with torch.no_grad():
        prop(wav, src, rec, models=models)


@pytest.mark.parametrize("dev", DEVICES)
def test_acoustic_still_returns_the_wavelet_gradient(dev):
    prop, shape = _solver(Acoustic, dev)
    wav, src, rec, models = _inputs(prop, shape, wavelet_grad=True)
    (prop(wav, src, rec, models=models).double() ** 2).sum().backward()
    assert wav.grad is not None and torch.isfinite(wav.grad).all() and float(wav.grad.abs().max()) > 0


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("cls", (Elastic, AcousticVTI1st), ids=lambda c: c.__name__)
def test_no_grad_per_shot_wavelet_is_not_refused(cls, dev):
    """A per-shot wavelet ``(nshots, nt)`` reaches the compiled forward as a
    view, and a view keeps ``requires_grad=True`` under no_grad. Nothing is
    asked of the backward there, so nothing may be refused -- the refusal has to
    follow grad mode, not the tensor flag."""
    prop, shape = _solver(cls, dev)
    wav, src, rec, models = _inputs(prop, shape, wavelet_grad=False)
    wav = wav[None].clone().requires_grad_(True)
    models = [m.detach() for m in models]
    with torch.no_grad():
        out = prop(wav, src, rec, models=models)
    assert torch.isfinite(out).all()
    with pytest.raises(NotImplementedError, match="does not return the wavelet gradient"):
        prop(wav, src, rec, models=models)
