"""A pml_type the equation was not written for is refused, not run.

Every equation ships one CPML formulation (``Acoustic1st`` also takes
``'spml'``), and its kernels read the profiles positionally. Nothing checked an
explicit ``pml_type``: ``Elastic`` with ``pml_type='cpmlr'`` and ``impl='c'``
reached the elastic core with six profiles where it unpacks eight, and the
core's ``Buf::data_ptr`` assert aborted the whole Python process.
"""
import numpy as np
import pytest
import torch

import sweep.equations as E
from sweep.equations import Elastic
from sweep.propagator.torch import PropTorch

PML_TYPES = ("cpmlr", "cpmls", "spml")


def _build(cls):
    try:
        return cls(device="cpu")
    except ImportError as exc:  # e.g. Acoustic1st needs jax
        pytest.skip(f"{cls.__name__}: {exc}")


@pytest.mark.parametrize("cls", sorted(E.equation_classes().values(), key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
def test_default_is_supported(cls):
    eq = _build(cls)
    assert eq.default_pml_type in eq.supported_pml
    assert set(eq.supported_pml) <= set(PML_TYPES)
    assert eq.defaults()["supported_pml"] == list(eq.supported_pml)


@pytest.mark.parametrize("cls", sorted(E.equation_classes().values(), key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
def test_unsupported_pml_type_is_refused(cls):
    eq = _build(cls)
    for pml_type in (t for t in PML_TYPES if t not in eq.supported_pml):
        with pytest.raises(ValueError, match=rf"{cls.__name__} does not support pml_type='{pml_type}'"):
            PropTorch(eq, shape=(32, 32), dh=10.0, dt=1e-3, dev=torch.device("cpu"),
                      pml_type=pml_type)


def test_acoustic1st_keeps_spml():
    pytest.importorskip("jax")
    assert E.Acoustic1st(device="cpu").supported_pml == ["cpmls", "spml"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_elastic_cpmlr_on_c_raises_instead_of_aborting():
    dev = torch.device("cuda")
    kw = dict(shape=(48, 56), dh=10.0, dt=1e-3, dev=dev, impl="c")
    with pytest.raises(ValueError, match="supported: \\['cpmls'\\]"):
        PropTorch(Elastic(device=dev), pml_type="cpmlr", **kw)

    # The formulation it ships, named or left unset, still runs.
    wavelet = np.zeros(200, dtype=np.float32)
    wavelet[10] = 1.0
    models = [torch.full((48, 56), v, device=dev) for v in (2000.0, 1100.0, 2000.0)]
    for pml_type in ("cpmls", None):
        solver = PropTorch(Elastic(device=dev), pml_type=pml_type, **kw)
        out = solver(wavelet, np.array([[14, 4]]), np.array([[[20, 4]]]), models=models)
        assert torch.isfinite(out).all()
