"""The eager step injects the source itself, and nothing about that recompiles.

The compiled step used to hand back the field, and the source went in after it
as two more full-grid ops (``wavefield + mask * wavelet``); it is now a scatter
inside the compiled graph. The step's wavelet sample is a graph input, laid out
time-major so that nothing about it -- nt included -- specializes the graph.
Boundary saving injects inside its autograd.Function and hands the wavelet its
gradient by hand, which must be autograd's.
"""
import numpy as np
import pytest
import torch

from conftest import requires_compile
from sweep.equations import Acoustic
from sweep.propagator import BoundarySaving, Full
from sweep.propagator.options import Ckpt, EagerOptions
from sweep.propagator.torch import PropTorch

SHAPE = (24, 28)
SRC = np.array([[10, 4], [18, 5]])
REC = np.repeat(np.array([[[ix, 3] for ix in range(2, 26, 3)]]), 2, axis=0)


def _wavelet(nt):
    t = np.arange(nt, dtype=np.float32) * 1e-3 - 0.012
    a = (np.pi * 30.0 * t) ** 2
    w = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)
    return np.stack([w, 0.7 * np.roll(w, 3)])                      # one per shot


def _prop(memory=Full, **eager):
    return PropTorch(Acoustic(spatial_order=4, device="cpu"), shape=SHAPE, dh=10.0, dt=1e-3,
                     nt=40, abcn=6, dev="cpu", impl="eager", memory=memory(),
                     eager_options=EagerOptions(**eager))


def _recorder():
    """A torch.compile backend that keeps each graph's ops and runs it as is,
    and the list it keeps them in.  A function, not an object: the options
    are deep-copied, and a copied object would fill a list nobody reads."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append([str(n.target) for n in gm.graph.nodes
                       if n.op in ("call_function", "call_method")])
        return gm.forward

    return backend, graphs


@requires_compile
def test_the_source_is_scattered_inside_the_compiled_step():
    backend, graphs = _recorder()
    with torch.no_grad():
        _prop(use_compile=True, compile_backend=backend)(_wavelet(30), SRC, REC,
                                                         models=[torch.full(SHAPE, 2000.0)])
    assert graphs, "the step was not compiled"
    assert all(any("index_put" in op for op in ops) for ops in graphs), graphs


@requires_compile
def test_a_new_nt_does_not_recompile_the_step():
    backend, graphs = _recorder()
    prop = _prop(use_compile=True, compile_backend=backend)
    vp = torch.full(SHAPE, 2000.0)
    with torch.no_grad():
        prop(_wavelet(30), SRC, REC, models=[vp])
        compiled = len(graphs)
        prop(_wavelet(45), SRC, REC, models=[vp])
    assert compiled and len(graphs) == compiled


def _grads(memory):
    wav = torch.tensor(_wavelet(40), requires_grad=True)
    vp = torch.full(SHAPE, 2000.0, requires_grad=True)
    out = _prop(memory, use_compile=False)(wav, SRC, REC, models=[vp])
    weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(0))
    (out * weights).sum().backward()
    return out.detach(), wav.grad, vp.grad


@pytest.mark.parametrize("memory", [lambda: Ckpt(chunks=8), lambda: BoundarySaving(storage="gpu")],
                         ids=["ckpt", "bs"])
def test_the_wavelet_gradient_matches_the_full_tape(memory):
    record, wav_grad, vp_grad = _grads(memory)
    ref_record, ref_wav_grad, ref_vp_grad = _grads(Full)
    assert torch.equal(record, ref_record)
    torch.testing.assert_close(wav_grad, ref_wav_grad, rtol=1e-5, atol=1e-7 * float(ref_wav_grad.abs().max()))
    assert float(vp_grad.abs().max()) > 0


def test_adjoint_mode_runs_the_wavelet_backwards():
    """It raised for one-source-per-shot input: the old indexed path unpacked
    ``(B, nsrc, ndim)`` coordinates it was never given."""
    prop = _prop(use_compile=False)
    vp, wav = torch.full(SHAPE, 2000.0), _wavelet(40)
    with torch.no_grad():
        adjoint = prop(wav, SRC, REC, models=[vp], adj=True)
        reversed_ = prop(np.ascontiguousarray(wav[:, ::-1]), SRC, REC, models=[vp])
    assert torch.equal(adjoint, reversed_)
