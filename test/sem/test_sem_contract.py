"""``contract``: the SEM reference-derivative einsum, unrolled while torch.compile
traces for a GPU.

Every ``contract("<spec>", D, F)`` in the SEM sources is checked against
``torch.einsum`` in its unrolled form (the specs are read off the sources, so a new
call site is covered without editing this file), and the switch itself: einsum
uncompiled and when tracing for the CPU, the multiply-adds when tracing for CUDA --
except where a call passes ``unroll=False``, as the 3-D deformed C-PML classes do.
"""
import pathlib
import re

import pytest
import torch

import sweep.sem._assembly as A

SEM = pathlib.Path(A.__file__).parent
SPECS = sorted({m.group(1) for p in SEM.rglob("*.py")
                for m in re.finditer(r'contract\(\s*"([^"]+)"', p.read_text())}
               | {"Xx,ezyX->ezyx", "Yy,ezYx->ezyx", "Zz,eZyx->ezyx"})   # stress_divergence3's loop


def _operands(spec, n=5, dtype=torch.float64, device="cpu"):
    a, b = spec.split("->")[0].split(",")
    c = next(ch for ch in a if ch in b)
    sizes = {ch: (n if ch == c else 3 + i % 2) for i, ch in enumerate(b)}
    g = torch.Generator().manual_seed(len(spec))
    D = torch.randn(n, n, generator=g, dtype=dtype).to(device)
    F = torch.randn(*[sizes[ch] for ch in b], generator=g, dtype=dtype).to(device)
    return D, F


def test_specs_found():
    assert len(SPECS) >= 10, SPECS


@pytest.mark.parametrize("spec", SPECS)
def test_unrolled_equals_einsum(spec):
    D, F = _operands(spec)
    got = A._contract_unrolled(spec, D, F)
    assert got is not None, f"{spec} is not of the unrollable form"
    ref = torch.einsum(spec, D, F)
    assert got.shape == ref.shape
    assert (got - ref).abs().max() <= 1e-13 * ref.abs().max()


def test_other_forms_are_left_to_einsum():
    D = torch.randn(5, 5, dtype=torch.float64)
    assert A._contract_unrolled("nij,nj->ni", torch.randn(4, 3, 3), torch.randn(4, 3)) is None
    assert A._contract_unrolled("ab,eab->e", D, torch.randn(2, 5, 5)) is None   # both indices contracted
    assert A._contract_unrolled("ab,eb->ea", D, torch.randn(2, 4)) is None      # wrong axis length


def test_uncompiled_is_the_einsum():
    for spec in SPECS:
        D, F = _operands(spec, dtype=torch.float32)
        assert torch.equal(A.contract(spec, D, F), torch.einsum(spec, D, F))


def _spy_unrolled(monkeypatch):
    """A list that records each call of the unrolled form."""
    taken = []
    real = A._contract_unrolled

    def spy(*args):
        taken.append(1)
        return real(*args)

    monkeypatch.setattr(A, "_contract_unrolled", spy)
    return taken


def _trace(spec, D, F, monkeypatch):
    """Run ``contract`` under Dynamo (eager backend: no C++/Triton compiler needed)
    and report whether the unrolled form was taken."""
    taken = _spy_unrolled(monkeypatch)
    torch._dynamo.reset()
    out = torch.compile(lambda d, f: A.contract(spec, d, f), backend="eager")(D, F)
    torch._dynamo.reset()
    return out, bool(taken)


def test_cpu_trace_keeps_the_einsum(monkeypatch):
    spec = "bg,eagk->eabk"
    D, F = _operands(spec, dtype=torch.float32)
    out, unrolled = _trace(spec, D, F, monkeypatch)
    assert not unrolled
    assert torch.allclose(out, torch.einsum(spec, D, F), rtol=1e-6, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the unrolled form is the CUDA compile path")
@pytest.mark.parametrize("spec", ["bg,eagk->eabk", "ga,egb->eab", "Zz,eZyx->ezyx"])
def test_cuda_trace_unrolls(monkeypatch, spec):
    D, F = _operands(spec, dtype=torch.float32, device="cuda")
    out, unrolled = _trace(spec, D, F, monkeypatch)
    assert unrolled
    ref = torch.einsum(spec, D, F)
    assert (out - ref).abs().max() <= 1e-5 * ref.abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the unrolled form is the CUDA compile path")
def test_cuda_trace_keeps_the_einsum_when_asked(monkeypatch):
    spec = "Zz,eZyx->ezyx"
    D, F = _operands(spec, dtype=torch.float32, device="cuda")
    taken = _spy_unrolled(monkeypatch)
    torch._dynamo.reset()
    out = torch.compile(lambda d, f: A.contract(spec, d, f, unroll=False), backend="eager")(D, F)
    torch._dynamo.reset()
    assert not taken
    assert torch.equal(out, torch.einsum(spec, D, F))


# The 3-D deformed classes with a C-PML keep the einsum when compiled for CUDA: unrolled, Inductor
# put the gathers of every field into each multiply-add of the 3x3 inverse-Jacobian chain rule (one
# kernel, ~100 indirect loads per GLL node, spilled registers) and the compiled step ran 1.4-6x
# slower than with the einsum. Their flat twins unroll.
_KEEP_EINSUM = ["AcousticSEM3DDeformed", "ElasticVSSEM3DDeformed", "ElasticVTISEM3DDeformed",
                "ElasticTTISEM3DDeformed", "ViscoElasticVSSEM3DDeformed", "BiotVSSEM3DDeformed"]
_VALUES = {"vp": 2500.0, "vs": 1400.0, "rho": 2000.0, "epsilon": 0.2, "delta": 0.1, "gamma": 0.15,
           "theta": 0.4, "phi": 0.6, "Qp": 40.0, "Qs": 30.0}


def _value(cls, name):
    if cls.__name__.startswith("Biot"):
        from sweep.equations.biot import biot_from_rock
        c = biot_from_rock(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3, k_d=9e9,
                           mu=7e9, phi=0.2, kappa=1e-10, tortuosity=2.0)
        biot = dict(zip(("H", "mu", "C", "M", "rho", "rho_f", "m", "b"), (float(x) for x in c)))
        return biot[name]
    return _VALUES[name]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the unrolled form is the CUDA compile path")
@pytest.mark.parametrize("name, unrolls", [(n, False) for n in _KEEP_EINSUM]
                         + [("AcousticSEM3D", True), ("ElasticVSSEM3D", True)])
def test_cuda_compile_unrolls_except_the_deformed_3d_cpml_classes(monkeypatch, name, unrolls):
    import sweep.sem as S
    from sweep.propagator import EagerOptions
    taken = _spy_unrolled(monkeypatch)
    monkeypatch.setattr(A, "is_compiling", lambda: True)     # contract's view: a CUDA trace
    cls = getattr(S, name)
    shape = (9, 9, 9)
    prop = S.PropSEM(cls, shape=shape, dh=10.0, dt=5e-4, nt=2, n_order=4, abcn=4,
                     free_surface=True, f_dominant=10.0, dev="cuda", impl="eager",
                     eager_options=EagerOptions(use_compile=False))
    models = [torch.full(shape, _value(cls, n), device="cuda") for n in prop.model_names]
    with torch.no_grad():
        prop(torch.ones(2, device="cuda"), [[4, 4, 2]], [[[4, 4, 1]]], models=models)
    assert bool(taken) == unrolls
