"""``contract``: the SEM reference-derivative einsum, unrolled while torch.compile
traces for a GPU.

Every ``contract("<spec>", D, F)`` in the SEM sources is checked against
``torch.einsum`` in its unrolled form (the specs are read off the sources, so a new
call site is covered without editing this file), and the switch itself: einsum
uncompiled and when tracing for the CPU, the multiply-adds when tracing for CUDA.
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


def _trace(spec, D, F, monkeypatch):
    """Run ``contract`` under Dynamo (eager backend: no C++/Triton compiler needed)
    and report whether the unrolled form was taken."""
    taken = []
    real = A._contract_unrolled

    def spy(*args):
        taken.append(1)
        return real(*args)

    monkeypatch.setattr(A, "_contract_unrolled", spy)
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
