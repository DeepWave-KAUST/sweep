"""`cpml_axis_update` must be the inline form, arithmetic for arithmetic.

The refactor's acceptance criterion is bit-exact output, and floating-point
addition is not associative: rewriting ``(1 + b) * lap + dbd * du`` as
``dbd * du + (1 + b) * lap`` is a different number, not a tidier spelling of the
same one. The bit-exact gate would catch that -- but only for equations a gate
actually covers, and only after a full propagation. Comparing the primitive
against a verbatim transcription of the block it replaced localises the check to
the arithmetic itself, on inputs chosen to make cancellation likely.
"""
from __future__ import annotations

import pytest
import torch

from sweep.equations._cpml import cpml_axis_update
from sweep.operators.torch import gradient as grad_op

CASES = [((1, 64, 72), -2), ((1, 64, 72), -1), ((1, 24, 20, 24), -3),
         ((1, 24, 20, 24), -2), ((1, 24, 20, 24), -1)]


def _inline(lap, du, psi, zeta, a, b, dbd, h, axis):
    """Transcribed verbatim from acoustic.py's step_cpml, pre-refactor."""
    tmp = ((1 + b) * lap + dbd * du) + grad_op(a * psi, h, axis, kernels=None)
    return ((1 + b) * tmp + a * zeta,
            b * du + a * psi,
            b * tmp + a * zeta)


@pytest.mark.parametrize("shape,axis", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_primitive_is_the_inline_form_bit_for_bit(shape, axis, dtype, device):
    g = torch.Generator(device=device).manual_seed(7)
    # Signed, O(1) inputs: same-magnitude opposite-sign terms are where a
    # reassociated sum stops agreeing in the last bits.
    args = [torch.rand(shape, generator=g, device=device, dtype=dtype) * 2 - 1
            for _ in range(7)]
    lap, du, psi, zeta, a, b, dbd = args
    h = 12.5
    got = cpml_axis_update(lap, du, psi, zeta, a, b, dbd, h, axis, grad_op)
    want = _inline(lap, du, psi, zeta, a, b, dbd, h, axis)
    for name, x, y in zip(("contrib", "psi_next", "zeta_next"), got, want):
        assert torch.equal(x, y), f"{name} differs at {dtype}/{device}/axis={axis}"


def test_psi_next_does_not_alias_psi():
    """The forward reads neighbours' psi before writing the next one, so the
    update must produce a NEW tensor -- an in-place one is a read-after-write
    race that shows up as a wrong wavefield only under some launch orders."""
    g = torch.Generator().manual_seed(0)
    t = lambda: torch.rand((1, 16, 16), generator=g) * 2 - 1
    lap, du, psi, zeta, a, b, dbd = (t() for _ in range(7))
    before = psi.clone()
    _, psi_next, _ = cpml_axis_update(lap, du, psi, zeta, a, b, dbd, 10.0, -1, grad_op)
    assert psi_next.data_ptr() != psi.data_ptr()
    assert torch.equal(psi, before), "psi was mutated in place"
