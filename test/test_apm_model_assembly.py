"""`build_apm_model_tensors` against a verbatim copy of the block it replaced.

This extraction could not be covered the usual way. The compiled APM FORWARD is
live -- `_C_apm()` wires `forward_func` -- but `_guard_apm_backward` refuses the
compiled backward, and every bitgate config records a record AND a gradient. A
compiled-APM gate config would therefore error in both the before and the after
run and "pass" by comparing one error message to another, which checks nothing.

So the arithmetic is checked directly instead: the pre-refactor block is
transcribed below and compared tensor for tensor with `torch.equal`. Approximate
agreement would not do -- `lam = rho * (vp**2 - 2*vs**2)` and the algebraically
equal `rho*vp**2 - 2*rho*vs**2` are different floating-point numbers, and the
whole refactor's contract is that the numbers do not move.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.core.topography import (
    APM_MODEL_NAMES_2D,
    APM_MODEL_NAMES_3D,
    build_apm_model_tensors,
)


def _models(shape, seed=3):
    g = torch.Generator().manual_seed(seed)
    vp = 1500.0 + 1500.0 * torch.rand(shape, generator=g)
    vs = vp / 1.9
    rho = 1800.0 + 400.0 * torch.rand(shape, generator=g)
    return [vp, vs, rho]


def _air_mask(shape):
    """A hill: air above a varying surface row, as the runtime mask has it."""
    mask = np.zeros(shape, dtype=np.float32)
    nz = shape[0]
    cols = np.ndindex(*shape[1:])
    for c in cols:
        surf = 2 + int(3 * (1 + np.cos(sum(c) * 0.7)))
        mask[(slice(0, min(surf, nz - 1)), *c)] = 1.0
    return mask


def _inline_2d(models, air_mask_rt):
    """Transcribed from _c.py::forward as it stood before the extraction."""
    from sweep.equations._topography import (
        classify_topography, precompute_apm_moduli,
    )
    vp_r, vs_r, rho_r = models[0], models[1], models[2]
    lam_r = rho_r * (vp_r ** 2 - 2 * vs_r ** 2)
    mu_r = rho_r * (vs_r ** 2)
    lam_2mu_r = lam_r + 2 * mu_r
    cat_np = classify_topography(air_mask_rt)
    cat_t = torch.from_numpy(cat_np).to(device=vp_r.device, dtype=torch.int32)
    lam_eff, mu_eff, mu_xz, rho_x, rho_z = precompute_apm_moduli(
        lam_r, mu_r, rho_r, cat_np)
    return (vp_r, vs_r, rho_r, lam_r, mu_r, lam_2mu_r,
            lam_eff, mu_eff, mu_xz, rho_x, rho_z), cat_t


def _inline_3d(models, air_mask_rt):
    from sweep.equations._topography import (
        classify_topography_3d, precompute_apm_moduli_3d,
    )
    vp_r, vs_r, rho_r = models[0], models[1], models[2]
    lam_r = rho_r * (vp_r ** 2 - 2 * vs_r ** 2)
    mu_r = rho_r * (vs_r ** 2)
    lam_2mu_r = lam_r + 2 * mu_r
    cat_np = classify_topography_3d(air_mask_rt)
    cat_t = torch.from_numpy(cat_np).to(device=vp_r.device, dtype=torch.int32)
    extra = precompute_apm_moduli_3d(lam_r, mu_r, rho_r, cat_np)
    return (vp_r, vs_r, rho_r, lam_r, mu_r, lam_2mu_r, *extra), cat_t


@pytest.mark.parametrize("shape,ndim,inline,names", [
    ((28, 34), 2, _inline_2d, APM_MODEL_NAMES_2D),
    ((16, 12, 14), 3, _inline_3d, APM_MODEL_NAMES_3D),
])
def test_matches_the_inline_block_bit_for_bit(shape, ndim, inline, names):
    models = _models(shape)
    mask = _air_mask(shape)
    got, got_cat = build_apm_model_tensors(models, mask, ndim)
    want, want_cat = inline(models, mask)

    assert len(got) == len(names), f"{ndim}-D built {len(got)}, contract has {len(names)}"
    assert len(want) == len(names)
    assert torch.equal(got_cat, want_cat)
    for name, a, b in zip(names, got, want):
        assert torch.equal(torch.as_tensor(a), torch.as_tensor(b)), f"{name} differs"


def test_category_tensor_is_int32():
    """The compiled side reads it as `int`; a wider dtype is a pointer
    reinterpretation, not a wider number."""
    _, cat = build_apm_model_tensors(_models((20, 20)), _air_mask((20, 20)), 2)
    assert cat.dtype == torch.int32


def test_the_first_six_are_the_bulk_parameters_in_order():
    """The CUDA side reads this tuple POSITIONALLY. Inserting anywhere but the
    end silently reinterprets every later tensor, so the prefix is pinned."""
    assert APM_MODEL_NAMES_2D[:6] == ("vp", "vs", "rho", "lam", "mu", "lam_2mu")
    assert APM_MODEL_NAMES_3D[:6] == ("vp", "vs", "rho", "lam", "mu", "lam_2mu")
    models = _models((20, 20))
    got, _ = build_apm_model_tensors(models, _air_mask((20, 20)), 2)
    for i in range(3):
        assert torch.equal(got[i], models[i]), f"slot {i} is not models[{i}]"


def test_a_wrong_tensor_count_is_refused_not_passed_on():
    """Guards the contract itself: if a precompute helper ever returns a
    different number of arrays, that must stop here rather than reach CUDA as a
    short bind list."""
    import sweep.core.topography as T
    real = T.build_apm_model_tensors
    from sweep.equations import _topography as TP
    orig = TP.precompute_apm_moduli
    TP.precompute_apm_moduli = lambda *a, **k: orig(*a, **k)[:-1]   # one short
    try:
        with pytest.raises(RuntimeError, match="model tensors"):
            real(_models((20, 20)), _air_mask((20, 20)), 2)
    finally:
        TP.precompute_apm_moduli = orig
