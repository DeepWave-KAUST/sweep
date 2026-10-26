"""One propagator, several models: nothing cached may go stale.

`PropSEM` caches work that is expensive and model-independent. The C-PML
profile is cached today (it was rebuilt on every forward call and was 62-77% of
`_resolve_model_batch`, which was in turn about half of a DD forward call at
nt=60); the per-strip consts are a candidate for the same treatment.

Every such cache is keyed on what the cached thing depends on, and the whole
risk is that the key is INCOMPLETE: the second call then silently reuses the
first model's tensors and returns a gradient for a medium that is not the one
passed in. Nothing else in the suite catches that -- the DD and propagator
tests build a fresh `PropSEM` per case, so a per-propagator cache is always
cold there and a missing key never shows.

These tests reuse ONE propagator across models chosen so that a stale cache
cannot hide:

* `test_second_model_is_not_the_first` -- two clearly different media, compared
  against the same models run on their own fresh propagators.
* `test_same_c_max_different_medium` -- the sharp case for the C-PML key, which
  is `c_max`. Two media with the SAME maximum velocity but different interiors
  must still give different results; a cache keyed only on `c_max` is correct
  here (the profile genuinely depends on nothing else about the model), and
  this test is what says so out loud rather than by assumption.
* `test_alternating_models_are_order_independent` -- A, B, A must reproduce A.
  A cache that is written but never invalidated passes the first two tests and
  fails this one.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0 = 4, 25.0, 8.0
NZ, NX, NT, DT = 65, 65, 150, 5e-4
SRC = np.array([[NX // 3, 6]], dtype=np.int64)
_RIX = np.arange(4, NX - 4, 4)
REC = np.stack([_RIX, np.full_like(_RIX, 10)], -1)[None]


def _prop():
    return S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=4 * NO, f_dominant=F0, memory=BoundarySaving(),
                     dev="cuda", impl="c")


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                        device="cuda")


def _model(kind):
    z = torch.arange(NZ, device="cuda")[:, None].expand(NZ, NX).float()
    x = torch.arange(NX, device="cuda")[None, :].expand(NZ, NX).float()
    if kind == "A":
        return 2000.0 + 8.0 * z + 60.0 * torch.sin(x / 7.0)
    if kind == "B":
        return 2600.0 - 6.0 * z + 90.0 * torch.cos(x / 5.0)
    # same MAXIMUM as A (the C-PML key) but a different interior
    a = 2000.0 + 8.0 * z + 60.0 * torch.sin(x / 7.0)
    c = 2000.0 + 3.0 * z + 20.0 * torch.cos(x / 11.0)
    return c - float(c.max()) + float(a.max())


def _run(p, kind):
    vp = _model(kind).clone().requires_grad_(True)
    s = p(_wav(), SRC, REC, models=[vp])            # boundary-saved (_prop)
    (s ** 2).sum().backward()
    return s.detach().clone(), vp.grad.detach().clone()


def _err(a, b):
    """1 - cos(a, b). Compares the FIELDS, not their worst single entry."""
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return 1.0 - float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))


# Boundary saving reconstructs through atomics, so the same model does not
# repeat exactly even on one propagator. Two earlier drafts of this file got
# the CRITERION wrong before the code was ever at fault:
#
#   max|a-b| / max|b|  -- the gradient has a huge dynamic range and many
#       near-zero entries, so this is set by whichever single entry happens to
#       be noisiest. Measured when written, over six A->B->A repeats, it ranged 1.5e-06 to
#       9.8e-05, i.e. it spans more than an order of magnitude run to run. A
#       bar anywhere inside that range is a coin flip, and the first draft's
#       1e-6 bar sat below the whole distribution.
#   relative L2 -- same disease, 2.1e-06 to 5.5e-05.
#
# Cosine separates cleanly. Measured, same six repeats: 1-cos(gradient) worst
# 1.2e-09, 1-cos(record) worst 1.2e-13; and two genuinely different models give
# 1-cos(gradient) = 1.4e-02, seven orders above the noise. The bars below sit
# ~80x over the worst observed noise and ~1e5x under a real staleness.
ERR_SEIS, ERR_GRAD = 1e-10, 1e-7


@pytest.mark.parametrize("kind", ["B", "C"])
def test_second_model_is_not_the_first(kind):
    """A then X on ONE propagator must equal X on a fresh one."""
    p = _prop()
    _run(p, "A")                                   # warms every cache
    got_s, got_g = _run(p, kind)
    ref_s, ref_g = _run(_prop(), kind)
    assert _err(got_s, ref_s) < ERR_SEIS, f"{kind}: record is stale"
    assert _err(got_g, ref_g) < ERR_GRAD, f"{kind}: gradient is stale"


def test_same_c_max_different_medium():
    """C is built to share A's maximum velocity -- the C-PML cache key -- while
    differing everywhere else. The cache hits, and that must still be right."""
    p = _prop()
    a_max = float(_model("A").max())
    c_max = float(_model("C").max())
    assert abs(a_max - c_max) < 1e-3, "the fixture stopped sharing c_max"
    s_a, _ = _run(p, "A")
    s_c, _ = _run(p, "C")                          # cache HIT by construction
    assert _err(s_c, s_a) > 1e-6, (
        "two different media produced the same record -- a cache returned the "
        "first model's result")
    ref_s, _ = _run(_prop(), "C")
    assert _err(s_c, ref_s) < ERR_SEIS, "cache hit returned a stale record"


def test_alternating_models_are_order_independent():
    """A, B, A. A cache that is written once and never rechecked passes the
    other tests and fails this one."""
    p = _prop()
    s1, g1 = _run(p, "A")
    _run(p, "B")
    s2, g2 = _run(p, "A")
    assert _err(s2, s1) < ERR_SEIS, "A after B no longer reproduces A"
    assert _err(g2, g1) < ERR_GRAD, "A's gradient after B no longer reproduces A"
