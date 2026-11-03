"""A source or receiver above PropSEM's topography is in no element; PropSEM
says which point and where its column's surface is, instead of the locator's
"could not locate ... in any Quad9 element of column ex=..".
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM
from sweep.sem.equations import AcousticSEM2DDeformed
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed

NZ, NY, NX, DH = 17, 13, 21, 25.0
NT = 8
TOPO2 = 1.0 + 0.5 * np.sin(np.arange(NX) / 4.0)
TOPO3 = TOPO2[None, :] * np.ones((NY, 1))
EAGER = dict(impl="eager", eager_options=dict(use_compile=False))


def _wav():
    w = torch.zeros(NT)
    w[2] = 1.0
    return w


def _p3():
    return PropSEM(AcousticSEM3DDeformed, shape=(NZ, NY, NX), dh=DH, dt=2e-3,
                   n_order=4, abcn=8, free_surface=True, topography=TOPO3, **EAGER)


def _p2(**kw):
    return PropSEM(AcousticSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=2e-3,
                   n_order=4, abcn=8, topography=TOPO2, **EAGER, **kw)


def _run(p, src, rec, **kw):
    vp = torch.full(p.shape, 2000.0)
    with torch.no_grad():
        return p(_wav(), np.asarray(src), np.asarray(rec), models=[vp], **kw)


@pytest.mark.parametrize("role", ["source", "receiver"])
def test_point_above_the_topography_2d_is_named(role):
    p = _p2()
    src, rec = [[10, 6]], [[[10, 6]]]
    if role == "source":
        src = [[5, 0]]
    else:
        rec = [[[10, 6], [5, 0]]]
    s = f"{1.0 + 0.5 * np.sin(5 / 4.0):.2f}"
    with pytest.raises(ValueError,
                       match=rf"1 {role}\(s\) above the topography, e\.g\. "
                             rf"\[5, 0\] where the surface of column \[5\] is row {s}"):
        _run(p, src, rec)


def test_point_above_the_topography_3d_is_named():
    p = _p3()
    with pytest.raises(ValueError, match=r"receiver\(s\) above the topography.*"
                                         r"\[3, 4, 0\].*column \[3, 4\]"):
        _run(p, [[10, 6, 6]], [[[10, 6, 6], [3, 4, 0]]])


def test_point_on_the_surface_row_is_accepted():
    """On the surface (fractional row, fractional column too) is in the
    model; the locator takes it."""
    p = _p2()
    x = np.array([4, 5, 9])
    rows = np.interp([4.5], np.arange(NX), TOPO2)
    rec = torch.tensor([[[4.5, float(rows[0])],
                         *[[float(i), float(TOPO2[i])] for i in x]]],
                       dtype=torch.float64)
    out = _run(p, [[10, 6]], rec)
    assert torch.isfinite(out).all()


def test_flat_model_does_not_check():
    """No topography: row 0 is the top of the model, nothing to refuse."""
    p = PropSEM(AcousticSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=2e-3,
                n_order=4, abcn=8, free_surface=True, **EAGER)
    assert torch.isfinite(_run(p, [[10, 0]], [[[5, 0]]])).all()
