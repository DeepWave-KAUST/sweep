"""Stability condition of the explicit pure-qP scheme (Liang et al. 2023,
appendix B, eqs. B-8 to B-10) and the warning ``PropTorch`` gives when ``dt``
breaks it."""
import math
import warnings

import numpy as np
import pytest
import torch

from sweep.equations import AcousticTTI, AcousticTTIBorn, AcousticVTI
from sweep.equations.qP_vti import qp_stability_ratio
from sweep.propagator.options import EagerOptions, Full
from sweep.propagator.torch import PropTorch

ONE = np.ones((4, 4), np.float32)


@pytest.mark.parametrize("order, courant", [(2, 1 / math.sqrt(2)), (4, math.sqrt(3 / 8))])
def test_isotropic_limit_is_the_textbook_courant_number(order, courant):
    # epsilon = delta = 0: B-9 reduces to the 2-D isotropic limit of the scheme
    dt = courant * 10.0 / 2000.0
    assert qp_stability_ratio(2000 * ONE, 0 * ONE, 0 * ONE, dt, (10.0, 10.0), order) == pytest.approx(1.0)


def test_anisotropy_tightens_the_limit_and_the_tti_bound_is_safe_side():
    iso = qp_stability_ratio(3000 * ONE, 0 * ONE, 0 * ONE, 1e-3, (10.0, 10.0), 8)
    vti = qp_stability_ratio(3000 * ONE, 0.2 * ONE, 0.1 * ONE, 1e-3, (10.0, 10.0), 8)
    tti = qp_stability_ratio(3000 * ONE, 0.2 * ONE, 0.1 * ONE, 1e-3, (10.0, 10.0), 8, tilted=True)
    assert iso < vti < tti


def _run(cls, dt, n=24, nt=4):
    eq = cls(spatial_order=4, device="cpu")
    solver = PropTorch(eq, shape=(n, n), dh=10.0, dt=dt, nt=nt, dev="cpu", abcn=6, impl="eager",
                       eager_options=EagerOptions(use_compile=False), memory=Full())
    values = (3000.0, 0.2, 0.1) + ((0.5,) if cls is AcousticTTI else ())
    models = [torch.full((n, n), v) for v in values]
    wavelet = torch.zeros(nt)
    wavelet[0] = 1.0
    src = np.array([[n // 2, n // 2]], np.int64)
    rec = np.array([[[n // 2, 2]]], np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver(wavelet, src, rec, models=models)
        solver(wavelet, src, rec, models=models)
    return [w for w in caught if "stability limit" in str(w.message)]


def _limit(cls):
    return 1e-3 / qp_stability_ratio(3000 * ONE, 0.2 * ONE, 0.1 * ONE, 1e-3, (10.0, 10.0), 4,
                                     tilted=cls is AcousticTTI)


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
def test_proptorch_warns_once_when_dt_breaks_the_limit(cls):
    hits = _run(cls, 1.5 * _limit(cls))
    assert len(hits) == 1, [str(w.message) for w in hits]
    assert "B-9" in str(hits[0].message)


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
def test_no_warning_below_the_limit(cls):
    assert _run(cls, 0.5 * _limit(cls)) == []


def test_born_pair_checks_too():
    eq = AcousticTTIBorn(spatial_order=4)
    models = [torch.full((8, 8), v) for v in (3000.0, 0.2, 0.1, 0.5, 0.0)]
    with pytest.warns(UserWarning, match="stability limit"):
        eq.check_stability(models, 3e-3, (10.0, 10.0))
