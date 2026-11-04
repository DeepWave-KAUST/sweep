"""The velocity-stress C-PML against a reflection-free reference, on CPU.

A half-space (free top) of 20 m elements with PropSEM's C-PML on the other three sides, five
elements thick -- a third of the fast P wavelength at 10 Hz -- and a larger box free on every
side, whose own echoes arrive after the record. Receivers 10 m from the layer; the reflection is
the relative L2 difference over the record, both components, at the worst receiver.

The velocity-stress classes used to convolve the stress itself in the momentum equation and to
take the weak divergence of ``sigma + m``, which differentiates the damping profile too: their
layer then reflected 0.54 (elastic) and 0.63 (Biot) here, against 3.6e-3 for the displacement
form's, and now 1.2e-3 and 9e-4 (on SPECFEM2D-like resolution 5e-2 against 3e-4 before, 1.5e-4
now; test/sem/specfem_compare/run_compare_absorb.py). The bound below holds them to the
displacement form's level, which that scheme fails by two orders of magnitude.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.equations.biot import biot_from_biot_constants
from sweep.sem import (BiotVSSEM2DDeformed, ElasticSEM2DDeformed, ElasticVSSEM2DDeformed,
                       PropSEM, ViscoElasticVSSEM2DDeformed)
from test_biot import DAI1

NO, DH, DT, NT, F0 = 4, 5.0, 3e-4, 1500, 10.0          # 20 m elements, 0.45 s
SHAPE, PAD = (81, 161), 120                             # 400 m x 800 m; 600 m more, x and z
ABCN = 5 * NO
SRC = np.array([[82, 6]])                               # inside an element, 30 m deep
REC = np.array([[[2, 2], [2, 40], [2, 78], [158, 2], [158, 40], [158, 78],
                 [40, 78], [82, 78], [120, 78]]])
ROCK = (3000.0, 1500.0, 2400.0)


def _ricker():
    a = (np.pi * F0 * (np.arange(NT) * DT - 1.2 / F0)) ** 2
    return torch.from_numpy((1.0 - 2.0 * a) * np.exp(-a))


def _record(eq, coef, absorbing, **kw):
    """vx, vz ``(nt, nrec, 2)``: the half-space with its layer, or the larger box with the same
    free top (Biot's open pores), source and receivers, and free sides."""
    shape = SHAPE if absorbing else (SHAPE[0] + PAD, SHAPE[1] + 2 * PAD)
    shift = np.array([0 if absorbing else PAD, 0])
    prop = PropSEM(eq, shape=shape, dh=DH, dt=DT, n_order=NO, abcn=ABCN if absorbing else 0,
                   free_surface=True, f_dominant=F0, source_type=["vz"],
                   receiver_type=["vx", "vz"], impl="eager", dtype=torch.float64, **kw)
    models = [torch.full(shape, float(c), dtype=torch.float64) for c in coef]
    with torch.no_grad():
        return prop(_ricker(), SRC + shift, REC + shift, models=models)[0].numpy()


def _reflection(eq, coef, **kw):
    test, ref = _record(eq, coef, True, **kw), _record(eq, coef, False, **kw)
    return float((np.linalg.norm(test - ref, axis=(0, 2)) / np.linalg.norm(ref, axis=(0, 2))).max())


@pytest.fixture(scope="module")
def displacement_reflection():
    return _reflection(ElasticSEM2DDeformed, ROCK)


@pytest.mark.parametrize("band_split", [False, True])
def test_velocity_stress_layer_reflects_like_the_displacement_one(band_split,
                                                                  displacement_reflection):
    refl = _reflection(ElasticVSSEM2DDeformed, ROCK,
                       equation_options={"band_split": band_split})
    assert displacement_reflection < 1e-2, displacement_reflection
    assert refl < 2.0 * displacement_reflection, (refl, displacement_reflection)


def test_biot_layer_absorbs_the_three_waves(displacement_reflection):
    """Dai's medium without drag: the slow P wave propagates and the flux memories matter."""
    coef = [float(c) for c in biot_from_biot_constants(**DAI1, b=0.0, phi=0.3)]
    refl = _reflection(BiotVSSEM2DDeformed, coef)
    assert refl < 2.0 * displacement_reflection, (refl, displacement_reflection)


def test_visco_elastic_layer_absorbs(displacement_reflection):
    """The attenuating class has its own step: Qp = Qs = 50 in both boxes."""
    refl = _reflection(ViscoElasticVSSEM2DDeformed, ROCK + (50.0, 50.0))
    assert refl < 2.0 * displacement_reflection, (refl, displacement_reflection)
