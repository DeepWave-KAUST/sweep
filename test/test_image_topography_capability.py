"""topography= (a staircase, not a flat free surface) must be refused by
equations that would silently ignore the rows and model a flat surface.

The support matrix is genuinely per-impl: 3-D Elastic applies the staircase in
its eager func but its CUDA kernels never read the rows (switching impl used to
switch the physics with no error); ElasticVRR is the exact reverse.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.equations import (
    Acoustic, Acoustic3D, AcousticVRZ, Elastic, Elastic3D, ElasticVRR,
    ViscoAcoustic,
)
from sweep.propagator.torch import PropTorch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rows(nx=56):
    x = np.arange(nx)
    return (3 + 2 * np.exp(-((x - nx / 2) ** 2) / 40)).round().astype(np.int64)


def _build(eq_cls, impl, dev, **extra):
    eq = eq_cls(spatial_order=4, device=dev, backend="torch")
    return PropTorch(eq, (48, 56), backend="torch", impl=impl, dev=dev,
                     dh=10.0, dt=1e-3, abcn=20, nt=50, B=1, **extra)


def test_the_support_matrix_is_declared_per_impl():
    assert Acoustic.supports_image_topography and Acoustic.supports_image_topography_c
    assert Acoustic3D.supports_image_topography and Acoustic3D.supports_image_topography_c
    assert Elastic.supports_image_topography and Elastic.supports_image_topography_c
    assert Elastic3D.supports_image_topography and not Elastic3D.supports_image_topography_c
    assert ViscoAcoustic.supports_image_topography and not ViscoAcoustic.supports_image_topography_c
    assert not ElasticVRR.supports_image_topography and ElasticVRR.supports_image_topography_c
    assert not AcousticVRZ.supports_image_topography and not AcousticVRZ.supports_image_topography_c


def test_eager_refuses_topography_on_an_equation_that_ignores_it():
    with pytest.raises(NotImplementedError, match="supports_image_topography"):
        _build(AcousticVRZ, "eager", "cpu", topography=_rows())


def test_a_flat_free_surface_never_consults_the_flag():
    p = _build(AcousticVRZ, "eager", "cpu", free_surface=True)
    assert p.free_surface


def test_eager_elastic3d_still_takes_image_topography():
    eq3 = Elastic3D(spatial_order=4, device="cpu", backend="torch")
    rows = np.tile(_rows(24)[None, :3], (20, 8))[:20, :24]
    p = PropTorch(eq3, (24, 20, 24), backend="torch", impl="eager", dev="cpu",
                  dh=10.0, dt=1e-3, abcn=6, nt=20, B=1,
                  topography=rows, topo_method="image")
    assert p._topo_method == "image"


@cuda_only
def test_c_elastic3d_refuses_what_its_kernels_ignore():
    eq3 = Elastic3D(spatial_order=4, device="cuda", backend="torch")
    rows = np.tile(_rows(24)[None, :3], (20, 8))[:20, :24]
    with pytest.raises(NotImplementedError, match="supports_image_topography_c"):
        PropTorch(eq3, (24, 20, 24), backend="torch", impl="c", dev="cuda",
                  dh=10.0, dt=1e-3, abcn=6, nt=20, B=1,
                  topography=rows, topo_method="image")


@cuda_only
def test_c_acoustic_still_takes_topography():
    p = _build(Acoustic, "c", "cuda", topography=_rows())
    assert p._topo_method == "image"
