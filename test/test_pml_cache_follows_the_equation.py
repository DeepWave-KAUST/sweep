"""The PML freshness key must live with the profiles it describes.

``PropBase.init_abc`` caches CPML/SPML profiles on the EQUATION (``equation.b``)
but used to hold the freshness key on the PROPAGATOR. Sharing one equation
between propagators is supported -- ``ModelParallel`` builds a second
propagator over the wrapped one's equation -- and anything that changes the pad
(a free surface, a different ``abcn``, a DD tile) changes the profile length.
Two propagators then each started with their own ``None`` key, both built, and
whichever ran last owned ``equation.b`` while the other's key still matched, so
its rebuild was skipped and it handed the kernel profiles built for a different
padded shape. Nothing validates the length on the way in.

Shipped example that hits it: ``docs/notebooks/02_fwi_elastic_marmousi.ipynb``
builds ``solver``, then ``solver_fs``, then runs ``solver`` again.

CPU + eager: the defect is in the cache bookkeeping, not in any kernel.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch

SHAPE = (40, 50)


def _prop(equation, abcn):
    return PropTorch(equation, backend="torch", impl="eager", shape=SHAPE, dh=10.0,
                     dt=1e-3, nt=10, abcn=abcn, B=1, dev="cpu",
                     source_type=["h1"], receiver_type=["h1"])


def _z_profile_len(equation):
    return int(np.shape(equation.b[0])[1])


def test_a_shared_equation_rebuilds_for_whichever_propagator_asks():
    equation = Acoustic(spatial_order=4, device="cpu")
    narrow, wide = _prop(equation, 10), _prop(equation, 30)
    assert narrow.shape != wide.shape          # different pads, different profiles

    narrow.init_abc()
    assert _z_profile_len(equation) == narrow.shape[0]
    wide.init_abc()
    assert _z_profile_len(equation) == wide.shape[0]

    narrow.init_abc()
    assert _z_profile_len(equation) == narrow.shape[0], (
        "the profiles on the equation are still the other propagator's; this "
        "propagator would step with a CPML profile built for a different grid"
    )


def test_the_key_lives_on_the_equation_next_to_the_value():
    equation = Acoustic(spatial_order=4, device="cpu")
    assert equation._abc_cache_key is None
    prop = _prop(equation, 10)
    prop.init_abc()
    assert equation._abc_cache_key is not None


def test_one_propagator_still_caches():
    """The point of the key is to skip a rebuild that would change nothing."""
    equation = Acoustic(spatial_order=4, device="cpu")
    prop = _prop(equation, 10)
    prop.init_abc()
    first = equation.b[0]
    prop.init_abc()
    assert equation.b[0] is first, "an unchanged configuration rebuilt the profiles"
