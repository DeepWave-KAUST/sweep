"""The boundary/layout spec cannot be changed after the propagator is built.

``free_surface`` and friends are consumed once, at construction: they size the
padded grid, set each face's PML width, build the profiles and -- on
``impl='c'`` -- fix the kernels' free-surface bitmask and image mirror.
Assigning them afterwards lands on the ``PropTorch`` wrapper, where an instance
attribute SHADOWS the ``__getattr__`` delegation to the backend: the read-back
reports the new value while every kernel keeps the old one.  That silence cost
a marine field-data campaign a free surface it believed was on, so the write
now raises.
"""
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch


def _prop(**kw):
    eq = Acoustic(spatial_order=4, device="cpu", backend="torch")
    return PropTorch(eq, backend="torch", impl="eager", shape=(48, 56), dev="cpu",
                     dh=10.0, dt=1e-3, nt=50, abcn=10, **kw)


@pytest.mark.parametrize("name,value", [
    ("free_surface", True),
    ("free_surface", ["top", "left"]),
    ("fs_faces", (True, False, False, False)),
    ("abcn", 30),
    ("pad", (0, 10, 10, 10)),
    ("pml_type", "cpmls"),
    ("topography", None),
])
def test_layout_attributes_are_construction_only(name, value):
    prop = _prop(free_surface=False)
    before = getattr(prop, name)
    with pytest.raises(AttributeError, match="fixed when the propagator is built"):
        setattr(prop, name, value)
    assert getattr(prop, name) == before, "the refused write must leave the value alone"


def test_the_shadowing_it_prevents():
    """Without the guard the wrapper's attribute wins the read while the
    backend -- and therefore the physics -- keeps the original."""
    prop = _prop(free_surface=False)
    backend = prop._backend_impl
    assert backend.free_surface is False and backend.fs_faces == (False,) * 4
    with pytest.raises(AttributeError):
        prop.free_surface = ["top", "left"]
    assert prop.free_surface is False
    assert backend.fs_faces == (False,) * 4
    assert backend.pad == (10, 10, 10, 10)


def test_construction_still_accepts_the_same_values():
    prop = _prop(free_surface=True)
    assert prop.free_surface is True
    assert prop.pad[0] == 0, "a top free surface means no PML pad on that face"


def test_illumination_flags_still_delegate():
    """The guard must not disturb the writes that DO delegate to the backend."""
    prop = _prop(free_surface=False)
    prop.compute_illumination = True
    assert prop._backend_impl.compute_illumination is True
