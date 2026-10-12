"""Acoustic1st must REFUSE a free surface.

Its time step has no free-surface code, yet the class inherited
``supports_free_surface = True``: ``PropTorch(Acoustic1st, free_surface=True)``
ran with the top left as the edge of the padded array (a reflector a few
cells above row 0 that moves with spatial_order) instead of a
pressure-release surface. The refusal names the real reason, not the
anisotropy text the VTI/TTI equations get.
"""
import pytest

torch = pytest.importorskip("torch")


def _build(free_surface, pml_type="cpmls", **extra):
    from sweep.equations import Acoustic1st
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    eq = Acoustic1st(spatial_order=4, device="cpu", backend="torch")
    return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False),
                     use_ckpt=False, dev="cpu", shape=(32, 40), abcn=8, dh=10.0, dt=8e-4,
                     free_surface=free_surface, nt=20, B=1, pml_type=pml_type, **extra)


@pytest.mark.parametrize("pml_type", ["cpmls", "spml"])
def test_free_surface_true_raises(pml_type):
    with pytest.raises(NotImplementedError, match="no free-surface implementation"):
        _build(True, pml_type=pml_type)


def test_per_edge_top_raises():
    with pytest.raises(NotImplementedError, match="no free-surface implementation"):
        _build(["top"])


def test_absorbing_top_still_builds():
    _build(False)


def test_flag_and_reason_on_the_class():
    from sweep.equations import Acoustic, Acoustic1st

    assert Acoustic1st.supports_free_surface is False
    assert "no free-surface implementation" in Acoustic1st.free_surface_unsupported_reason
    assert Acoustic.supports_free_surface is True
    assert Acoustic.free_surface_unsupported_reason is None
