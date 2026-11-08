"""JAX boundary saving's ring, as ``test_eager_bs_ring_placement`` holds the
eager one: taken after the source goes in (a source on the free-surface
face's ring, which lies inside the model, was dropped: a vp gradient 18-21 %
off plain autodiff's, against 1e-3 a few cells away), and in each face's own
PML (with every ring at ``max(abcn) + halo``, a face thinner than the others
was 6 % off).  Away from the outermost ``M + 1`` cells next to an absorbing
face the gradient matches plain autodiff's to round-off, whatever each face's
width."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from sweep.equations import Acoustic  # noqa: E402
from sweep.propagator.jax import PropJax  # noqa: E402
from sweep.propagator.options import BoundaryOptions, MemoryOptions  # noqa: E402

SHAPE, NT, ABCN, HALO = (40, 44), 160, 8, 2
EDGE = HALO + 1                # the cells next to an absorbing face that keep an error


def _vp_gradient(boundary_saving, src, abcn=ABCN, free_surface=False):
    t = np.arange(NT) * 1e-3 - 0.03
    a = (np.pi * 25.0 * t) ** 2
    wavelet = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)[None, :]
    vp = (2000 + 300 * np.random.default_rng(1).random(SHAPE)).astype(np.float32)
    memory = (MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage="gpu"))
              if boundary_saving else None)
    prop = PropJax(Acoustic(spatial_order=4, device="cpu", backend="jax"), SHAPE, memory=memory, dh=10.0,
                   dt=1e-3, source_type=["h1"], receiver_type=["h1"], abcn=abcn, free_surface=free_surface,
                   nt=NT, B=1)
    rec = np.array([[[ix, 12] for ix in range(2, SHAPE[1] - 2, 3)]], dtype=np.int32)
    s = np.array([[src]], dtype=np.int32)
    return np.asarray(jax.grad(lambda m: jnp.sum(prop(wavelet, s, rec, models=[m]) ** 2))(jnp.asarray(vp)))


def _bs_error(src, inner=False, **kw):
    full, bs = _vp_gradient(False, src, **kw), _vp_gradient(True, src, **kw)
    if inner:
        full, bs = full[EDGE:-EDGE, EDGE:-EDGE], bs[EDGE:-EDGE, EDGE:-EDGE]
    return float(np.linalg.norm(bs - full) / np.linalg.norm(full))


def test_a_source_on_the_ring_of_a_free_surface_face():
    control = _bs_error((22, ABCN + 3), free_surface=True)
    for z in (ABCN - HALO, ABCN - 1):
        assert _bs_error((22, z), free_surface=True) < 3 * control


@pytest.mark.parametrize("abcn", [(8, 8, 4, 8), (12, 8, 8, 8)], ids=["thin_x_lo", "thick_z_lo"])
def test_a_thinner_pml_face_keeps_its_ring_out_of_the_model(abcn):
    src = (6, 20)
    assert _bs_error(src, abcn=abcn, inner=True) < 2e-5
