"""``ModelParallel`` must inherit the wrapped propagator's PML formulation.

It used to compute ``pml_type or ("cpmls" if family == "elastic" else "cpmlr")``.
Two things were wrong with that:

* the fallback was unreachable — ``PropBase.__init__`` resolves a ``None``
  ``pml_type`` to ``equation.default_pml_type`` before ``ModelParallel`` reads
  ``prop.pml_type``, so the right-hand side never ran;
* the rule it encoded was wrong. ``_family_of`` classifies by a substring of
  the class name, so ``AcousticVTI1st`` — whose staggered step unpacks the 8
  profiles of ``cpmls`` — lands in the "acoustic" family and would have been
  handed ``cpmlr`` (6 profiles). The profile list is bound positionally, so
  that is silently wrong physics under ``impl='c'``, not a crash.

Constructing the mesh and reading the resolved string needs no GPU, so the
per-tile check below is CPU-only; only the end-to-end DD build needs CUDA.
"""

import pytest

torch = pytest.importorskip("torch")

DEV = "cpu"


def _prop(cls, pml_type=None):
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    return PropTorch(cls(device=DEV, backend="torch"), impl="eager",
                     eager_options=EagerOptions(use_compile=False),
                     use_ckpt=False, dev=DEV, shape=(48, 56), abcn=16, dh=10.0,
                     dt=8e-4, free_surface=False, nt=50, B=1, pml_type=pml_type)


def test_prop_pml_type_is_always_resolved():
    """The premise of the fix: ModelParallel never sees a None to fall back on."""
    from sweep.equations import Acoustic, AcousticVTI1st, Elastic

    for cls in (Acoustic, Elastic, AcousticVTI1st):
        p = _prop(cls)                      # no pml_type passed at all
        assert p.pml_type is not None, f"{cls.__name__} left pml_type unresolved"
        assert p.pml_type == cls.default_pml_type


def test_family_guess_would_have_been_wrong_for_acoustic_vti1st():
    """Pin the reason the fallback had to go, not just that it went.

    If someone reintroduces a family-based default, this is the case that
    breaks: an equation whose NAME says acoustic but whose STEP is staggered.
    """
    from sweep.equations import AcousticVTI1st
    from sweep.parallel.dd_propagator import _family_of

    eq = AcousticVTI1st(device=DEV, backend="torch")
    family = _family_of(eq)
    would_have_guessed = "cpmls" if family == "elastic" else "cpmlr"
    assert family == "acoustic"
    assert eq.default_pml_type == "cpmls"
    assert would_have_guessed != eq.default_pml_type, (
        "the class-name family guess happens to agree here; this test is only "
        "meaningful while it disagrees")


def test_source_has_no_family_based_pml_fallback():
    """The guess must not come back — check the source, since the branch it
    lived on is unreachable and no runtime test can observe it."""
    import pathlib
    import re

    import sweep.parallel.dd_propagator as m

    src = pathlib.Path(m.__file__).read_text()
    # Narrow on purpose: ``self.family`` is used legitimately elsewhere in this
    # file (wavefield-count constants, the elastic half-step protocol). Only a
    # fallback on the pml assignment itself is the bug.
    assert not re.search(r"pml\s*=\s*pml_type\s+or", src), (
        "a pml_type fallback reappeared in dd_propagator; the wrapped prop's "
        "already-resolved pml_type is the only correct source")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_model_parallel_tile_inherits_the_outer_pml_type():
    """End to end, world=1: the per-tile PropTorch carries the same string."""
    from sweep.equations import Acoustic
    from sweep.parallel import MeshTopology
    from sweep.parallel.dd_propagator import ModelParallel
    from sweep.propagator.torch import PropTorch

    dev = torch.device("cuda:0")
    eq = Acoustic(spatial_order=4, device=dev, backend="torch")
    prop = PropTorch(eq, backend="torch", impl="c", shape=(48, 56), dh=10.0,
                     dt=8e-4, nt=50, abcn=16, source_type=["h1"],
                     receiver_type=["h1"], dev=dev, free_surface=False)
    mesh = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    ddp = ModelParallel(prop, mesh)
    assert ddp.prop.pml_type == prop.pml_type == "cpmlr"
