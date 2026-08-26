"""The slot table must reproduce every constant it is meant to replace.

``equations/slot_table.py`` declares each compiled equation's CUDA wavefield
bind order once, so that three separate hand-maintained copies of the same
positional knowledge can be derived instead of transcribed:

* ``propagator/_stepped.py`` -- the buffer-role rotation's index pairs;
* ``parallel/dd_propagator.py`` -- ``_FAMILIES``' per-family counts;
* ``equations/cuda_layout.py`` -- ``pml_slot_axes`` / ``base_nvar`` / ``pml_nvar``.

Until every consumer reads the table, both copies exist, and this test is what
stops them drifting apart. Once the consumers are migrated the *hard-coded* side
of each assertion goes away and the table stands alone; while both exist, a
mismatch here is the cheapest possible place to find out.

None of this needs a GPU: the tables are pure data.
"""
from __future__ import annotations

import pytest

from sweep.equations import slot_table as ST


# --------------------------------------------------------------------------- #
# vs propagator/_stepped.py
# --------------------------------------------------------------------------- #
def test_psi_pairs_match_stepped():
    from sweep.propagator import _stepped as S

    assert ST.ACOUSTIC2D.pairs(adjoint=False) == S.ACOUSTIC2D_PSI_PAIRS
    assert ST.ACOUSTIC3D.pairs(adjoint=False) == S.ACOUSTIC3D_PSI_PAIRS


def test_adjoint_pairs_match_stepped():
    """The 3-D adjoint tuple is NOT sorted by its first component.

    ``swap_aux`` shadows psi and zeta, and the 3-D bind order inserts the y
    slots mid-list, so the pairs come out ``(3,4,7,5,6,8)`` in their first
    components. Ordering the derivation by the SHADOW index reproduces that for
    free; ordering it by the slot index would silently produce a different --
    and wrong -- rotation.
    """
    from sweep.propagator import _stepped as S

    assert ST.ACOUSTIC2D.pairs(adjoint=True) == S.ACOUSTIC2D_ADJ_PAIRS
    assert ST.ACOUSTIC3D.pairs(adjoint=True) == S.ACOUSTIC3D_ADJ_PAIRS


def test_first_order_families_rotate_nothing():
    """Elastic slots are fixed; the DD driver passes ``()`` for both, and the
    table has to agree or a stepped continuation would re-order live buffers."""
    for t in (ST.ELASTIC2D, ST.ELASTIC3D):
        assert t.pairs(adjoint=True) == ()
        assert t.pairs(adjoint=False) == ()
        assert t.u_blocks == ()


def test_vrz_needs_no_adjoint_extra_discriminator():
    """VRZ has no fused adjoint, so it has no adjoint-only shadow slots.

    The DD driver currently picks its pair set with
    ``acoustic_adj_pairs if adjoint_extra_nvar else acoustic_psi_pairs`` -- a
    discriminator that exists only because the pair sets were hand-written.
    From the table it falls out.
    """
    for t in (ST.ACOUSTIC_VRZ2D, ST.ACOUSTIC_VRZ3D):
        assert t.adjoint_extra_nvar == 0
        assert t.pairs(adjoint=True) == t.pairs(adjoint=False)


# --------------------------------------------------------------------------- #
# vs equations/cuda_layout.py
# --------------------------------------------------------------------------- #
DECLARED = [
    ("Acoustic", "ACOUSTIC2D"),
    ("Acoustic3D", "ACOUSTIC3D"),
    ("AcousticVRZ", "ACOUSTIC_VRZ2D"),
    ("AcousticVRZ3D", "ACOUSTIC_VRZ3D"),
    ("Elastic", "ELASTIC2D"),
    ("Elastic3D", "ELASTIC3D"),
]


@pytest.mark.parametrize("eq_name,table_name", DECLARED)
def test_derived_counts_match_cuda_layout(eq_name, table_name):
    import sweep.equations as E

    eq = getattr(E, eq_name)(device="cpu", backend="torch")
    spec, t = eq.cuda_layout, getattr(ST, table_name)

    assert t.base_nvar == spec.base_nvar
    assert t.pml_nvar == spec.pml_nvar
    assert t.adjoint_extra_nvar == getattr(spec, "adjoint_extra_nvar", 0)
    assert t.n_forward == spec.base_nvar + spec.pml_nvar


@pytest.mark.parametrize("eq_name,table_name", DECLARED)
def test_pml_slot_axes_match_cuda_layout(eq_name, table_name):
    """``axis`` is a fact; slab allocation is a policy.

    An equation that has not opted into ``aux_storage='slab'`` must derive
    ``None`` even though its slots carry real axis letters -- otherwise
    declaring a table would silently switch its aux buffers from full grids to
    per-axis slabs and change what gets allocated.
    """
    import sweep.equations as E

    eq = getattr(E, eq_name)(device="cpu", backend="torch")
    spec, t = eq.cuda_layout, getattr(ST, table_name)
    assert t.pml_slot_axes == spec.pml_slot_axes


# --------------------------------------------------------------------------- #
# vs parallel/dd_propagator.py
# --------------------------------------------------------------------------- #
def test_dd_wavefield_counts():
    """Pin the counts the DD driver derives from the table.

    These used to be asserted against ``dd_propagator._FAMILIES``, the
    hand-written copy. That copy is gone -- the driver reads the table -- so
    with nothing on the other side of the equals sign the table could drift
    silently. The expected values are therefore spelled out here, taken from
    the C++ bind order they describe:

      acoustic 2-D/3-D  9 / 12 forward, 11 / 15 adjoint (the fused adjoint
                        double-buffers zeta as well as psi), 3 reconstruction
      elastic  2-D/3-D  15 / 36 forward, same adjoint (no shadow slots),
                        7 / 12 reconstruction (physical + velocity carries)
    """
    import sweep.equations as E

    expected = {
        # name: (n_forward, n_adjoint, nrecon, n_velocity)
        "Acoustic": (9, 11, 3, 0),
        "Acoustic3D": (12, 15, 3, 0),
        "AcousticVRZ": (9, 9, 3, 0),
        "AcousticVRZ3D": (12, 12, 3, 0),
        "Elastic": (15, 15, 7, 2),
        "Elastic3D": (36, 36, 12, 3),
    }
    for name, want in expected.items():
        t = getattr(E, name)(device="cpu", backend="torch").cuda_layout.slots
        got = (t.n_forward, t.n_adjoint, t.nrecon, len(t.vel_idx))
        assert got == want, f"{name}: {got} != {want}"


def test_adjoint_list_is_longer_than_the_forward_one_for_acoustic():
    """``dd_propagator`` sizes its adjoint fallback list with ``_nwf``.

    For acoustic the compiled backward wants 11 (2-D) / 15 (3-D) tensors, not
    the forward's 9 / 12 -- see the TORCH_CHECKs in ``acoustic2d/backward.cu``
    and ``acoustic3d/backward.cu``. Latent today only because the adjoint list
    is never actually empty. Two different quantities sharing one name is
    exactly what the table separates.
    """
    assert (ST.ACOUSTIC2D.n_forward, ST.ACOUSTIC2D.n_adjoint) == (9, 11)
    assert (ST.ACOUSTIC3D.n_forward, ST.ACOUSTIC3D.n_adjoint) == (12, 15)
    assert ST.ELASTIC2D.n_adjoint == ST.ELASTIC2D.n_forward == 15
    assert ST.ELASTIC3D.n_adjoint == ST.ELASTIC3D.n_forward == 36


def test_elastic_velocity_slots_are_a_prefix_but_say_so_explicitly():
    """The DD driver slices ``[0, nv)`` for velocity and ``[nv, nphys)`` for
    stress. That holds for elastic, but it is an unstated invariant -- and it is
    false for DAS-Zhao, whose physical block is split around the PML block. The
    table states the indices instead of assuming the layout."""
    assert ST.ELASTIC2D.vel_idx == (0, 1)
    assert ST.ELASTIC2D.stress_idx == (2, 3, 4)
    assert ST.ELASTIC3D.vel_idx == (0, 1, 2)
    assert ST.ELASTIC3D.stress_idx == (3, 4, 5, 6, 7, 8)


# --------------------------------------------------------------------------- #
# internal consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("table_name", [n for _, n in DECLARED])
def test_dbuf_links_resolve(table_name):
    t = getattr(ST, table_name)
    names = {s.name for s in t.slots}
    for s in t.slots:
        if s.role == "dbuf":
            assert s.dbuf_of in names, f"{s.name} shadows unknown slot {s.dbuf_of}"
        else:
            assert s.dbuf_of is None


@pytest.mark.parametrize("table_name", [n for _, n in DECLARED])
def test_adjoint_only_slots_come_last(table_name):
    """The forward list is a PREFIX of the adjoint list -- that is what lets
    ``bind()`` infer which variant it was handed from the list length alone."""
    t = getattr(ST, table_name)
    flags = [s.adjoint_only for s in t.slots]
    assert flags == sorted(flags), "an adjoint-only slot precedes a forward one"


@pytest.mark.parametrize("table_name", [n for _, n in DECLARED])
def test_recon_names_are_real_slots_or_declared_carries(table_name):
    t = getattr(ST, table_name)
    names = {s.name for s in t.slots}
    for r in t.recon:
        assert r in names or r.endswith("_prev"), (
            f"{r} is neither a bind slot nor a reconstruction carry")


# --------------------------------------------------------------------------- #
# output-binding declarations vs the compiled backward's TORCH_CHECKs
# --------------------------------------------------------------------------- #
GRADS_OUT = {
    # equation -> (grads_out_has_wavelet, illum_nvar), from the TORCH_CHECKs in
    # each backward.cu. These used to be inferred from the DD driver's notion of
    # "family"; a wrong value is a length/offset error in the gradient list, so
    # it is worth pinning independently of the driver.
    "Acoustic": (True, 2),        # acoustic2d/backward.cu:101, :104
    "Acoustic3D": (True, 2),      # acoustic3d/backward.cu:125, :128
    "AcousticVRZ3D": (True, 2),   # acoustic_vrz3d/backward.cu:329; illum unused
    "Elastic": (False, 0),        # elastic2d/backward.cu:306, :310
    "Elastic3D": (False, 0),      # elastic3d/backward.cu:528, :532
}


@pytest.mark.parametrize("eq_name,expected", sorted(GRADS_OUT.items()))
def test_grads_out_and_illum_declarations(eq_name, expected):
    import sweep.equations as E

    spec = getattr(E, eq_name)(device="cpu", backend="torch").cuda_layout
    assert (spec.grads_out_has_wavelet, spec.illum_nvar) == expected
