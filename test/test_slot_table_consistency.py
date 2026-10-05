"""The slot table must reproduce every constant it is meant to replace.

``equations/slot_table.py`` declares each compiled equation's CUDA wavefield
bind order once, so that three separate hand-maintained copies of the same
positional knowledge can be derived instead of transcribed:

* ``propagator/_stepped.py`` -- the buffer-role rotation's index pairs
  (**migrated**: it now derives them; the literals below are the pin);
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
#: The values ``_stepped.py`` used to hard-code. It now derives them from the
#: table, so the literals live HERE, as the expectation -- one source of truth
#: in the library, and a pin in the suite so the derivation cannot drift.
EXPECTED_PSI = {2: ((3, 7), (4, 8)),
                3: ((3, 9), (4, 10), (7, 11))}
EXPECTED_ADJ = {2: ((3, 7), (4, 8), (5, 9), (6, 10)),
                3: ((3, 9), (4, 10), (7, 11), (5, 12), (6, 13), (8, 14))}


@pytest.mark.parametrize("ndim", [2, 3])
def test_psi_pairs_match_stepped(ndim):
    from sweep.propagator import _stepped as S

    table = ST.ACOUSTIC2D if ndim == 2 else ST.ACOUSTIC3D
    assert S.acoustic_psi_pairs(ndim) == EXPECTED_PSI[ndim]
    assert table.pairs(adjoint=False) == EXPECTED_PSI[ndim]


@pytest.mark.parametrize("ndim", [2, 3])
def test_adjoint_pairs_match_stepped(ndim):
    """The 3-D adjoint tuple is NOT sorted by its first component.

    ``swap_aux`` shadows psi and zeta, and the 3-D bind order inserts the y
    slots mid-list, so the pairs come out ``(3,4,7,5,6,8)`` in their first
    components. Ordering the derivation by the SHADOW index reproduces that for
    free; ordering it by the slot index would silently produce a different --
    and wrong -- rotation.
    """
    from sweep.propagator import _stepped as S

    table = ST.ACOUSTIC2D if ndim == 2 else ST.ACOUSTIC3D
    assert S.acoustic_adj_pairs(ndim) == EXPECTED_ADJ[ndim]
    assert table.pairs(adjoint=True) == EXPECTED_ADJ[ndim]


def test_stepped_keeps_no_second_copy_of_the_pair_tables():
    """The point of the migration: the literals must not come back."""
    import sweep.propagator._stepped as S

    leftovers = [n for n in dir(S) if n.endswith("_PSI_PAIRS") or n.endswith("_ADJ_PAIRS")]
    assert not leftovers, f"_stepped.py hard-codes pair tables again: {leftovers}"


def test_first_order_families_rotate_nothing():
    """Elastic slots are fixed; the DD driver passes ``()`` for both, and the
    table has to agree or a stepped continuation would re-order live buffers."""
    for t in (ST.ELASTIC2D, ST.ELASTIC3D):
        assert t.pairs(adjoint=True) == ()
        assert t.pairs(adjoint=False) == ()
        assert t.u_blocks == ()


def test_vrz_shadow_slots_follow_the_exact_adjoint():
    """The exact CPML adjoint double-buffers zeta as well as psi (rotated
    with swap_aux), so the VRZ tables carry acoustic's adjoint-only shadow
    slots and the same adjoint pair sets.

    The DD driver picks its pair set with
    ``acoustic_adj_pairs if adjoint_extra_nvar else acoustic_psi_pairs``;
    from the table it falls out.
    """
    for vrz, ac, extra in ((ST.ACOUSTIC_VRZ2D, ST.ACOUSTIC2D, 2),
                           (ST.ACOUSTIC_VRZ3D, ST.ACOUSTIC3D, 3)):
        assert vrz.adjoint_extra_nvar == extra
        assert vrz.pairs(adjoint=True) == ac.pairs(adjoint=True)
        assert vrz.pairs(adjoint=False) == ac.pairs(adjoint=False)
        assert vrz.pairs(adjoint=True) != vrz.pairs(adjoint=False)


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
    ("AcousticLSRTM3D", "ACOUSTIC_LSRTM3D"),
    ("AcousticLSRTM", "ACOUSTIC_LSRTM2D"),
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
        "AcousticVRZ": (9, 11, 3, 0),     # exact CPML adjoint: zeta shadow slots
        "AcousticVRZ3D": (12, 15, 3, 0),
        "Elastic": (15, 15, 7, 2),
        "Elastic3D": (36, 36, 12, 3),
        # two coupled acoustic fields (bg + scattered), psi-only double buffer,
        # so adjoint == forward; reconstruction is 3 + 3
        "AcousticLSRTM3D": (24, 24, 6, 0),
        "AcousticLSRTM": (18, 18, 6, 0),
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


# --------------------------------------------------------------------------- #
# vs the shared test wavelet
# --------------------------------------------------------------------------- #
def test_the_local_ricker_copies_that_remain_are_genuinely_different():
    """Sixteen modules were migrated onto ``conftest.ricker``; five were not.

    Each of those five keeps its own because it is a DIFFERENT wavelet, not
    because nobody got round to it -- and this test says which is which, so a
    later reader does not "finish the job" and silently change those tests'
    inputs. If one of them ever becomes equivalent, this fails and it can be
    migrated deliberately.
    """
    import ast
    import pathlib

    import numpy as np

    from conftest import ricker

    keep = {"test_das_equations.py": "its own defaults (fm=12.0, delay=0.04)",
            "test_elastic_tti_2nd.py": "array form ricker(t, f)",
            "test_elastic_tti_sg3d.py": "array form ricker(t, f)",
            "test_sweep_pytorch.py": "array form, (1-0.5x^2)exp(-0.25x^2)",
            "test_visco_acoustic.py": "signature (nt, dt, f0), own delay rule"}
    here = pathlib.Path(__file__).parent
    still_local = {p.name for p in here.glob("test_*.py")
                   if any(isinstance(n, ast.FunctionDef) and n.name in ("ricker", "_ricker")
                          for n in ast.walk(ast.parse(p.read_text())))}
    assert still_local == set(keep), (
        f"the set of modules with a local ricker changed: "
        f"unexpected {sorted(still_local - set(keep))}, "
        f"gone {sorted(set(keep) - still_local)}")

    # and the shared one still produces what the migrated modules were built on
    expected = np.array([-0.1748605, -0.19641757, -0.21919768, -0.24298875],
                        dtype=np.float32)
    np.testing.assert_array_equal(ricker(4, 1.5e-3, 10.0, 0.06), expected)


def test_the_local_capture_copies_that_remain_are_deliberate():
    """Ten `capture` / `capture_backward` / `capture_both` bodies moved to
    conftest; six definitions stay, and each stays for a stated reason.

    Same rule as the ricker census above: the point is not the count, it is that
    a later reader can tell "not yet migrated" from "must not be migrated".
    """
    import ast
    import pathlib

    keep = {
        "conftest.py": "the shared home",
        # torchrun entry points -- they run under `torch.distributed.run` from
        # test/, so `from conftest import ...` would resolve via sys.path[0],
        # but `python -m test.dd_nccl_check` would not. They need two GPUs, so
        # a migration here could not be verified before landing.
        "dd_nccl_bench.py": "torchrun script, 2 GPUs, unverifiable here",
        "dd_nccl_check.py": "torchrun script, 2 GPUs, unverifiable here",
        "dd_nccl_elastic_bench.py": "torchrun script, 2 GPUs, unverifiable here",
        "dd_nccl_elastic_check.py": "torchrun script, 2 GPUs, unverifiable here",
        # genuinely different bodies
        "dd_cross_gpu_diag.py": "different body (takes `p`, not `prop`)",
        "test_dd_backward_two_tile_3d.py": "a different capture_both",
    }
    here = pathlib.Path(__file__).parent
    local = {p.name for p in here.glob("*.py")
             if any(isinstance(n, ast.FunctionDef)
                    and n.name in ("capture", "capture_backward", "capture_both")
                    for n in ast.parse(p.read_text()).body)}
    assert local == set(keep), (
        f"the set of modules defining their own capture helper changed: "
        f"unexpected {sorted(local - set(keep))}, gone {sorted(set(keep) - local)}")


def test_lsrtm2d_rotates_and_ships_both_fields():
    """2-D twin of the LSRTM3D table: background 0..8, scattered 9..17, 3+3 recon."""
    t = ST.ACOUSTIC_LSRTM2D
    assert t.u_blocks == (0, 9)
    assert t.recon_u_blocks == (0, 3)
    bg = ((3, 7), (4, 8))
    assert t.pairs(adjoint=False) == bg + tuple((a + 9, b + 9) for a, b in bg)
    assert t.pairs(adjoint=True) == t.pairs(adjoint=False)


def test_lsrtm3d_rotates_and_ships_both_fields():
    """AcousticLSRTM3D is the one multi-field rotating table.

    Every count the DD driver derives must cover BOTH fields: two rotating
    blocks in the forward/adjoint list (background 0..11, scattered 12..23) and
    two in the 3+3 reconstruction list -- which is why the reconstruction blocks
    are resolved by name (``recon_u_blocks``) instead of reusing ``u_blocks``.
    The scattered field's psi pairs are the background's shifted by 12.
    """
    t = ST.ACOUSTIC_LSRTM3D
    assert t.u_blocks == (0, 12)
    assert t.recon_u_blocks == (0, 3)
    bg = ((3, 9), (4, 10), (7, 11))
    assert t.pairs(adjoint=False) == bg + tuple((a + 12, b + 12) for a, b in bg)
    assert t.pairs(adjoint=True) == t.pairs(adjoint=False)


@pytest.mark.parametrize("eq_name, rwi_table, mp_table", [
    ("AcousticLSRTM", "ACOUSTIC_LSRTM2D", "ACOUSTIC_LSRTM2D_MP"),
    ("AcousticLSRTM3D", "ACOUSTIC_LSRTM3D", "ACOUSTIC_LSRTM3D_MP"),
])
def test_lsrtm_layout_follows_vp_grad(eq_name, rwi_table, mp_table):
    """Without vp's RWI gradient the layout reconstructs the background alone:
    the per-call table keeps the forward/adjoint geometry and drops the
    scattered reconstruction block, so ModelParallel rotates 3 grids, not 6."""
    import sweep.equations as eqs
    eq = getattr(eqs, eq_name)(device="cpu", backend="torch")
    rwi, mp = getattr(ST, rwi_table), getattr(ST, mp_table)
    assert eq.cuda_layout.slots is rwi
    assert eq.cuda_layout_for_grads(None).slots is rwi
    assert eq.cuda_layout_for_grads((True, True)).slots is rwi
    lay = eq.cuda_layout_for_grads((False, True))
    assert lay.slots is mp
    assert (lay.bs_reconstruction_nvar, lay.boundary_save_nvar, lay.last_two_storage_nvar) == (3, 1, 1)
    assert mp.slots == rwi.slots and mp.u_blocks == rwi.u_blocks
    assert mp.pairs(adjoint=True) == rwi.pairs(adjoint=True)
    assert (mp.nrecon, mp.recon_u_blocks) == (3, (0,))
    assert (rwi.nrecon, rwi.recon_u_blocks) == (6, (0, 3))


def test_recon_blocks_equal_u_blocks_for_single_field_tables():
    """``recon_u_blocks`` replaced ``u_blocks`` for the reconstruction list in the
    DD driver; for every single-field table the two must agree, so that switch
    changes nothing outside LSRTM."""
    for name in dir(ST):
        t = getattr(ST, name)
        if isinstance(t, ST.SlotTable) and len(t.u_blocks) <= 1:
            assert t.recon_u_blocks == t.u_blocks, name
