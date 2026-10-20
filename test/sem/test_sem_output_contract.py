"""Every SEM propagator returns the FD seismogram tensor, not a dict.

``SEMPropagator.forward`` used to return ``{"seis_x": ..., "seis_z": ...}``, with
the key set depending on which equation you had built -- ``seis_x``/``seis_z``
for displacement elastic, ``seis_vx``/``seis_vz`` for velocity-stress,
``seis_chi`` or ``seismograms`` for acoustic, plus ``snapshots*`` and
``snapshot_times``. Eight key names for one concept, all string-typed, so a
wrong key was a runtime ``KeyError`` and there was nothing to autocomplete.

It now returns ``(nt, nrec, ncomp)`` -- the FD ``PropTorch`` layout minus the
shot axis, since this propagator is single-shot -- with the components selected
and ORDERED by ``receiver_type``. ``PropSEM`` stacks shots on top of that to
give the full ``(nshots, nt, nrec, ncomp)``.

These tests pin the contract itself. The physics is covered elsewhere; what is
easy to break here is the AXIS ORDER, because a transposed seismogram of a
square-ish test case still has a plausible shape and silently broadcasts.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import (PropSEM, SEMPropagator, ElasticSEM2DDeformed,
                       build_axis_aligned_quad9, build_elastic_cpml)
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D

NE, NO, DH = 12, 4, 12.5
L = NE * NO * DH
VP, VS, RHO, F0 = 2400.0, 1250.0, 2000.0, 12.0
DT = 0.30 * (0.172 * NO * DH) / (VP * 1.2)          # GLL-limited
NT, NREC = 160, 9
# nt != nrec on purpose: a transposed record would still have a legal shape if
# they matched, and every assertion below would pass on a broken layout.
assert NT != NREC


def _mesh():
    return build_axis_aligned_quad9(nex=NE, nez=NE, x_max=L, z_max=L, n_order=NO)


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))


def _rec():
    return [(float(x), 3 * DH) for x in np.linspace(0.2 * L, 0.8 * L, NREC)]


def _elastic(**kw):
    m = _mesh()
    shp = (m.n_elements, m.ngll, m.ngll)
    cpml = build_elastic_cpml(m, dt=DT, sides=("left", "right", "bottom"),
                              pml_thickness_elements=3, c_max=VP * 1.2,
                              f_dominant=F0)
    eq = ElasticSEM2DDeformed.build(m, torch.full(shp, VP), torch.full(shp, VS),
                                    torch.full(shp, RHO), cpml=cpml)
    return SEMPropagator.build(eq, dt=DT, nt=NT, source_xz=[(0.5 * L, 3 * DH)],
                               receiver_xz=_rec(), source_kinds=["force"],
                               source_payloads=[(0.0, 1.0)], pml_guard="off",
                               **kw)


def _acoustic(**kw):
    m = _mesh()
    shp = (m.n_elements, m.ngll, m.ngll)
    cpml = build_cpml(m, dt=DT, sides=("left", "right", "bottom"),
                      pml_thickness_elements=3, c_max=VP, f_dominant=F0)
    eq = AcousticPotentialSEM2D.build(m, torch.full(shp, VP),
                                      torch.full(shp, RHO),
                                      dirichlet=("top",), cpml=cpml)
    return SEMPropagator.build(eq, dt=DT, nt=NT, source_xz=[(0.5 * L, 3 * DH)],
                               receiver_xz=_rec(), pml_guard="off", **kw)


# ------------------------------ the contract ------------------------------- #
def test_forward_returns_a_tensor_not_a_mapping():
    out = _elastic().forward(_wav())
    assert torch.is_tensor(out)
    assert not hasattr(out, "keys"), "still dict-like"


@pytest.mark.parametrize("build,ncomp", [(_acoustic, 1), (_elastic, 2)])
def test_axis_order_is_nt_nrec_ncomp(build, ncomp):
    out = build().forward(_wav())
    assert out.shape == (NT, NREC, ncomp), (
        f"expected (nt={NT}, nrec={NREC}, ncomp={ncomp}), got {tuple(out.shape)}")


def test_the_time_axis_really_is_the_time_axis():
    """Shape alone cannot catch a transpose that happens to fit. A causal run
    is quiet before the source arrives, so the FIRST samples along axis 0 must
    be near zero while a whole receiver column is not."""
    out = _elastic().forward(_wav())
    early = out[:5].abs().max()
    total = out.abs().max()
    assert early < 1e-3 * total, (
        f"axis 0 is not time: |first 5 samples| = {early:.3e} vs peak "
        f"{total:.3e}; a transposed record would look like this")


def test_receiver_type_selects_and_orders_components():
    full = _elastic().forward(_wav())
    z_only = _elastic(receiver_type=["u_z"]).forward(_wav())
    swapped = _elastic(receiver_type=["u_z", "u_x"]).forward(_wav())
    assert z_only.shape == (NT, NREC, 1)
    assert torch.equal(z_only[..., 0], full[..., 1])
    assert torch.equal(swapped[..., 0], full[..., 1])
    assert torch.equal(swapped[..., 1], full[..., 0])
    assert not torch.equal(full[..., 0], full[..., 1]), "components are identical"


def test_snapshots_come_back_as_a_pair():
    """``record_field_every`` returns ``(seis, snaps)`` -- the same shape as the
    FD/JAX ``(record, wavefield)`` -- rather than extra dict keys."""
    every = 40
    seis, snaps = _elastic().forward(_wav(), record_field_every=every)
    n_snap = len(range(0, NT, every))
    assert seis.shape == (NT, NREC, 2)
    assert snaps.shape == (n_snap, 2, _mesh().n_global)
    assert torch.isfinite(snaps).all() and snaps.abs().max() > 0
    # the seismogram must be unaffected by asking for snapshots
    assert torch.equal(seis, _elastic().forward(_wav()))


def test_snapshot_times_are_derivable_not_returned():
    """They are exactly ``range(0, nt, every)``; returning them was the only
    reason ``snapshot_times`` existed as a ninth output key."""
    every = 30
    _, snaps = _elastic().forward(_wav(), record_field_every=every)
    assert snaps.shape[0] == len(range(0, NT, every))


def test_boundary_saving_keeps_the_same_contract():
    out = _acoustic().forward(_wav(), memory="boundary")
    assert out.shape == (NT, NREC, 1)
    ref = _acoustic().forward(_wav())
    rel = float((out - ref).abs().max() / ref.abs().max())
    assert rel < 1e-5, f"BS record disagrees with the plain one: rel={rel:.3e}"


def test_boundary_saving_rejects_snapshots():
    """It cannot serve them -- the forward state is discarded -- so it must say
    so rather than return a bare tensor where the caller unpacks a pair."""
    with pytest.raises(ValueError, match="record_field_every"):
        _acoustic().forward(_wav(), memory="boundary", record_field_every=10)


# ------------------------- PropSEM stacks the shots ------------------------ #
@pytest.mark.parametrize("nshots", [1, 3])
def test_propsem_adds_only_a_shot_axis(nshots):
    nz = nx = NE * NO + 1
    p = PropSEM(ElasticSEM2DDeformed, shape=(nz, nx), dh=DH, dt=DT, nt=NT,
                n_order=NO, free_surface=("top",),
                abcn=3 * NO, f_dominant=F0, dev="cpu", impl="eager")
    src = np.stack([np.linspace(8, nx - 8, nshots).astype(int),
                    np.full(nshots, 3)], -1)
    rix = np.linspace(6, nx - 6, NREC).astype(int)
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    models = [torch.full((nz, nx), v) for v in (VP, VS, RHO)]
    with torch.no_grad():
        out = p(_wav(), src, rec, models=models)
    assert out.shape == (nshots, NT, NREC, 2)
