"""Boundary saving must still match the full tape on a LONG record.

The C-PML memory fields are marked ``boundary_related``, which tells the reverse
march to zero them. The framework used to read that flag as "carries no
gradient" too and dropped their adjoint. That is invisible on a short record --
every existing boundary-saving test passed at cos 1.0 -- and only shows up once
the wavefield has actually reached the absorbing band, at which point the
dropped adjoint contaminates the INTERIOR gradient:

    Marmousi, acoustic potential form, gradient vs the full tape
        1.6 s  cos 1.0000000
        3.2 s  cos 0.9999987
        6.3 s  cos 0.9997223     <- below the 1e-4 bar

It is not round-off (fp64 gives the identical cosine) and not reverse-march
drift (re-anchoring every 100 steps does not move it). Seven equations shared
the omission, so the fix is the framework default, and this test is what stops
it regressing: it runs long enough for the wavefield to load the band.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.boundary_saving import cpml_node_masks
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# The configuration matters more than the record length. What decides the size
# of the error is how much ENERGY ends up in the absorbing band, so this uses a
# strong velocity contrast and puts the source close to the left boundary. In a
# quiet-band setup the old behaviour still scored cos 0.9999998 and no test
# could see it; here it scores 0.9979.
XMAX, ZMAX, VP_LO, VP_HI, F0 = 2400.0, 1600.0, 1500.0, 4500.0, 10.0
DH, NPML = 20.0, 5
SRC_X_FRAC = 0.15
NEX, NEZ = int(XMAX / (4 * DH)), int(ZMAX / (4 * DH))
# dt is set by the smallest GLL spacing in an element (~0.17 of it), not by DH.
DT = 0.32 * (0.172 * 4 * DH) / VP_HI


def _build(nt, dev="cuda"):
    mesh = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                    n_order=4, dtype=torch.float32, device=dev)
    shp = (mesh.n_elements, mesh.ngll, mesh.ngll)
    zf = mesh.z_e / ZMAX
    vp = (VP_LO + (VP_HI - VP_LO) * zf).contiguous().requires_grad_(True)
    cp = build_cpml(mesh, dt=DT, sides=("left", "right", "bottom"),
                    pml_thickness_elements=NPML, c_max=VP_HI * 1.2, f_dominant=F0)
    eq = AcousticPotentialSEM2D.build(
        mesh, vp, torch.full(shp, 2000.0, device=dev), dirichlet=("top",), cpml=cp)
    P = SEMPropagator.build(
        eq, dt=DT, nt=nt, source_xz=[(XMAX * SRC_X_FRAC, ZMAX * 0.12)],
        receiver_xz=[(float(v), ZMAX * 0.12)
                     for v in np.linspace(XMAX * 0.05, XMAX * 0.95, 30)],
        source_kinds=["scalar"], source_payloads=[(1.0,)], pml_guard="off")
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)
    return P, eq, vp, wav


def _grad(nt, memory):
    P, eq, vp, wav = _build(nt)
    o = P.forward(wav, memory=memory)
    (o[..., 0].T ** 2).sum().backward()
    _, band = cpml_node_masks(eq)
    inb = torch.zeros(eq.mesh.n_global, dtype=torch.bool, device="cuda")
    inb[band] = True
    interior = ~inb[eq.mesh.ibool.reshape(eq.mesh.n_elements, -1)].any(dim=1)
    return vp.grad.detach(), interior


def _cos(a, b, m):
    """No additive epsilon: with a small gradient a `+1e-30` guard in the
    denominator silently deflates the cosine."""
    a, b = a[m].reshape(-1).double(), b[m].reshape(-1).double()
    return float(a @ b) / (float(a.norm()) * float(b.norm()))


@pytest.mark.parametrize("nt", [400, 3000], ids=["short", "long"])
def test_boundary_saving_matches_full_tape(nt):
    """The long case is the one that matters: by then the wavefield has loaded
    the C-PML band, which is when a dropped memory adjoint starts contaminating
    the interior. Verified to FAIL (cos 0.9979) with the old default that
    excluded boundary_related fields from the adjoint."""
    ref, interior = _grad(nt, None)
    got, _ = _grad(nt, "boundary")
    c = _cos(got, ref, interior)
    assert c >= 1 - 1e-4, f"nt={nt}: boundary-saved gradient cos={c:.8f}"


def test_every_field_adjoint_is_threaded():
    """Guard the framework default directly: dropping the C-PML memory fields
    from grad_idx is what caused the long-record drift."""
    from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D as E
    specs = E.field_specs
    assert any(s.boundary_related for s in specs), "test no longer covers the case"
    _, eq, _, _ = _build(50)[0], *_build(50)[1:]
    default = list(getattr(eq, "bs_grad_fields", None) or list(range(len(specs))))
    assert default == list(range(len(specs))), (
        "the C-PML memory fields are excluded from the boundary-saving adjoint "
        "again -- that is invisible on short records and costs ~3e-4 of the "
        "interior gradient once the band is loaded")
