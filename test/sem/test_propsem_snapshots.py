"""``PropSEM(..., return_wavefield=True)`` -- PropTorch's snapshot contract.

``(seis, snapshots)`` with the same ``snapshot_times`` / ``snapshot_interval``
rule as PropTorch (default: every step), the snapshots detached on the host as
``(nsnap, nfield, nshots, 1, *grid)`` on the grid of the model and its
absorbing layer: the fields the receivers record from, each grid point taking
the SEM field through its element's basis. Eager only.

The strongest check needs no reference solver: a receiver records the field
through the very same basis at its point, so its trace at step n must be the
snapshot at step n at its grid point -- anywhere, on a node or not, flat or
under a topography. That pins the time alignment, the field identity and the
grid frame at once.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Ckpt
from sweep.sem import (PropSEM, SEMPropagator, ElasticSEM2DDeformed,
                       ElasticVSSEM2DDeformed)
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

NZ, NX, NO, DH, F0, PAD = 33, 41, 4, 10.0, 12.0, 3
DT, NT = 6e-4, 200
P = PAD * NO                    # the absorbing layer, in cells


def _prop(equation=AcousticPotentialSEM2D, **kw):
    kw.setdefault("free_surface", ("top",))
    kw.setdefault("impl", "eager")
    return PropSEM(equation, shape=(NZ, NX), dh=DH, dt=DT, nt=NT, n_order=NO,
                   abcn=P, f_dominant=F0, dev="cpu", dtype=torch.float64, **kw)


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 0.08)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def _models(equation):
    if equation is AcousticPotentialSEM2D:
        return [torch.full((NZ, NX), 2000.0, dtype=torch.float64)]
    return [torch.full((NZ, NX), v, dtype=torch.float64)
            for v in (2400.0, 1300.0, 2000.0)]


SRC = np.array([[NX // 2, 14]])
# an element corner, a point inside an element and one on an element edge
REC = np.array([[[8, 6], [21, 7], [32, 10]]])
# a hill 2 to 5 rows down; the receivers sit below it
TOPO = 2.0 + 3.0 * np.exp(-((np.arange(NX) - NX / 2) / 8.0) ** 2)


@pytest.mark.parametrize("equation,topo", [(AcousticPotentialSEM2D, None),
                                           (ElasticVSSEM2DDeformed, None),
                                           (AcousticPotentialSEM2D, TOPO)])
def test_receiver_trace_is_the_snapshot_at_its_grid_point(equation, topo):
    p = _prop(equation, topography=topo)
    times = [0, 37, 120, NT - 1]
    with torch.no_grad():
        seis, snaps = p(_wav(), SRC, REC, models=_models(equation),
                        return_wavefield=True, snapshot_times=times)
    assert snaps.shape == (len(times), len(p.snapshot_fields), 1, 1,
                           NZ + P, NX + 2 * P)       # no pad on the free top
    assert p.snapshot_times_last == tuple(times)
    iz, ix = torch.as_tensor(REC[0, :, 1]), torch.as_tensor(REC[0, :, 0] + P)
    for f, name in enumerate(p.snapshot_fields):
        col = p.receiver_type.index(name)
        for i, n in enumerate(times):
            at_points = snaps[i, f, 0, 0, iz, ix]
            # the receiver's element is located by search (Newton on Quad9)
            # and the snapshot's by construction: the same basis to round-off
            tol = 1e-10 * float(snaps[i, f].abs().max()) + 1e-300
            err = float((at_points - seis[0, n, :, col]).abs().max())
            assert err <= tol, (name, n, err, tol)
    assert float(snaps.abs().max()) > 0
    # the check is not vacuous: by the last step the receivers have signal
    assert float(seis[0, -1].abs().max()) > 1e-6 * float(seis.abs().max())


def test_the_air_above_a_topography_is_zero():
    p = _prop(topography=TOPO)
    with torch.no_grad():
        _, snaps = p(_wav(), SRC, REC, models=_models(AcousticPotentialSEM2D),
                     return_wavefield=True, snapshot_times=[NT - 1])
    rows = np.arange(NZ + P)[:, None]
    surf = np.interp(np.arange(NX + 2 * P) - P, np.arange(NX), TOPO)[None, :]
    air = torch.as_tensor(rows < surf - 0.5)          # clear of the curved edge
    assert air.any()
    assert (snaps[0, 0, 0, 0][air] == 0).all()
    assert float(snaps[0, 0, 0, 0][~air].abs().max()) > 0


def test_asking_for_snapshots_does_not_change_the_record():
    p = _prop()
    with torch.no_grad():
        plain = p(_wav(), SRC, REC, models=_models(AcousticPotentialSEM2D))
        seis, _ = p(_wav(), SRC, REC, models=_models(AcousticPotentialSEM2D),
                    return_wavefield=True, snapshot_interval=50)
    assert torch.equal(plain, seis)


def test_default_is_every_step_and_interval_is_its_subsample():
    p = _prop()
    m = _models(AcousticPotentialSEM2D)
    with torch.no_grad():
        _, every = p(_wav(), SRC, REC, models=m, return_wavefield=True)
        _, some = p(_wav(), SRC, REC, models=m, return_wavefield=True,
                    snapshot_interval=40)
    assert every.shape[0] == NT
    assert p.snapshot_times_last == tuple(range(0, NT, 40))
    assert torch.equal(every[::40], some)


def test_snapshots_are_detached_host_copies_and_the_record_differentiates():
    p = _prop()
    vp = _models(AcousticPotentialSEM2D)[0].requires_grad_(True)
    seis, snaps = p(_wav(), SRC, REC, models=[vp], return_wavefield=True,
                    snapshot_times=[100])
    assert not snaps.requires_grad and snaps.device.type == "cpu"
    (seis ** 2).sum().backward()
    assert vp.grad is not None and float(vp.grad.abs().max()) > 0


def test_fields_follow_the_receivers_base_fields():
    """A derived receiver (vz = d u_z / dt on the displacement elastic) is
    recorded from u_z; the snapshot is that base field, and says so."""
    p = _prop(ElasticSEM2DDeformed, receiver_type=["vz"])
    assert p.snapshot_fields == ["u_z"]
    with torch.no_grad():
        _, snaps = p(_wav(), SRC, REC, models=_models(ElasticSEM2DDeformed),
                     return_wavefield=True, snapshot_times=[60])
    assert snaps.shape[1] == 1


def test_shots_keep_their_own_snapshots():
    """Two shots run one by one; snapshot b belongs to shot b."""
    p = _prop()
    m = _models(AcousticPotentialSEM2D)
    src = np.array([[12, 14], [28, 14]])
    with torch.no_grad():
        seis, snaps = p(_wav(), src, REC, models=m, return_wavefield=True,
                        snapshot_times=[90])
        for b in range(2):
            one, alone = p(_wav(), src[b:b + 1], REC, models=m,
                           return_wavefield=True, snapshot_times=[90])
            assert torch.equal(snaps[:, :, b], alone[:, :, 0])
            assert torch.equal(seis[b], one[0])
    assert not torch.equal(snaps[:, :, 0], snaps[:, :, 1])


def test_the_grid_is_proptorchs():
    """The model and its absorbing layer, (z, x), the model's first cell at
    the layer's width -- what a PropTorch snapshot with the same abcn and
    free surface is."""
    p = _prop(free_surface=False)
    with torch.no_grad():
        _, snaps = p(_wav(), SRC, REC, models=_models(AcousticPotentialSEM2D),
                     return_wavefield=True, snapshot_times=[5])
    assert snaps.shape[-2:] == (NZ + 2 * P, NX + 2 * P)


# ------------------------------ refusals ------------------------------------ #
def _call(p, **kw):
    return p(_wav(), SRC, REC, models=_models(AcousticPotentialSEM2D), **kw)


@pytest.mark.parametrize("kw,exc,match", [
    (dict(snapshot_times=[3]), ValueError, "return_wavefield=True"),
    (dict(snapshot_interval=3), ValueError, "return_wavefield=True"),
    (dict(return_wavefield=True, snapshot_times=[1], snapshot_interval=2),
     ValueError, "either"),
    (dict(return_wavefield=True, snapshot_times=[NT]), ValueError, "outside"),
    (dict(return_wavefield=True, snapshot_interval=0), ValueError, ">= 1"),
])
def test_refusals(kw, exc, match):
    with pytest.raises(exc, match=match):
        _call(_prop(), **kw)


def test_boundary_saving_is_refused_with_snapshots():
    with pytest.raises(ValueError, match="BoundarySaving"):
        _call(_prop(memory=BoundarySaving()), return_wavefield=True)


def test_ckpt_is_refused_with_snapshots():
    p = _prop(memory=Ckpt(chunks=20))
    src = np.array([[12, 14], [28, 14]])
    with pytest.raises(ValueError, match="Ckpt"):
        p(_wav(), src, REC, models=_models(AcousticPotentialSEM2D),
          return_wavefield=True)


@pytest.mark.skipif(not (torch.cuda.is_available() and sem_c_available()),
                    reason="needs CUDA + the SEM core")
def test_impl_c_refuses_snapshots():
    p = PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                n_order=NO, free_surface=("top",), abcn=P, f_dominant=F0,
                dev="cuda", impl="c")
    with pytest.raises(NotImplementedError, match="impl='eager'"):
        p(_wav().float().cuda(), SRC, REC,
          models=[torch.full((NZ, NX), 2000.0, device="cuda")],
          return_wavefield=True)


# ---------------------- the low-level snapshot_times ------------------------ #
def _low():
    p = _prop()
    eq = p._build_eq(_models(AcousticPotentialSEM2D))
    return SEMPropagator.build(eq, dt=DT, nt=NT, source_xz=[p._coords(SRC[0])],
                               receiver_xz=[p._coords(REC[0, 0])])


def test_low_level_snapshot_times_match_record_field_every():
    sp = _low()
    with torch.no_grad():
        s1, every = sp.forward(_wav(), record_field_every=25)
        s2, named = sp.forward(_wav(), snapshot_times=[150, 25, 25, 0])
    assert torch.equal(s1, s2)
    assert torch.equal(named, every[[0, 1, 6]])


def test_low_level_snapshot_refusals():
    sp = _low()
    with pytest.raises(ValueError, match="exclusive"):
        sp.forward(_wav(), record_field_every=5, snapshot_times=[1])
    with pytest.raises(ValueError, match="outside"):
        sp.forward(_wav(), snapshot_times=[-1])
    with torch.no_grad():
        _, none = sp.forward(_wav(), snapshot_times=[])
    assert none.shape[0] == 0
