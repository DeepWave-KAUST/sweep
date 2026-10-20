"""FD-aligned derived receiver components: p, vx, vy, vz.

Field data arrives as pressure / particle velocity, so PropSEM converts
internally -- ``receiver_type=['p', 'vx', 'vz']`` works like the FD side --
instead of every caller differentiating traces by hand. The contract:

* ``p`` on the potential-form acoustics is ``-d2(chi)/dt2`` and ``vx``/``vz``
  on the displacement elastics is ``d(u)/dt``, both with the MATCHED leapfrog
  stencils (central differences, quiescent-start left pad);
* on the scalar-wave acoustics ``p`` is a pure name alias (the field IS the
  pressure-like wavefield -- FD's ``Acoustic`` aliases the same equation's
  field to 'p'); on the velocity-stress family ``vx`` aliases ``v_x``;
* derived names are receivers ONLY -- as a source_type ``p`` is refused (on
  the elastics ``vx``/``vz`` do name a source, FD's force; that is the source
  vocabulary, see test_sem_source_type_fd.py, not a derived component);
* the transform is a post-op on the recorded traces, so both backends, the
  shot batch and boundary saving inherit it unchanged, and the default
  receiver_type path returns the exact same tensor object as before.

Bit-equality trap this file avoids: neither CUDA backend is run-to-run
deterministic (atomics), so every "derived equals transformed base" assertion
requests the base and the derived component IN THE SAME CALL and compares
within that one record.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import sweep.sem as S
import sweep.sem.equations as E
from sweep.propagator import BoundarySaving
from sweep.sem import prop_c
from sweep.sem.equations import (AcousticElasticSEM2D, AcousticPotentialSEM2D,
                                 ElasticVSSEM3D)
from sweep.sem.fields import resolve_field_types
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NT, DT = 4, 25.0, 6.0, 400, 0.0016
NZ, NX, NW = 49, 81, 8
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                        device="cuda")


SRC = np.array([[NX // 2, 3]], dtype=np.int64)
RIX = np.arange(6, NX - 6, 3)
REC_WATER = np.stack([RIX, np.full_like(RIX, 3)], -1)[None]
REC_SOLID = np.stack([RIX, np.full_like(RIX, NW + 2)], -1)[None]


def _fs_prop(rtype, impl="c"):
    return S.PropSEM(AcousticElasticSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=4 * NO, f_dominant=F0,
                     dev="cuda", impl=impl, receiver_type=rtype)


def _fs_models(p):
    m = p.mesh
    fl = (m.z_e.flatten(1).mean(1) < NW * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    vp = torch.where(fl, 1500.0, 2200.0).float().cuda()
    vs = torch.where(fl, 0.0, 1200.0).float().cuda()
    rho = torch.where(fl, 1000.0, 2100.0).float().cuda()
    return [vp, vs, rho]


def _rel(a, b, scale=None):
    a, b = a.double(), b.double()
    d = float(a.abs().max()) if scale is None else float(scale)
    return float((a - b).abs().max() / d) if d else 0.0


def test_every_declaration_resolves():
    """Each equation's derived names resolve as receiver_type; a bad mapping
    (base not receiver-capable) would raise here, not at first use."""
    classes = [c for _, c in inspect.getmembers(E, inspect.isclass)
               if hasattr(c, "field_specs")]
    assert len(classes) >= 14
    for c in classes:
        names = sorted(getattr(c, "derived_receiver_types", {}))
        if names:
            assert resolve_field_types(c, names, "receiver") == names, c.__name__


def test_p_is_the_matched_second_difference():
    p = _fs_prop(["chi", "p"])
    with torch.no_grad():
        out = p(_wav(), SRC, REC_WATER, models=_fs_models(p))
    chi, pr = out[..., 0], out[..., 1]
    assert float(chi.abs().max()) > 0, "hydrophones recorded nothing"
    assert torch.equal(pr, S.PropSEM._trace_neg_d2dt2(chi, DT))
    assert float((pr - chi).abs().max()) > 0, "p was not transformed"


def test_v_is_the_matched_first_difference():
    p = _fs_prop(["u_x", "vx", "u_z", "vz"])
    with torch.no_grad():
        out = p(_wav(), SRC, REC_SOLID, models=_fs_models(p))
    for ub, vb in ((0, 1), (2, 3)):
        u, v = out[..., ub], out[..., vb]
        assert float(u.abs().max()) > 0
        assert torch.equal(v, S.PropSEM._trace_ddt(u, DT))
        assert float((v - u).abs().max()) > 0


def test_vs_family_alias_is_identity():
    """On the velocity-stress family 'vx' is a NAME for v_x, not a
    transform -- the columns must be bit-identical within one call."""
    p = S.PropSEM(ElasticVSSEM3D, shape=(17, 17, 17), dh=DH, dt=DT, nt=120,
                  n_order=NO, free_surface=("top",),
                  abcn=3 * NO, f_dominant=8.0, dev="cuda", impl="c",
                  receiver_type=["v_x", "vx", "v_z", "vz"])
    src = np.array([[8, 8, 3]], dtype=np.int64)
    rix = np.arange(4, 13)
    rec = np.stack([rix, np.full_like(rix, 8), np.full_like(rix, 2)], -1)[None]
    vp = torch.full((17, 17, 17), 2400.0, device="cuda")
    vs = torch.full((17, 17, 17), 1250.0, device="cuda")
    rho = torch.full((17, 17, 17), 2000.0, device="cuda")
    t = np.arange(120) * DT
    a = (np.pi * 8.0 * (t - 1.5 / 8.0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                     device="cuda")
    with torch.no_grad():
        out = p(w, src, rec, models=[vp, vs, rho])
    assert torch.equal(out[..., 0], out[..., 1])
    assert torch.equal(out[..., 2], out[..., 3])
    assert float(out[..., 0].abs().max()) > 0


def test_c_matches_eager_on_derived_components():
    ms = _fs_models(_fs_prop(["p"]))
    args = (_wav(), SRC, REC_SOLID)
    with torch.no_grad():
        c1 = _fs_prop(["p", "vx", "vz"])(*args, models=ms)
        c2 = _fs_prop(["p", "vx", "vz"])(*args, models=ms)
        e = _fs_prop(["p", "vx", "vz"], impl="eager")(*args, models=ms)
    floor, got = _rel(c1, c2), _rel(c1, e)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"derived comps, c vs eager {got:.2e}, floor {floor:.2e}")


def test_shot_batch_composes_with_derived(monkeypatch):
    """receiver_type=['p'] on the batched acoustic C-PML path: the batch must
    still engage (the transform is downstream of it) and shot k must equal an
    independent single-shot run."""
    calls = {"n": 0}
    orig = prop_c.sem_c_forward_shot_batch

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(prop_c, "sem_c_forward_shot_batch", spy)
    vp = torch.full((NZ, NX), 2400.0, device="cuda")
    srcs = np.stack([np.linspace(8, NX - 9, 4).astype(np.int64),
                     np.full(4, 3, dtype=np.int64)], -1)

    def prop(memory=None):
        return S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                         nt=NT, n_order=NO, free_surface=("top",),
                         abcn=3 * NO, f_dominant=F0, memory=memory,
                         dev="cuda", impl="c", receiver_type=["p"])

    with torch.no_grad():
        got = prop(BoundarySaving())(_wav(), srcs, REC_WATER, models=[vp])
        again = prop(BoundarySaving())(_wav(), srcs, REC_WATER, models=[vp])
    assert calls["n"] == 2, "batched path declined under a derived component"
    with torch.no_grad():
        ref = torch.cat([prop()(_wav(), srcs[b:b + 1], REC_WATER, models=[vp])
                         for b in range(4)], 0)
    floor = _rel(got, again)
    scale = float(got.abs().max())
    for b in range(4):
        d = _rel(got[b], ref[b], scale)
        assert d <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"shot {b}: batched-derived vs independent {d:.2e}")


def test_gradients_flow_through_the_transform():
    p = _fs_prop(["p", "vx", "vz"])
    ms = [m.requires_grad_(True) for m in _fs_models(p)]
    s = p(_wav(), SRC, REC_SOLID, models=ms)
    (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30)).backward()
    assert all(m.grad is not None and torch.isfinite(m.grad).all() for m in ms)
    assert float(ms[0].grad.abs().max()) > 0


def test_derived_names_refused_as_source_type():
    with pytest.raises(ValueError, match="source_type"):
        _fs_prop(["chi"]).__class__(
            AcousticElasticSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
            n_order=NO, free_surface=("top",),
            abcn=4 * NO, f_dominant=F0, dev="cuda", impl="c",
            source_type=["p"])


def test_unknown_receiver_error_lists_the_derived_names():
    with pytest.raises(ValueError, match="Derived receiver components:.*p"):
        _fs_prop(["totally_bogus"])


def test_default_path_returns_the_untouched_tensor():
    """No derived request -> _apply_derived_receivers must be the identity on
    the SAME object, so the historical default path stays bit-untouched."""
    p = _fs_prop(None)
    assert p._recv_identity
    x = torch.randn(2, 8, 3, 3)
    assert p._apply_derived_receivers(x) is x
