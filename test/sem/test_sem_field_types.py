"""``PropSEM(source_type=..., receiver_type=...)`` -- the FD field-role API.

FD names the fields a source drives and a receiver samples
(``source_type=["sxx","szz"]``, ``receiver_type=["vx","vz"]``); SEM used to
return a dict keyed by ``seis_*`` and take a raw force vector. These tests pin
the aligned surface, and one property that is easy to break while aligning it:
the DEFAULT source must stay a unit vertical force. Defaulting ``source_type``
to "every field that supports a source" -- the natural reading, and what the
receiver side does -- silently turns it into a diagonal force instead.

The valid names are the equation's, and SEM has elastic in BOTH formulations:
``ElasticSEM*`` is 2nd-order displacement (``u_x``/``u_z``) and ``ElasticVSSEM*``
is 1st-order velocity-stress (``v_x``/``v_z``). Both families are covered here,
because a test written against only one would let a wrong assumption about
"the" SEM elastic field names through.

Determinism note: the eager path scatters sources with ``index_add_``, which is
atomic and therefore NOT bit-reproducible on CUDA (~2e-6 relative between two
identical runs). The bit-exact assertions run on CPU, where the same code is
deterministic; asserting them on CUDA would be testing the noise floor.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import (PropSEM, ElasticSEM2DDeformed, ElasticSEM3D,
                       ElasticVSSEM2DDeformed)
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D

NZ = NX = 41
NO, DH, F0 = 4, 12.5, 8.0
VP, VS, RHO = 2400.0, 1250.0, 2000.0
DT = 0.30 * (0.172 * NO * DH) / (VP * 1.2)      # GLL-limited, not the FD CFL
NT = 120


def _prop(equation=ElasticSEM2DDeformed, dev="cpu", **kw):
    # Two absorbing elements a side (four made the layer 152 of the mesh's
    # 252 elements): in NT steps the P wave only just reaches the model's
    # edge, so no reflection from the layer is in any record compared here.
    return PropSEM(equation, shape=(NZ, NX), dh=DH, dt=DT, nt=NT, n_order=NO,
                   free_surface=("top",), abcn=2 * NO,
                   f_dominant=F0, dev=dev, dtype=torch.float32, impl="eager",
                   **kw)


def _shot(dev="cpu"):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)
    src = np.array([[NX // 2, 3]], dtype=np.int64)
    rix = np.arange(6, NX - 6)
    rec = np.stack([rix, np.zeros_like(rix)], -1)[None]
    models = [torch.full((NZ, NX), v, device=dev) for v in (VP, VS, RHO)]
    return w, src, rec, models


# ------------------------------- defaults ---------------------------------- #
def test_defaults_match_the_equation():
    p = _prop()
    assert p.receiver_type == ["u_x", "u_z"]     # every receiver-capable field
    assert p.source_type == ["u_z"]              # the LAST one = vertical force
    a = PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                n_order=NO, free_surface=("top",),
                abcn=4 * NO, f_dominant=F0, dev="cpu", impl="eager")
    assert a.receiver_type == ["chi"] and a.source_type == ["chi"]
    # abcn=0: ElasticSEM3D.build takes no `cpml`, so it is a bare reflecting
    # box. It used to accept a 2-element layer and pad the mesh while
    # building NO absorbing layer -- an absorbing layer silently not applied.
    p3 = PropSEM(ElasticSEM3D, shape=(17, 17, 17), dh=20.0, dt=1e-3, nt=4,
                 n_order=4, free_surface=("top",), abcn=0,
                 f_dominant=8.0, dev="cpu", impl="eager")
    assert p3.receiver_type == ["u_x", "u_y", "u_z"]
    assert p3.source_type == ["u_z"]
    assert p3._source_vector() == (0.0, 0.0, 1.0)


def test_default_source_is_still_a_unit_vertical_force():
    """The regression this whole file exists for.

    Before ``source_type`` existed the default was ``(0,)*(nd-1) + (1,)``. If
    the default resolved to "all source fields" it would become ``(1, 1)`` --
    a different experiment, with no error anywhere.
    """
    w, src, rec, models = _shot()
    with torch.no_grad():
        d = _prop()(w, src, rec, models=models)
        v = _prop()(w, src, rec, models=models, source_vector=(0.0, 1.0))
        n = _prop(source_type=["u_z"])(w, src, rec, models=models)
    assert torch.equal(d, v), "default is no longer a unit vertical force"
    assert torch.equal(d, n)
    # ... and the wrong default would be caught: (1,1) is a different record.
    with torch.no_grad():
        diag = _prop(source_type=["u_x", "u_z"])(w, src, rec, models=models)
    rel = float((d - diag).abs().max() / d.abs().max())
    assert rel > 1e-2, (
        f"a diagonal force differs from a vertical one by only {rel:.2e}; "
        "this test cannot detect the regression it is written for")


# ------------------------------- receivers --------------------------------- #
@pytest.mark.parametrize("rt,ncomp", [(["u_z"], 1), (["u_x"], 1),
                                      (["u_x", "u_z"], 2), (["u_z", "u_x"], 2)])
def test_receiver_type_selects_components(rt, ncomp):
    w, src, rec, models = _shot()
    with torch.no_grad():
        s = _prop(receiver_type=rt)(w, src, rec, models=models)
    assert s.shape == (1, NT, rec.shape[1], ncomp)


def test_receiver_type_order_is_respected():
    """``["u_z","u_x"]`` must be the reverse of ``["u_x","u_z"]``, not the same
    thing sorted -- otherwise the order is silently ignored."""
    w, src, rec, models = _shot()
    with torch.no_grad():
        xz = _prop(receiver_type=["u_x", "u_z"])(w, src, rec, models=models)
        zx = _prop(receiver_type=["u_z", "u_x"])(w, src, rec, models=models)
    assert torch.equal(xz[..., 0], zx[..., 1])
    assert torch.equal(xz[..., 1], zx[..., 0])
    # the two components must actually differ, or the check above is vacuous
    assert not torch.equal(xz[..., 0], xz[..., 1])


def test_selected_component_matches_the_full_record():
    """Selecting one component must not change it -- it is a projection of the
    same run, not a different run."""
    w, src, rec, models = _shot()
    with torch.no_grad():
        both = _prop(receiver_type=["u_x", "u_z"])(w, src, rec, models=models)
        only = _prop(receiver_type=["u_z"])(w, src, rec, models=models)
    assert torch.equal(both[..., 1:2], only)


# -------------------------------- sources ---------------------------------- #
def test_source_type_sets_the_force_direction():
    w, src, rec, models = _shot()
    with torch.no_grad():
        fz = _prop(source_type=["u_z"])(w, src, rec, models=models)
        fx = _prop(source_type=["u_x"])(w, src, rec, models=models)
    assert not torch.equal(fz, fx)
    # a vertical force on a horizontal free surface puts more into u_z than a
    # horizontal one does -- a sanity check on which direction actually moved
    assert fz[..., 1].abs().max() > fx[..., 1].abs().max()


def test_source_vector_overrides_source_type():
    """``source_vector`` can express directions no field list can (30 deg), so
    it must win when both are given rather than being silently dropped."""
    w, src, rec, models = _shot()
    with torch.no_grad():
        a = _prop(source_type=["u_x"])(w, src, rec, models=models,
                                       source_vector=(0.0, 1.0))
        b = _prop(source_type=["u_z"])(w, src, rec, models=models)
    assert torch.equal(a, b)


# ------------------------------- validation -------------------------------- #
# NOTE: ``vz`` used to sit in this list ("FD names are not transferable"). It
# is now a DERIVED receiver on the displacement elastics (vz = d(u_z)/dt; see
# test_sem_derived_receivers.py) and, as a source, FD's vertical force -- as
# are sxx/szz/sxz its stress sources (test_sem_source_type_fd.py).
@pytest.mark.parametrize("role,bad", [("receiver_type", ["sxx"]),
                                      ("source_type", ["txx"]),
                                      ("source_type", ["u_y"]),
                                      ("receiver_type", ["v_z"])])
def test_unknown_field_names_rejected_at_construction(role, bad):
    """Stress names are never recordable (no SEM family stores a receiver-
    capable stress), and the OTHER SEM family's names stay untransferable:
    ``v_z`` is real in the velocity-stress equations but not in this
    displacement one."""
    with pytest.raises(ValueError, match="Unknown"):
        _prop(**{role: bad})


# ------------------- the two elastic formulations -------------------------- #
def test_the_two_elastic_families_name_different_fields():
    """SEM has elastic in both formulations and they do NOT share field names.

    Assuming "SEM elastic records u_x/u_z" is wrong for half the equations, so
    this pins both. ``ElasticSEM*`` integrates displacement 2nd-order in time;
    ``ElasticVSSEM*`` is a 1st-order velocity-stress system.
    """
    assert _prop(equation=ElasticSEM2DDeformed).receiver_type == ["u_x", "u_z"]
    assert _prop(equation=ElasticVSSEM2DDeformed).receiver_type == ["v_x", "v_z"]
    assert _prop(equation=ElasticVSSEM2DDeformed).source_type == ["v_z"]
    # the displacement family has no stress state at all; the velocity-stress
    # family has one, but element-local, so it is not receiver-capable either
    sig = [f for f in ElasticVSSEM2DDeformed.field_specs
           if f.name.startswith("sigma")]
    assert sig, "expected sigma_* fields in the velocity-stress equation"
    assert not any(f.supports_receiver for f in sig)


def test_velocity_stress_family_records_a_real_wavefield():
    """Not just the right names -- the VS path must actually run through
    PropSEM, since it takes a different C-PML profile type."""
    w, src, rec, models = _shot()
    with torch.no_grad():
        s = _prop(equation=ElasticVSSEM2DDeformed,
                  source_type=["v_z"], receiver_type=["v_x", "v_z"])(
                      w, src, rec, models=models)
    assert s.shape == (1, NT, rec.shape[1], 2)
    assert torch.isfinite(s).all() and s.abs().max() > 0


def test_error_lists_the_valid_options():
    # "sxx" stays unknown (stress is never a state here); "vz" moved out of
    # this test when it became a derived receiver component.
    with pytest.raises(ValueError) as e:
        _prop(receiver_type=["sxx"])
    assert "u_x" in str(e.value) and "u_z" in str(e.value)
    # the derived names are part of the valid vocabulary and must be listed
    assert "Derived receiver components" in str(e.value)
    assert "vx" in str(e.value)


@pytest.mark.parametrize("role", ["source_type", "receiver_type"])
def test_duplicates_rejected(role):
    with pytest.raises(ValueError, match="duplicate"):
        _prop(**{role: ["u_z", "u_z"]})


def test_internal_fields_are_not_selectable():
    """``u_x_prev`` is an internal history field; naming it must fail rather
    than hand back a half-step-old wavefield that looks almost right."""
    with pytest.raises(ValueError):
        _prop(receiver_type=["u_x_prev"])


# --------------------- 2-D elastic through PropSEM ------------------------- #
@pytest.mark.parametrize("equation", [ElasticSEM2DDeformed, ElasticVSSEM2DDeformed])
def test_2d_elastic_returns_an_fd_layout_tensor(equation):
    """It used to raise NotImplementedError and force callers down to
    ``SEMPropagator``, whose output is a dict keyed ``seis_x``/``seis_z``."""
    w, src, rec, models = _shot()
    with torch.no_grad():
        s = _prop(equation=equation)(w, src, rec, models=models)
    assert torch.is_tensor(s)
    assert s.shape == (1, NT, rec.shape[1], 2)          # (B, nt, nrec, ncomp)
    assert torch.isfinite(s).all()


def test_2d_elastic_gradient_flows_to_all_three_models():
    w, src, rec, models = _shot()
    models = [m.clone().requires_grad_(True) for m in models]
    s = _prop()(w, src, rec, models=models)
    (s ** 2).sum().backward()
    for name, m in zip(("vp", "vs", "rho"), models):
        assert m.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(m.grad).all(), f"non-finite {name} gradient"
        assert m.grad.abs().max() > 0, f"identically-zero {name} gradient"
