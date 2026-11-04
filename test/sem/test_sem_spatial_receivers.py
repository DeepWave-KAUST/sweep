"""Receivers that record a spatial derivative (sweep.sem.receivers): strain, rotation, the
pressure in a solid and a potential fluid's particle motion, against finite differences of plain
receivers 0.1 m apart, and the normal velocity across a sea floor."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import (AcousticElasticSEM2D, AcousticPotentialSEM2D, ElasticSEM2DDeformed,
                       ElasticSEM3D, ElasticVSSEM2DDeformed, ElasticVTISEM2DDeformed, PropSEM)

DH, DT = 5.0, 5e-4
H = 0.02                                   # grid cells: the finite-difference step, 0.1 m
ROCK = (3000.0, 1500.0, 2400.0)
KAPPA = 2400.0 * (3000.0 ** 2 - 4.0 / 3.0 * 1500.0 ** 2)


def _ricker(nt, f0=15.0, dt=DT):
    a = (np.pi * f0 * (np.arange(nt) * dt - 1.2 / f0)) ** 2
    return torch.from_numpy((1.0 - 2.0 * a) * np.exp(-a))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _prop(eq, shape, names, **kw):
    return PropSEM(eq, shape=shape, dh=DH, dt=DT, abcn=0, receiver_type=names, impl="eager",
                   dtype=torch.float64, eager_options={"use_compile": False}, **kw)


def _star(p):
    """A receiver and its four neighbours 0.1 m away along x and z (grid indices)."""
    x, z = p
    return np.array([[[x, z], [x + H, z], [x - H, z], [x, z + H], [x, z - H]]])


@pytest.mark.parametrize("eq, fx, fz, rate", [
    (ElasticSEM2DDeformed, "u_x", "u_z", ""), (ElasticVSSEM2DDeformed, "v_x", "v_z", "_t")])
def test_strain_rotation_and_pressure_are_the_field_derivatives(eq, fx, fz, rate):
    shape = (61, 81)
    names = [fx, fz] + [n + rate for n in ("exx", "ezz", "exz", "rot_y")] + ["p"]
    out = _prop(eq, shape, names, source_type=["vz"])(
        _ricker(400), np.array([[40, 20]]), _star((30.3, 25.7)),
        models=[torch.full(shape, v, dtype=torch.float64) for v in ROCK])[0].numpy()
    d = lambda c, i, j: (out[:, i, c] - out[:, j, c]) / (2 * H * DH)
    dudx, dwdz, dudz, dwdx = d(0, 1, 2), d(1, 3, 4), d(0, 3, 4), d(1, 1, 2)
    for k, ref in enumerate((dudx, dwdz, 0.5 * (dudz + dwdx), 0.5 * (dudz - dwdx))):
        assert _rel(out[:, 0, 2 + k], ref) < 1e-4, names[2 + k]
    div = dudx + dwdz if not rate else np.cumsum(dudx + dwdz) * DT     # pressure from u
    assert _rel(out[:, 0, 6], -KAPPA * div) < 1e-4


def test_velocity_stress_strain_is_its_rate_integrated():
    shape = (61, 81)
    out = _prop(ElasticVSSEM2DDeformed, shape, ["exx", "exx_t"], source_type=["vz"])(
        _ricker(300), np.array([[40, 20]]), np.array([[[30.3, 25.7]]]),
        models=[torch.full(shape, v, dtype=torch.float64) for v in ROCK])[0].numpy()
    assert np.array_equal(out[:, 0, 0], np.cumsum(out[:, 0, 1]) * DT)


def test_a_potential_fluids_particle_motion_is_grad_chi_over_rho():
    shape = (61, 81)
    out = _prop(AcousticPotentialSEM2D, shape, ["chi", "ux", "vx", "vz"], density=1020.0)(
        _ricker(400), np.array([[40, 20]]), _star((30.3, 25.7)),
        models=[torch.full(shape, 1500.0, dtype=torch.float64)])[0].numpy()
    gx = (out[:, 1, 0] - out[:, 2, 0]) / (2 * H * DH) / 1020.0
    gz = (out[:, 3, 0] - out[:, 4, 0]) / (2 * H * DH) / 1020.0
    ddt = lambda x: (x[2:] - x[:-2]) / (2 * DT)
    assert _rel(out[:, 0, 1], gx) < 1e-4
    assert _rel(out[1:-1, 0, 2], ddt(gx)) < 1e-4
    assert _rel(out[1:-1, 0, 3], ddt(gz)) < 1e-4


def test_the_normal_velocity_is_continuous_across_a_sea_floor():
    """Each receiver takes the medium it sits in: vz from grad(chi) in the water just above the
    floor equals the rock's just below (0.02 m each side; 0.2 m gave 10 times the difference,
    the field's own gradient), the tangential one does not, and the strain is zero in the water."""
    shape = (61, 81)
    xs = np.array([10.0, 30.0, 50.0, 70.0])
    rec = np.array([np.concatenate([np.stack([xs, np.full_like(xs, 20.0 - 0.004)], 1),
                                    np.stack([xs, np.full_like(xs, 20.0 + 0.004)], 1)])])
    prop = _prop(AcousticElasticSEM2D, shape, ["vx", "vz", "p", "exx"], source_type=["chi"],
                 free_surface={"top": True, "left": True, "right": True, "bottom": False})
    m = prop.mesh
    wet = m.z_e.mean((1, 2)) < 100.0
    E, n = m.n_elements, m.ngll
    models = [torch.where(wet, torch.tensor(w, dtype=torch.float64), torch.tensor(r, dtype=torch.float64))
              .view(E, 1, 1).expand(E, n, n).contiguous()
              for w, r in zip((1500.0, 0.0, 1020.0), ROCK)]
    out = prop(_ricker(500, 12.0) / (1020.0 * 1500.0 ** 2), np.array([[40, 10]]), rec,
               models=models)[0].numpy()
    k = len(xs)
    for j in range(k):
        assert _rel(out[:, j, 1], out[:, k + j, 1]) < 1e-2
        assert np.linalg.norm(out[:, j, 0]) > 1.5 * np.linalg.norm(out[:, k + j, 0])
    assert np.abs(out[:, :k, 3]).max() == 0.0 and np.abs(out[:, k:, 3]).max() > 0.0


def test_3d_strain_and_rotation():
    shape = (25, 25, 25)
    names = ["u_x", "u_y", "u_z", "exy", "rot_y", "p"]   # a vertical force: no rot_z
    x, y, z = 12.3, 11.6, 10.7
    rec = np.array([[[x, y, z], [x + H, y, z], [x - H, y, z], [x, y + H, z], [x, y - H, z],
                     [x, y, z + H], [x, y, z - H]]])
    out = PropSEM(ElasticSEM3D, shape=shape, dh=DH, dt=DT, abcn=0, receiver_type=names,
                  source_type=["vz"], impl="eager", dtype=torch.float64,
                  eager_options={"use_compile": False})(
        _ricker(150, 25.0), np.array([[14, 13, 16]]), rec,
        models=[torch.full(shape, v, dtype=torch.float64) for v in ROCK])[0].numpy()
    d = lambda c, i, j: (out[:, i, c] - out[:, j, c]) / (2 * H * DH)
    dux_dy, duy_dx, dux_dz, duz_dx = d(0, 3, 4), d(1, 1, 2), d(0, 5, 6), d(2, 1, 2)
    assert _rel(out[:, 0, 3], 0.5 * (dux_dy + duy_dx)) < 1e-4
    assert _rel(out[:, 0, 4], 0.5 * (dux_dz - duz_dx)) < 1e-4
    div = d(0, 1, 2) + d(1, 3, 4) + d(2, 5, 6)
    assert _rel(out[:, 0, 5], -KAPPA * div) < 1e-4


def test_an_anisotropic_class_has_no_pressure_receiver():
    with pytest.raises(ValueError, match="p"):
        PropSEM(ElasticVTISEM2DDeformed, shape=(21, 21), dh=DH, dt=DT, receiver_type=["p"])
    PropSEM(ElasticVTISEM2DDeformed, shape=(21, 21), dh=DH, dt=DT, receiver_type=["exx_t"])


def test_the_gradient_reaches_the_model_through_the_material_factor():
    """p = -kappa div u carries kappa at the receiver: d p / d vp includes it."""
    shape = (41, 41)
    vp = torch.full(shape, 3000.0, dtype=torch.float64, requires_grad=True)
    out = _prop(ElasticSEM2DDeformed, shape, ["p"], source_type=["vz"])(
        _ricker(150), np.array([[20, 10]]), np.array([[[22.3, 25.7]]]),
        models=[vp, torch.full(shape, 1500.0, dtype=torch.float64),
                torch.full(shape, 2400.0, dtype=torch.float64)])
    out.square().sum().backward()
    assert torch.isfinite(vp.grad).all() and vp.grad.abs().max() > 0


def test_each_shot_takes_its_own_receivers():
    """Two shots in one call (the batched path) give what each gives alone."""
    shape = (41, 61)
    models = [torch.full(shape, v, dtype=torch.float64) for v in ROCK]
    src = np.array([[20, 10], [40, 12]])
    rec = np.array([[[10.3, 20.7], [30.6, 25.1]], [[15.2, 22.4], [45.9, 18.3]]])
    names = ["vz", "exz_t", "rot_y", "p"]
    prop = _prop(ElasticSEM2DDeformed, shape, names, source_type=["vz"])
    both = prop(_ricker(200), src, rec, models=models).numpy()
    for b in range(2):
        one = prop(_ricker(200), src[b:b + 1], rec[b:b + 1], models=models).numpy()
        assert np.allclose(both[b], one[0], rtol=1e-12, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("eq", [ElasticSEM2DDeformed, AcousticElasticSEM2D])
def test_impl_c_records_the_same_derivatives(eq):
    from sweep.sem.prop_c import sem_c_available
    if not sem_c_available():
        pytest.skip("needs the compiled sweep._C SEM binding")
    shape = (41, 81)
    names = ["vx", "vz", "p", "exx", "rot_y_t"]
    kw = dict(shape=shape, dh=DH, dt=DT, abcn=8, receiver_type=names, dtype=torch.float32,
              dev="cuda", f_dominant=15.0)
    rec = np.array([[[12.3, 6.7], [30.1, 9.4], [52.6, 30.3], [70.2, 34.9]]])
    out = {}
    for impl in ("eager", "c"):
        if eq is AcousticElasticSEM2D:
            prop = PropSEM(eq, impl=impl, free_surface={"top": True}, source_type=["chi"], **kw)
            m = prop.mesh
            wet = (m.z_e.mean((1, 2)) < 60.0).to(m.z_e.device)
            E, n = m.n_elements, m.ngll
            models = [torch.where(wet, torch.tensor(w, device="cuda"), torch.tensor(r, device="cuda"))
                      .view(E, 1, 1).expand(E, n, n).contiguous().float()
                      for w, r in zip((1500.0, 0.0, 1020.0), ROCK)]
            wav = _ricker(300).float() / (1020.0 * 1500.0 ** 2)
        else:
            prop = PropSEM(eq, impl=impl, free_surface=True, source_type=["vz"], **kw)
            models = [torch.full(shape, v, device="cuda") for v in ROCK]
            wav = _ricker(300).float()
        out[impl] = prop(wav.cuda(), np.array([[40, 8]]), rec, models=models)[0].double().cpu().numpy()
    for k, nm in enumerate(names):
        assert _rel(out["c"][..., k], out["eager"][..., k]) < 1e-3, nm      # fp32


def test_a_receiver_on_an_element_corner_takes_every_element_there():
    """Grid receivers land on element edges every n_order cells, where the gradient jumps: one on
    a corner takes the four elements by their shares and matches the difference across it."""
    shape = (61, 81)
    out = _prop(ElasticSEM2DDeformed, shape, ["u_x", "u_z", "exx", "ezz", "exz"],
                source_type=["vz"])(
        _ricker(500, 20.0), np.array([[40, 20]]), _star((32.0, 24.0)),
        models=[torch.full(shape, v, dtype=torch.float64) for v in ROCK])[0].numpy()
    d = lambda c, i, j: (out[:, i, c] - out[:, j, c]) / (2 * H * DH)
    assert _rel(out[:, 0, 2], d(0, 1, 2)) < 1e-3
    assert _rel(out[:, 0, 3], d(1, 3, 4)) < 1e-3
    assert _rel(out[:, 0, 4], 0.5 * (d(0, 3, 4) + d(1, 1, 2))) < 1e-3
