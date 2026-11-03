"""PropSEM's wavelet is the pressure source on the potential forms too.

On the pressure form (and PropTorch's ``Acoustic``) the wavelet is added to
the pressure equation, and the recorded field -- ``p`` -- is the pressure.
The potential forms (``AcousticPotentialSEM2D``, the fluid of
``AcousticElasticSEM2D``) record ``p = -d2(chi)/dt2``; fed the wavelet
itself, that ``p`` was two time derivatives off the pressure form's for the
same wavelet. PropSEM now drives chi with minus the wavelet's double time
integral, so both forms record the same pressure waveform -- exactly, up to
the scale ``kappa = rho*c^2`` at the source the potential form carries -- and
the fluid-solid ``p`` stays consistent with its ``vx``/``vz``.
"""
from __future__ import annotations

import numpy as np
import torch

from sweep.sem import PropSEM
from sweep.sem.equations import (AcousticElasticSEM2D, AcousticPotentialSEM2D,
                                 AcousticSEM2DDeformed)
from sweep.sem.prop_base import _pressure_to_potential_wavelet

NZ, NX, DH, DT, NT, F0 = 41, 81, 10.0, 8e-4, 360, 15.0
VP, RHO = 2000.0, 1000.0
SRC = np.array([[40, 16]], dtype=np.int64)
_RIX = np.arange(8, 74, 8)
REC = np.stack([_RIX, np.full_like(_RIX, 6)], -1)[None]
KW = dict(shape=(NZ, NX), dh=DH, dt=DT, n_order=4, abcn=16, f_dominant=F0,
          impl="eager", eager_options=dict(use_compile=False),
          dtype=torch.float64)


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def _vp():
    return torch.full((NZ, NX), VP, dtype=torch.float64)


def _pressure_form():
    with torch.no_grad():
        return PropSEM(AcousticSEM2DDeformed, **KW)(
            _wav(), SRC, REC, models=[_vp()])[..., 0]


def _potential(rtype):
    with torch.no_grad():
        return PropSEM(AcousticPotentialSEM2D, density=RHO, receiver_type=rtype,
                       **KW)(_wav(), SRC, REC, models=[_vp()])


def test_conversion_is_the_wavelet_under_the_matched_stencil():
    """-(s[n+1] - 2 s[n] + s[n-1]) / dt^2 is the wavelet sample n, for every
    row of a (nshots, nt) / (nsrc, nt) wavelet."""
    w = torch.randn(3, 50, dtype=torch.float64)
    s = _pressure_to_potential_wavelet(w, DT)
    pad = torch.nn.functional.pad(s, (1, 1))
    d2 = -(pad[:, 2:] - 2 * s + pad[:, :-2]) / DT ** 2
    torch.testing.assert_close(d2[:, :-1], w[:, :-1], rtol=1e-9, atol=1e-9)
    assert torch.equal(_pressure_to_potential_wavelet(w[1], DT), s[1])


def test_potential_p_is_the_pressure_form_record():
    u = _pressure_form()
    p = _potential(["p"])[..., 0]
    scale = float(u.abs().max())
    assert scale > 0
    kappa = RHO * VP ** 2
    # every sample but the last (a one-sided difference there) to round-off
    err = float((p[:, :-1] / kappa - u[:, :-1]).abs().max()) / scale
    assert err < 1e-10, err


def test_chi_is_driven_by_the_converted_wavelet():
    """The chi record is what the converted wavelet makes, not the wavelet."""
    chi = _potential(["chi"])[..., 0]
    u = _pressure_form()
    kappa = RHO * VP ** 2
    # chi is the pressure's negated double integral: far from u's waveform
    corr = float((chi * u).sum() / (chi.norm() * u.norm()))
    assert abs(corr) < 0.9, corr
    lap = (chi[:, 2:] - 2 * chi[:, 1:-1] + chi[:, :-2]) / DT ** 2
    torch.testing.assert_close(-lap / kappa, u[:, 1:-1], rtol=0,
                               atol=1e-10 * float(u.abs().max()))


def test_fluid_solid_p_matches_the_potential_form_in_water():
    """All water (vs = 0, rho = RHO): the coupled class's fluid is the
    potential form, wavelet convention included."""
    with torch.no_grad():
        pf = PropSEM(AcousticElasticSEM2D, receiver_type=["p"], **KW)(
            _wav(), SRC, REC,
            models=[_vp(), torch.zeros(NZ, NX, dtype=torch.float64),
                    torch.full((NZ, NX), RHO, dtype=torch.float64)])[..., 0]
    p = _potential(["p"])[..., 0]
    err = float((pf - p).abs().max()) / float(p.abs().max())
    assert err < 1e-10, err


def test_snapshot_run_records_the_same():
    """With snapshots the run really is driven by the converted wavelet (its
    fields must be chi / u); without, the wavelet's own run is converted on
    the way out. Same records -- the snapshot route's p and v take a
    one-sided last sample, so that one is left out."""
    rtype = ["chi", "p"]
    p = PropSEM(AcousticPotentialSEM2D, density=RHO, receiver_type=rtype, **KW)
    with torch.no_grad():
        plain = p(_wav(), SRC, REC, models=[_vp()])
        snap, fields = p(_wav(), SRC, REC, models=[_vp()],
                         return_wavefield=True, snapshot_times=[NT // 2])
    scale = plain.abs().amax(dim=(0, 1, 2))
    err = ((snap - plain)[:, :-1].abs().amax(dim=(0, 1, 2)) / scale)
    assert float(err.max()) < 1e-9, err
    assert fields.shape[1] == 1          # the chi field


def test_fluid_solid_velocity_is_the_displacement_rate():
    """u and v of the fluid-solid class come out of the same conversion:
    v is still the central difference of u."""
    prop = PropSEM(AcousticElasticSEM2D, receiver_type=["u_z", "vz", "p"],
                   free_surface=("top",), **KW)
    m = prop.mesh
    fl = (m.z_e.flatten(1).mean(1) < 8 * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    models = [torch.where(fl, VP, 2600.0).double(),
              torch.where(fl, 0.0, 1300.0).double(),
              torch.where(fl, RHO, 2100.0).double()]
    rec = np.stack([_RIX, np.full_like(_RIX, 20)], -1)[None]
    with torch.no_grad():
        out = prop(_wav(), np.array([[40, 4]]), rec, models=models)
    u, v = out[..., 0], out[..., 1]
    assert float(v.abs().max()) > 0
    ref = PropSEM._trace_ddt(u, DT)
    err = float((v - ref)[:, :-1].abs().max()) / float(v.abs().max())
    assert err < 1e-9, err


def test_propsem_records_the_pressure_by_default():
    """As PropTorch: p (and vx, vz on the fluid-solid class) unless asked;
    the low-level propagators keep recording chi."""
    from sweep.sem.fields import resolve_field_types
    p = PropSEM(AcousticPotentialSEM2D, **KW)
    assert p.receiver_type == ["p"]
    fs = PropSEM(AcousticElasticSEM2D, free_surface=("top",), **KW)
    assert fs.receiver_type == ["p", "vx", "vz"]
    assert PropSEM(AcousticSEM2DDeformed, **KW).receiver_type == ["u"]
    assert resolve_field_types(AcousticPotentialSEM2D, None, "receiver") == ["chi"]
    with torch.no_grad():
        dflt = p(_wav(), SRC, REC, models=[_vp()])
    torch.testing.assert_close(dflt, _potential(["p"]), rtol=0, atol=0)
