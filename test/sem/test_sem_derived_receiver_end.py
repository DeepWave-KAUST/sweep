"""The last sample of a derived receiver component (``p`` = -d2chi/dt2,
``v`` = du/dt).

The record ends mid-signal, so the stencil's right neighbour u(nt) is not
known. Repeating u(nt-1) there made the last pressure sample a FIRST
difference over dt^2 -- a spike tens of times the trace (717 after 103 on a
97x193 potential-form run). It is now extrapolated (second-order one-sided
differences), so the last sample follows the trace: a record cut at nt must
end on the value the longer record has at that sample.

Both backends, the shot batch and the gradient reach the transform through
``PropSEM.forward``; the CUDA test below runs them on a cut record.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM
from sweep.sem.equations import AcousticPotentialSEM2D, ElasticSEM2DDeformed
from sweep.sem.prop_c import sem_c_available

DT = 1e-3


def _t(nt):
    return torch.arange(nt, dtype=torch.float64) * DT


def test_end_is_exact_on_low_order_polynomials():
    """The one-sided ends are exact where the central interior is: d/dt on a
    quadratic, d2/dt2 on a cubic (with the quiescent left pad, so start at 0
    with zero slope and curvature)."""
    t = _t(40)[None, :, None]
    u2 = (t / 0.04) ** 2
    u3 = (t / 0.04) ** 3
    d1 = PropSEM._trace_ddt(u2, DT)
    d2 = PropSEM._trace_neg_d2dt2(u3, DT)
    torch.testing.assert_close(d1[0, 1:, 0], (2 * t / 0.04 ** 2)[0, 1:, 0],
                               rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(d2[0, 1:, 0], (-6 * t / 0.04 ** 3)[0, 1:, 0],
                               rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("op", ["ddt", "neg_d2dt2"])
def test_end_error_is_second_order(op):
    """On a smooth trace cut mid-cycle, halving dt quarters the end error."""
    errs = []
    for k in (4, 8, 16):
        dt = 2e-3 / k
        t = torch.arange(round(0.37 / dt) + 1, dtype=torch.float64) * dt
        u = torch.sin(2 * np.pi * 7.0 * t) ** 3          # quiescent start
        w = 2 * np.pi * 7.0
        s, c = torch.sin(w * t[-1]), torch.cos(w * t[-1])
        if op == "ddt":
            got = PropSEM._trace_ddt(u[None, :, None], dt)[0, -1, 0]
            ref = 3 * w * s ** 2 * c
        else:
            got = PropSEM._trace_neg_d2dt2(u[None, :, None], dt)[0, -1, 0]
            ref = -(3 * w ** 2 * s * (2 * c ** 2 - s ** 2))
        errs.append(float(abs(got - ref)))
    assert errs[0] / errs[1] > 3.2 and errs[1] / errs[2] > 3.2, errs


def test_short_traces_do_not_fail():
    """Fewer samples than the stencil wants: the extrapolation drops order."""
    for nt in (1, 2, 3):
        x = torch.ones(1, nt, 2)
        assert PropSEM._trace_ddt(x, DT).shape == x.shape
        assert PropSEM._trace_neg_d2dt2(x, DT).shape == x.shape


# --------------------------------------------------------------------------- #
# PropSEM: a record cut at nt ends where the longer record is at nt-1.
# --------------------------------------------------------------------------- #
NZ, NX, DH, DT_RUN, F0 = 41, 81, 10.0, 8e-4, 15.0
NT_FULL, CUTS = 360, (200, 260, 320)
SRC = np.array([[40, 12], [20, 12]], dtype=np.int64)
_RIX = np.arange(8, 74, 6)
REC = np.stack([_RIX, np.full_like(_RIX, 6)], -1)[None]


def _wav(dev):
    t = np.arange(NT_FULL) * DT_RUN
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32,
                        device=dev)


def _prop(impl, dev, equation=AcousticPotentialSEM2D, rtype=("p",)):
    kw = dict(eager_options=dict(use_compile=False)) if impl == "eager" else {}
    return PropSEM(equation, shape=(NZ, NX), dh=DH, dt=DT_RUN,
                   n_order=4, abcn=16, f_dominant=F0, impl=impl, dev=dev,
                   receiver_type=list(rtype), **kw)


def _models(equation, dev):
    vals = (2000.0,) if equation is AcousticPotentialSEM2D else (2000.0, 1100.0, 2000.0)
    return [torch.full((NZ, NX), v, device=dev) for v in vals]


def _end_errors(impl, dev, equation=AcousticPotentialSEM2D, rtype=("p",)):
    """On the potential form p is the recorded run itself (no difference at
    all, test_sem_pressure_wavelet.py); on the displacement elastics vx/vz
    are the differenced displacement, end sample included."""
    p = _prop(impl, dev, equation, rtype)
    vp = _models(equation, dev)
    w = _wav(dev)
    with torch.no_grad():
        full = p(w, SRC, REC, models=vp).double().cpu()
        scale = float(full.abs().max())
        out = []
        for n in CUTS:
            cut = p(w[:n], SRC, REC, models=vp).double().cpu()
            # the cut changes nothing before its last sample...
            torch.testing.assert_close(cut[:, :-1], full[:, :n - 1],
                                       rtol=0, atol=1e-4 * scale)
            # ...and its last sample follows the trace
            out.append(float((cut[:, -1] - full[:, n - 1]).abs().max()) / scale)
    return out, scale


@pytest.mark.parametrize("equation, rtype", [
    (AcousticPotentialSEM2D, ("p",)),
    (ElasticSEM2DDeformed, ("vx", "vz"))])
def test_eager_cut_record_ends_on_the_trace(equation, rtype):
    errs, scale = _end_errors("eager", "cpu", equation, rtype)
    assert scale > 0
    # repeating u(nt-1) gave 2-9x the trace's peak for p here
    assert max(errs) < 0.03, errs


@pytest.mark.skipif(not (torch.cuda.is_available() and sem_c_available()),
                    reason="needs CUDA + the compiled sweep._C SEM binding")
def test_c_cut_record_ends_on_the_trace_and_matches_eager():
    """impl='c', two shots (the batched path), gradient through the cut end."""
    errs, _ = _end_errors("c", "cuda")
    assert max(errs) < 0.03, errs
    n = CUTS[0]
    w = _wav("cuda")[:n]
    vp = torch.full((NZ, NX), 2000.0, device="cuda", requires_grad=True)
    c = _prop("c", "cuda")(w, SRC, REC, models=[vp])
    with torch.no_grad():
        e = _prop("eager", "cuda")(w, SRC, REC, models=[vp.detach()])
    scale = float(e.abs().max())
    assert float((c.detach() - e).abs().max()) / scale < 1e-4
    assert float((c.detach()[:, -1] - e[:, -1]).abs().max()) / scale < 1e-4
    c[:, -1].pow(2).sum().backward()                     # only the end sample
    assert torch.isfinite(vp.grad).all() and float(vp.grad.abs().max()) > 0
