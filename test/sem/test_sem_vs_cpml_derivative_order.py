"""The velocity-stress C-PML recurses on the stress (and Biot flux) DERIVATIVES.

The velocity update used to recurse on the stress itself and take the weak-form
derivative of ``sigma + psi``, i.e. ``d(psi * sigma)`` instead of ``psi * d(sigma)``:
an extra term in the profile's gradient that sent a low-frequency echo back from
the top of the layer, the same at any resolution. Here: a free-surface box with
a 160 m layer below and beside it against the same box in a model large enough
that nothing comes back within the record (twice as deep, wider by the box's
half-width on each side). Measured (CPU, float64): elastic
4.0e-3 before, 9.5e-5 now; Biot vz 2.7e-2 / 2.6e-4, p 7.1e-3 / 6.3e-5.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from sweep.equations.biot import biot_from_biot_constants
from sweep.sem import BiotVSSEM2DDeformed, ElasticVSSEM2DDeformed, PropSEM
from test_biot import DAI1

ABCN = 16                        # four elements of layer


def _echo(eq, models, dh, dt, nt, f0, receivers):
    """max |record - record of the large model| / its peak, per receiver type."""
    a = (math.pi * f0 * (np.arange(nt) * dt - 1.2 / f0)) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))

    def run(nz, nx, x0):
        p = PropSEM(eq, shape=(nz, nx), dh=dh, dt=dt, n_order=4, abcn=ABCN, f_dominant=f0,
                    dev="cpu", dtype=torch.float64, impl="eager", free_surface=True,
                    source_type=["sxx", "szz"], receiver_type=receivers)
        ms = [torch.as_tensor(c, dtype=torch.float64).expand((nz, nx)).contiguous()
              for c in models]
        rx = np.arange(10, 71, 4) + x0
        with torch.no_grad():
            r = p(w, np.array([[40 + x0, 3.0]]), np.stack([rx, np.full_like(rx, 1)], 1)[None],
                  models=ms)
        return r[0].numpy()

    small, large = run(41, 81, 0), run(81, 161, 40)    # nothing returns from the large one
    return [float(np.abs(small[..., k] - large[..., k]).max() / np.abs(large[..., k]).max())
            for k in range(len(receivers))]


def test_elastic_layer_echo():
    (e,) = _echo(ElasticVSSEM2DDeformed, (2400.0, 1300.0, 2000.0), 10.0, 6e-4, 700, 15.0,
                 ["vz"])
    assert e < 1e-3, f"echo {e:.2e}"


def test_biot_layer_echo():
    """The flux q of the pressure equation took the same route as the stress."""
    models = biot_from_biot_constants(**DAI1, b=7e4, phi=0.3)
    e_vz, e_p = _echo(BiotVSSEM2DDeformed, models, 4.0, 2e-4, 900, 30.0, ["vz", "p"])
    assert e_vz < 2e-3, f"vz echo {e_vz:.2e}"
    assert e_p < 1e-3, f"p echo {e_p:.2e}"
