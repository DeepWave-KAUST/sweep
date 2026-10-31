"""The acoustic C-PML profile's ``db/dx`` and ``db/dz`` on graded meshes.

The double-convolution step needs the derivative of the recursion coefficient ``b`` across
the layer. It used to be taken with row 0's x axis and column 0's z axis for the whole mesh;
a topography mesh grades every column differently, so ``db/dz`` came out off by the ratio of
the column heights there (here up to 1.4 under the hill).
"""
from __future__ import annotations

import numpy as np
import torch

from sweep.sem import build_topography_quad9
from sweep.sem.cpml._axis import gradient_along
from sweep.sem.cpml.acoustic import build_cpml


def test_gradient_along_is_np_gradient_line_by_line():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.uniform(0.5, 1.5, size=(9, 13)), axis=0)     # every column its own axis
    f = np.sin(0.3 * x) + x ** 2
    ref = np.stack([np.gradient(f[:, j], x[:, j]) for j in range(x.shape[1])], 1)
    assert np.array_equal(gradient_along(f, x, axis=0), ref)
    assert np.array_equal(gradient_along(f.T, x.T, axis=1), ref.T)


def test_db_dz_follows_every_column_of_a_topography_mesh():
    hill = lambda x: 400.0 * (1.0 - torch.exp(-((x - 1000.0) / 250.0) ** 2))   # z down
    mesh = build_topography_quad9(nex=20, nez=10, x_max=2000.0, z_bottom=1400.0,
                                  z_top_fn=hill, n_order=4, dtype=torch.float64)
    cp = build_cpml(mesh, dt=1e-3, sides=("left", "right", "bottom"), pml_thickness_elements=3,
                    c_max=3000.0, f_dominant=10.0)
    nz, nx = mesh.nez * mesh.n_order + 1, mesh.nex * mesh.n_order + 1
    z = mesh.z_global.reshape(nz, nx).numpy()
    b = cp.b_z.reshape(nz, nx).numpy()
    db = cp.dbz_dz.reshape(nz, nx).numpy()
    heights = z[-1] - z[0]
    assert heights.max() / heights.min() > 1.3                      # the columns do differ
    for j in range(nx):
        assert np.array_equal(db[:, j], np.gradient(b[:, j], z[:, j]))
    old = np.gradient(b, z[:, 0], axis=0)                             # column 0's axis
    j = int(np.argmax(heights))                                       # under the summit
    band = np.abs(db[:, j]) > 0.1 * np.abs(db[:, j]).max()
    assert np.abs(old[band, j] / db[band, j] - heights[j] / heights[0]).max() < 1e-6
