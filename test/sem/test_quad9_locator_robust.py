"""Regression tests for the Quad9 source/receiver locator.

Guards against the curved-edge non-convergence bug: on a topography
mesh, points near a curved inter-element boundary used to make the
Newton solver stall on the reference-square edge (it clamped ξ, η every
iteration) and then wrongly raise "did not converge". The robust locator
(a) runs Newton without mid-iteration clamping so an overshoot reveals
the point is in a neighbour, and (b) searches the column's rows by
centre-z proximity until one lands inside [-1, 1].

Run::

    PYTHONPATH=src python test/sem/test_quad9_locator_robust.py
    # or: pytest test/sem/test_quad9_locator_robust.py
"""
from __future__ import annotations

import functools

import numpy as np
import torch

from sweep.sem import build_topography_quad9, build_axis_aligned_quad9
from sweep.sem.propagator import build_point_stamp, _locate_in_quad9, _q2_eval_one


X_MAX, Z_BOTTOM = 1270.0, 890.0
HILL_BASE, HILL_X0, HILL_SIG = 50.0, X_MAX / 2, 160.0


@functools.lru_cache(maxsize=None)
def _hill_mesh():
    """Built once: the locator only reads the mesh, so the three tests that
    use it share one 64x44 topography build."""
    def z_top_fn(x):
        return HILL_BASE * (1.0 - torch.exp(-((x - HILL_X0) ** 2)
                                            / (2.0 * HILL_SIG ** 2)))
    return build_topography_quad9(nex=64, nez=44, x_max=X_MAX, z_bottom=Z_BOTTOM,
                                  z_top_fn=z_top_fn, n_order=4, dtype=torch.float64)


def _backmap_err(mesh, x, z):
    e, xi, eta = _locate_in_quad9(mesh, float(x), float(z))
    pos = _q2_eval_one(mesh.control_xy[e], xi, eta)
    tgt = torch.tensor([x, z], dtype=torch.float64)
    return e, float((pos - tgt).norm())


def test_previously_failing_point():
    """(452.93, 300.0) on the hill mesh — the exact point that used to
    raise. It lies just below the ez=13/ez=14 curved edge, so it belongs
    to the lower element; the locator must find it to machine precision."""
    mesh = _hill_mesh()
    e, err = _backmap_err(mesh, 452.93, 300.0)
    assert err < 1e-9, f"back-map err {err:.2e} m (elem {e})"


def test_grid_scan_all_locate():
    """Every interior point of the hill mesh must locate, with a back-map
    error at machine precision (the curved-edge cases included)."""
    mesh = _hill_mesh()
    def z_top(x):
        return HILL_BASE * (1.0 - np.exp(-((x - HILL_X0) ** 2)
                                         / (2.0 * HILL_SIG ** 2)))
    n_fail = 0
    max_err = 0.0
    n = 0
    for x in np.linspace(15.0, X_MAX - 15.0, 100):
        zt = z_top(x)
        for z in np.linspace(zt + 8.0, Z_BOTTOM - 8.0, 60):
            try:
                _, err = _backmap_err(mesh, x, z)
                max_err = max(max_err, err)
                n += 1
            except Exception:
                n_fail += 1
    assert n_fail == 0, f"{n_fail}/{n + n_fail} points failed to locate"
    assert max_err < 1e-8, f"max back-map err {max_err:.2e} m over {n} points"


def test_receiver_line_at_curved_depths():
    """A dense horizontal receiver line at several depths must build
    stamps everywhere (the benchmark-triggering scenario)."""
    mesh = _hill_mesh()
    for z in (60.0, 150.0, 300.0, 445.0, 700.0):
        for x in np.linspace(120.0, X_MAX - 120.0, 120):
            build_point_stamp(mesh, float(x), z)   # must not raise


def test_flat_quad9_unaffected():
    """On an axis-aligned Quad9, the locator still returns exact (ξ, η)."""
    mesh = build_axis_aligned_quad9(nex=32, nez=22, x_max=X_MAX, z_max=Z_BOTTOM,
                                    n_order=4, dtype=torch.float64)
    for (x, z) in [(100.0, 100.0), (635.0, 445.0), (1200.0, 800.0)]:
        _, err = _backmap_err(mesh, x, z)
        assert err < 1e-9, f"flat Q9 back-map err {err:.2e} at ({x},{z})"


def test_float32_grid_scan_locates():
    """The locator must work on a float32 mesh too. A fixed absolute tol
    (1e-5 m) sits below the float32 residual floor (~|coord|·eps ≈ 1e-4 for
    O(1000 m) coordinates), so Newton could never reach it and ~half the
    points were wrongly rejected. The tolerance now scales with the
    coordinate magnitude + dtype eps."""
    def z_top_t(x):
        return HILL_BASE * (1.0 - torch.exp(-((x - HILL_X0) ** 2)
                                            / (2.0 * HILL_SIG ** 2)))
    meshes = {
        "topo": build_topography_quad9(nex=40, nez=30, x_max=X_MAX,
                                       z_bottom=Z_BOTTOM, z_top_fn=z_top_t,
                                       n_order=4, dtype=torch.float32),
        "flat": build_axis_aligned_quad9(nex=40, nez=30, x_max=X_MAX,
                                         z_max=Z_BOTTOM, n_order=4,
                                         dtype=torch.float32),
    }
    for name, mesh in meshes.items():
        n_fail = max_err = n = 0
        for x in np.linspace(15.0, X_MAX - 15.0, 50):
            for z in np.linspace(60.0, Z_BOTTOM - 15.0, 40):
                try:
                    e, xi, eta = _locate_in_quad9(mesh, float(x), float(z))
                    pos = _q2_eval_one(mesh.control_xy[e], xi, eta)
                    err = float((pos - torch.tensor([x, z],
                                                    dtype=torch.float32)).norm())
                    max_err = max(max_err, err); n += 1
                except Exception:
                    n_fail += 1
        assert n_fail == 0, f"{name}: {n_fail}/{n + n_fail} float32 points failed"
        assert max_err < 0.5, f"{name}: float32 back-map err {max_err:.2e} m"


if __name__ == "__main__":
    test_previously_failing_point()
    print("test_previously_failing_point: OK")
    test_grid_scan_all_locate()
    print("test_grid_scan_all_locate: OK")
    test_receiver_line_at_curved_depths()
    print("test_receiver_line_at_curved_depths: OK")
    test_flat_quad9_unaffected()
    print("test_flat_quad9_unaffected: OK")
    test_float32_grid_scan_locates()
    print("test_float32_grid_scan_locates: OK")
    print("ALL LOCATOR REGRESSION TESTS PASSED")
