"""suggest_discretization: sensible values + the suggested dt is CFL-stable."""
import numpy as np
import torch

from sweep.sem import (
    suggest_discretization,
    build_topography_quad9,
    ElasticVSSEM2DDeformed,
    SEMPropagator,
    build_simple_elastic_cpml,
)

X, Z, F0 = 1280.0, 900.0, 12.0
VP, VS, RHO = 2400.0, 1200.0, 2000.0


def test_values_sensible():
    d = suggest_discretization(X, Z, v_min=VS, v_max=VP, f0=F0, record_length=0.6)
    assert d.n_order == 4
    assert d.points_per_wavelength >= 5.0 - 1e-9          # meets the ppw target
    assert d.dt > 0 and d.nt == int(np.ceil(0.6 / d.dt))
    assert 20.0 <= d.h_elem_x <= 50.0 and 20.0 <= d.h_elem_z <= 50.0
    # coarser ppw → coarser mesh (fewer elements); finer ppw → more
    d_fine = suggest_discretization(X, Z, v_min=VS, v_max=VP, f0=F0, ppw=10.0)
    assert d_fine.nex > d.nex and d_fine.nez > d.nez


def test_suggested_dt_is_stable_on_hill():
    """A deformed-mesh elastic run with the suggested dt stays bounded."""
    d = suggest_discretization(X, Z, v_min=VS, v_max=VP, f0=F0, record_length=0.5)

    def z_top(x):
        return 50.0 * (1.0 - torch.exp(-((x - X / 2) ** 2) / (2.0 * 150.0 ** 2)))

    mesh = build_topography_quad9(nex=d.nex, nez=d.nez, x_max=X, z_bottom=Z,
                                  z_top_fn=z_top, n_order=d.n_order, dtype=torch.float64)
    sh = (mesh.n_elements, mesh.ngll, mesh.ngll)
    cpml = build_simple_elastic_cpml(mesh, dt=d.dt, sides=("left", "right", "bottom"),
                                     pml_thickness_elements=8, c_max=VP, f_dominant=F0)
    eq = ElasticVSSEM2DDeformed.build(
        mesh,
        torch.full(sh, VP, dtype=torch.float64),
        torch.full(sh, VS, dtype=torch.float64),
        torch.full(sh, RHO, dtype=torch.float64),
        free_top=False, cpml=cpml)
    src_z = float(z_top(torch.tensor([X / 2]))[0]) + 40.0
    prop = SEMPropagator.build(eq, dt=d.dt, nt=d.nt, source_xz=[(X / 2, src_z)],
                               receiver_xz=[(X / 2, src_z + 50.0)],
                               source_kinds=["moment_tensor"], source_payloads=[(1.0, 1.0, 0.0)])
    t = np.arange(d.nt) * d.dt - 0.1
    a = (np.pi * F0 * t) ** 2
    wav = torch.tensor((1.0 - 2.0 * a) * np.exp(-a), dtype=torch.float64)
    seis = prop.forward(wav)[..., 1].T
    assert torch.isfinite(seis).all()
    assert seis.abs().max().item() < 1.0          # stable ~1e-8; CFL blow-up ≫ 1
