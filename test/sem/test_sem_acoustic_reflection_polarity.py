"""Two-layer reflection polarity for the four constant-density acoustic SEM
classes, bare (no C-PML) and with a C-PML.

The strong form is ``ü = c² ∇²u``: u and ∂u/∂n are continuous at a velocity
jump and a plane wave from c1 into c2 reflects with R = (c2 - c1)/(c2 + c1).
The bare path used to bake c² into the stiffness integral, i.e. solve
``∇·(c² ∇u)``, which keeps ``c² ∂u/∂n`` continuous instead and gives -R.
Homogeneous tests cannot tell the two apart.

Setup: a plane source (uniform acceleration on one node plane) in a column one
element wide with natural (Neumann) side walls, so the wave is exactly 1-D and
does not spread: the receiver above the source sees the up-going direct wave
and then the reflection off the interface below, in the ratio R. The 1-D
response is the time integral of the Ricker, an ODD pulse, so a signed peak
pick cannot see the polarity (both lobes have the same size, and flipping
the sign just moves the negative one to the other side); R is the projection of the reflection
window onto the direct window shifted by the known traveltime difference.
"""
from __future__ import annotations

import math

import pytest
import torch

from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.acoustic3d import build_cpml3d
from sweep.sem.equations.acoustic2d import AcousticSEM2D
from sweep.sem.equations.acoustic2d_deformed import AcousticSEM2DDeformed
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.hex27 import build_axis_aligned_hex27
from sweep.sem.mesh.quad4 import build_quad4_mesh
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DT64 = torch.float64
H, NEZ, NO = 25.0, 60, 4                  # 1500 m column, 25 m elements
C1, C2, Z_IFACE = 1000.0, 3000.0, 900.0   # interface on an element edge
Z_SRC, Z_REC = 700.0, 600.0
F0, T0, DT, NT = 10.0, 0.15, 2.5e-4, 3200  # 0.8 s: the top echo comes at 1.45 s
R_EXPECT = (C2 - C1) / (C2 + C1)          # +0.5

FAMILIES = {
    "2d": (AcousticSEM2D,
           lambda dt=DT64: build_quad4_mesh(1, NEZ, H, NEZ * H, NO, dtype=dt)),
    "2d_deformed": (AcousticSEM2DDeformed,
                    lambda dt=DT64: build_axis_aligned_quad9(
                        1, NEZ, H, NEZ * H, n_order=NO, dtype=dt)),
    "3d": (AcousticSEM3D,
           lambda dt=DT64: build_hex8_mesh(1, 1, NEZ, H, H, NEZ * H, NO, dtype=dt)),
    "3d_deformed": (AcousticSEM3DDeformed,
                    lambda dt=DT64: build_axis_aligned_hex27(
                        1, 1, NEZ, H, H, NEZ * H, n_order=NO, dtype=dt)),
}


def _ricker(t):
    a = (math.pi * F0 * (t - T0)) ** 2
    return (1.0 - 2.0 * a) * math.exp(-a)


def _two_layer(name, cpml, dtype=DT64):
    cls, mk = FAMILIES[name]
    m = mk(dtype)
    # element-wise constant layers (decided by the element's mean depth)
    zc = m.z_e.reshape(m.n_elements, -1).mean(-1)
    shape = (m.n_elements,) + (1,) * (m.z_e.dim() - 1)
    vp_e = torch.where(zc < Z_IFACE, C1, C2).to(dtype).reshape(shape).expand_as(m.z_e)
    kw = {}
    if cpml:
        build = build_cpml3d if m.z_e.dim() == 4 else build_cpml
        kw["cpml"] = build(m, dt=DT, sides=("bottom",), pml_thickness_elements=8,
                           c_max=C2, f_dominant=F0)
    return cls.build(m, vp_e.contiguous(), dirichlet=(), **kw)


def _trace(name, cpml):
    eq = _two_layer(name, cpml)
    m = eq.mesh

    zg = m.z_global
    src_nodes = (zg - Z_SRC).abs() < 1e-6
    rec_nodes = ((zg - Z_REC).abs() < 1e-6).nonzero().reshape(-1)
    assert src_nodes.any() and rec_nodes.numel()
    pattern = torch.where(src_nodes, eq.mass, torch.zeros_like(eq.mass))  # M⁻¹b uniform

    fields = eq.init_state(DT, zg.device, DT64)
    out = torch.empty(NT, dtype=DT64)
    for n in range(NT):
        fields = eq.step(fields, (pattern * _ricker(n * DT),), DT)
        out[n] = fields[0][rec_nodes].mean()
    return out


def _reflection_ratio(tr):
    t_dir = T0 + (Z_SRC - Z_REC) / C1                        # 0.25 s
    t_ref = T0 + (2 * Z_IFACE - Z_SRC - Z_REC) / C1          # 0.65 s
    i0, i1 = round((t_dir - 0.1) / DT), round((t_dir + 0.1) / DT)
    k = round((t_ref - t_dir) / DT)
    d, r = tr[i0:i1], tr[i0 + k:i1 + k]
    return float((r * d).sum() / (d * d).sum())


@pytest.mark.parametrize("cpml", [False, True], ids=["bare", "cpml"])
@pytest.mark.parametrize("name", list(FAMILIES))
def test_two_layer_reflection_has_the_c2_laplacian_polarity(name, cpml):
    ratio = _reflection_ratio(_trace(name, cpml))
    assert abs(ratio - R_EXPECT) < 0.03, (
        f"{name} {'cpml' if cpml else 'bare'}: reflection/direct = {ratio:+.4f}, "
        f"expected {R_EXPECT:+.3f} (div(c² grad u) gives {-R_EXPECT:+.3f})")


@pytest.mark.parametrize("name", list(FAMILIES))
def test_bare_and_cpml_paths_agree_before_the_pml_echo(name):
    """Away from the absorbing band the two paths are one scheme: identical
    until anything could have come back from the bottom (the transmitted wave
    reaches the 1300 m band top at ~0.48 s; nothing returns before ~0.6 s)."""
    a, b = _trace(name, False), _trace(name, True)
    n = int(0.55 / DT)
    rel = float((a[:n] - b[:n]).norm() / a[:n].norm())
    assert rel < 1e-12, f"{name}: bare vs cpml rel {rel:.2e}"



# ------------------------- impl='c' (3-D, CUDA) ---------------------------- #
# The CUDA kernels against the eager classes pinned above, on the same jump: a
# point source at the top of the column, all of the direct wave and the
# interface reflection. A smooth model cannot catch a div(c² grad u) kernel
# (the two forms differ only where c varies); this jump flips the reflection.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cpml", [False, True], ids=["bare", "cpml"])
@pytest.mark.parametrize("name", ["3d", "3d_deformed"])
def test_cuda_two_layer_matches_eager(name, cpml):
    from sweep.sem import SEMPropagator
    from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available
    if not sem_c_available():
        pytest.skip("needs the compiled SEM core")
    nt = 2800                     # the reflection ends at ~0.7 s, the top echo is later
    wav = torch.tensor([_ricker(n * DT) for n in range(nt)], dtype=DT64)
    kw = dict(dt=DT, nt=nt, source_xz=[(H / 2, H / 2, Z_SRC)],
              receiver_xz=[(H / 2, H / 2, Z_REC)])
    with torch.no_grad():
        ref = SEMPropagator.build(_two_layer(name, cpml), **kw).forward(wav)
        out = SEMCudaPropagator.build(_two_layer(name, cpml, torch.float32),
                                      **kw).forward(wav.float().cuda())
    a, b = ref.reshape(-1), out.reshape(-1).double().cpu()
    cos = float((a @ b) / (a.norm() * b.norm()))
    assert cos > 1 - 1e-5, f"{name} {'cpml' if cpml else 'bare'}: c vs eager cos {cos:.7f}"
