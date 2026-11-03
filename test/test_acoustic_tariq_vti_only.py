"""AcousticTariq: ``vv`` is a velocity, and the class is VTI only.

The docstring called ``vv`` the *squared* vertical velocity while the step
squares it (``vv**2 * lap_z``), so a user following the docstring passed
``vp**2`` and got a medium ``vp`` times too fast vertically.  The factory also
returned this class for ``AcousticAniso(method="alkhalifah", symmetry="tti")``
and registered it as ``AcousticTTIAlkhalifah``, with a comment promising a
"rotated eta" that does not exist: the class has no tilt, so a TTI request
silently solved the VTI equation.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

NZ, NX, NT, DT, DH = 40, 48, 300, 1.0e-3, 10.0


def test_factory_refuses_alkhalifah_tti():
    from sweep.equations import AcousticAniso

    with pytest.raises(NotImplementedError, match="VTI only"):
        AcousticAniso(method="alkhalifah", symmetry="tti")
    with pytest.raises(NotImplementedError, match="VTI only"):
        AcousticAniso.equation_class(method="alkhalifah", symmetry="tti")
    assert ("alkhalifah", "tti", 2) not in AcousticAniso.supported()
    assert type(AcousticAniso(method="alkhalifah", symmetry="vti")).__name__ == "AcousticTariq"


def test_no_tti_alias():
    from sweep.equations import get_equation

    assert get_equation("AcousticVTIAlkhalifah").__name__ == "AcousticTariq"
    with pytest.raises(KeyError):
        get_equation("AcousticTTIAlkhalifah")


def test_vv_spec_is_a_velocity():
    from sweep.equations import AcousticTariq

    spec = {s.name: s for s in AcousticTariq.MODEL_SPECS}["vv"]
    assert spec.unit == "m/s"
    assert "squared" not in spec.description.lower()


def _record(name, models):
    import sweep.equations as E
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    eq = getattr(E, name)(spatial_order=4, device="cpu", backend="torch")
    prop = PropTorch(eq, backend="torch", impl="eager",
                     eager_options=EagerOptions(use_compile=False),
                     shape=(NZ, NX), dev="cpu", dh=DH, dt=DT,
                     source_type=["h1"], receiver_type=["h1"],
                     abcn=20, pml_type="cpmlr", nt=NT, B=1, use_ckpt=False)
    t = torch.arange(NT, dtype=torch.float32) * DT - 0.06
    wav = (1 - 2 * (np.pi * 15 * t) ** 2) * torch.exp(-(np.pi * 15 * t) ** 2)
    src = np.array([[NX // 2, NZ // 2]], dtype=np.int32)
    rx = np.arange(4, NX - 4, dtype=np.int32)
    rec = np.stack([rx, np.full_like(rx, 5)], axis=-1)[None, ...]
    with torch.no_grad():
        out = prop(wav, src, rec, models=models)
    return out[0] if isinstance(out, (tuple, list)) else out


def test_isotropic_limit_matches_acoustic():
    """eta = 0, vv = v = c is the isotropic wave equation with velocity c:
    the record must be the Acoustic one (vv = c**2 would blow up instead)."""
    c = torch.full((NZ, NX), 2000.0)
    ref = _record("Acoustic", [c])
    out = _record("AcousticTariq", [c, c, torch.zeros((NZ, NX))])
    assert torch.isfinite(out).all()
    scale = float(ref.abs().max())
    assert scale > 0
    # ~1e-3 apart (the two steps are written differently); vv = c**2 is inf
    assert float((out - ref).abs().max()) <= 1e-2 * scale
