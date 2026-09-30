"""``AcousticVTI1st`` on impl='c' read the grid spacing with the axes swapped.

``PropBase`` stores spacing in model-axis order ``(dz, dx)`` and
``_c.py::_cuda_spacing()`` reverses it, so ``p.spacing`` reaches the kernels in
Cartesian order ``[dx, dz]`` -- as every other driver in the tree reads it, and
as this equation's own 3-D twin reads it. ``acoustic_vti_1st_2d`` read it as
``(dz, dx)`` in all four entry points (forward, backward, backward_bs,
backward_ckpt), which is the same thing whenever ``dx == dz`` and wrong the
moment they differ.

Nothing caught it because nothing ever passed a non-uniform ``dh``: the gate,
the gradient matrix and every test use a scalar. Measured before the fix, on a
60x72 grid with ``dh=(dz=10, dx=25)``: the c record's cosine against eager was
**-0.259** and all four model gradients were ~0. With ``dh=10`` it was 1.000000
to six digits, which is exactly why it hid.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, AcousticVTI1st, AcousticVTI1st3D
from sweep.propagator.options import MemoryOptions
from sweep.propagator.torch import PropTorch

def requires_binding(*symbols):
    """Skip unless the compiled extension really is there, with these symbols.

    Deliberately not ``try: hasattr(sweep._C, s) except Exception`` -- attribute
    access on that lazy shim triggers the JIT, so the except would turn a
    COMPILE FAILURE into a green skip.
    """
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="no CUDA device")
    import sweep
    if not sweep.is_torch_binding_available():
        return pytest.mark.skip(reason="sweep._C is not available here")
    import sweep._C as _C
    missing = [s for s in symbols if not hasattr(_C, s)]
    if missing:
        raise RuntimeError(
            f"sweep._C is built but lacks {missing} -- a stale or partial "
            f"build, not a missing capability")
    return pytest.mark.skipif(False, reason="")


# Hold the PHYSICAL extent fixed and vary only the spacing, so every case is as
# well-excited as the isotropic one. And propagate long enough to ARRIVE: the
# source sits ~400 m deep and the far receiver is ~720 m offset, so a short run
# records the pre-arrival and compares two numerical zeros -- the first version
# of this test did exactly that and reported a failure that was not there.
EXTENT_Z, EXTENT_X = 1200.0, 1440.0        # metres
NT, DT, FREQ, ABCN = 1500, 2e-4, 8.0, 15   # 0.30 s; 8 Hz keeps ~15 ppw at dx=24
SPACINGS = [(10.0, 10.0), (10.0, 20.0), (20.0, 10.0), (8.0, 24.0)]


def _run(cls, impl, dh, source, receiver, pml, models):
    dz, dx = dh
    nz, nx = int(round(EXTENT_Z / dz)), int(round(EXTENT_X / dx))
    shape = (nz, nx)
    prop = PropTorch(cls(spatial_order=4, device="cuda"), backend="torch", impl=impl,
                     shape=shape, dh=dh, dt=DT, nt=NT, abcn=ABCN, B=1, dev="cuda",
                     pml_type=pml, source_type=source, receiver_type=receiver,
                     memory=MemoryOptions(strategy="full"))
    src = np.array([[nx // 2, nz // 3]], np.int64)
    rxx = np.arange(3, nx - 3, max(1, nx // 14), np.int64)
    rec = np.stack([rxx, np.full(rxx.size, 4)], -1)[None]
    t = np.arange(NT, dtype=np.float32) * DT - 1.5 / FREQ
    a = np.pi * FREQ * t
    wav = torch.tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e3).astype(np.float32),
                       device="cuda")
    ms = [torch.full(shape, v, device="cuda", requires_grad=True) for v in models]
    out = prop(wav, src, rec, models=ms)
    out.pow(2).mean().backward()
    return out.detach(), [m.grad.detach() for m in ms]


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


VTI = (AcousticVTI1st, ["sV"], ["sV"], "cpmls", (3000.0, 0.15, 0.05, 2200.0))
AC = (Acoustic, ["h1"], ["h1"], "cpmlr", (3000.0,))


@requires_binding("acoustic_vti_1st_2d_forward")
@pytest.mark.parametrize("dh", SPACINGS, ids=lambda d: f"dz{d[0]:g}_dx{d[1]:g}")
def test_vti1st_2d_matches_eager_whatever_the_spacing(dh):
    label = "isotropic" if dh[0] == dh[1] else "anisotropic"
    cls, s, r, pml, models = VTI
    rec_c, grads_c = _run(cls, "c", dh, s, r, pml, models)
    rec_e, grads_e = _run(cls, "eager", dh, s, r, pml, models)
    # Guard the comparison before trusting it: a record at the fp32 floor makes
    # every cosine below meaningless. Before the fix the swapped axes ALSO
    # collapsed the amplitude, so this assert is part of the diagnosis.
    assert float(rec_e.abs().max()) > 1.0, (
        f"{label} dh={dh}: the eager record peaks at "
        f"{float(rec_e.abs().max()):.3e} -- nothing arrived, so this case "
        f"proves nothing")
    assert _cos(rec_c, rec_e) > 0.999, (
        f"{label} dh={dh}: the compiled record disagrees with eager, "
        f"cosine {_cos(rec_c, rec_e):.6f}")
    for name, gc, ge in zip(("vp", "epsilon", "delta", "rho"), grads_c, grads_e):
        assert _cos(gc, ge) > 0.999, (
            f"{label} dh={dh}: grad[{name}] cosine {_cos(gc, ge):.6f}")


@requires_binding("acoustic2d_forward")
def test_the_anisotropic_path_itself_is_fine():
    """The control. Acoustic reads the same ``p.spacing`` in Cartesian order and
    agrees with eager at dx != dz, so a VTI failure is that equation's own axis
    handling and not something structural about non-uniform grids."""
    cls, s, r, pml, models = AC
    rec_c, grads_c = _run(cls, "c", (10.0, 20.0), s, r, pml, models)
    rec_e, grads_e = _run(cls, "eager", (10.0, 20.0), s, r, pml, models)
    assert _cos(rec_c, rec_e) > 0.999
    assert _cos(grads_c[0], grads_e[0]) > 0.999


@requires_binding("acoustic_vti_1st_2d_forward")
def test_swapping_dh_actually_changes_the_answer():
    """Guards the test itself: if dh=(10, 25) and dh=(25, 10) gave the same
    record, the two cases above would agree for a reason that has nothing to do
    with the axes being read correctly."""
    cls, s, r, pml, models = VTI
    a, _ = _run(cls, "eager", (10.0, 20.0), s, r, pml, models)
    b, _ = _run(cls, "eager", (20.0, 10.0), s, r, pml, models)
    assert _cos(a, b) < 0.99, (
        f"transposing dh left the eager record essentially unchanged "
        f"(cosine {_cos(a, b):.6f}) -- this grid cannot see an axis swap")


@requires_binding("acoustic_vti_1st_3d_forward")
def test_the_3d_twin_was_always_right():
    """The sharpest control: same equation, same physics, one more dimension.

    ``acoustic_vti_1st_3d`` reads ``dx = spacing[0], dy = spacing[1],
    dz = spacing[2]`` -- the Cartesian order -- and agrees with eager at a fully
    anisotropic spacing. So the 2-D file was not following a different local
    convention that happened to work; it was the only one of the pair reading
    the array backwards.
    """
    shape = (36, 40, 44)
    nz, ny, nx = shape

    def run(impl):
        prop = PropTorch(AcousticVTI1st3D(spatial_order=4, device="cuda"),
                         backend="torch", impl=impl, shape=shape,
                         dh=(10.0, 18.0, 25.0), dt=DT, nt=90, abcn=12, B=1,
                         dev="cuda", pml_type="cpmls", source_type=["sV"],
                         receiver_type=["sV"], memory=MemoryOptions(strategy="full"))
        src = np.array([[nx // 2, ny // 2, nz // 3]], np.int64)
        rxx = np.arange(4, nx - 4, 5, np.int64)
        rec = np.stack([rxx, np.full(rxx.size, ny // 2),
                        np.full(rxx.size, 5)], -1)[None]
        t = np.arange(90, dtype=np.float32) * DT - 0.008
        a = np.pi * 20.0 * t
        wav = torch.tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e3).astype(np.float32),
                           device="cuda")
        ms = [torch.full(shape, v, device="cuda", requires_grad=True)
              for v in (3000.0, 0.15, 0.05, 2200.0)]
        out = prop(wav, src, rec, models=ms)
        out.pow(2).mean().backward()
        return out.detach(), [m.grad.detach() for m in ms]

    rec_c, grads_c = run("c")
    rec_e, grads_e = run("eager")
    assert _cos(rec_c, rec_e) > 0.999, _cos(rec_c, rec_e)
    for name, gc, ge in zip(("vp", "epsilon", "delta", "rho"), grads_c, grads_e):
        assert _cos(gc, ge) > 0.999, f"grad[{name}] {_cos(gc, ge):.6f}"


def test_no_driver_reads_the_spacing_array_backwards():
    """A census, not a spot check -- this bug had no siblings and must keep none.

    ``p.spacing`` reaches the kernels in Cartesian order because
    ``_c.py::_cuda_spacing()`` reverses PropBase's model-axis order. So
    ``p.spacing[0]`` is dx in 2-D and in 3-D, always. At the time this test was
    written 33 driver files read it that way and exactly two did not -- both of
    them ``acoustic_vti_1st_2d``, which is what this file is about.

    Grepping is the right instrument here: the failure is invisible to any
    isotropic run, so a numerical test can only cover the equations someone
    thought to parametrise, while this covers every driver that exists.
    """
    import pathlib
    import re

    import sweep

    root = pathlib.Path(sweep.__file__).resolve().parent / "csrc" / "cuda" / "equations"
    if not root.exists():
        pytest.skip("csrc is not present in this checkout")

    pat = re.compile(r"\b(?:const\s+)?float\s+(\w+)\s*=\s*p\.spacing\[0\]")
    offenders, seen = [], 0
    for f in sorted(root.rglob("*.cu")) + sorted(root.rglob("*.cuh")):
        for name in pat.findall(f.read_text()):
            seen += 1
            if name != "dx":
                offenders.append(f"{f.relative_to(root)}: float {name} = p.spacing[0]")
    assert seen >= 20, f"the pattern matched only {seen} sites -- has the idiom changed?"
    assert not offenders, (
        "p.spacing arrives in Cartesian order [dx, dz] (see _c.py::_cuda_spacing), "
        "so spacing[0] is dx. These read it as something else, which is a silent "
        f"axis swap on any non-uniform grid:\n  " + "\n  ".join(offenders))
