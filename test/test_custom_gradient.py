"""Test the register_gradient public API (eager path).

  1. gradient mode + standard imaging condition → vp.grad matches autograd (cos≈1)
  2. gradient mode + raw uf*ub → a different gradient (the customization works)
  3. imaging mode → prop.imaging() returns an image, param.grad untouched
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, Acoustic1st, Elastic
from sweep.propagator import BoundarySaving, Ckpt, Full, PropTorch


from conftest import ricker

def cosine(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def build(device, nz=48, nx=56, abcn=30, M=2, dt=0.0015, dh=10.0, nt=120,
          memory=None, **kw):
    eq = Acoustic(spatial_order=2 * M, device=device, backend="torch")
    prop = PropTorch(eq, shape=(nz, nx), abcn=abcn, dh=dh, dt=dt, nt=nt,
                     dev=device, impl="eager", use_compile=False,
                     memory=Full() if memory is None else memory, **kw)
    return eq, prop


def standard_imaging(eq):
    def fn(forward, adjoint, models, dt, h):
        vp = models[0]
        hz, hx = eq._spacings_2d(h)
        lz, lx = eq.separable_d2_2d(forward['h1'], eq.laplace_kernels, hz, hx)
        return 2.0 * dt * dt * vp * (lz + lx) * adjoint['h1']
    return fn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    nt = 120
    vp_np = np.full((48, 56), 2000.0, dtype=np.float32)
    vp_np[16:, :] += 200.0
    wavelet = torch.as_tensor(ricker(nt, 0.0015, 10.0, 0.06), device=device)
    sources = torch.as_tensor([[[28, 2]]], device=device)
    receivers = torch.as_tensor([[[rx, 2] for rx in range(0, 56, 4)]], device=device)

    # ---------- reference: pure autograd gradient ----------
    eq, prop = build(device)
    vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
    rec = prop(wavelet, sources, receivers, models=[vp])
    (0.5 * rec.pow(2).sum()).backward()
    ref_grad = vp.grad.detach().clone()
    print(f"[ref] autograd grad norm={ref_grad.norm().item():.4e}")

    # ---------- 1. gradient mode + standard imaging ----------
    eq, prop = build(device)
    prop.register_gradient("vp", standard_imaging(eq), mode="gradient")
    vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
    rec_c = prop(wavelet, sources, receivers, models=[vp])
    (0.5 * rec_c.pow(2).sum()).backward()
    g_custom = vp.grad.detach().clone()
    print(f"\n[1] gradient mode, standard imaging:")
    print(f"    record match vs autograd fwd: max|Δ|={(rec_c - rec).abs().max().item():.2e}")
    print(f"    grad norm={g_custom.norm().item():.4e}  "
          f"cosine vs autograd = {cosine(g_custom, ref_grad):.4f}")

    # ---------- 2. gradient mode + raw uf*ub ----------
    eq, prop = build(device)
    prop.register_gradient("vp", lambda fwd, adj, m, dt, h: fwd['h1'] * adj['h1'], mode="gradient")
    vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
    rec_x = prop(wavelet, sources, receivers, models=[vp])
    (0.5 * rec_x.pow(2).sum()).backward()
    g_xcorr = vp.grad.detach().clone()
    print(f"\n[2] gradient mode, raw uf*ub:")
    print(f"    grad norm={g_xcorr.norm().item():.4e}  "
          f"cosine vs autograd = {cosine(g_xcorr, ref_grad):.4f}  (expected != 1)")

    # ---------- 3. imaging mode ----------
    eq, prop = build(device)
    prop.register_gradient("vp", lambda fwd, adj, m, dt, h: fwd['h1'] * adj['h1'], mode="imaging")
    vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
    images = prop.imaging(wavelet, sources, receivers, models=[vp])
    print(f"\n[3] imaging mode:")
    print(f"    returned keys: {list(images.keys())}")
    print(f"    image['vp'] shape={tuple(images['vp'].shape)} "
          f"norm={images['vp'].norm().item():.4e}")
    print(f"    vp.grad is None (untouched): {vp.grad is None}")


def test_gradient_mode_under_a_free_surface_is_cropped_like_autograd():
    """A free surface has no pad on top, so the runtime pads are not centred:
    the registered gradient is cropped per face, as EdgePadding crops the
    autograd one (a centred crop put it half the bottom pad too deep)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nt = 120
    vp_np = np.full((48, 56), 2000.0, dtype=np.float32)
    vp_np[16:, :] += 200.0
    wavelet = torch.as_tensor(ricker(nt, 0.0015, 10.0, 0.06), device=device)
    sources = torch.as_tensor([[[28, 2]]], device=device)
    receivers = torch.as_tensor([[[rx, 2] for rx in range(0, 56, 4)]], device=device)
    grads = []
    for custom in (False, True):
        eq, prop = build(device, free_surface=True)
        if custom:
            prop.register_gradient("vp", standard_imaging(eq), mode="gradient")
        vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
        (0.5 * prop(wavelet, sources, receivers, models=[vp]).pow(2).sum()).backward()
        grads.append(vp.grad.detach())
    assert cosine(*grads) > 0.99, cosine(*grads)


# --------------------------------------------------------------------------- #
# When the registration applies. On CPU: these check routing and plumbing.
# --------------------------------------------------------------------------- #
def _acquisition(device="cpu", nt=120):
    vp = np.full((48, 56), 2000.0, dtype=np.float32)
    vp[16:, :] += 200.0
    wavelet = torch.as_tensor(ricker(nt, 0.0015, 10.0, 0.06), dtype=torch.float32,
                              device=device)
    sources = torch.as_tensor([[[28, 2]]], device=device)
    receivers = torch.as_tensor([[[rx, 2] for rx in range(0, 56, 4)]], device=device)
    return torch.as_tensor(vp, device=device), wavelet, sources, receivers


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("memory", [Ckpt(), BoundarySaving()], ids=["ckpt", "boundary"])
def test_a_gradient_registration_needs_the_full_tape(memory):
    """Chunk checkpointing ignored the registration and gave autograd's
    gradient; boundary saving took it but kept the whole tape anyway."""
    eq, prop = build("cpu", memory=memory)
    with pytest.raises(ValueError, match=r"memory=Full\(\)"):
        prop.register_gradient("vp", standard_imaging(eq), mode="gradient")
    # imaging() runs its own full tape, whatever the strategy
    prop.register_gradient("vp", standard_imaging(eq), mode="imaging")


def test_a_recording_call_refuses_a_strategy_set_after_the_registration():
    eq, prop = build("cpu")
    prop.register_gradient("vp", standard_imaging(eq), mode="gradient")
    prop._backend_impl.memory_strategy = "ckpt"
    vp, wavelet, sources, receivers = _acquisition()
    vp.requires_grad_(True)
    with pytest.raises(ValueError, match=r"memory=Full\(\)"):
        prop(wavelet, sources, receivers, models=[vp])
    with torch.no_grad():                    # nothing to replace: runs
        prop(wavelet, sources, receivers, models=[vp])


@pytest.mark.parametrize("kw", [{"adj": True}, {"return_wavefield": True}],
                         ids=["adj", "return_wavefield"])
def test_a_recording_adjoint_or_snapshot_call_refuses_a_registration(kw):
    """Both ran the standard path and gave autograd's gradient."""
    eq, prop = build("cpu")
    prop.register_gradient("vp", standard_imaging(eq), mode="gradient")
    vp, wavelet, sources, receivers = _acquisition()
    vp.requires_grad_(True)
    with pytest.raises(ValueError, match="register_gradient"):
        prop(wavelet, sources, receivers, models=[vp], **kw)
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=[vp], **kw)


def test_a_call_that_records_no_registered_model_takes_the_standard_path(monkeypatch):
    """The hooked path ran (per-step snapshots, a record that requires grad)
    when nothing was recorded at all."""
    vp, wavelet, sources, receivers = _acquisition()
    _, plain = build("cpu")
    with torch.no_grad():
        ref = plain(wavelet, sources, receivers, models=[vp])
    w_ref = wavelet.clone().requires_grad_(True)
    plain(w_ref, sources, receivers, models=[vp]).square().sum().backward()

    eq, prop = build("cpu")
    prop.register_gradient("vp", standard_imaging(eq), mode="gradient")
    impl = type(prop._backend_impl)
    hooked = []
    real = impl._run_with_hooks
    monkeypatch.setattr(impl, "_run_with_hooks",
                        lambda self, *a, **k: hooked.append(1) or real(self, *a, **k))

    out = prop(wavelet, sources, receivers, models=[vp])        # grad on, no leaf
    assert not out.requires_grad
    assert torch.equal(out, ref)
    with torch.no_grad():
        out = prop(wavelet, sources, receivers, models=[vp.clone().requires_grad_(True)])
    assert not out.requires_grad
    w = wavelet.clone().requires_grad_(True)                    # an unregistered leaf
    prop(w, sources, receivers, models=[vp]).square().sum().backward()
    assert torch.equal(w.grad, w_ref.grad)
    assert not hooked


def _zero(fwd, adj, m, dt, h):
    return torch.zeros_like(next(iter(fwd.values())))


@pytest.mark.parametrize("equation,source_type,receiver_type,values", [
    (Acoustic1st, ["p"], ["p"], (2000.0, 1000.0)),
    (Elastic, ["sxx", "szz"], ["vz"], (2500.0, 1400.0, 2000.0)),
], ids=["acoustic1st", "elastic_two_source_fields"])
def test_the_hooked_path_keeps_the_rest_of_autograd(equation, source_type,
                                                    receiver_type, values):
    """With vp registered, rho / vs and the wavelet had no gradient, and the
    record injected the source into ``source_type[0]`` alone: the Elastic
    one with ``['sxx', 'szz']`` differed from the standard record by as much
    as the record itself."""
    shape, nt, dt = (32, 40), 64, 0.001
    t = np.arange(nt) * dt - 0.04
    a = (np.pi * 15 * t) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))
    src = torch.tensor([[20, 8]])
    rx = torch.arange(2, 38, 6)
    rec = torch.stack([rx, torch.full_like(rx, 2)], -1)[None]
    runs = []
    for registered in (False, True):
        eq = equation(spatial_order=4, device="cpu", backend="torch")
        prop = PropTorch(eq, impl="eager", shape=shape, dev="cpu", dh=10.0, dt=dt,
                         abcn=20, nt=nt, memory=Full(), use_compile=False,
                         source_type=source_type, receiver_type=receiver_type)
        if registered:
            prop.register_gradient("vp", _zero, mode="gradient")
        ms = [torch.full(shape, v, requires_grad=True) for v in values]
        w = wav.clone().requires_grad_(True)
        out = prop(w, src, rec, models=ms)
        out.square().sum().backward()
        runs.append((out.detach(), [m.grad for m in ms], w.grad))
    (r0, g0, w0), (r1, g1, w1) = runs
    assert _rel(r1, r0) < 1e-6
    assert torch.count_nonzero(g1[0]) == 0               # the registration applies
    for name, a_, b_ in zip(prop.model_names[1:], g1[1:], g0[1:]):
        assert a_ is not None, f"{name} lost its gradient"
        assert _rel(a_, b_) < 1e-6, name
    assert w1 is not None, "the wavelet lost its gradient"
    assert _rel(w1, w0) < 1e-6


def test_imaging_under_no_grad():
    eq, prop = build("cpu")
    prop.register_gradient("vp", lambda fwd, adj, m, dt, h: fwd["h1"] * adj["h1"],
                           mode="imaging")
    vp, wavelet, sources, receivers = _acquisition()
    image = prop.imaging(wavelet, sources, receivers, models=[vp])["vp"]
    with torch.no_grad():
        again = prop.imaging(wavelet, sources, receivers, models=[vp])["vp"]
    assert image.abs().max() > 0
    assert torch.equal(again, image)


if __name__ == "__main__":
    main()
