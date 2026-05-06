import numpy as np
import torch

from sweep.equations import DAS, DAS3D, DASElastic, DASElastic3D, gauge_average, helical_das_response
from sweep.propagator.torch import PropTorch


def _ricker(nt, dt, fm=12.0, delay=0.04):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def test_helical_response_weights_and_gauge_average():
    exx = torch.tensor([1.0, 2.0, 4.0])
    eyy = torch.tensor([10.0, 20.0, 40.0])
    ezz = torch.tensor([100.0, 200.0, 400.0])

    assert torch.allclose(helical_das_response(exx, ezz, eyy, angle=35.3), exx + eyy + ezz)
    assert torch.allclose(helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="x"), 4 * exx + eyy + ezz)
    assert torch.allclose(helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="z"), exx + eyy + 4 * ezz)

    averaged = gauge_average(torch.arange(5.0), gauge_cells=3)
    expected = torch.tensor([(0 + 0 + 1) / 3, 1.0, 2.0, 3.0, (3 + 4 + 4) / 3])
    assert torch.allclose(averaged, expected)


def test_das_equations_are_exported_without_cuda_binding():
    import sweep.equations as eq

    classes = eq._equation_classes()
    assert classes["DASElastic"] is DASElastic
    assert classes["DASElastic3D"] is DASElastic3D
    assert DAS is DASElastic
    assert DAS3D is DASElastic3D
    assert not DASElastic.supports_torch_binding()
    assert not DASElastic3D.supports_torch_binding()


def test_das_elastic_2d_eager_forward_and_gradient():
    device = torch.device("cpu")
    nt = 28
    dt = 0.001
    shape = (24, 28)
    wavelet = _ricker(nt, dt).reshape(1, 1, nt)
    sources = np.array([[[shape[1] // 2, 3]]], dtype=np.int32)
    receivers = np.stack(
        [
            np.arange(4, shape[1] - 4, 6, dtype=np.int32),
            np.full(len(np.arange(4, shape[1] - 4, 6)), 5, dtype=np.int32),
        ],
        axis=-1,
    )[None, ...]

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32)

    solver = PropTorch(
        DASElastic(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "szz"],
        receiver_type=["exx", "ezz", "das35", "das54x"],
        abcn=4,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 4)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    loss = out.pow(2).mean()
    loss.backward()
    assert vp.grad is not None and torch.isfinite(vp.grad).all()
    assert vs.grad is not None and torch.isfinite(vs.grad).all()


def test_das_elastic_3d_field_metadata():
    equation = DASElastic3D(spatial_order=2, backend="torch")
    assert equation.models == ["vp", "vs", "rho"]
    assert equation.default_source_fields == ["sxx", "syy", "szz"]
    assert equation.default_receiver_fields == ["exx", "eyy", "ezz"]
    assert "das54z" in equation.wavefields
    assert "m_sxx_xf" in equation.wavefields
    assert "m_tzz_yb" in equation.wavefields


def test_das_elastic_3d_eager_forward_smoke():
    device = torch.device("cpu")
    nt = 8
    dt = 0.001
    shape = (8, 8, 8)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.003).reshape(1, 1, nt)
    sources = np.array([[[4, 4, 3]]], dtype=np.int32)
    receivers = np.array([[[3, 3, 3], [5, 5, 3]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32)
    vs = torch.full(shape, 1200.0, dtype=torch.float32)
    rho = torch.full(shape, 2100.0, dtype=torch.float32)

    solver = PropTorch(
        DASElastic3D(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "syy", "szz"],
        receiver_type=["exx", "eyy", "ezz", "das35", "das54x"],
        abcn=2,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 5)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0


def _assert_nonzero_finite_gradient(*models):
    for model in models:
        assert model.grad is not None
        assert torch.isfinite(model.grad).all()
        assert model.grad.abs().max() > 0


def test_das_final_channels_backward_2d():
    device = torch.device("cpu")
    nt = 20
    dt = 0.001
    shape = (24, 28)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, 1, nt)
    sources = np.array([[[shape[1] // 2, 5]]], dtype=np.int32)
    receivers = np.array([[[8, 6], [14, 6], [20, 8], [14, 12]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, requires_grad=True)

    solver = PropTorch(
        DASElastic(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "szz"],
        receiver_type=["das35", "das54x", "das54z"],
        abcn=4,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 3)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    out.pow(2).mean().backward()
    _assert_nonzero_finite_gradient(vp, vs, rho)


def test_das_final_channels_backward_3d():
    device = torch.device("cpu")
    nt = 14
    dt = 0.001
    shape = (10, 10, 10)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, 1, nt)
    sources = np.array([[[5, 5, 5]]], dtype=np.int32)
    receivers = np.array([[[4, 5, 5], [6, 5, 5], [5, 4, 5], [5, 5, 6]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, requires_grad=True)

    solver = PropTorch(
        DASElastic3D(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "syy", "szz"],
        receiver_type=["das35", "das54x", "das54y", "das54z"],
        abcn=2,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 4)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    out.pow(2).mean().backward()
    _assert_nonzero_finite_gradient(vp, vs, rho)
