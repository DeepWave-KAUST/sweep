import numpy as np
import pytest
import torch
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from sweep.equations import (
    Elastic,
    Elastic3D,
    build_elastic_das_receivers,
    elastic_velocity_record_to_das,
)
from sweep.propagator.torch import PropTorch


def _ricker(nt, dt, fm=15.0, delay=0.006):
    t = np.arange(nt, dtype=np.float32) * np.float32(dt) - np.float32(delay)
    arg = np.pi * np.float32(fm) * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def _assert_nonzero_finite_gradient(*models):
    for model in models:
        assert model.grad is not None
        assert torch.isfinite(model.grad).all()
        assert model.grad.abs().max() > 0


def _load_reproduce_layered_das():
    path = Path(__file__).resolve().parents[1] / "examples" / "wavefields" / "das" / "reproduce_layered_das.py"
    spec = importlib.util.spec_from_file_location("reproduce_layered_das_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_elastic_derived_das_2d_maps_velocity_derivatives():
    receivers = np.array([[[5, 7], [8, 9]]], dtype=np.int32)
    receiver_map = build_elastic_das_receivers(receivers, spatial_order=2)
    augmented = receiver_map.augmented_receivers[0]

    record = torch.zeros((1, 1, augmented.shape[0], 2), dtype=torch.float32, requires_grad=True)
    record_data = record.detach().clone()
    record_data[0, 0, :, 0] = torch.as_tensor(augmented[:, 0], dtype=torch.float32) ** 2
    record_data[0, 0, :, 1] = 2.0 * torch.as_tensor(augmented[:, 1], dtype=torch.float32)
    record = record_data.requires_grad_()

    das, fields = elastic_velocity_record_to_das(record, receiver_map, dh=1.0)

    exx = das[0, 0, :, fields.index("exx")]
    ezz = das[0, 0, :, fields.index("ezz")]
    assert torch.allclose(exx, torch.tensor([9.0, 15.0]))
    assert torch.allclose(ezz, torch.tensor([2.0, 2.0]))

    das[..., fields.index("das35")].sum().backward()
    assert record.grad is not None
    assert record.grad.abs().max() > 0


def _run_2d_backward(backend, device):
    nt = 24
    dt = 0.001
    shape = (24, 28)
    dh = 10.0
    wavelet = torch.as_tensor(_ricker(nt, dt).reshape(1, 1, nt), dtype=torch.float32, device=device)
    sources = np.array([[[shape[1] // 2, 6]]], dtype=np.int32)
    receivers = np.array([[[8, 7], [14, 8], [20, 9]]], dtype=np.int32)
    receiver_map = build_elastic_das_receivers(receivers, spatial_order=2)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, device=device, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, device=device, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, device=device, requires_grad=True)

    solver = PropTorch(
        Elastic(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "szz"],
        receiver_type=list(receiver_map.elastic_receiver_type),
        abcn=4,
        dh=dh,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
        backend=backend,
    )

    velocity_record = solver(
        wavelet,
        sources=sources,
        receivers=receiver_map.augmented_receivers,
        models=[vp, vs, rho],
    )
    das, fields = elastic_velocity_record_to_das(
        velocity_record,
        receiver_map,
        dh=dh,
        receiver_type=["das35", "das54x", "das54z"],
    )

    assert das.shape == (1, nt, receivers.shape[1], 3)
    assert fields == ("das35", "das54x", "das54z")
    assert torch.isfinite(das).all()
    assert das.abs().max() > 0

    das.pow(2).mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _assert_nonzero_finite_gradient(vp, vs, rho)


def test_elastic_derived_das_2d_eager_backward():
    _run_2d_backward("eager", torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_elastic_derived_das_2d_cuda_backward():
    _run_2d_backward("cuda", torch.device("cuda:0"))


def _run_3d_backward(backend, device):
    nt = 10
    dt = 0.001
    shape = (8, 8, 8)
    dh = 10.0
    wavelet = torch.as_tensor(_ricker(nt, dt).reshape(1, 1, nt), dtype=torch.float32, device=device)
    sources = np.array([[[4, 4, 4]]], dtype=np.int32)
    receivers = np.array([[[3, 4, 4], [4, 3, 4], [4, 4, 5]]], dtype=np.int32)
    receiver_map = build_elastic_das_receivers(receivers, spatial_order=2)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, device=device, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, device=device, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, device=device, requires_grad=True)

    solver = PropTorch(
        Elastic3D(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "syy", "szz"],
        receiver_type=list(receiver_map.elastic_receiver_type),
        abcn=2,
        dh=dh,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
        backend=backend,
    )

    velocity_record = solver(
        wavelet,
        sources=sources,
        receivers=receiver_map.augmented_receivers,
        models=[vp, vs, rho],
    )
    das, fields = elastic_velocity_record_to_das(
        velocity_record,
        receiver_map,
        dh=dh,
        receiver_type=["das35", "das54x", "das54y", "das54z"],
    )

    assert das.shape == (1, nt, receivers.shape[1], 4)
    assert fields == ("das35", "das54x", "das54y", "das54z")
    assert torch.isfinite(das).all()
    assert das.abs().max() > 0

    das.pow(2).mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _assert_nonzero_finite_gradient(vp, vs, rho)


def test_elastic_derived_das_3d_eager_backward():
    _run_3d_backward("eager", torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_elastic_derived_das_3d_cuda_backward():
    _run_3d_backward("cuda", torch.device("cuda:0"))


def _run_reproduce_solver_backward(backend):
    module = _load_reproduce_layered_das()
    nt = 18
    dt = 0.001
    shape = (24, 28)
    args = SimpleNamespace(
        device="cuda:0" if backend == "cuda" else "cpu",
        spatial_order=2,
        nz=shape[0],
        nx=shape[1],
        abcn=4,
        dh=10.0,
        dt=dt,
        check_backward=True,
    )
    geometry = {
        "source": np.array([[shape[1] // 2, 6]], dtype=np.int32),
        "receivers": np.array([[8, 7], [14, 8], [20, 9]], dtype=np.int32),
        "slices": {},
    }
    models = (
        np.full(shape, 2200.0, dtype=np.float32),
        np.full(shape, 1200.0, dtype=np.float32),
        np.full(shape, 2100.0, dtype=np.float32),
    )
    wavelet = _ricker(nt, dt).reshape(1, 1, nt)

    records, channels, elapsed = module.run_elastic_derived_das_solver(
        backend=backend,
        geometry=geometry,
        models=models,
        wavelet=wavelet,
        elastic_receiver_fields=["vx", "vz"],
        das_receiver_fields=["exx", "ezz"],
        args=args,
    )
    assert records.shape == (3, nt, 4)
    assert channels == {"vx": 0, "vz": 1, "exx": 2, "ezz": 3}
    assert elapsed > 0
    assert np.isfinite(records).all()
    assert np.abs(records).max() > 0


def test_reproduce_layered_das_solver_eager_backward():
    _run_reproduce_solver_backward("eager")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_reproduce_layered_das_solver_cuda_backward():
    _run_reproduce_solver_backward("cuda")
