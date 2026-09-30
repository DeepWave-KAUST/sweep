"""The compiled core launches on sweep::current_stream(); the binding sets it
per entry from torch's current stream for the input's device.  This pins that
wiring: the probe returns the handle the core sees, and it must be torch's
current stream both on the default stream and under ``torch.cuda.stream(s)``.

Without the StreamScope in the binding every launch would silently land on the
legacy default stream -- correct on the default stream, racy under a side
stream, and invisible to every bit-exact gate (they all run on the default
stream).  So the check is on the handle, not on numbers.
"""
import pytest

torch = pytest.importorskip("torch")


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _binding_ready(), reason="needs the CUDA binding")


def _forward_input(device):
    import sweep._C as _C
    fi = _C.ForwardInput()
    fi.models = [torch.zeros(1, 1, 8, 8, device=device)]
    return fi


def test_core_sees_torch_default_stream():
    import sweep._C as _C
    dev = torch.device("cuda", torch.cuda.current_device())
    fi = _forward_input(dev)
    assert _C._core_stream_for(fi) == torch.cuda.current_stream(dev).cuda_stream


def test_core_sees_torch_side_stream():
    import sweep._C as _C
    dev = torch.device("cuda", torch.cuda.current_device())
    fi = _forward_input(dev)
    s = torch.cuda.Stream(device=dev)
    with torch.cuda.stream(s):
        seen = _C._core_stream_for(fi)
    assert seen == s.cuda_stream
    assert seen != torch.cuda.default_stream(dev).cuda_stream
    # and it is restored: back on the default stream after the scope
    assert _C._core_stream_for(fi) == torch.cuda.current_stream(dev).cuda_stream


def test_core_stream_is_nullptr_for_host_models():
    import sweep._C as _C
    fi = _C.ForwardInput()
    fi.models = [torch.zeros(1, 1, 8, 8)]
    assert _C._core_stream_for(fi) == 0
