"""3-D DAS has no compiled boundary-saving path, and must say so.

`das3d::backward_bs` re-runs the forward and allocates the whole
`{nt, 3, B, nz, ny, nx}` strain history -- what full storage holds -- while the
Python side allocated a 13-field boundary ring the forward never writes. Since
boundary saving is the implicit `impl='c'` default, every plain 3-D DAS gradient
paid strictly MORE than `'full'` for a "memory-saving" strategy. Declaring the
capability routes the default to `'full'` and makes an explicit request raise,
the same contract `ViscoAcoustic` already has.
"""
import pytest

from sweep.equations import DASMu3D, DASZhao, DASZhao3D
from sweep.propagator.torch import _normalize_cuda_memory_kwargs


def _resolve(equation, **knobs):
    return _normalize_cuda_memory_kwargs(dict(knobs), equation=equation)


def test_das3d_declares_that_it_has_no_boundary_saving():
    assert DASZhao3D.supports_boundary_saving_c is False


@pytest.mark.parametrize("cls", [DASZhao, DASMu3D])
def test_the_das_equations_that_do_have_it_are_untouched(cls):
    """das2d writes real strips; DASMu/DASMu3D get them from the staggered
    skeleton. Only the 3-D Zhao equation lacks the path."""
    assert cls.supports_boundary_saving_c is True


def test_the_implicit_default_becomes_full_storage():
    """No memory knob passed: the user gets full storage rather than a
    'saving' strategy that costs more than full."""
    merged = _resolve(DASZhao3D())
    assert merged["use_ckpt"] is False
    assert merged["boundary_saving_config"]["enabled"] is False


def test_an_explicit_boundary_request_raises_and_names_the_alternative():
    with pytest.raises(NotImplementedError) as exc:
        _resolve(DASZhao3D(), boundary_saving_config={"enabled": True})
    msg = str(exc.value)
    assert "DASZhao3D" in msg
    # Match the spelling a caller would actually type. The message used to say
    # "ckpt"/"full"; the typed strategies renamed those to Ckpt()/Full(), and a
    # lowercase substring test kept passing for neither.
    assert "Ckpt()" in msg or "Full()" in msg, msg


def test_checkpointing_is_still_offered():
    """Only the boundary path is withdrawn; ckpt still resolves."""
    merged = _resolve(DASZhao3D(), use_ckpt=True)
    assert merged["use_ckpt"] is True
