"""The first compile of a process can run inside a checkpointed CPU forward.

Dynamo saves and restores the CUDA RNG state around every trace when a GPU is
visible, which initializes CUDA on the first trace of the process -- of a step on
CPU tensors too -- and ``torch.utils.checkpoint`` raises when CUDA comes up in its
forward ("PyTorch's device state was initialized in the forward pass of a
Checkpoint"). The eager default, ``memory=Ckpt`` with torch.compile on, compiles
inside the first checkpointed chunk, so the first forward of a CPU process in grad
mode failed where nothing had asked for the GPU before: no shipped core (its fit
check reads the GPU's architecture) and TORCH_CUDA_ARCH_LIST set (else the build
check does). The second call passed, CUDA being up by then.

That takes a fresh process with CUDA down, a GPU visible and a real torch.compile, so
each rollout runs in a subprocess, with SWEEP_CORE naming no file (no shipped core)
and TORCH_CUDA_ARCH_LIST set. The child checks that CUDA is still down when the
forward starts: if construction brought it up, there is nothing to test.
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from conftest import SRC, requires_compile

_CHILD = textwrap.dedent(
    """
    import sys
    import numpy as np
    import torch
    from sweep.propagator import Ckpt, EagerOptions

    nt = 24
    w = torch.zeros(nt)
    w[2] = 1.0
    on = EagerOptions(use_compile=True)
    if sys.argv[1] == "fd":
        from sweep.equations import Acoustic
        from sweep.propagator import PropTorch
        shape = (30, 40)
        p = PropTorch(Acoustic(), impl="eager", eager_options=on, memory=Ckpt(chunks=8),
                      shape=shape, dh=10.0, dt=1e-3, nt=nt, abcn=10, dev="cpu")
        src, rec = torch.tensor([[20, 5]]), torch.tensor([[[5, 5], [35, 5]]])
    else:
        # two shots: the vmapped rollout, which torch.utils.checkpoint chunks
        from sweep.sem import AcousticSEM2D, PropSEM
        shape = (21, 33)
        p = PropSEM(AcousticSEM2D, impl="eager", eager_options=on, memory=Ckpt(chunks=8),
                    shape=shape, dh=10.0, dt=1e-3, n_order=4, abcn=8, free_surface=True,
                    f_dominant=15.0, dev="cpu")
        src = np.array([[12, 2.0], [20, 2.0]])
        rec = np.array([[[5, 2.0], [28, 2.0]]] * 2)
    vp = torch.full(shape, 2000.0, requires_grad=True)
    if torch.cuda.is_initialized():
        print("CUDA came up before the forward")
        raise SystemExit(3)
    p(w, src, rec, models=[vp]).square().sum().backward()
    print("grad", bool(torch.isfinite(vp.grad).all() and vp.grad.abs().sum() > 0))
    """
)


@requires_compile
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="without a visible GPU the trace leaves CUDA down")
@pytest.mark.parametrize("rollout", ["fd", "sem"])
def test_first_compile_of_the_process_in_a_checkpointed_cpu_forward(rollout, tmp_path):
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]),
               TORCH_CUDA_ARCH_LIST="8.0",
               SWEEP_CORE=str(tmp_path / "libsweep_core.so"))     # no file: no shipped core
    out = subprocess.run([sys.executable, "-c", _CHILD, rollout], capture_output=True,
                         text=True, env=env, timeout=600)
    assert out.returncode != 3, (
        "CUDA came up while the propagator was built, so the first compile no longer "
        "brings it up in the checkpointed forward: this test checks nothing as it stands")
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split()[-2:] == ["grad", "True"], out.stdout + out.stderr[-2000:]
