"""JAX must not preallocate the GPU in the test process.

torch and jax share one pytest process. JAX's GPU client takes 75% of the
device on its first op and keeps it, so the jax parity check in
test_acoustic_vti_1st left torch ~7.5 GiB of a 32 GiB V100 for the rest of the
run, and the 3-D elastic c tests after it OOMed -- a different handful every
run, none of them when run alone. conftest turns preallocation off; this pins it.

The GPU check runs in a fresh process: in this one an earlier test may already
have brought JAX up, and then whatever it grabbed is already gone.
"""
import os
import subprocess
import sys
import textwrap

import pytest


def test_conftest_turns_jax_preallocation_off():
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"


_PROBE = textwrap.dedent(
    """
    import torch, jax, jax.numpy as jnp
    torch.zeros(1, device="cuda")
    free0, total = torch.cuda.mem_get_info()
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("nogpu")
        raise SystemExit(0)
    jax.device_put(jnp.zeros(1), gpu).block_until_ready()
    free1, _ = torch.cuda.mem_get_info()
    print(free0 - free1, total)
    """
)


def test_jax_gpu_client_leaves_torch_the_device():
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    last = out.stdout.split()
    if last == ["nogpu"]:
        pytest.skip("this jax has no GPU backend")
    taken, total = (int(v) for v in last[-2:])
    # Preallocation takes 0.75 * total; without it JAX's first op costs its
    # context and a few MiB.
    assert taken < 0.25 * total, (
        f"JAX's first GPU op took {taken / 2**30:.2f} of {total / 2**30:.2f} GiB"
    )
