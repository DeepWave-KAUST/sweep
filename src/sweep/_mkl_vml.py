"""Settle oneMKL's VML CPU dispatch before any multi-threaded CPU math.

PyTorch's x86 wheels link oneMKL statically into ``libtorch_cpu.so``; on CPU
float tensors ``torch.log`` / ``sqrt`` / ``exp`` / ``sin`` / ``tanh`` / ``erf`` ...
run MKL VML (``vmsLn``, ``vmsSqrt``, ...).  Every VML call picks its kernel
through ``mkl_vml_serv_cpu_detect``, which caches the CPU type in an unlocked
static.  Its first call stores the *raw* MKL CPU type and only then the remapped
VML table column (oneMKL 2024.2 in torch 2.9.1: 9, then 5 on AVX-512).  A
thread that reads the static in between indexes the kernel table with the raw
value and runs another kernel: on AVX-512 the AVX2 "enhanced performance" one
-- sqrt off by up to 3e-4 relative, exp by 1.5e-4, ln by 31 ulp.

torch splits a VML op over the OpenMP threads above 2048 elements, so the first
such op of a process -- a model the caller derives (``vp = sqrt(H / rho)``),
ElasticVRR's ``log(rho * vp)``, ViscoElastic's moduli ``sqrt`` -- now and then
hands a worker's chunk to that kernel: a different model, hence a different
record and gradient.  Measured with torch 2.14.1 at 4 threads on a Xeon Gold
6148: one fresh process in five; never on AMD EPYC, where MKL dispatches
otherwise.  One serial call (a 1-element tensor is below torch's grain, so it
runs on the calling thread alone) completes the detection first; every later
call reads the settled value.  The kernel a settled process runs is the one an
unraced process ran before, so results do not change.  Math a process runs
before it imports sweep is out of its reach.
"""
from __future__ import annotations

_settled = False


def settle_mkl_vml_dispatch(force: bool = False) -> None:
    """Run oneMKL's VML CPU detection once, serially, on this thread.

    Idempotent and cheap (a few microseconds); a no-op where torch is missing or
    does not use MKL VML.  ``force`` repeats it (tests)."""
    global _settled
    if _settled and not force:
        return
    _settled = True
    try:
        import torch

        with torch.no_grad():
            for dtype in (torch.float32, torch.float64):
                torch.log(torch.ones(1, dtype=dtype))
    except Exception:  # pragma: no cover - never let a warm-up break an import
        pass
