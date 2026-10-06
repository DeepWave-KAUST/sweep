"""The gradient matrix's eager column must be the uncompiled path.

``test/backend_gradient_matrix.py`` runs every case in one process.  Its eager
solvers used to take the compiled default; they all share one ``step_func``
code object, so from the 11th case on Dynamo hit its recompile limit and the
rest of the reference column silently ran uncompiled.  The reference is pinned
to the uncompiled path, like those of ``gate/bitgate.py`` and
``solver_gradient_mode_suite.py``.
"""
import pytest
import torch

from backend_gradient_matrix import EAGER_MODES, make_eager_solver
from gradient_cases import CASES, CUDA_SUITE_CONFIG


@pytest.mark.parametrize("mode", sorted(EAGER_MODES))
def test_eager_reference_is_uncompiled(mode):
    solver = make_eager_solver(CASES[0], mode, torch.device("cpu"), CUDA_SUITE_CONFIG)
    assert solver.use_compile is False
