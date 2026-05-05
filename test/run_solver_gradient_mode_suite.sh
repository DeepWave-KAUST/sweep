#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/wangs0j/miniconda3/etc/profile.d/conda.sh
conda activate ifwitorch

python "${REPO_ROOT}/test/solver_gradient_mode_suite.py" "$@"
