#!/usr/bin/env bash
# Push the machine-INDEPENDENT gate files to the ibex worktree.
#
# Why this exists: gate/ lives in two places (this box and the ibex worktree),
# both get edited, and nothing kept them in step. A stale ibex copy of
# check_equations_api.py -- missing one entry from its ALLOWED_NEW list -- turned
# an otherwise clean multi-GPU verification into RC=1. A stale copy in the other
# direction would do the opposite and pass something it should have failed.
#
# Machine-DEPENDENT files are NOT synced: the .sbatch scripts and _common.sh are
# ibex-only, and the .pt baselines are per-GPU-model and per-run.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=/ibex/user/wangs0j/dd-refactor-ibex/gate

FILES=(bitgate.py ddgate.py check_equations_api.py golden_equations_api.json
       run_gate.sh noise_floors.json)

for f in "${FILES[@]}"; do
    [ -f "gate/$f" ] || { echo "FATAL: gate/$f missing locally"; exit 1; }
    if cmp -s "gate/$f" "$REMOTE/$f" 2>/dev/null; then
        printf '  same     %s\n' "$f"
    else
        cp "gate/$f" "$REMOTE/$f"
        printf '  UPDATED  %s\n' "$f"
    fi
done
# Print the stamp the sbatch scripts echo, so a mismatch is visible in the log.
echo "gate-stamp: $(cat "${FILES[@]/#/gate/}" | md5sum | cut -c1-12)"
