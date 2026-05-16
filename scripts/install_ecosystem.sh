#!/usr/bin/env bash
#
# install_ecosystem.sh — editable install for the sweep + companion-repo
# ecosystem when no companion is on PyPI yet.
#
# Usage:
#     ./scripts/install_ecosystem.sh                          # install everything in dep order
#     ./scripts/install_ecosystem.sh --repo-root <path>       # override repo root (default: /ibex/user/wangs0j/repo)
#     ./scripts/install_ecosystem.sh --dry-run                # print pip commands, don't execute
#     ./scripts/install_ecosystem.sh --skip-sweep             # don't touch the sweep base install
#
# Why this exists:
#     `pip install sweep[full]` resolves the companion package names from
#     PyPI. None of them are published yet, so it would fail. Local editable
#     installs satisfy each repo's `dependencies = ["sweep", ...]` declaration
#     from the *currently installed* environment, so we install bottom-up.
#
# Dependency order (each repo's `pyproject.toml` -> `dependencies`):
#     [no sweep dep]  sweep-io, sweep-preproc, sweep-loss, sweep-opt, sweep-viz, sweep-nn
#     [sweep-io dep]  sweep-zoo
#     [sweep dep]     sweep-runner, sweep-tasks

set -euo pipefail

REPO_ROOT="/ibex/user/wangs0j/repo"
SWEEP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
SKIP_SWEEP=0

while (( "$#" )); do
    case "$1" in
        --repo-root)
            REPO_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-sweep)
            SKIP_SWEEP=1
            shift
            ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

run() {
    echo "+ $*"
    if (( DRY_RUN == 0 )); then
        "$@"
    fi
}

install_one() {
    local pkg="$1"
    local path="$REPO_ROOT/$pkg"
    if [[ ! -d "$path" ]]; then
        echo "skip $pkg — directory $path does not exist" >&2
        return 0
    fi
    if [[ ! -f "$path/pyproject.toml" ]]; then
        echo "skip $pkg — no pyproject.toml at $path" >&2
        return 0
    fi
    echo
    echo "=== $pkg ==="
    run pip install -e "$path"
}

# 0. sweep itself (the engine) — base install only, no extras.
if (( SKIP_SWEEP == 0 )); then
    echo
    echo "=== sweep (base, no CUDA) ==="
    run pip install -e "$SWEEP_ROOT"
fi

# 1. Tier-2 framework-agnostic primitives — no sweep dep.
install_one sweep-io
install_one sweep-preproc
install_one sweep-loss
install_one sweep-opt
install_one sweep-viz
install_one sweep-nn

# 2. Tier-3 integration (needs sweep-io / sweep).
install_one sweep-zoo
install_one sweep-runner
install_one sweep-tasks

echo
echo "ecosystem editable install complete."
echo
echo "smoke-check (companion namespace aliasing):"
echo "  python -c 'import sweep; [getattr(sweep, n) for n in (\"io\",\"preproc\",\"loss\",\"opt\",\"runner\",\"tasks\",\"viz\",\"nn\",\"zoo\")]; print(\"ok\")'"
echo
echo "Recommended import idiom — prefer the unified namespace:"
echo "  from sweep.tasks import TaskRunner    # not 'from sweep_tasks import ...'"
echo "  from sweep.io.segy import SEGYReader"
echo "  from sweep.runner.multiscale import FrequencyBand"
