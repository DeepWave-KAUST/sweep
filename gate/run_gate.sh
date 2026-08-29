#!/usr/bin/env bash
# The single entry point for running gates. Use this instead of invoking
# bitgate/ddgate by hand.
#
# Why it exists: a gate piped through `head` gets SIGPIPEd the moment head has
# its N lines, so it dies part-way through -- and because a pipeline's exit
# status comes from the LAST command, `head` succeeding makes the truncated run
# look successful. That happened once here: tier A ran 20 of 27 configs, tier C
# 8 of 30, neither produced a verdict line, and the wrapper still printed DONE.
# A gate that can fail silently is worse than no gate, because it is trusted.
#
# So: never pipe, always write the full output to a file, and treat "produced no
# verdict line" as a FAILURE rather than as nothing to report.
#
#   gate/run_gate.sh A base_A.pt [C base_C.pt ...]        # pairs of tier/baseline
#   gate/run_gate.sh dd1 base_dd1.pt                      # dd1/dd2 use ddgate
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f gate/env.sh ] && . gate/env.sh

# Delete every output this invocation will write, BEFORE writing any of them.
# Otherwise a tier that crashes (or is still running when someone looks) leaves
# the PREVIOUS run's file in place, and a stale green is indistinguishable from
# a fresh one. Reading `run_A.out` from the last round and calling it this
# round's result is exactly the kind of mistake a gate exists to prevent.
for t in "$@"; do
    case "$t" in *.pt) ;; *) rm -f "gate/run_${t}.out" ;; esac
done

rc=0
while [ $# -gt 0 ]; do
    tier=$1; base=$2; shift 2
    out="gate/run_${tier}.out"
    case "$tier" in
        dd*)  "$PY" gate/ddgate.py --compare "gate/$base" > "$out" 2>&1 ;;
        *)    "$PY" gate/bitgate.py --tier "$tier" --compare "gate/$base" > "$out" 2>&1 ;;
    esac
    exit_code=$?

    verdict=$(grep -E "^PASS [0-9]+" "$out" | head -1)
    # Progress lines look like "[ 3/27] .  1.2s cfg" -- require the
    # status mark so a JIT rebuild's ninja "[N/48] nvcc ..." lines
    # (first gate after an extension wipe) do not inflate the count.
    ran=$(grep -cE "^\[ *[0-9]+/[0-9]+\] +[.EF]" "$out")
    if [ -z "$verdict" ]; then
        echo "GATE $tier: NO VERDICT LINE after $ran configs -- the run did not"
        echo "  complete (crash, OOM, or a truncated pipe). This is a FAILURE."
        tail -5 "$out" | sed 's/^/    /'
        rc=1
        continue
    fi
    # The verdict must account for every config that ran; if it does not, the
    # comparison itself was cut short.
    # ran == PASS+FAIL+MISSING+NEW is the only automatic check that the run was
    # not cut short; NEW must be in the sum or an added config breaks it.
    counted=$(echo "$verdict" | grep -oE "(PASS|FAIL|MISSING|NEW) [0-9]+" \
              | grep -oE "[0-9]+" | paste -sd+ | bc)
    echo "GATE $tier: exit=$exit_code  ran=$ran  $verdict"
    if [ "$exit_code" -ne 0 ]; then
        grep -A3 "^FAIL" "$out" | head -40 | sed 's/^/    /'
        rc=1
    fi
    if [ -n "$counted" ] && [ "$counted" -ne "$ran" ]; then
        echo "  TRUNCATED: ran $ran configs but the verdict accounts for $counted"
        rc=1
    fi
done

echo "GATE-RC=$rc"
exit $rc
