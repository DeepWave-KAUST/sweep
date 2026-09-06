#!/bin/bash
cd /home/wangs0j/sweep-local/dd-refactor
./gate/run_gate.sh A base_A.pt C base_C.pt T base_T.pt dd1 base_dd1.pt
echo GATES_RC=$?
. gate/env.sh
for sub in elastic das_mu acoustic vrz; do
  echo "--- B --only $sub ---"
  "$PY" gate/bitgate.py --tier B --only $sub --compare gate/base_B.pt 2>&1 | tail -2
done
"$PY" gate/evr_ab.py --out gate/evr_reorder.pt --compare gate/evr_base.pt
echo EVR_RC=$?
