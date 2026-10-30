#!/bin/bash
export FANAR_ROOT="${FANAR_ROOT:-$(pwd)}"   # repo root (run from it)
# Windows-side reviewer experiments, run one after another (GPU is not shared with any other job while these run)
cd /d/Fanar
for e in zsweep k100 batch grid seeds encoder scale platform; do
  echo "=== $e"; .venv/Scripts/python -W ignore -u src/rev_fanardb.py $e 2>&1 | tail -n 400 > results/rev_fanardb_$e.log; tail -2 results/rev_fanardb_$e.log
done
for e in ee; do echo "=== theory $e"; .venv/Scripts/python -W ignore -u src/rev_theory.py $e > results/rev_theory_$e.log 2>&1; tail -2 results/rev_theory_$e.log; done

echo "=== platform under WSL2"; powershell.exe -NoProfile -Command "wsl -e bash $FANAR_ROOT/baselines/cuvs/fanar_wsl.sh" > /d/Fanar/results/rev_fanardb_platform_wsl.log 2>&1; tail -3 /d/Fanar/results/rev_fanardb_platform_wsl.log
echo CHAIN2_DONE
for ds in t3 sparse; do for r in rot norot; do echo "=== cfail $ds $r"; cd /d/Fanar; .venv/Scripts/python -W ignore -u src/rev_cfail.py $ds 50000 1 $r > results/rev_cfail_${ds}_$r.log 2>&1; tail -1 results/rev_cfail_${ds}_$r.log; done; done
echo CHAIN3_DONE
echo "=== highrecall R384"; cd /d/Fanar; FANAR_R=384 .venv/Scripts/python -W ignore -u src/rev_fanardb.py highrecall > results/rev_fanardb_highrecall_R384.log 2>&1; tail -2 results/rev_fanardb_highrecall_R384.log
echo CHAIN4_DONE
