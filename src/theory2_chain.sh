#!/bin/bash
cd /d/Fanar
for e in rotpair cond cover iso ident2; do
  echo "=== $e"; .venv/Scripts/python -W ignore -u src/theory2.py $e > results/theory2_$e.log 2>&1; tail -n 3 results/theory2_$e.log
done
echo CHAIN_DONE
