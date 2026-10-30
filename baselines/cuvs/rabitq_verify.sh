#!/bin/bash
export FANAR_ROOT="${FANAR_ROOT:-$(pwd)}"   # repo root (run from it)
for cfg in "9 1024" "8 1024" "7 1024" "9 4096" "8 4096" "7 4096" "6 1024"; do bash $FANAR_ROOT/baselines/cuvs/run.sh $FANAR_ROOT/baselines/cuvs/rabitq_verify.py $cfg; done
echo ALLDONE
