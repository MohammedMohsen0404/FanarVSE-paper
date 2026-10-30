#!/bin/bash
export FANAR_ROOT="${FANAR_ROOT:-$(pwd)}"   # repo root (run from it)
for ds in glove dbp; do bash $FANAR_ROOT/baselines/cuvs/run.sh $FANAR_ROOT/baselines/cuvs/small_ds.py $ds; done
echo SMALL_DONE
