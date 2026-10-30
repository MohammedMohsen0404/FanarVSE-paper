#!/bin/bash
export FANAR_ROOT="${FANAR_ROOT:-$(pwd)}"   # repo root (run from it)
bash $FANAR_ROOT/baselines/cuvs/run.sh $FANAR_ROOT/baselines/cuvs/faiss_k100.py
bash $FANAR_ROOT/baselines/cuvs/run.sh $FANAR_ROOT/baselines/cuvs/cagra_diag.py
bash $FANAR_ROOT/baselines/cuvs/rabitq_verify.sh
