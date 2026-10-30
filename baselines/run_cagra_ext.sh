#!/bin/bash
export FANAR_ROOT="${FANAR_ROOT:-$(pwd)}"   # repo root (run from it)
source ~/.local/bin/env; cd ~/gpuvec; source .venv/bin/activate
for ds in glove dbp; do python -u $FANAR_ROOT/baselines/cagra_ext.py $ds 2>&1 | grep -v -i warn; done
