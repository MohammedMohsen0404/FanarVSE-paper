#!/bin/bash
export FANAR_ROOT="${FANAR_ROOT:-$(pwd)}"   # repo root (run from it)
source ~/.local/bin/env; cd ~/gpuvec; source .venv/bin/activate
python -u $FANAR_ROOT/baselines/cagra_batch_hi.py 2>&1 | grep -v -i warn
