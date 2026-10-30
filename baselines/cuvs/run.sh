#!/bin/bash
export FANAR_ROOT="${FANAR_ROOT:-$(pwd)}"   # repo root (run from it)
# usage: run.sh script.py [args]   -> runs in the gpuvec venv
source ~/.local/bin/env; cd ~/gpuvec; source .venv/bin/activate
export PYTHONPATH=$FANAR_ROOT/baselines/cuvs
python -u "$@" 2>&1 | grep -v -i warn
