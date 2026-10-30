#!/bin/bash
set -e
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
source ~/.local/bin/env
mkdir -p ~/gpuvec && cd ~/gpuvec
[ -d .venv ] || uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install faiss-gpu-cu12 numpy pyarrow h5py 2>&1 | tail -2
python -c 'import faiss; print("faiss", faiss.__version__, "gpus", faiss.get_num_gpus())'
