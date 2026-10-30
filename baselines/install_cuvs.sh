#!/bin/bash
source ~/.local/bin/env; cd ~/gpuvec; source .venv/bin/activate
export UV_HTTP_TIMEOUT=900; uv pip install cuvs-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com 2>&1 | tail -6
python -c "import cuvs, cupy; from cuvs.neighbors import cagra; print('cuvs', cuvs.__version__)" 2>&1 | tail -2
