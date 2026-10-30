# FanarVSE-paper

Research archive: code and raw results for the paper **"FanarVSE: Fast, Memory-Light GPU Vector Search with Error Bars from Coding Theory"** (Mohammed Mohsen, Fayoum University).

FanarVSE is a GPU vector index that keeps about 570 bytes per vector on the GPU (under 8 bytes in its host-resident mode) and reads full-precision vectors only for the few candidates whose error bars overlap the top-k boundary. On one million OpenAI embeddings and a 6 GB laptop GPU it reaches recall@10 0.992 at 22,649 queries per second; its optional throughput mode reaches 0.99 at 83,600 queries per second at a batch of 2,000.

This repository is the reproducibility archive of the paper: the experiment scripts and raw results exactly as they produced the published numbers. It is not maintained and does not take pull requests. For the FanarVSE library, which is developed and accepts contributions, see the **FanarVSE** repository.

## Repository layout

| Folder | Contents | Paper |
|---|---|---|
| `src/` | The engine (trellis codes, cell-residual stage 1, threshold-and-emit selection, tiered refinement, zero-copy stage 3, NVMe reader) and the scripts that measure it | Sections 3–6, most tables |
| `throughput/` | Throughput mode FanarVSE-T: int8 stage 1 on tensor cores, lookup-table stage 2 | Sections 5.1, 6.6; Supplementary Note "Throughput mode" |
| `host_resident/` | Mode with every per-vector array in host RAM (under 8 B/vector on the GPU), memory accounting | Section 6.7 |
| `graph/` | Optional graph front-end (pruned kNN graph, CUDA beam search) | Section 6.7, Table 10 |
| `baselines/` | CAGRA, IVF-RaBitQ, IVF-PQ and FAISS-GPU baselines, run in WSL2 / Linux with cuVS | Sections 6.1–6.9 |
| `figures/` | Scripts that draw every figure from `results/` | Figures 1–6 |
| `results/` | Raw JSON and log output behind every number in the paper | — |
| `paper/` | LaTeX source, figures and PDFs of the paper and its supplementary information | — |

`src/hvb.py` is kept only for its `Prep` data-loading helper, which three experiment scripts use.

## Main entry points

| What | Command (run from the repository root) |
|---|---|
| Final engine, 1M OpenAI vectors: recall/QPS grid, latency, k=100, z sweep, seeds, lean and NVMe variants | `python src/v19_final.py {sweep,lat,k100,z,seeds,lean,nvme}` |
| Final engine on the other datasets | `python src/v19_final.py data {oai3,arxiv,arxivL,siglip,dino}` |
| FanarVSE-v1 measurements, ablation, rotations, error-bar width | `python src/rev_fanardb.py {grid,zsweep,rot,seeds,k100,batch,encoder,scale,platform}` |
| Direct tests of the theory | `python src/theory2.py {law,rotpair,cond,cover,iso,ident2}` |
| Lattice codebooks, codebook order | `python src/exp35_lattice_rabitq_general.py`, `python src/rev_cfail.py` |
| Trellis codes | `python src/trellis.py`, `python src/trellis_recall.py` |
| Throughput mode, final frontier at batch B | `python throughput/e_v3final.py B` |
| Throughput mode, GPU memory | `python throughput/e_mem.py` |
| Host-resident mode | `python host_resident/space_x.py` |
| Graph front-end | `python graph/knn.py oai1m`, `python graph/rng.py ...`, then `python graph/gpipe.py` |
| Baselines (WSL2 / Linux) | `bash baselines/setup.sh`, `bash baselines/install_cuvs.sh`, then the `baselines/run_*.sh` and `baselines/cuvs/*.py` scripts |
| Figures | `python figures/make_figs.py`, `rev_figs.py`, `fig_1m_v4.py`, `theory_figs.py`, `make_fig4.py`; output in `figures/out/` |

Every script reads and writes paths relative to the repository root, so run it from there, or set `FANAR_ROOT` to the root.

## Setup

Tested on Windows 10 with an NVIDIA RTX 3050 Laptop GPU (6 GB), Python 3.12, PyTorch 2.6 (CUDA 12.4) and Triton 3.2 (`triton-windows`). The baselines ran in WSL2 Ubuntu on the same GPU with cuVS 26.10 and FAISS-GPU 1.14.

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The NVMe reader (`src/native/nvme_reader.c`) is Windows-only; `src/nvme_async.py` compiles it on first use with the TCC compiler that ships with Triton.

## Data

The datasets are not included. Put them in `data/` (or link `data/` to where they live) and run the preparation scripts, which write normalised fp16 vectors, 2,000 held-out queries and exact top-10/100 ground truth:

| Dataset | Source | Preparation |
|---|---|---|
| DBpedia OpenAI text-embedding-ada-002, 998k × 1536 | Hugging Face `KShivendu/dbpedia-entities-openai-1M`, the 26 parquet files saved as `data/dbp/shard00.parquet` … `shard25.parquet` | `python src/prep_1m.py` |
| DBpedia OpenAI text-embedding-3-large, 98k × 1536 | `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K` → `data/oai3/*.parquet` | `python src/v13_prep.py` |
| arXiv InstructorXL titles, 98k and 2.25M × 768 | `Qdrant/arxiv-titles-instructorxl-embeddings` → `data/arxiv/*.parquet` | `python src/v13_prep.py`, `python src/v16_prep.py arxivL` |
| SigLIP, ImageNet-1k, 1.28M × 768 | `mlnomad/imagenet-1k-224-clip-embeddings` | `python src/v16_prep.py siglip` |
| DINOv2-giant, Met open access, 258k × 1536 | `metmuseum/openaccess-embeddings-dinov2-giant` | `python src/v16_prep.py dino` |
| GloVe-100 (angular), 1.18M × 100 | ann-benchmarks `glove-100-angular.hdf5` → `data/` | read directly |

The scripts cache built indexes in `data/` and `cache/`; both folders are ignored by git.

Script and result file names such as `src/rev_fanardb.py` and `results/rev_fanardb_*.json` keep the system's earlier working name (`fanardb`); they are the files that produced the paper's numbers.

## License

Apache License 2.0, see `LICENSE`.

## Citation

If you use this code, please cite the paper (full reference to be added on publication):

```
Mohammed Mohsen. FanarVSE: Fast, Memory-Light GPU Vector Search with Error Bars from Coding Theory.
```

## Contact

Mohammed Mohsen, Information Systems Department, Faculty of Computers and Artificial Intelligence, Fayoum University — mm5987@fayoum.edu.eg
