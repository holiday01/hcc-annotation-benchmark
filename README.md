# HCC scRNA-seq Cell Type Annotation Benchmark

Companion code for the manuscript on benchmarking cell type annotation
tools for hepatocellular carcinoma (HCC) single-cell RNA sequencing,
spanning reference-based classifiers, marker-database scoring, a Bayesian
marker-gene model, a large language model API, and a single-cell
foundation model, evaluated under both author-curated and database-derived
ground truth, with an independent inferCNV-based malignant-cell validation.

> **Note**: This repository contains analysis code only. Manuscript and
> figures will be released after acceptance. See `CITATION.cff` for how to
> cite.

---

## Repository structure

```
.
├── scripts/                  # Analysis pipeline (numbered execution order)
├── ground_truth/             # Author-curated, EnrichR snapshot labels, and CellAssign marker matrix
│   ├── GSE125449_ground_truth.tsv       (author-curated)
│   ├── GSE149614_ground_truth.tsv       (author-curated)
│   ├── GSE125449_enrichr_ground_truth.tsv
│   ├── GSE149614_enrichr_ground_truth.tsv
│   ├── GSE156625_enrichr_ground_truth.tsv
│   ├── GSE156625_enrichr_cluster_labels.tsv
│   ├── GSE162616_enrichr_ground_truth.tsv
│   ├── GSE202642_enrichr_ground_truth.tsv
│   ├── GSE223204_enrichr_ground_truth.tsv
│   └── cellassign_markers.tsv           (CellAssign marker matrix used in the paper)
├── cytetype_responses/       # Frozen per-cluster CyteType API JSON (4 validation datasets)
│   ├── GSE156625_cytetype_response.json
│   ├── GSE162616_cytetype_response.json
│   ├── GSE202642_cytetype_response.json
│   ├── GSE223204_cytetype_response.json
│   └── manifest.json
├── expected_results/         # CSVs the manuscript reports (for diff-style verification)
├── external/sc-type/         # Third-party ScType R scripts and marker DB
├── download_data.sh          # Fetch all 6 GEO datasets into ./data/
├── run_all.sh                # One-command driver for the full pipeline
├── install_r_deps.R          # Pinned R package installation
├── requirements.txt          # Pinned Python dependencies
├── CITATION.cff              # Citation metadata (Zenodo / GitHub)
├── LICENSE                   # MIT
└── README.md
```

The 6 EnrichR ground-truth TSVs in `ground_truth/` are **frozen snapshots**
of the EnrichR query results used in the manuscript. They are loaded by
default by `scripts/10_build_enrichr_groundtruth.py`; pass `--refresh` to
re-query the live EnrichR API (results may drift if upstream databases
have changed).

---

## Datasets

All data are publicly available from NCBI Gene Expression Omnibus (GEO).
The two-step `download_data.sh` script will pull every needed supplementary
file:

```bash
bash download_data.sh                  # all 6 datasets
bash download_data.sh GSE125449        # just one
```

| Dataset | GEO accession | Cells | Author labels |
|---|---|---:|---|
| GSE125449 | [GSE125449](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125449) | 9,752 | Yes |
| GSE149614 | [GSE149614](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149614) | 71,915 | Yes |
| GSE156625 | [GSE156625](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156625) | 73,589 | No |
| GSE162616 | [GSE162616](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162616) | 57,261 | No |
| GSE202642 | [GSE202642](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE202642) | 122,092 | No |
| GSE223204 | [GSE223204](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223204) | 23,165 | No |

By default `download_data.sh` puts everything under `./data/<GSE>/suppl/`.
Override the location with `BENCHMARK_DATA_DIR=/some/path bash download_data.sh`.

---

## Installation

### Python (3.11 tested, conda recommended)

```bash
conda create -n hcc-bench python=3.11 -y
conda activate hcc-bench
pip install -r requirements.txt
```

### R (4.5 tested)

```bash
Rscript install_r_deps.R
```

`install_r_deps.R` pins Seurat, SignacX, SingleR, splatter, celldex, and
SeuratDisk to the versions used for the manuscript.

### scGPT pretrained weights

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('wanglab/scGPT-human', local_dir='models/scGPT-human')"
```

---

## Reproducing the pipeline

The single-command path:

```bash
bash run_all.sh
```

To run individual stages (defined in `run_all.sh`):

```bash
bash run_all.sh preprocess         # 01
bash run_all.sh groundtruth        # 10 (loads EnrichR snapshot by default)
bash run_all.sh annotate           # 03–06, 17 (7 tools on benchmark data)
bash run_all.sh external           # 16, 18–20, 27 (GSE156625 generalisation; 27 adds CellAssign)
bash run_all.sh cnv                # 22, 26, 29 (inferCNV: baseline + T+B-only sensitivity + GSE149614 extension)
bash run_all.sh simulate           # 21, 23, 24 (Splatter negative control)
bash run_all.sh evaluate           # 02, 09, 12, 13 (metrics)
bash run_all.sh robustness         # 25 (bootstrap CI, permutation null, Friedman/Nemenyi)
bash run_all.sh figures            # 07, 15 (figure assets)
# Optional (require scgpt + working CUDA env; not part of `all`):
bash run_all.sh scgpt_sweep        # 28 (KNN k sensitivity on cached scGPT embeddings)
bash run_all.sh scgpt_ft           # 30 (scGPT supervised fine-tune skeleton)
```

The numbered scripts in `scripts/` can also be run individually:

```bash
python scripts/01_preprocess_benchmark_datasets.py
python scripts/10_build_enrichr_groundtruth.py             # snapshot (deterministic)
python scripts/10_build_enrichr_groundtruth.py --refresh   # re-query EnrichR
Rscript scripts/03_run_signacX_on_benchmark.R
# ... etc.
```

After `evaluate`, compare regenerated CSVs to the ones in
`expected_results/` to verify the pipeline reproduces the manuscript
numbers — see `expected_results/README.md` for tolerance guidance.

---

## Acknowledgements

Third-party tool code under `external/sc-type/` is from
<https://github.com/IanevskiAleksandr/sc-type> and is included for
reproducibility under its original licence.

---

## License

MIT — see [LICENSE](LICENSE).
