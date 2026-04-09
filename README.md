# HCC scRNA-seq Cell Type Annotation Benchmark

This repository contains the analysis scripts, ground truth labels, and evaluation results for the paper:

> **Benchmarking Cell Type Annotation Tools for Hepatocellular Carcinoma Single-Cell RNA Sequencing: A Comparison of Traditional, Large Language Model, and Foundation Model Approaches with Database-Derived Ground Truth**
> Yen-Jung Chiu, Department of Biomedical Engineering, Chang Gung University
> *NAR Genomics and Bioinformatics* (submitted)

---

## Repository Structure

```
hcc-annotation-benchmark/
├── scripts/                          # Analysis pipeline (run in order)
│   ├── 01_preprocess_benchmark_datasets.py   # QC, normalisation, clustering
│   ├── 02_benchmark_evaluation.py            # Compute metrics (F1, Kappa, etc.)
│   ├── 03_run_signacX_on_benchmark.R         # SignacX annotation (R)
│   ├── 04_run_CyteType_on_benchmark.py       # CyteType LLM annotation
│   ├── 05_run_CellAssign_on_benchmark.py     # CellAssign Bayesian annotation
│   ├── 05_run_scGPT_annotation.py            # scGPT foundation model annotation
│   ├── 06_run_CellTypist_on_benchmark.py     # CellTypist classifier annotation
│   ├── 07_generate_figures.py                # Generate all manuscript figures
│   ├── 09_per_class_analysis.py              # Per-class F1 / confusion matrices
│   └── 10_build_enrichr_groundtruth.py       # enrichR-based ground truth construction
├── ground_truth/                     # Ground truth label files
│   ├── GSE125449_ground_truth.tsv            # Published author annotations
│   ├── GSE149614_ground_truth.tsv            # Published author annotations
│   ├── GSE125449_enrichr_ground_truth.tsv    # enrichR-derived GT
│   ├── GSE149614_enrichr_ground_truth.tsv    # enrichR-derived GT
│   ├── GSE162616_enrichr_ground_truth.tsv    # enrichR-derived GT
│   ├── GSE202642_enrichr_ground_truth.tsv    # enrichR-derived GT
│   └── GSE223204_enrichr_ground_truth.tsv    # enrichR-derived GT
├── results/                          # Evaluation output
│   ├── benchmark_summary.csv                 # All metrics, all tools, all datasets
│   └── TableS1_perclass_metrics.csv          # Per-class F1/precision/recall
```

---

## Data Sources

Raw scRNA-seq data are publicly available from NCBI Gene Expression Omnibus (GEO). Download each dataset before running the analysis pipeline.

| Dataset | GEO Accession | Cells | Published GT | Download |
|---------|--------------|-------|--------------|----------|
| GSE125449 | [GSE125449](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125449) | 9,752 | Yes | See below |
| GSE149614 | [GSE149614](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149614) | 71,915 | Yes | See below |
| GSE162616 | [GSE162616](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162616) | 57,261 | No | See below |
| GSE202642 | [GSE202642](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE202642) | 122,092 | No | See below |
| GSE223204 | [GSE223204](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223204) | 23,165 | No | See below |

### Download Instructions

**Option 1: GEO web browser**
Visit each accession link above → click "Download" → download the supplementary count matrix files (`.h5`, `.mtx.gz`, or `_matrix.tar.gz`).

**Option 2: GEO command line (recommended)**

```bash
# Install GEO download tool
pip install GEOparse

# Or use wget directly (replace GSExxxxxx with each accession)
wget -r -np -nd "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/GSE125449/suppl/" -P data/GSE125449/
wget -r -np -nd "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149614/suppl/" -P data/GSE149614/
wget -r -np -nd "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162616/suppl/" -P data/GSE162616/
wget -r -np -nd "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202642/suppl/" -P data/GSE202642/
wget -r -np -nd "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn/GSE223204/suppl/" -P data/GSE223204/
```

Place downloaded files under a `data/` directory:
```
data/
├── GSE125449/    # raw count matrix files
├── GSE149614/
├── GSE162616/
├── GSE202642/
└── GSE223204/
```

---

## Installation

### Python dependencies

```bash
pip install scanpy==1.10 celltypist==1.6 scvi-tools==0.20 gseapy==1.1.12 scgpt==0.2.4 \
    anndata pandas numpy matplotlib seaborn scipy scikit-learn
```

### R dependencies (for SignacX)

```r
install.packages("Seurat")
install.packages("remotes")
remotes::install_github("mathewchamberlain/SignacX")
install.packages("SeuratDisk")   # for AnnData → Seurat conversion
```

### scGPT pretrained weights

Download `scGPT-human` from HuggingFace:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('wanglab/scGPT-human', local_dir='models/scGPT-human')"
```

---

## Usage

Run scripts in numbered order:

```bash
# Step 1: Preprocess datasets (QC, normalisation, clustering)
python scripts/01_preprocess_benchmark_datasets.py

# Step 2: Build enrichR ground truth
python scripts/10_build_enrichr_groundtruth.py

# Step 3: Run annotation tools
python scripts/06_run_CellTypist_on_benchmark.py
python scripts/05_run_CellAssign_on_benchmark.py
Rscript scripts/03_run_signacX_on_benchmark.R
python scripts/04_run_CyteType_on_benchmark.py
python scripts/05_run_scGPT_annotation.py

# Step 4: Evaluate
python scripts/02_benchmark_evaluation.py
python scripts/09_per_class_analysis.py

# Step 5: Generate figures
python scripts/07_generate_figures.py
```

---

## Citation

If you use this code or data, please cite:

```
Chiu, Y.-J. (2025). Benchmarking Cell Type Annotation Tools for Hepatocellular Carcinoma
Single-Cell RNA Sequencing. NAR Genomics and Bioinformatics.
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
