"""
scGPT cell-type annotation via fine-tuning (Cell-Type Annotation task).

Skeleton script for the configuration described in Cui et al. 2024
Section: scGPT Tutorial: Cell Type Annotation
(https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Annotation.ipynb).

This complements the zero-shot embedding + external KNN evaluation in
05_run_scGPT_annotation.py by exposing the supervised fine-tuning mode of
scGPT, where a classification head on top of the [CLS] token is trained
with cross-entropy on a labelled reference. The fine-tuned scGPT
classifier is the configuration that Cui et al. report as competitive
with the strongest reference-based tools on standard benchmarks and is
the expected mitigation for class-imbalance collapse failures such as
the one observed on GSE162616.

Status: SCRIPT WRITTEN, EXECUTION DEFERRED.
Execution requires:
  - a Python env with scgpt >= 0.2.4 AND a working torch CUDA build that
    matches the host driver. As of the revision date, the system driver
    is CUDA 12.8; the base miniconda env ships scgpt + torch 2.11.0+cu130
    which fails to initialise CUDA at runtime, and the drug_sc env has
    working CUDA + torch 2.10.0+cu128 but has scvi-tools 1.4.2 which
    conflicts with scgpt's pinned dependencies.
  - DISCO liver reference for training labels
  - GPU with ≥16 GB VRAM

When the env is fixed, run:
  /path/to/working/python 30_scgpt_finetune.py

Outputs:
  release/expected_results/scGPT_finetune_GSE149614.csv
  release/expected_results/scGPT_finetune_GSE125449.csv
  release/expected_results/FigS_scGPT_finetune_vs_zeroshot.pdf
"""

import os, random, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, classification_report
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE        = Path("/mnt/10t/assi_result/HCC")
BENCHMARK   = BASE / "benchmark_datasets"
MODEL_DIR   = BENCHMARK / "scgpt_model"
DISCO_H5AD  = Path("/mnt/10t/assi_result/DISCO_DATA/disco_adata_mt5_cell200gen5_no_combat_with_celltype_20240415.h5ad")
OUT_DIR     = Path("/mnt/10t/holiday/hcc_paper/release/expected_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = {
    "GSE149614": BENCHMARK / "GSE149614_benchmark.h5ad",
    "GSE125449": BENCHMARK / "GSE125449_benchmark.h5ad",
}

BROAD6 = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]


def fine_tune_scgpt(ref_adata, query_adata, n_epochs=30, lr=1e-4,
                    batch_size=64, n_classes=6, model_dir=MODEL_DIR):
    """
    Fine-tune scGPT [CLS]-token classifier on ref, predict on query.

    Reference: scGPT Tutorial_Annotation.ipynb
      1. Load pretrained scGPT-human encoder
      2. Add a linear classification head on top of [CLS] token
      3. Train cross-entropy on ref with class balanced sampling
      4. Predict on query

    The exact API depends on scgpt version. For 0.2.4, the canonical
    pattern is to import scgpt.model.TransformerModel, build a wrapper
    that exposes [CLS] embedding, attach a torch.nn.Linear head, and
    train with AdamW + cross-entropy. See the tutorial notebook in
    bowang-lab/scGPT for the full reference implementation.
    """
    raise NotImplementedError(
        "Fine-tune skeleton: substitute the model and training loop from "
        "scgpt's Tutorial_Annotation.ipynb. Required env: scgpt + working "
        "GPU torch build (see header)."
    )


def main():
    print(f"[seed={SEED}] scGPT fine-tune skeleton")
    # ── ref ─────────────────────────────────────────────────────────────────
    ref = sc.read_h5ad(DISCO_H5AD)
    print(f"  DISCO ref: {ref.shape}")
    # apply DISCO → BROAD6 mapping (see 05_run_scGPT_annotation.py for full map)
    # ...
    raise NotImplementedError(
        "Complete the data prep / fine-tune loop before running. "
        "This script is a placeholder until the scgpt+torch+CUDA env is "
        "reconciled."
    )


if __name__ == "__main__":
    main()
