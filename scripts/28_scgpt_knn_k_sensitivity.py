"""
scGPT KNN k sensitivity analysis on GSE149614 (the larger published-GT
benchmark cohort, where scGPT achieved the headline Macro F1 = 0.942
with k = 15).

Approach: embed the reference (DISCO liver, 3000-cell-per-type
stratified subsample) and the query (GSE149614) once with scGPT, then
evaluate KNN classification at k ∈ {5, 10, 15, 20, 30} from the same
embeddings, producing a sensitivity panel of (k, Macro F1, Accuracy).
Embeddings are saved to disk so future analyses can re-run the KNN sweep
without re-embedding.

Outputs:
  release/expected_results/scGPT_knn_k_sensitivity_GSE149614.csv
  release/expected_results/FigS_scGPT_knn_k_sensitivity.pdf
  cache/scgpt_ref_emb.npy, cache/scgpt_GSE149614_query_emb.npy

Usage:
  /home/holiday01/miniconda/bin/python 28_scgpt_knn_k_sensitivity.py
"""

import os, random, warnings, json
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
QUERY_H5AD  = BENCHMARK / "GSE149614_benchmark.h5ad"
GT_PATH     = BENCHMARK / "ground_truth" / "GSE149614_ground_truth.tsv"

OUT_DIR     = Path("/mnt/10t/holiday/hcc_paper/release/expected_results")
CACHE_DIR   = Path("/mnt/10t/holiday/hcc_paper/release/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DISCO_TO_BROAD6_MIN = {
    # T/NK
    "CD56 NK cell": "T/NK", "CD16 NK cell": "T/NK", "NK cell": "T/NK",
    "INF-activated T cell": "T/NK", "GZMB CD8 T cell": "T/NK",
    "GZMK CD8 T cell": "T/NK", "CD4 T cell": "T/NK", "CD8 T cell": "T/NK",
    "Memory CD4 T cell": "T/NK", "Naive CD4 T cell": "T/NK",
    "MAIT cell": "T/NK", "Tfh cell": "T/NK", "ILC": "T/NK",
    "Cycling CXCL13 exhausted CD8 T cell": "T/NK",
    "NK cell / NKT cell": "T/NK",
    # Myeloid
    "Monocyte": "Myeloid", "CD14 monocyte": "Myeloid",
    "CD16 monocyte": "Myeloid", "Macrophage": "Myeloid",
    "SPP1 macrophage": "Myeloid", "M1 macrophage": "Myeloid",
    "LYVE1 macrophage": "Myeloid", "MMP19 macrophage": "Myeloid",
    "Cycling macrophage": "Myeloid", "Neutrophil": "Myeloid",
    "pDC": "Myeloid", "Myeloid pre-pDC": "Myeloid",
    "cDC1": "Myeloid", "cDC2": "Myeloid", "Dendritic cell": "Myeloid",
    "M2 macrophage": "Myeloid", "CD16 macrophage": "Myeloid",
    # B
    "B cell": "B", "Plasma cell": "B",
    # Hepatocyte
    "Hepatocyte": "Hepatocyte", "AFP+ fetal hepatocyte": "Hepatocyte",
    "CCL2+ pancreatic ductal cell": "Hepatocyte",
    "CXCL6+ cholangiocyte": "Hepatocyte",
    # Endothelial
    "Capillary EC": "Endothelial", "Lymphatic EC": "Endothelial",
    "Venous EC": "Endothelial", "Arterial EC": "Endothelial",
    "Endothelial Cells": "Endothelial",
    # Fibroblast / stromal
    "Pericyte": "Fibroblast",
    "BNC2+ZFPM2+ fibroblast": "Fibroblast",
    "Vascular smooth muscle cell": "Fibroblast",
}
BROAD6 = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
K_VALUES = [5, 10, 15, 20, 30]


def prepare_reference(max_cells_per_type=3000):
    print("Loading DISCO reference...")
    ref = sc.read_h5ad(DISCO_H5AD)
    ref.obs["broad6"] = ref.obs["cell_type"].map(DISCO_TO_BROAD6_MIN)
    ref = ref[ref.obs["broad6"].notna()].copy()
    print(f"  after mapping: {ref.shape[0]} cells")

    idx = []
    for label in BROAD6:
        mask = ref.obs["broad6"] == label
        n = int(mask.sum()); take = min(n, max_cells_per_type)
        idx.extend(ref.obs.index[mask].to_series().sample(take, random_state=SEED).tolist())
    ref = ref[idx].copy()
    print(f"  subsampled: {ref.shape[0]} cells")
    return ref


def embed(adata, batch_size, cache_path):
    if cache_path.exists():
        emb = np.load(cache_path)
        if emb.shape[0] == adata.shape[0]:
            print(f"  using cached embeddings from {cache_path}  shape={emb.shape}")
            return emb
        print(f"  cache shape mismatch ({emb.shape[0]} vs {adata.shape[0]}); re-embedding")
    from scgpt.tasks import embed_data
    print(f"  embedding {adata.shape[0]} cells with scGPT (batch_size={batch_size}) ...")
    out = embed_data(
        adata, model_dir=str(MODEL_DIR), gene_col="index",
        max_length=1200, batch_size=batch_size, device="cuda",
        use_fast_transformer=False, return_new_adata=True,
    )
    emb = out.X
    np.save(cache_path, emb)
    print(f"  saved cache: {cache_path}")
    return emb


def main():
    print(f"[seed={SEED}] scGPT KNN k sensitivity on GSE149614")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    # 1. Reference embeddings
    ref = prepare_reference(max_cells_per_type=3000)
    if "gene_symbols" in ref.var.columns and ref.var.index[0].startswith("ENS"):
        ref.var.index = ref.var["gene_symbols"].values
        ref.var_names_make_unique()
    ref_cache = CACHE_DIR / "scgpt_ref_emb.npy"
    ref_emb   = embed(ref, batch_size=128, cache_path=ref_cache)
    ref_labels = ref.obs["broad6"].values
    print(f"  ref_emb shape: {ref_emb.shape}")

    # 2. Query embeddings
    print(f"\nLoading query {QUERY_H5AD}")
    query = sc.read_h5ad(QUERY_H5AD)
    if "gene_symbols" in query.var.columns and query.var.index[0].startswith("ENS"):
        query.var.index = query.var["gene_symbols"].values
        query.var_names_make_unique()
    qcache = CACHE_DIR / "scgpt_GSE149614_query_emb.npy"
    qemb   = embed(query, batch_size=64, cache_path=qcache)
    print(f"  query_emb shape: {qemb.shape}")

    # 3. Ground truth
    gt = pd.read_csv(GT_PATH, sep="\t", index_col=0)["celltype"]
    common = query.obs_names.intersection(gt.index)
    print(f"\n  GT cells common to query: {len(common)}")
    gt_idx = pd.Series(query.obs_names).isin(common).values
    yt = gt.reindex(query.obs_names[gt_idx]).values

    # 4. KNN sweep
    results = []
    for k in K_VALUES:
        print(f"\n  KNN k={k} ...")
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(ref_emb, ref_labels)
        yp = knn.predict(qemb)[gt_idx]
        mask = yt != "Unknown"
        ytm  = yt[mask]; ypm = yp[mask]
        f1   = f1_score(ytm, ypm, average="macro", zero_division=0)
        acc  = accuracy_score(ytm, ypm)
        kap  = cohen_kappa_score(ytm, ypm)
        results.append({"k": k, "Macro_F1": round(f1, 4),
                        "Accuracy": round(acc, 4), "Kappa": round(kap, 4),
                        "N_evaluated": int(mask.sum())})
        print(f"    Macro F1 = {f1:.4f}  Acc = {acc:.4f}  Kappa = {kap:.4f}")

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "scGPT_knn_k_sensitivity_GSE149614.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))

    # 5. Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["k"], df["Macro_F1"], marker="o", label="Macro F1",  color="#4C72B0")
    ax.plot(df["k"], df["Accuracy"], marker="s", label="Accuracy",  color="#DD8452")
    ax.plot(df["k"], df["Kappa"],    marker="^", label="Cohen Kappa", color="#55A868")
    ax.set_xlabel("KNN k")
    ax.set_ylabel("Metric")
    ax.set_title("scGPT KNN classifier k sensitivity on GSE149614 (published GT)")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(K_VALUES)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    for k, f1 in zip(df["k"], df["Macro_F1"]):
        ax.annotate(f"{f1:.3f}", (k, f1), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    fig_out = OUT_DIR / "FigS_scGPT_knn_k_sensitivity.pdf"
    plt.savefig(fig_out, bbox_inches="tight")
    print(f"Figure: {fig_out}")


if __name__ == "__main__":
    main()
