"""
Run CellTypist, ScType, scGPT, and SingleR (via subprocess) on
simulated_benchmark.h5ad and save predictions in standard format.

Output:
  annotation_results/simulated_CellTypist_pred.csv
  annotation_results/simulated_ScType_pred.csv
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

H5AD = "/mnt/10t/assi_result/HCC/benchmark_datasets/simulated_benchmark.h5ad"
OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Label map: CellTypist Healthy_Human_Liver labels → Broad-6
CELLTYPIST_TO_BROAD6 = {
    "T cells": "T/NK", "Circulating NK/NKT": "T/NK", "Resident NK": "T/NK",
    "Macrophages": "Myeloid", "Mono+mono derived cells": "Myeloid",
    "Mig.cDCs": "Myeloid", "cDC1s": "Myeloid", "cDC2s": "Myeloid",
    "pDCs": "Myeloid", "Basophils": "Myeloid", "Neutrophils": "Myeloid",
    "Hepatocytes": "Hepatocyte", "Cholangiocytes": "Hepatocyte",
    "B cells": "B", "Plasma cells": "B",
    "Endothelial cells": "Endothelial",
    "Fibroblasts": "Fibroblast",
}


# ── CellTypist ─────────────────────────────────────────────────────────────
def run_celltypist():
    import celltypist
    from celltypist import models

    print("Loading simulated h5ad...")
    adata = ad.read_h5ad(H5AD)
    # CellTypist expects log1p-normalised; simulate from raw counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Try project path first, fall back to CellTypist cache
    model_path = "/mnt/10t/assi_result/HCC/open_source_model/celltypist/Healthy_Human_Liver.pkl"
    cache_path = "/home/holiday01/.celltypist/data/models/Healthy_Human_Liver.pkl"
    if os.path.exists(model_path):
        model = models.Model.load(model=model_path)
    elif os.path.exists(cache_path):
        model = models.Model.load(model=cache_path)
    else:
        print("  Downloading Healthy_Human_Liver model...")
        models.download_models(model="Healthy_Human_Liver")
        model = models.Model.load(model="Healthy_Human_Liver")

    print("Running CellTypist...")
    pred = celltypist.annotate(adata, model=model, majority_voting=True)
    raw_labels = pred.predicted_labels["majority_voting"]
    broad = raw_labels.map(lambda x: CELLTYPIST_TO_BROAD6.get(x, "Unknown"))

    out_df = pd.DataFrame({
        "pred_celltype": broad,
        "pred_celltypist_raw": raw_labels,
    }, index=adata.obs_names)
    out_path = f"{OUT_DIR}/simulated_CellTypist_pred.csv"
    out_df.to_csv(out_path)
    print(f"  Saved: {out_path}")
    print(out_df["pred_celltype"].value_counts())


# ── ScType ──────────────────────────────────────────────────────────────────
def run_sctype():
    # ScType uses marker gene scoring; re-use the existing script's approach
    import sys
    sys.path.insert(0, "/mnt/10t/holiday/hcc-annotation-benchmark/scripts")

    print("\nLoading simulated h5ad for ScType...")
    adata = ad.read_h5ad(H5AD)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5)

    # ScType marker gene sets (same as script 17)
    SCTYPE_MARKERS = {
        "T/NK":        ["CD3D", "CD3E", "CD3G", "CD8A", "CD4", "NKG7", "GNLY", "NCAM1"],
        "Myeloid":     ["LYZ", "CD14", "CST3", "AIF1", "FCGR3A", "CD68", "S100A8", "S100A9"],
        "Hepatocyte":  ["ALB", "APOA1", "APOB", "HP", "TF", "CYP3A4", "TTR", "FABP1"],
        "B":           ["CD79A", "CD79B", "MS4A1", "CD19", "BANK1", "PAX5"],
        "Endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "ENG", "RAMP2"],
        "Fibroblast":  ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "PDGFRA"],
    }

    # Score each cluster per cell type
    var_names = set(adata.var_names)
    cell_types = list(SCTYPE_MARKERS.keys())
    scores = np.zeros((adata.n_obs, len(cell_types)))
    for j, ct in enumerate(cell_types):
        markers = [g for g in SCTYPE_MARKERS[ct] if g in var_names]
        if not markers:
            continue
        sc.tl.score_genes(adata, gene_list=markers, score_name="_tmp_score")
        scores[:, j] = adata.obs["_tmp_score"].values
        del adata.obs["_tmp_score"]

    pred_idx = scores.argmax(axis=1)
    pred_celltype = pd.Series([cell_types[i] for i in pred_idx], index=adata.obs_names)

    out_df = pd.DataFrame({"pred_celltype": pred_celltype}, index=adata.obs_names)
    out_path = f"{OUT_DIR}/simulated_ScType_pred.csv"
    out_df.to_csv(out_path)
    print(f"  Saved: {out_path}")
    print(out_df["pred_celltype"].value_counts())


if __name__ == "__main__":
    run_celltypist()
    run_sctype()
    print("\nDone.")
