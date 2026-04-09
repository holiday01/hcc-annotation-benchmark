"""
Run CellTypist on GSE149614 and GSE125449 benchmark datasets.
Uses Healthy_Human_Liver.pkl model (liver-specific, publicly available).

CellTypist is a logistic regression-based classifier trained on curated
single-cell atlases. No markers needed — uses full transcriptome.
"""

import anndata, scanpy as sc, pandas as pd, numpy as np, warnings, os
warnings.filterwarnings('ignore')
import celltypist
from celltypist import models

OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Label mapping: CellTypist → broad categories
# ─────────────────────────────────────────────
# For GSE149614 (6 broad classes: T/NK, Myeloid, Hepatocyte, B, Endothelial, Fibroblast)
CELLTYPIST_TO_BROAD = {
    # T/NK
    "T cells":              "T/NK",
    "Circulating NK/NKT":   "T/NK",
    "Resident NK":          "T/NK",
    # Myeloid
    "Macrophages":          "Myeloid",
    "Mono+mono derived cells": "Myeloid",
    "Mig.cDCs":             "Myeloid",
    "cDC1s":                "Myeloid",
    "cDC2s":                "Myeloid",
    "pDCs":                 "Myeloid",
    "Basophils":            "Myeloid",
    "Neutrophils":          "Myeloid",
    # Hepatocyte
    "Hepatocytes":          "Hepatocyte",
    # B
    "B cells":              "B",
    "Plasma cells":         "B",
    # Endothelial
    "Endothelial cells":    "Endothelial",
    "Cholangiocytes":       "Hepatocyte",   # biliary epithelium → parenchymal liver cells
    # Fibroblast
    "Fibroblasts":          "Fibroblast",
}

# For GSE125449 (7 classes: T cell, CAF, Malignant cell, TAM, B cell, TEC, HPC-like)
CELLTYPIST_TO_125449 = {
    "T cells":              "T cell",
    "Circulating NK/NKT":   "T cell",
    "Resident NK":          "T cell",
    "Macrophages":          "TAM",
    "Mono+mono derived cells": "TAM",
    "Mig.cDCs":             "TAM",
    "cDC1s":                "TAM",
    "cDC2s":                "TAM",
    "pDCs":                 "TAM",
    "Basophils":            "TAM",
    "Neutrophils":          "TAM",
    "Hepatocytes":          "Malignant cell",  # liver origin; model has no malignant class
    "Cholangiocytes":       "HPC-like",
    "B cells":              "B cell",
    "Plasma cells":         "B cell",
    "Endothelial cells":    "TEC",
    "Fibroblasts":          "CAF",
}


def run_celltypist(h5ad_path, dataset_name, label_map, majority_voting=True):
    print(f"\n=== {dataset_name} ===")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    # CellTypist requires log-normalized data (already done in preprocessing)
    # Normalise to 10,000 counts if not already done (data should be log1p)
    adata_ct = adata.copy()

    # Load Healthy_Human_Liver model
    model = models.Model.load("Healthy_Human_Liver.pkl")
    print(f"  Model: Healthy_Human_Liver.pkl ({len(model.cell_types)} cell types)")

    # Run CellTypist with majority voting (cluster-level smoothing)
    print(f"  Running CellTypist (majority_voting={majority_voting})...")
    predictions = celltypist.annotate(
        adata_ct,
        model=model,
        majority_voting=majority_voting,
        over_clustering=None,  # use existing neighbors graph if available
    )

    # Get predicted labels
    if majority_voting:
        pred_raw = predictions.predicted_labels["majority_voting"].values
    else:
        pred_raw = predictions.predicted_labels["predicted_labels"].values

    # Map to broad/benchmark categories
    pred_mapped = pd.Series(pred_raw).map(label_map).fillna("Unknown").values

    # Save probability scores for top prediction
    prob_df = predictions.probability_matrix

    pred_df = pd.DataFrame({
        "barcode":        adata_ct.obs_names,
        "pred_raw":       pred_raw,
        "pred_celltype":  pred_mapped,
        "ground_truth":   adata_ct.obs["ground_truth"].values,
    }, index=adata_ct.obs_names)

    out = f"{OUT_DIR}/{dataset_name}_CellTypist_pred.csv"
    pred_df.to_csv(out)
    print(f"  Saved: {out}")
    print(f"  Raw prediction distribution:\n{pd.Series(pred_raw).value_counts()}")
    print(f"  Mapped prediction distribution:\n{pd.Series(pred_mapped).value_counts()}")


if __name__ == "__main__":
    run_celltypist(
        "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE125449_benchmark.h5ad",
        "GSE125449",
        CELLTYPIST_TO_125449,
        majority_voting=True,
    )
    run_celltypist(
        "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE149614_benchmark.h5ad",
        "GSE149614",
        CELLTYPIST_TO_BROAD,
        majority_voting=True,
    )
