"""
Run CyteType on GSE149614 and GSE125449 benchmark datasets.
CyteType is cluster-based: cluster → DE → API call → per-cluster label → map to cells.

Usage:
    python 04_run_CyteType_on_benchmark.py
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os

OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
os.makedirs(OUT_DIR, exist_ok=True)


def run_cytetype(h5ad_path, dataset_name):
    print(f"\n=== {dataset_name} ===")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.shape}")

    # ── Preprocessing ────────────────────────────────────────────────────────
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

    # ── Leiden clustering ─────────────────────────────────────────────────────
    sc.tl.leiden(adata, key_added="clusters", resolution=0.5)
    n_clusters = adata.obs["clusters"].nunique()
    print(f"  Leiden clusters: {n_clusters}")

    # ── Rank genes per cluster (required by CyteType) ─────────────────────────
    sc.tl.rank_genes_groups(adata, groupby="clusters", method="t-test",
                            n_genes=50)

    # gene_symbols must be in adata.var
    adata.var["gene_symbols"] = adata.var_names

    # ── CyteType ──────────────────────────────────────────────────────────────
    from cytetype import CyteType

    print("  Calling CyteType API...")
    annotator = CyteType(adata, group_key="clusters")
    adata = annotator.run(
        study_context="Human liver hepatocellular carcinoma single-cell RNA-seq"
    )

    pred_col = "cytetype_annotation_clusters"
    print(f"  CyteType done. Cluster distribution:\n"
          f"{adata.obs[pred_col].value_counts()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "barcode":       adata.obs_names,
        "pred_celltype": adata.obs[pred_col].values,
        "ground_truth":  adata.obs["ground_truth"].values,
    }, index=adata.obs_names)

    out_path = f"{OUT_DIR}/{dataset_name}_CyteType_pred.csv"
    out_df.to_csv(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Prediction distribution:\n{out_df['pred_celltype'].value_counts()}")


if __name__ == "__main__":
    BASE = "/mnt/10t/assi_result/HCC/benchmark_datasets"
    run_cytetype(f"{BASE}/GSE125449_benchmark.h5ad", "GSE125449")
    run_cytetype(f"{BASE}/GSE149614_benchmark.h5ad", "GSE149614")
    print("\nAll done.")
