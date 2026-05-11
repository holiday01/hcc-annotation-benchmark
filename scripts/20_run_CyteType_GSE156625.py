"""
Run CyteType on GSE156625 external HCC cohort.

Re-uses pre-computed Louvain clusters and rank_genes_groups stored in the
benchmark h5ad (computed during script 16), avoiding redundant clustering.

Output:
    /mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results/GSE156625_CyteType_pred.csv
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os

BENCHMARK_H5AD = "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE156625_benchmark.h5ad"
OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("Loading GSE156625 benchmark h5ad...")
    adata = ad.read_h5ad(BENCHMARK_H5AD)
    print(f"  Shape: {adata.shape}")

    # Use pre-computed louvain clusters
    if "louvain" not in adata.obs.columns:
        raise ValueError("'louvain' cluster column not found in obs. Run script 16 first.")
    n_clusters = adata.obs["louvain"].nunique()
    print(f"  Using pre-computed Louvain clusters: {n_clusters}")

    # Re-compute rank_genes_groups if missing or stale
    if "rank_genes_groups" not in adata.uns:
        print("  rank_genes_groups not found, recomputing...")
        sc.tl.rank_genes_groups(adata, groupby="louvain", method="t-test", n_genes=50)
    else:
        print("  Using pre-computed rank_genes_groups")

    adata.var["gene_symbols"] = adata.var_names

    # Run CyteType
    from cytetype import CyteType
    print("  Calling CyteType API (cluster-level)...")
    annotator = CyteType(adata, group_key="louvain")
    adata = annotator.run(
        study_context="Human liver hepatocellular carcinoma single-cell RNA-seq"
    )

    pred_col = "cytetype_annotation_louvain"
    if pred_col not in adata.obs.columns:
        # Fallback: try the generic annotation column name
        candidates = [c for c in adata.obs.columns if "cytetype" in c.lower()]
        if candidates:
            pred_col = candidates[0]
        else:
            raise KeyError(f"CyteType annotation column not found. obs columns: {list(adata.obs.columns)}")

    print(f"  CyteType done. Using column: {pred_col}")
    print(f"  Cluster distribution:\n{adata.obs[pred_col].value_counts()}")

    out_df = pd.DataFrame({
        "barcode":       adata.obs_names,
        "pred_celltype": adata.obs[pred_col].values,
        "ground_truth":  adata.obs["ground_truth"].values,
    }, index=adata.obs_names)

    out_path = f"{OUT_DIR}/GSE156625_CyteType_pred.csv"
    out_df.to_csv(out_path)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
