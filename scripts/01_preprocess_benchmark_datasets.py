"""
Preprocess GSE149614 and GSE125449 for benchmark evaluation.
Loads raw count matrices, applies standard QC, and saves as h5ad with ground truth labels.
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.io
import gzip
import os

sc.settings.verbosity = 1

OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets"
GT_DIR  = f"{OUT_DIR}/ground_truth"

# ─────────────────────────────────────────────
# 1. GSE149614
# ─────────────────────────────────────────────
print("=" * 60)
print("Processing GSE149614 (71,915 cells, 6 broad cell types)")
print("=" * 60)

SUPPL_149614 = "/mnt/10t/assi_result/GEO_DATA/GSE149614/suppl"

# Load raw count matrix (genes x cells)
# Note: normalized.txt.gz is corrupted (only 570 lines); use count.txt (3.5 GB, complete)
print("  Loading count matrix (3.5 GB, ~15 min)...")
count_file = f"{SUPPL_149614}/GSE149614_HCC.scRNAseq.S71915.count.txt"
df_count = pd.read_csv(count_file, sep="\t", index_col=None, header=0)
print(f"  Matrix shape (genes x cells): {df_count.shape}")

# AnnData: cells x genes
adata_149614 = ad.AnnData(X=df_count.values.T.astype("float32"))
adata_149614.obs_names = df_count.columns.tolist()
adata_149614.var_names = df_count.index.tolist()
del df_count  # free memory

# Attach ground truth
gt_149614 = pd.read_csv(f"{GT_DIR}/GSE149614_ground_truth.tsv", sep="\t", index_col=0)
adata_149614.obs = adata_149614.obs.join(gt_149614[["sample", "site", "celltype"]], how="left")
adata_149614.obs.rename(columns={"celltype": "ground_truth"}, inplace=True)
adata_149614 = adata_149614[adata_149614.obs["ground_truth"].notna()].copy()

print(f"  Cells with ground truth: {adata_149614.n_obs}")
print(f"  Cell type distribution:\n{adata_149614.obs['ground_truth'].value_counts()}")

# Basic QC
sc.pp.filter_genes(adata_149614, min_cells=10)
print(f"  After gene filter: {adata_149614.shape}")

# Normalise and log-transform (standard pipeline)
sc.pp.normalize_total(adata_149614, target_sum=1e4)
sc.pp.log1p(adata_149614)

# HVG + PCA + UMAP for visualisation
sc.pp.highly_variable_genes(adata_149614, n_top_genes=2000)
sc.pp.pca(adata_149614, use_highly_variable=True)
sc.pp.neighbors(adata_149614, n_pcs=30)
sc.tl.umap(adata_149614)

out_149614 = f"{OUT_DIR}/GSE149614_benchmark.h5ad"
adata_149614.write_h5ad(out_149614)
print(f"  Saved: {out_149614}")


# ─────────────────────────────────────────────
# 2. GSE125449
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Processing GSE125449 (9,946 cells, 7 cell types)")
print("=" * 60)

SUPPL_125449 = "/mnt/10t/assi_result/GEO_DATA/GSE125449/suppl"

adatas = []
for setname in ["Set1", "Set2"]:
    print(f"  Loading {setname}...")
    barcodes = pd.read_csv(f"{SUPPL_125449}/GSE125449_{setname}_barcodes.tsv", header=None)[0].tolist()
    genes    = pd.read_csv(f"{SUPPL_125449}/GSE125449_{setname}_genes.tsv", header=None)[0].tolist()
    mat      = scipy.io.mmread(f"{SUPPL_125449}/GSE125449_{setname}_matrix.mtx").T.tocsr()

    adata_set = ad.AnnData(X=mat)
    adata_set.obs_names = barcodes
    adata_set.var_names = genes
    adata_set.obs["set"] = setname
    adatas.append(adata_set)
    print(f"    {setname}: {adata_set.shape}")

adata_125449 = ad.concat(adatas, join="outer", fill_value=0)
adata_125449.obs_names_make_unique()

# Attach ground truth
gt_125449 = pd.read_csv(f"{GT_DIR}/GSE125449_ground_truth.tsv", sep="\t", index_col=0)
adata_125449.obs = adata_125449.obs.join(gt_125449[["sample", "celltype", "set"]], how="left", rsuffix="_gt")
adata_125449.obs.rename(columns={"celltype": "ground_truth"}, inplace=True)

# Drop unclassified cells from benchmark evaluation
n_before = adata_125449.n_obs
adata_125449 = adata_125449[adata_125449.obs["ground_truth"] != "unclassified"].copy()
print(f"  Dropped {n_before - adata_125449.n_obs} unclassified cells")
print(f"  Cell type distribution:\n{adata_125449.obs['ground_truth'].value_counts()}")

# Standard QC
sc.pp.filter_cells(adata_125449, min_genes=200)
sc.pp.filter_genes(adata_125449, min_cells=5)
sc.pp.normalize_total(adata_125449, target_sum=1e4)
sc.pp.log1p(adata_125449)
print(f"  After QC: {adata_125449.shape}")

# HVG + PCA + UMAP
sc.pp.highly_variable_genes(adata_125449, n_top_genes=2000)
sc.pp.pca(adata_125449, use_highly_variable=True)
sc.pp.neighbors(adata_125449, n_pcs=30)
sc.tl.umap(adata_125449)

out_125449 = f"{OUT_DIR}/GSE125449_benchmark.h5ad"
adata_125449.write_h5ad(out_125449)
print(f"  Saved: {out_125449}")

print("\nDone. Both datasets preprocessed and saved.")
