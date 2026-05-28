"""
Run CellAssign on the external atlas cohort GSE156625.

The original main analysis script (05_run_CellAssign_on_benchmark.py) ran
CellAssign on GSE125449 and GSE149614 only; GSE156625 was excluded because
the active environment used Python 3.13 + scvi-tools whose API surface was
not compatible at the time. This script runs CellAssign on GSE156625
inside the dedicated conda env `drug_sc` (Python 3.11, scvi-tools 1.4.2)
where the import is verified.

Outputs:
  /mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results/
      GSE156625_CellAssign_pred.csv

Usage:
  /home/holiday01/miniconda/envs/drug_sc/bin/python 27_run_CellAssign_GSE156625.py
"""

import os, warnings, random
import anndata, scanpy as sc, pandas as pd, numpy as np
warnings.filterwarnings('ignore')

# Deterministic seeding
SEED = 42
random.seed(SEED); np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

import scvi
from scvi.external import CellAssign
import scipy.io as sio

OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Broad6 markers (identical to MARKERS_BROAD in 05_run_CellAssign_on_benchmark.py)
MARKERS_BROAD = {
    "T/NK":        ["CD3D", "CD3E", "CD3G", "CD8A", "CD4", "GNLY", "NKG7", "KLRD1", "GZMB", "GZMK", "PRF1", "NCAM1"],
    "Myeloid":     ["CD14", "CD68", "LYZ", "CSF1R", "ITGAM", "FCGR3A", "S100A8", "S100A9", "HLA-DRA", "MARCO"],
    "Hepatocyte":  ["ALB", "APOE", "APOA1", "CYP3A4", "CYP2E1", "FABP1", "HPX", "FGB", "FGA", "AFP", "TF"],
    "B":           ["CD79A", "CD79B", "MS4A1", "CD19", "BANK1", "IGHM", "CD27", "JCHAIN", "MZB1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP", "ENG", "LYVE1", "CLEC4G", "FCN2", "FCN3"],
    "Fibroblast":  ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "FAP", "PDGFRA", "DCN", "LUM", "PDPN", "THY1"],
}

BENCHMARK_H5AD = "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE156625_benchmark.h5ad"
SUPPL_DIR      = "/mnt/10t/assi_result/GEO_DATA/GSE156625/suppl"
RAW_MTX        = f"{SUPPL_DIR}/GSE156625_HCCmatrix.mtx"
RAW_BARCODES   = f"{SUPPL_DIR}/GSE156625_HCCbarcodes.tsv"
RAW_GENES      = f"{SUPPL_DIR}/GSE156625_HCCgenes.tsv"


def build_marker_matrix(adata, marker_dict):
    available = set(adata.var_names)
    filtered  = {ct: [g for g in genes if g in available] for ct, genes in marker_dict.items()}
    for ct, genes in filtered.items():
        print(f"  {ct}: {len(genes)}/{len(marker_dict[ct])} markers found")
    all_markers = sorted(set(g for genes in filtered.values() for g in genes))
    mat = pd.DataFrame(0, index=all_markers, columns=list(filtered.keys()))
    for ct, genes in filtered.items():
        mat.loc[genes, ct] = 1
    print(f"  Marker matrix: {mat.shape[0]} genes × {mat.shape[1]} cell types")
    return mat


def main():
    print(f"[seed={SEED}] CellAssign on GSE156625 (drug_sc env)")
    print(f"Loading benchmark h5ad: {BENCHMARK_H5AD}")
    adata = anndata.read_h5ad(BENCHMARK_H5AD)
    print(f"  shape: {adata.shape}")
    print(f"  obs cols: {list(adata.obs.columns)[:8]}...")

    # CellAssign needs raw integer counts; the benchmark h5ad is log-normalised
    print(f"\nLoading raw counts from {RAW_MTX} ...")
    m = sio.mmread(RAW_MTX).T.tocsr()
    barcodes = pd.read_csv(RAW_BARCODES, header=None, sep="\t")[0].tolist()
    genes_df = pd.read_csv(RAW_GENES, header=None, sep="\t")
    genes = genes_df[1].tolist() if genes_df.shape[1] > 1 else genes_df[0].tolist()
    raw = anndata.AnnData(X=m)
    raw.obs_names = barcodes
    raw.var_names = genes
    raw.var_names_make_unique()
    print(f"  raw shape: {raw.shape}")

    # Keep only cells in benchmark adata
    common = raw.obs_names.intersection(adata.obs_names)
    print(f"  cells common to raw+benchmark: {len(common)}")
    raw = raw[common].copy()

    # Align to benchmark order
    adata_raw = raw[adata.obs_names.intersection(common)].copy()
    if adata_raw.shape[0] < adata.shape[0]:
        print(f"  WARNING: only {adata_raw.shape[0]}/{adata.shape[0]} cells matched")

    print("\nBuilding marker matrix...")
    marker_mat = build_marker_matrix(adata_raw, MARKERS_BROAD)

    # Subset adata to marker genes only (CellAssign requirement)
    shared = [g for g in marker_mat.index if g in adata_raw.var_names]
    adata_sub = adata_raw[:, shared].copy()
    marker_sub = marker_mat.loc[shared]
    print(f"  using {len(shared)} marker genes after intersection")

    # Library-size offset
    sc.pp.normalize_total(adata_sub, target_sum=None, inplace=True)
    adata_sub.obs["size_factor"] = (
        adata_sub.obs["total_counts"] if "total_counts" in adata_sub.obs else 1.0
    )

    print("\nFitting CellAssign model...")
    scvi.external.CellAssign.setup_anndata(adata_sub, size_factor_key="size_factor")
    model = CellAssign(adata_sub, marker_sub)
    model.train(max_epochs=100)
    probs = model.predict()
    pred  = probs.idxmax(axis=1).values

    out = pd.DataFrame({
        "barcode":       adata_sub.obs_names,
        "pred_celltype": pred,
        "ground_truth":  "NA",
    }, index=adata_sub.obs_names)

    out_path = f"{OUT_DIR}/GSE156625_CellAssign_pred.csv"
    out.to_csv(out_path)
    print(f"\nSaved: {out_path}")
    print("\nPrediction distribution:")
    print(pd.Series(pred).value_counts())


if __name__ == "__main__":
    main()
