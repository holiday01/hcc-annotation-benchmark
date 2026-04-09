"""
Run CellAssign on GSE149614 and GSE125449 benchmark datasets.
Uses literature-based marker genes for each cell type.

This mirrors real-world usage: users provide marker genes from literature,
not from the dataset itself (which would be data leakage).
"""

import anndata, scanpy as sc, pandas as pd, numpy as np, warnings, os
warnings.filterwarnings('ignore')
import scvi
from scvi.external import CellAssign

OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Marker gene matrices (literature-based)
# ─────────────────────────────────────────────

# For GSE149614: 6 broad cell types
# Source: CellMarker 2.0, HCA liver atlas, various liver cancer papers
MARKERS_BROAD = {
    "T/NK":        ["CD3D", "CD3E", "CD3G", "CD8A", "CD4", "GNLY", "NKG7", "KLRD1", "GZMB", "GZMK", "PRF1", "NCAM1"],
    "Myeloid":     ["CD14", "CD68", "LYZ", "CSF1R", "ITGAM", "FCGR3A", "S100A8", "S100A9", "HLA-DRA", "MARCO"],
    "Hepatocyte":  ["ALB", "APOE", "APOA1", "CYP3A4", "CYP2E1", "FABP1", "HPX", "FGB", "FGA", "AFP", "TF"],
    "B":           ["CD79A", "CD79B", "MS4A1", "CD19", "BANK1", "IGHM", "CD27", "JCHAIN", "MZB1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP", "ENG", "LYVE1", "CLEC4G", "FCN2", "FCN3"],
    "Fibroblast":  ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "FAP", "PDGFRA", "DCN", "LUM", "PDPN", "THY1"],
}

# For GSE125449: 7 cell types
# T cell, CAF, Malignant cell, TAM, B cell, TEC, HPC-like
MARKERS_125449 = {
    "T cell":         ["CD3D", "CD3E", "CD8A", "CD4", "GNLY", "NKG7", "GZMB", "FOXP3"],
    "CAF":            ["COL1A1", "COL1A2", "ACTA2", "FAP", "PDGFRA", "DCN", "LUM"],
    "Malignant cell": ["AFP", "GPC3", "EPCAM", "KRT8", "KRT18", "KRT19", "ALB", "APOE"],
    "TAM":            ["CD68", "CD163", "MRC1", "CSF1R", "MARCO", "ADGRE1", "SPP1", "APOC1"],
    "B cell":         ["CD79A", "MS4A1", "CD19", "IGHM", "CD27", "JCHAIN", "MZB1"],
    "TEC":            ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP", "ENG", "MMRN1"],
    "HPC-like":       ["SOX9", "KRT7", "KRT19", "PROM1", "EPCAM", "CD24", "ANXA4", "TACSTD2"],
}


def build_marker_matrix(adata, marker_dict):
    """Create binary gene × cell_type marker matrix, keeping only genes in adata."""
    available_genes = set(adata.var_names)
    filtered = {ct: [g for g in genes if g in available_genes]
                for ct, genes in marker_dict.items()}

    # Report coverage
    for ct, genes in filtered.items():
        print(f"  {ct}: {len(genes)}/{len(marker_dict[ct])} markers found")

    all_markers = sorted(set(g for genes in filtered.values() for g in genes))
    mat = pd.DataFrame(0, index=all_markers, columns=list(filtered.keys()))
    for ct, genes in filtered.items():
        mat.loc[genes, ct] = 1

    print(f"  Marker matrix: {mat.shape[0]} genes × {mat.shape[1]} cell types")
    return mat


def run_cellassign(h5ad_path, dataset_name, marker_dict, use_gpu=False):
    print(f"\n=== {dataset_name} ===")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    # CellAssign needs raw integer counts
    # Our h5ad has log-normalized data; reload from count matrix
    if dataset_name == "GSE125449":
        import scipy.io
        SUPPL = "/mnt/10t/assi_result/GEO_DATA/GSE125449/suppl"
        adatas = []
        for s in ["Set1", "Set2"]:
            b = pd.read_csv(f"{SUPPL}/GSE125449_{s}_barcodes.tsv", header=None)[0].tolist()
            g = pd.read_csv(f"{SUPPL}/GSE125449_{s}_genes.tsv",    header=None)[0].tolist()
            m = scipy.io.mmread(f"{SUPPL}/GSE125449_{s}_matrix.mtx").T.tocsr()
            a = anndata.AnnData(X=m)
            a.obs_names = b; a.var_names = g
            adatas.append(a)
        raw = anndata.concat(adatas, join="outer", fill_value=0)
        raw.obs_names_make_unique()
        # Fix gene names: ENSGXXX\tSYMBOL → SYMBOL
        syms = [g.split('\t')[1] if '\t' in g else g for g in raw.var_names]
        sym_s = pd.Series(syms)
        cum = sym_s.groupby(sym_s).cumcount()
        raw.var_names = (sym_s + cum.replace(0,'').apply(lambda x: '' if x=='' else f'_{x}')).values
        raw.var_names_make_unique()
        # Keep only cells in benchmark set
        raw = raw[raw.obs_names.isin(adata.obs_names)].copy()
        adata_raw = raw[adata.obs_names].copy()
        adata_raw.obs = adata.obs.copy()
    elif dataset_name == "GSE149614":
        SUPPL = "/mnt/10t/assi_result/GEO_DATA/GSE149614/suppl"
        import pandas as pd_
        print("  Loading count matrix for CellAssign (raw counts needed)...")
        df = pd_.read_csv(f"{SUPPL}/GSE149614_HCC.scRNAseq.S71915.count.txt",
                          sep="\t", index_col=None, header=0)
        import scipy.sparse as sp_
        adata_raw = anndata.AnnData(X=sp_.csr_matrix(df.values.T.astype("float32")))
        adata_raw.obs_names = df.columns.tolist()
        adata_raw.var_names = df.index.tolist()
        del df
        adata_raw = adata_raw[adata.obs_names].copy()
        adata_raw.obs = adata.obs.copy()
    else:
        adata_raw = adata.copy()

    print(f"  Raw counts shape: {adata_raw.shape}")

    # Build marker matrix
    print("  Building marker matrix:")
    marker_mat = build_marker_matrix(adata_raw, marker_dict)

    # Subset to marker genes only
    shared_genes = [g for g in marker_mat.index if g in adata_raw.var_names]
    adata_sub = adata_raw[:, shared_genes].copy()
    marker_sub = marker_mat.loc[shared_genes]
    print(f"  Using {len(shared_genes)} marker genes")

    # Size factor normalization
    sc.pp.normalize_total(adata_sub, target_sum=None, inplace=True)
    adata_sub.obs["size_factor"] = adata_sub.obs["total_counts"] if "total_counts" in adata_sub.obs else 1.0

    # Setup CellAssign
    scvi.external.CellAssign.setup_anndata(adata_sub, size_factor_key="size_factor")

    # Train
    print("  Training CellAssign...")
    model = CellAssign(adata_sub, marker_sub)
    model.train(max_epochs=100)

    # Predict
    predictions = model.predict()
    pred_labels = predictions.idxmax(axis=1)
    adata_sub.obs["pred_celltype"] = pred_labels.values

    pred_df = pd.DataFrame({
        "barcode":       adata_sub.obs_names,
        "pred_celltype": adata_sub.obs["pred_celltype"].values,
        "ground_truth":  adata_sub.obs["ground_truth"].values,
    }, index=adata_sub.obs_names)

    out = f"{OUT_DIR}/{dataset_name}_CellAssign_pred.csv"
    pred_df.to_csv(out)
    print(f"  Saved: {out}")
    print(f"  Prediction distribution:\n{pred_df['pred_celltype'].value_counts()}")


if __name__ == "__main__":
    run_cellassign(
        "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE125449_benchmark.h5ad",
        "GSE125449", MARKERS_125449
    )
    run_cellassign(
        "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE149614_benchmark.h5ad",
        "GSE149614", MARKERS_BROAD
    )
