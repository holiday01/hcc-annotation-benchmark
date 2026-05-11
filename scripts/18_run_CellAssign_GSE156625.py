"""
Run CellAssign on GSE156625 external HCC cohort.

The benchmark h5ad contains log-normalized data (log1p, target_sum=1e4).
We back-approximate raw counts using n_counts from obs, which is sufficient
for CellAssign's size-factor-normalised NB model.

Output:
    /mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results/GSE156625_CellAssign_pred.csv
"""

import anndata, scanpy as sc, pandas as pd, numpy as np, warnings, os
warnings.filterwarnings("ignore")
import scvi
from scvi.external import CellAssign

BENCHMARK_H5AD = "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE156625_benchmark.h5ad"
OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Marker genes (same broad 6-class vocab as other datasets) ──────────────
MARKERS_BROAD = {
    "T/NK":        ["CD3D", "CD3E", "CD3G", "CD8A", "CD4", "GNLY", "NKG7",
                    "KLRD1", "GZMB", "GZMK", "PRF1", "NCAM1"],
    "Myeloid":     ["CD14", "CD68", "LYZ", "CSF1R", "ITGAM", "FCGR3A",
                    "S100A8", "S100A9", "HLA-DRA", "MARCO"],
    "Hepatocyte":  ["ALB", "APOE", "APOA1", "CYP3A4", "CYP2E1", "FABP1",
                    "HPX", "FGB", "FGA", "AFP", "TF"],
    "B":           ["CD79A", "CD79B", "MS4A1", "CD19", "BANK1", "IGHM",
                    "CD27", "JCHAIN", "MZB1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP", "ENG",
                    "LYVE1", "CLEC4G", "FCN2", "FCN3"],
    "Fibroblast":  ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "FAP", "PDGFRA",
                    "DCN", "LUM", "PDPN", "THY1"],
}


def build_marker_matrix(adata, marker_dict):
    available = set(adata.var_names)
    filtered = {ct: [g for g in genes if g in available]
                for ct, genes in marker_dict.items()}
    for ct, genes in filtered.items():
        print(f"  {ct}: {len(genes)}/{len(marker_dict[ct])} markers found")
    all_markers = sorted({g for genes in filtered.values() for g in genes})
    mat = pd.DataFrame(0, index=all_markers, columns=list(filtered.keys()))
    for ct, genes in filtered.items():
        mat.loc[genes, ct] = 1
    print(f"  Marker matrix: {mat.shape[0]} genes × {mat.shape[1]} cell types")
    return mat


def main():
    print("Loading GSE156625 benchmark h5ad...")
    adata = anndata.read_h5ad(BENCHMARK_H5AD)
    print(f"  Shape: {adata.shape}")

    # Back-approximate raw counts from log-normalized data
    # X = log1p(count / n_counts * 1e4)  → count ≈ expm1(X) * n_counts / 1e4
    print("  Back-transforming log-normalized data to approximate raw counts...")
    import scipy.sparse as sp
    X_expm1 = np.expm1(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    n_counts = adata.obs["n_counts"].values[:, None]
    X_raw = np.round(X_expm1 * n_counts / 1e4).astype(np.float32)

    adata_raw = anndata.AnnData(
        X=sp.csr_matrix(X_raw),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    adata_raw.obs["size_factor"] = adata.obs["n_counts"].values / 1e4

    # Build marker matrix and subset
    print("  Building marker matrix:")
    marker_mat = build_marker_matrix(adata_raw, MARKERS_BROAD)
    shared_genes = [g for g in marker_mat.index if g in adata_raw.var_names]
    adata_sub = adata_raw[:, shared_genes].copy()
    marker_sub = marker_mat.loc[shared_genes]
    print(f"  Using {len(shared_genes)} marker genes, {adata_sub.n_obs} cells")

    # Setup and train CellAssign
    scvi.external.CellAssign.setup_anndata(adata_sub, size_factor_key="size_factor")
    print("  Training CellAssign...")
    model = CellAssign(adata_sub, marker_sub)
    model.train(max_epochs=100)

    # Predict
    predictions = model.predict()
    pred_labels = predictions.idxmax(axis=1)

    pred_df = pd.DataFrame({
        "barcode":       adata_sub.obs_names,
        "pred_celltype": pred_labels.values,
        "ground_truth":  adata_sub.obs["ground_truth"].values,
    }, index=adata_sub.obs_names)

    out = f"{OUT_DIR}/GSE156625_CellAssign_pred.csv"
    pred_df.to_csv(out)
    print(f"  Saved: {out}")
    print(f"  Prediction distribution:\n{pred_df['pred_celltype'].value_counts()}")


if __name__ == "__main__":
    main()
