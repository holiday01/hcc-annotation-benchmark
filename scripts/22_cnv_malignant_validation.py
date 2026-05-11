"""
CNV-based malignant cell validation using infercnvpy.

Strategy:
  1. Run infercnvpy on GSE125449 (has HCC-specific populations including
     Malignant cell and HPC-like in published GT).
  2. Cells with high CNV signal → predicted malignant.
  3. Compare CNV-predicted malignant cells to:
     (a) Published GT malignant labels
     (b) Each tool's predicted malignant/hepatocyte labels
  4. Report precision/recall of each tool for malignant cell detection
     using CNV signal as an independent (non-database) reference.

Output:
  /mnt/10t/holiday/hcc-annotation-benchmark/results/cnv_malignant_validation.csv
  /mnt/10t/assi_result/HCC/benchmark_datasets/figures/FigS_cnv_validation.pdf
"""

import anndata as ad
import scanpy as sc
import infercnvpy as cnv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

BENCHMARK_H5AD = "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE125449_benchmark.h5ad"
ANNO_DIR       = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
OUT_DIR        = "/mnt/10t/holiday/hcc-annotation-benchmark/results"
FIG_DIR        = "/mnt/10t/assi_result/HCC/benchmark_datasets/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Tools whose predictions we want to compare
TOOLS = {
    "CellTypist": f"{ANNO_DIR}/GSE125449_CellTypist_pred.csv",
    "scGPT":      f"{ANNO_DIR}/GSE125449_scGPT_pred.csv",
    "SignacX":    f"{ANNO_DIR}/GSE125449_signacX_pred.csv",
    "CellAssign": f"{ANNO_DIR}/GSE125449_CellAssign_pred.csv",
    "SingleR":    f"{ANNO_DIR}/GSE125449_SingleR_pred.csv",
    "ScType":     f"{ANNO_DIR}/GSE125449_ScType_pred.csv",
    "CyteType":   f"{ANNO_DIR}/GSE125449_CyteType_pred.csv",
}

# Labels in each tool's output that correspond to malignant/tumour cells
# (after broad vocabulary mapping in the evaluation pipeline)
MALIGNANT_LABELS = {
    "published_gt": ["Malignant cell", "HPC-like"],
    "tools": ["Malignant cell", "HPC-like", "Hepatocyte"],  # Hepatocyte included
    # because some tools mis-assign malignant cells as hepatocytes
}


def load_tool_predictions(tool_name, pred_path, valid_barcodes):
    if not os.path.exists(pred_path):
        print(f"  {tool_name}: file not found, skipping")
        return None
    df = pd.read_csv(pred_path, index_col=0)
    df = df[df.index.isin(valid_barcodes)]
    return df["pred_celltype"]


def main():
    print("Loading GSE125449 benchmark h5ad...")
    adata = ad.read_h5ad(BENCHMARK_H5AD)
    print(f"  Shape: {adata.shape}")
    print(f"  Published GT distribution:\n{adata.obs['ground_truth'].value_counts()}")

    # ── infercnvpy requires chromosome positions ──────────────────────────────
    GTF_PATH = "/mnt/10t/holiday/hnsc_analysis/gencode.v38.annotation.gtf"
    print(f"\nAnnotating gene positions from GTF ({GTF_PATH})...")
    cnv.io.genomic_position_from_gtf(
        GTF_PATH,
        adata,
        gtf_gene_id="gene_name",
        inplace=True,
    )
    n_ann = adata.var["chromosome"].notna().sum()
    print(f"  Annotated {n_ann} genes with positions")
    if n_ann < 100:
        raise RuntimeError(f"Too few genes annotated ({n_ann}); check GTF gene_name field")

    print(f"  Genes with chr annotation: {adata.var['chromosome'].notna().sum()}")

    # ── Run inferCNV ──────────────────────────────────────────────────────────
    # Reference cells = non-malignant (T/NK, B, Myeloid = confident normal cells)
    reference_cats = ["T cell", "B cell", "TAM"]  # GSE125449 published GT labels
    reference_mask = adata.obs["ground_truth"].isin(reference_cats)
    print(f"\n  Reference cells (normal): {reference_mask.sum()}")
    print(f"  Query cells: {(~reference_mask).sum()}")

    print("  Running infercnvpy (window=100, step=10)...")
    cnv.tl.infercnv(
        adata,
        reference_key="ground_truth",
        reference_cat=reference_cats,
        window_size=100,
        step=10,
        dynamic_threshold=1.5,
        n_jobs=4,
    )

    # ── CNV score: mean absolute CNV deviation ────────────────────────────────
    cnv_matrix = adata.obsm["X_cnv"]   # cells × genomic_windows
    cnv_score = np.abs(cnv_matrix).mean(axis=1)
    adata.obs["cnv_score"] = np.array(cnv_score).ravel()

    # ── Threshold: cells in top quartile of CNV score = CNV-predicted malignant
    threshold = np.percentile(adata.obs["cnv_score"], 75)
    adata.obs["cnv_malignant"] = (adata.obs["cnv_score"] >= threshold).astype(int)
    print(f"\n  CNV malignant threshold (75th percentile): {threshold:.4f}")
    print(f"  CNV-predicted malignant: {adata.obs['cnv_malignant'].sum()} cells")

    # ── Compare CNV labels vs published GT ───────────────────────────────────
    published_malignant = adata.obs["ground_truth"].isin(
        ["Malignant cell", "HPC-like"]
    ).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score
    cnv_labels  = adata.obs["cnv_malignant"].values
    pub_labels  = published_malignant.values

    print("\n  CNV vs Published GT:")
    print(f"    Precision: {precision_score(pub_labels, cnv_labels):.3f}")
    print(f"    Recall:    {recall_score(pub_labels, cnv_labels):.3f}")
    print(f"    F1:        {f1_score(pub_labels, cnv_labels):.3f}")

    # ── Compare each tool's malignant/hepatocyte predictions vs CNV ──────────
    valid_barcodes = set(adata.obs_names)
    rows = []

    for tool_name, pred_path in TOOLS.items():
        preds = load_tool_predictions(tool_name, pred_path, valid_barcodes)
        if preds is None:
            continue

        # Align to adata
        preds = preds.reindex(adata.obs_names)
        tool_malignant_labels = ["Malignant cell", "HPC-like", "Hepatocyte"]
        tool_malignant = preds.isin(tool_malignant_labels).fillna(False).astype(int)

        # Compare tool's malignant prediction to CNV-predicted malignant
        tp = int(((tool_malignant == 1) & (cnv_labels == 1)).sum())
        fp = int(((tool_malignant == 1) & (cnv_labels == 0)).sum())
        fn = int(((tool_malignant == 0) & (cnv_labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        # Also compare vs published GT directly
        tp_pub = int(((tool_malignant == 1) & (pub_labels == 1)).sum())
        fp_pub = int(((tool_malignant == 1) & (pub_labels == 0)).sum())
        fn_pub = int(((tool_malignant == 0) & (pub_labels == 1)).sum())
        prec_pub = tp_pub / (tp_pub + fp_pub) if (tp_pub + fp_pub) > 0 else 0
        rec_pub  = tp_pub / (tp_pub + fn_pub) if (tp_pub + fn_pub) > 0 else 0
        f1_pub   = 2 * prec_pub * rec_pub / (prec_pub + rec_pub) if (prec_pub + rec_pub) > 0 else 0

        rows.append({
            "Tool": tool_name,
            "CNV_Precision": round(prec, 3),
            "CNV_Recall":    round(rec, 3),
            "CNV_F1":        round(f1, 3),
            "PubGT_Precision": round(prec_pub, 3),
            "PubGT_Recall":    round(rec_pub, 3),
            "PubGT_F1":        round(f1_pub, 3),
        })
        print(f"  {tool_name:12s}: CNV F1={f1:.3f}  PubGT F1={f1_pub:.3f}")

    results_df = pd.DataFrame(rows)
    out_csv = f"{OUT_DIR}/cnv_malignant_validation.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── Figure: CNV score distribution by published GT ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: violin plot of CNV score by cell type
    cell_types = adata.obs["ground_truth"].unique()
    cnv_by_type = [
        adata.obs.loc[adata.obs["ground_truth"] == ct, "cnv_score"].values
        for ct in cell_types
    ]
    axes[0].violinplot(cnv_by_type, showmedians=True)
    axes[0].set_xticks(range(1, len(cell_types) + 1))
    axes[0].set_xticklabels(cell_types, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean |CNV score|", fontsize=10)
    axes[0].set_title("CNV score distribution by cell type (GSE125449)", fontsize=10)
    axes[0].axhline(threshold, color="red", linestyle="--", label=f"75th pct ({threshold:.3f})")
    axes[0].legend(fontsize=8)

    # Right: bar chart of CNV F1 vs Published GT F1 per tool
    if rows:
        x = np.arange(len(results_df))
        w = 0.35
        axes[1].bar(x - w/2, results_df["CNV_F1"],    width=w, label="vs CNV", color="#4C72B0")
        axes[1].bar(x + w/2, results_df["PubGT_F1"], width=w, label="vs PubGT", color="#DD8452")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results_df["Tool"], rotation=45, ha="right", fontsize=9)
        axes[1].set_ylabel("Malignant detection F1", fontsize=10)
        axes[1].set_title("Tool malignant cell detection\n(Malignant+HPC-like+Hepatocyte vs CNV/PubGT)", fontsize=9)
        axes[1].legend(fontsize=9)
        axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    fig_path = f"{FIG_DIR}/FigS_cnv_malignant_validation.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    print("\nDone.")
    return results_df


if __name__ == "__main__":
    main()
