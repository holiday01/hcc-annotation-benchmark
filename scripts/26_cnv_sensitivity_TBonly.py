"""
Sensitivity analysis for the CNV-based malignant cell validation.

The main analysis (22_cnv_malignant_validation.py) uses T+B+TAM as the
infercnvpy normal-cell reference. In HCC, however, TAMs are known to
carry ambient-RNA contamination from neighbouring malignant cells and
can therefore confound the CNV null reference (cf. Tirosh 2016,
Patel 2014 inferCNV practice). This script re-runs the same pipeline
with T+B only and reports the resulting tool×CNV F1, alongside a
direct comparison against the T+B+TAM baseline.

Output:
  release/expected_results/cnv_malignant_validation_TBonly.csv
  release/expected_results/cnv_TBonly_vs_TBTAM_diff.csv  (per-tool delta)
  release/expected_results/FigS_cnv_TBonly_sensitivity.pdf

Usage: python 26_cnv_sensitivity_TBonly.py
"""

import os, warnings, json
import anndata as ad
import infercnvpy as cnv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BENCHMARK_H5AD = "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE125449_benchmark.h5ad"
ANNO_DIR       = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
GTF_PATH       = "/mnt/10t/holiday/hnsc_analysis/gencode.v38.annotation.gtf"
OUT_DIR        = "/mnt/10t/holiday/hcc_paper/release/expected_results"
FIG_OUT        = f"{OUT_DIR}/FigS_cnv_TBonly_sensitivity.pdf"
BASELINE_CSV   = "/mnt/10t/holiday/hcc-annotation-benchmark/results/cnv_malignant_validation.csv"
os.makedirs(OUT_DIR, exist_ok=True)

TOOLS = {
    "CellTypist": f"{ANNO_DIR}/GSE125449_CellTypist_pred.csv",
    "scGPT":      f"{ANNO_DIR}/GSE125449_scGPT_pred.csv",
    "SignacX":    f"{ANNO_DIR}/GSE125449_signacX_pred.csv",
    "CellAssign": f"{ANNO_DIR}/GSE125449_CellAssign_pred.csv",
    "SingleR":    f"{ANNO_DIR}/GSE125449_SingleR_pred.csv",
    "ScType":     f"{ANNO_DIR}/GSE125449_ScType_pred.csv",
    "CyteType":   f"{ANNO_DIR}/GSE125449_CyteType_pred.csv",
}
TOOL_MALIGNANT_LABELS = ["Malignant cell", "HPC-like", "Hepatocyte"]


def load_tool(tool_name, path, all_barcodes):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df["pred_celltype"].reindex(all_barcodes)


def main():
    print(f"[seed={SEED}] CNV sensitivity: T+B-only normal reference")
    adata = ad.read_h5ad(BENCHMARK_H5AD)
    print(f"  GSE125449 shape: {adata.shape}")

    cnv.io.genomic_position_from_gtf(
        GTF_PATH, adata, gtf_gene_id="gene_name", inplace=True
    )
    n_ann = int(adata.var["chromosome"].notna().sum())
    if n_ann < 100:
        raise RuntimeError(f"Too few genes annotated ({n_ann})")
    print(f"  Annotated genes: {n_ann}")

    # ── T + B only reference (TAM excluded) ─────────────────────────────────
    ref_cats = ["T cell", "B cell"]
    ref_mask = adata.obs["ground_truth"].isin(ref_cats)
    print(f"\n  Reference cells (T+B only): {int(ref_mask.sum())}")
    print(f"  Query cells:                 {int((~ref_mask).sum())}")
    print(f"  (Baseline T+B+TAM ran with {(adata.obs['ground_truth'].isin(['T cell','B cell','TAM'])).sum()})")

    print("  Running infercnvpy (T+B only, window=100, step=10)...")
    cnv.tl.infercnv(
        adata,
        reference_key="ground_truth",
        reference_cat=ref_cats,
        window_size=100, step=10, dynamic_threshold=1.5,
        n_jobs=4,
    )

    cnv_mat = adata.obsm["X_cnv"]
    cnv_score = np.abs(cnv_mat).mean(axis=1)
    adata.obs["cnv_score"] = np.array(cnv_score).ravel()
    thr = float(np.percentile(adata.obs["cnv_score"], 75))
    adata.obs["cnv_malignant"] = (adata.obs["cnv_score"] >= thr).astype(int)
    print(f"  75th-pct threshold: {thr:.4f}")
    n_mal = int(adata.obs["cnv_malignant"].sum())
    print(f"  CNV-predicted malignant: {n_mal} cells")

    # vs published GT
    pub_malignant = adata.obs["ground_truth"].isin(["Malignant cell", "HPC-like"]).astype(int).values
    cnv_labels    = adata.obs["cnv_malignant"].values
    cnv_vs_pub = {
        "Precision": float(precision_score(pub_malignant, cnv_labels)),
        "Recall":    float(recall_score(pub_malignant, cnv_labels)),
        "F1":        float(f1_score(pub_malignant, cnv_labels)),
    }
    print(f"\n  CNV (T+B only) vs Published GT:  P={cnv_vs_pub['Precision']:.3f}  "
          f"R={cnv_vs_pub['Recall']:.3f}  F1={cnv_vs_pub['F1']:.3f}")

    # Per-tool
    rows = []
    for tool, path in TOOLS.items():
        preds = load_tool(tool, path, adata.obs_names)
        if preds is None:
            continue
        tool_mal = preds.isin(TOOL_MALIGNANT_LABELS).fillna(False).astype(int).values
        tp = int(((tool_mal == 1) & (cnv_labels == 1)).sum())
        fp = int(((tool_mal == 1) & (cnv_labels == 0)).sum())
        fn = int(((tool_mal == 0) & (cnv_labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "Tool": tool,
            "CNV_TBonly_Precision": round(prec, 3),
            "CNV_TBonly_Recall":    round(rec, 3),
            "CNV_TBonly_F1":        round(f1, 3),
            "N_CNV_predicted_malignant": n_mal,
            "Threshold_75p": round(thr, 4),
        })
        print(f"  {tool:12s}  CNV F1 = {f1:.3f}")

    out_df = pd.DataFrame(rows)
    out_path = f"{OUT_DIR}/cnv_malignant_validation_TBonly.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Compare against baseline (T+B+TAM) ──────────────────────────────────
    diff_df = None
    if os.path.exists(BASELINE_CSV):
        base = pd.read_csv(BASELINE_CSV)
        merge = base.merge(out_df[["Tool", "CNV_TBonly_F1"]], on="Tool", how="left")
        merge["delta_F1"] = merge["CNV_TBonly_F1"] - merge["CNV_F1"]
        diff_path = f"{OUT_DIR}/cnv_TBonly_vs_TBTAM_diff.csv"
        merge.to_csv(diff_path, index=False)
        print(f"Saved: {diff_path}")
        print(merge[["Tool", "CNV_F1", "CNV_TBonly_F1", "delta_F1"]].to_string(index=False))
        diff_df = merge

    # ── Figure: bar chart comparison ────────────────────────────────────────
    if diff_df is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(diff_df))
        w = 0.35
        ax.bar(x - w/2, diff_df["CNV_F1"],        width=w, label="T+B+TAM (baseline)", color="#4C72B0")
        ax.bar(x + w/2, diff_df["CNV_TBonly_F1"], width=w, label="T+B only (sensitivity)", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(diff_df["Tool"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Malignant-detection F1 (vs CNV labels)", fontsize=10)
        ax.set_title("CNV reference-panel sensitivity (GSE125449)", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, loc="upper right")
        for i, (a, b) in enumerate(zip(diff_df["CNV_F1"], diff_df["CNV_TBonly_F1"])):
            ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_OUT, bbox_inches="tight")
        print(f"Figure: {FIG_OUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()
