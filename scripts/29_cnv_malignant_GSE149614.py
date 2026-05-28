"""
Extend the CNV-based malignant cell validation from GSE125449 to
GSE149614, the larger benchmark cohort.

GSE149614 carries broad-6 published labels (T/NK, Myeloid, Hepatocyte,
B, Endothelial, Fibroblast) and does NOT distinguish malignant from
normal hepatocytes. Following Expert reviewer feedback that the
six-class harmonisation collapses two clinically meaningful classes
(Malignant vs Normal Hepatocyte), we use infercnvpy to derive a
Malignant-cell label from CNV signal and evaluate each tool's
Hepatocyte-class predictions against it. T and B cells alone serve as
the normal reference (the T+B-only design from the sensitivity analysis
on GSE125449 in script 26).

Outputs:
  release/expected_results/cnv_malignant_validation_GSE149614.csv
  release/expected_results/FigS_cnv_malignant_GSE149614.pdf

Usage:
  /home/holiday01/miniconda/bin/python 29_cnv_malignant_GSE149614.py
"""

import os, warnings
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

BENCHMARK_H5AD = "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE149614_benchmark.h5ad"
ANNO_DIR       = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
GTF_PATH       = "/mnt/10t/holiday/hnsc_analysis/gencode.v38.annotation.gtf"
OUT_DIR        = "/mnt/10t/holiday/hcc_paper/release/expected_results"
FIG_OUT        = f"{OUT_DIR}/FigS_cnv_malignant_GSE149614.pdf"
os.makedirs(OUT_DIR, exist_ok=True)

# Tools whose Hepatocyte predictions we compare against CNV-derived Malignant labels
TOOLS = {
    "CellTypist": f"{ANNO_DIR}/GSE149614_CellTypist_pred.csv",
    "scGPT":      f"{ANNO_DIR}/GSE149614_scGPT_pred.csv",
    "SignacX":    f"{ANNO_DIR}/GSE149614_signacX_pred.csv",
    "CellAssign": f"{ANNO_DIR}/GSE149614_CellAssign_pred.csv",
    "SingleR":    f"{ANNO_DIR}/GSE149614_SingleR_pred.csv",
    "ScType":     f"{ANNO_DIR}/GSE149614_ScType_pred.csv",
    "CyteType":   f"{ANNO_DIR}/GSE149614_CyteType_pred.csv",
}
TOOL_HEPATOCYTE_LABEL = "Hepatocyte"


def load_tool(tool_name, path, all_barcodes):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df["pred_celltype"].reindex(all_barcodes)


def main():
    print(f"[seed={SEED}] CNV-based Malignant validation on GSE149614 (T+B normal ref)")
    adata = ad.read_h5ad(BENCHMARK_H5AD)
    print(f"  GSE149614 shape: {adata.shape}")
    print(f"  Published GT distribution:\n{adata.obs['ground_truth'].value_counts()}")

    cnv.io.genomic_position_from_gtf(GTF_PATH, adata, gtf_gene_id="gene_name", inplace=True)
    n_ann = int(adata.var["chromosome"].notna().sum())
    if n_ann < 100:
        raise RuntimeError(f"Too few genes annotated ({n_ann})")
    print(f"  Annotated genes: {n_ann}")

    # T+B only as normal panel
    ref_cats = ["T/NK", "B"]
    ref_mask = adata.obs["ground_truth"].isin(ref_cats)
    print(f"\n  Reference cells (T/NK+B): {int(ref_mask.sum())}")
    print(f"  Query cells: {int((~ref_mask).sum())}")

    print("  Running infercnvpy (window=100, step=10, T+B normals)...")
    cnv.tl.infercnv(
        adata, reference_key="ground_truth", reference_cat=ref_cats,
        window_size=100, step=10, dynamic_threshold=1.5, n_jobs=4,
    )

    cnv_mat = adata.obsm["X_cnv"]
    cnv_score = np.abs(cnv_mat).mean(axis=1)
    adata.obs["cnv_score"] = np.array(cnv_score).ravel()
    thr = float(np.percentile(adata.obs["cnv_score"], 75))
    adata.obs["cnv_malignant"] = (adata.obs["cnv_score"] >= thr).astype(int)
    n_mal = int(adata.obs["cnv_malignant"].sum())
    print(f"  75th-pct threshold = {thr:.4f}; CNV-malignant cells = {n_mal}")

    # CNV-Malignant cell types breakdown (sanity check)
    print("\n  CNV-Malignant cells stratified by published GT:")
    print(adata.obs.loc[adata.obs["cnv_malignant"] == 1, "ground_truth"].value_counts())

    # CNV-predicted malignant labels (as binary)
    cnv_labels = adata.obs["cnv_malignant"].values
    # Published Hepatocyte cells are the ones we expect to be split into Malignant/Normal
    hep_mask = (adata.obs["ground_truth"] == "Hepatocyte").values

    # Within the Hepatocyte subset: how many are CNV-malignant?
    hep_malignant = int(((hep_mask == True) & (cnv_labels == 1)).sum())
    hep_normal    = int(((hep_mask == True) & (cnv_labels == 0)).sum())
    print(f"\n  Published-Hepatocyte cells: {int(hep_mask.sum())}")
    print(f"    CNV-Malignant subset: {hep_malignant}")
    print(f"    CNV-Normal subset:    {hep_normal}")

    # Per-tool: of the tool's Hepatocyte predictions, F1 against CNV-Malignant
    # interpretation: high F1 means the tool's Hepatocyte calls track CNV-malignancy;
    # low F1 means tool predicts Hepatocyte regardless of malignancy (= collapsed class).
    rows = []
    for tool, path in TOOLS.items():
        preds = load_tool(tool, path, adata.obs_names)
        if preds is None:
            continue
        tool_hep = (preds == TOOL_HEPATOCYTE_LABEL).fillna(False).astype(int).values
        # F1 against CNV-malignant binary
        tp = int(((tool_hep == 1) & (cnv_labels == 1)).sum())
        fp = int(((tool_hep == 1) & (cnv_labels == 0)).sum())
        fn = int(((tool_hep == 0) & (cnv_labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        # Also: of cells the tool labels Hepatocyte, what fraction are CNV-malignant?
        n_tool_hep = int(tool_hep.sum())
        frac_mal   = (tp / n_tool_hep) if n_tool_hep > 0 else 0.0

        rows.append({
            "Tool": tool,
            "N_tool_Hepatocyte": n_tool_hep,
            "TP_CNV_Mal":   tp, "FP_CNV_Norm": fp, "FN_CNV_Mal":  fn,
            "Precision_CNV_Mal": round(prec, 3),
            "Recall_CNV_Mal":    round(rec, 3),
            "F1_CNV_Mal":        round(f1, 3),
            "Frac_Hepatocyte_pred_that_is_CNV_Mal": round(frac_mal, 3),
        })
        print(f"  {tool:11s}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}  "
              f"({n_tool_hep} hep preds, {frac_mal:.0%} CNV-malignant)")

    df = pd.DataFrame(rows)
    csv_out = f"{OUT_DIR}/cnv_malignant_validation_GSE149614.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    # Figure: per-tool F1 against CNV-Malignant
    if not df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df))
        ax.bar(x, df["F1_CNV_Mal"], color="#4C72B0")
        for i, (tool, f1, frac) in enumerate(zip(df["Tool"], df["F1_CNV_Mal"],
                                                  df["Frac_Hepatocyte_pred_that_is_CNV_Mal"])):
            ax.text(i, f1 + 0.01, f"{f1:.3f}\n({frac:.0%} mal)",
                    ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(df["Tool"], rotation=45, ha="right")
        ax.set_ylabel("F1 (tool Hepatocyte prediction vs CNV-Malignant)")
        ax.set_title("Tool Hepatocyte-class predictions vs CNV-derived Malignant labels (GSE149614)")
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(FIG_OUT, bbox_inches="tight")
        print(f"Figure: {FIG_OUT}")


if __name__ == "__main__":
    main()
