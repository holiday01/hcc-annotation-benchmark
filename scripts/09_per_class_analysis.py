"""
Per-class F1 analysis and supplementary figures for benchmark paper.

Outputs:
  - FigS1: Per-class F1 heatmap for GSE149614 benchmark
  - FigS2: Per-class F1 heatmap for GSE125449 benchmark
  - Table S1: Full per-class metrics CSV
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import os

ANNOT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
FIG_DIR   = "/mnt/10t/assi_result/HCC/benchmark_datasets/evaluation/figures"
EVAL_DIR  = "/mnt/10t/assi_result/HCC/benchmark_datasets/evaluation"
os.makedirs(FIG_DIR, exist_ok=True)

BROAD_LABELS    = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
LABELS_125449   = ["T cell", "CAF", "Malignant cell", "TAM", "B cell", "TEC", "HPC-like"]

TOOL_COLORS = {
    "CellAssign": "#00A087",
    "CellTypist":  "#F39B7F",
}


def per_class_metrics(pred_path, tool_name, dataset_name, labels):
    df = pd.read_csv(pred_path, index_col=0).dropna(subset=["pred_celltype", "ground_truth"])
    y_true = df["ground_truth"].values
    y_pred = df["pred_celltype"].values

    # Restrict to known labels
    mask = np.isin(y_true, labels) & np.isin(y_pred, labels + ["Unknown"])
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Exclude Unknown from scoring
    mask2 = y_pred != "Unknown"
    yt = y_true[mask2]
    yp = y_pred[mask2]

    report = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
    rows = []
    for lbl in labels:
        r = report.get(lbl, {})
        gt_count = int((y_true == lbl).sum())
        pred_count = int((y_pred == lbl).sum())
        rows.append({
            "Tool":      tool_name,
            "Dataset":   dataset_name,
            "CellType":  lbl,
            "GT_count":  gt_count,
            "Pred_count": pred_count,
            "F1":        round(r.get("f1-score", 0), 4),
            "Precision": round(r.get("precision", 0), 4),
            "Recall":    round(r.get("recall", 0), 4),
        })
    return pd.DataFrame(rows)


def plot_perclass_heatmap(dfs, dataset_name, labels, out_prefix):
    """dfs: list of DataFrames from per_class_metrics for different tools."""
    combined = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, len(labels)//2 + 3)))
    metrics = ["F1", "Precision", "Recall"]

    for ax, metric in zip(axes, metrics):
        mat = combined.pivot(index="CellType", columns="Tool", values=metric)
        mat = mat.reindex(labels)

        sns.heatmap(
            mat, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0, vmax=1, linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=10, rotation=30)
        ax.tick_params(axis="y", labelsize=9, rotation=0)

    plt.suptitle(f"Per-class Performance: {dataset_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = f"{FIG_DIR}/{out_prefix}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_perclass_bars(dfs, dataset_name, labels, out_prefix):
    """Bar chart for F1 per class per tool."""
    combined = pd.concat(dfs, ignore_index=True)

    tools = combined["Tool"].unique()
    x = np.arange(len(labels))
    width = 0.8 / len(tools)

    fig, ax = plt.subplots(figsize=(max(10, len(labels)*1.5), 5))
    for i, tool in enumerate(tools):
        subset = combined[combined["Tool"] == tool]
        f1_vals = [subset[subset["CellType"] == lbl]["F1"].values[0]
                   if len(subset[subset["CellType"] == lbl]) > 0 else 0
                   for lbl in labels]
        offset = (i - len(tools)/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_vals, width*0.9,
                      label=tool, color=TOOL_COLORS.get(tool, "#888888"),
                      edgecolor="white")
        for bar, val in zip(bars, f1_vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_title(f"Per-class F1: {dataset_name}", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = f"{FIG_DIR}/{out_prefix}_bars.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    all_perclass = []

    # ─── GSE149614 ───────────────────────────────────────────────────────────
    dfs_149614 = []
    for tool, fname in [
        ("CellTypist", "GSE149614_CellTypist_pred.csv"),
        ("CellAssign", "GSE149614_CellAssign_pred.csv"),
    ]:
        path = f"{ANNOT_DIR}/{fname}"
        if os.path.exists(path):
            df_pc = per_class_metrics(path, tool, "GSE149614", BROAD_LABELS)
            dfs_149614.append(df_pc)
            all_perclass.append(df_pc)
            print(f"\n{tool} on GSE149614:")
            print(df_pc[["CellType", "GT_count", "Pred_count", "F1", "Precision", "Recall"]].to_string(index=False))

    if dfs_149614:
        plot_perclass_heatmap(dfs_149614, "GSE149614", BROAD_LABELS, "FigS1_perclass_GSE149614")
        plot_perclass_bars(dfs_149614, "GSE149614", BROAD_LABELS, "FigS1_perclass_GSE149614")

    # ─── GSE125449 ───────────────────────────────────────────────────────────
    dfs_125449 = []
    for tool, fname in [
        ("CellTypist", "GSE125449_CellTypist_pred.csv"),
        ("CellAssign", "GSE125449_CellAssign_pred.csv"),
    ]:
        path = f"{ANNOT_DIR}/{fname}"
        if os.path.exists(path):
            df_pc = per_class_metrics(path, tool, "GSE125449", LABELS_125449)
            dfs_125449.append(df_pc)
            all_perclass.append(df_pc)
            print(f"\n{tool} on GSE125449:")
            print(df_pc[["CellType", "GT_count", "Pred_count", "F1", "Precision", "Recall"]].to_string(index=False))

    if dfs_125449:
        plot_perclass_heatmap(dfs_125449, "GSE125449", LABELS_125449, "FigS2_perclass_GSE125449")
        plot_perclass_bars(dfs_125449, "GSE125449", LABELS_125449, "FigS2_perclass_GSE125449")

    # ─── Save full table ──────────────────────────────────────────────────────
    if all_perclass:
        full_df = pd.concat(all_perclass, ignore_index=True)
        out = f"{EVAL_DIR}/TableS1_perclass_metrics.csv"
        full_df.to_csv(out, index=False)
        print(f"\nSaved: {out}")
