"""
Generate publication-quality figures for HCC scRNA-seq annotation benchmark paper.

Primary GT: enrichR-based (up to 6 datasets)
Secondary GT: published author annotations (GSE125449, GSE149614 only)

Figures:
  Fig 1: Heatmap of key metrics (enrichR GT, all datasets)
  Fig 2: Bar plot Accuracy + Macro-F1 (enrichR GT, all datasets)
  Fig 3: Unknown / abstention rate comparison
  Fig 4: Confusion matrices for top tools (published GT, GSE125449 + GSE149614)
  Fig 5: Radar chart — average performance on benchmark datasets (enrichR GT)
  FigS1: Heatmap using published GT (GSE125449, GSE149614)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import os

EVAL_DIR   = "/mnt/10t/assi_result/HCC/benchmark_datasets/evaluation"
ANNOT_DIR  = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
GT_DIR     = "/mnt/10t/assi_result/HCC/benchmark_datasets/ground_truth"
FIG_DIR    = f"{EVAL_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Load summary ─────────────────────────────────────────────────────────────
_all = pd.read_csv(f"{EVAL_DIR}/benchmark_summary.csv")
df_enrichr  = _all[_all["GT_type"] == "enrichR"].copy()
df_published = _all[_all["GT_type"] == "published"].copy()
print(f"Loaded: {len(df_enrichr)} enrichR rows, {len(df_published)} published rows")

# ─── Colour / label scheme ───────────────────────────────────────────────────
TOOL_COLORS = {
    "CellTypist": "#F39B7F",   # orange
    "CyteType":   "#4DBBD5",   # cyan
    "signacX":    "#E64B35",   # red
    "CellAssign": "#00A087",   # teal
    "scGPT":      "#8B6FAE",   # purple
    "SingleR":    "#3C5488",   # blue
    "ScType":     "#7E6148",   # brown
}
TOOL_ORDER = ["CellTypist", "ScType", "CyteType", "scGPT", "SingleR", "signacX", "CellAssign"]

DS_ORDER_ALL   = ["GSE125449", "GSE149614", "GSE156625",
                  "GSE162616", "GSE202642", "GSE223204"]
DS_ORDER_BENCH = ["GSE125449", "GSE149614"]

DS_LABELS = {
    "GSE125449": "GSE125449\n(9.8K cells)",
    "GSE149614": "GSE149614\n(71.9K cells)",
    "GSE156625": "GSE156625\n(73.6K cells)",
    "GSE162616": "GSE162616\n(57.3K cells)",
    "GSE202642": "GSE202642\n(122K cells)",
    "GSE223204": "GSE223204\n(23.2K cells)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build metric matrix from df
# ─────────────────────────────────────────────────────────────────────────────
def build_matrix(df, metric, tools, datasets):
    mat = pd.DataFrame(index=tools, columns=datasets, dtype=float)
    for _, row in df.iterrows():
        if row["Tool"] in tools and row["Dataset"] in datasets:
            mat.loc[row["Tool"], row["Dataset"]] = row[metric]
    return mat.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Heatmap — enrichR GT, all datasets
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmap(df, gt_label, suffix=""):
    tools = [t for t in TOOL_ORDER if t in df["Tool"].values]
    datasets = [d for d in DS_ORDER_ALL if d in df["Dataset"].values]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title in zip(axes,
                                  ["Accuracy", "Macro_F1", "Kappa"],
                                  ["Accuracy", "Macro F1", "Cohen's Kappa"]):
        mat = build_matrix(df, metric, tools, datasets)
        ds_labels = [DS_LABELS[d].replace("\n", " ") for d in datasets]
        sns.heatmap(mat, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.8},
                    xticklabels=ds_labels, yticklabels=tools)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9, rotation=30)
        ax.tick_params(axis="y", labelsize=10, rotation=0)

    plt.suptitle(
        f"Cell Type Annotation Tool Benchmark ({gt_label} GT)",
        fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = f"{FIG_DIR}/Fig1_heatmap_metrics{suffix}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Grouped bar chart — enrichR GT, all datasets
# ─────────────────────────────────────────────────────────────────────────────
def plot_bar_all(df, gt_label, suffix=""):
    datasets = [d for d in DS_ORDER_ALL if d in df["Dataset"].values]
    tools    = [t for t in TOOL_ORDER if t in df["Tool"].values]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, metric, ylabel in zip(axes,
                                   ["Accuracy", "Macro_F1"],
                                   ["Accuracy", "Macro F1 Score"]):
        x     = np.arange(len(datasets))
        width = 0.8 / len(tools)
        for i, tool in enumerate(tools):
            vals = [df[(df["Tool"]==tool) & (df["Dataset"]==ds)][metric].values
                    for ds in datasets]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            offset = (i - len(tools)/2 + 0.5) * width
            bars = ax.bar(x + offset, [v if not np.isnan(v) else 0 for v in vals],
                          width*0.9, label=tool,
                          color=TOOL_COLORS.get(tool, "#999999"),
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([DS_LABELS[d] for d in datasets], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9)
        ax.set_title(f"{ylabel} — {gt_label} GT", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{FIG_DIR}/Fig2_bar_all{suffix}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Unknown rate comparison (enrichR GT, de-duplicated)
# ─────────────────────────────────────────────────────────────────────────────
def plot_unknown_rate():
    df = df_enrichr.copy()
    datasets = [d for d in DS_ORDER_ALL if d in df["Dataset"].values]
    tools    = [t for t in TOOL_ORDER if t in df["Tool"].values]

    fig, ax = plt.subplots(figsize=(13, 5))
    x     = np.arange(len(datasets))
    width = 0.8 / len(tools)

    for i, tool in enumerate(tools):
        vals = []
        for ds in datasets:
            row = df[(df["Tool"]==tool) & (df["Dataset"]==ds)]["Unknown_rate"].values
            vals.append(row[0] * 100 if len(row) > 0 else np.nan)
        offset = (i - len(tools)/2 + 0.5) * width
        bar_vals = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(x + offset, bar_vals, width*0.9,
               label=tool, color=TOOL_COLORS.get(tool, "#999999"),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[d] for d in datasets], fontsize=9)
    ax.set_ylabel("Unknown / Abstention Rate (%)", fontsize=11)
    ax.set_title("Abstention Rate per Tool per Dataset (enrichR GT)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{FIG_DIR}/Fig3_unknown_rate.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Confusion matrices (published GT, GSE125449 + GSE149614)
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(pred_path, dataset_name, tool_name, labels):
    pred_df = pd.read_csv(pred_path, index_col=0)

    # Load published GT from separate file
    gt_path = f"{GT_DIR}/{dataset_name}_ground_truth.tsv"
    gt_df   = pd.read_csv(gt_path, sep="\t", index_col=0)
    gt_col  = "celltype"
    # Join pred with GT on barcode index
    merged  = pred_df.join(gt_df[[gt_col]], how="inner").dropna(subset=["pred_celltype", gt_col])
    # Exclude "unclassified" from GT (GSE125449)
    merged  = merged[merged[gt_col] != "unclassified"]

    y_true = merged[gt_col].values
    y_pred = merged["pred_celltype"].values

    # Only keep rows where GT is a known label
    mask_gt = np.isin(y_true, labels)
    y_true_f = y_true[mask_gt]
    y_pred_f = y_pred[mask_gt]
    # Map pred to known labels or "Unknown"
    y_pred_filtered = np.where(np.isin(y_pred_f, labels), y_pred_f, "Unknown")
    display_labels = labels + (["Unknown"] if "Unknown" in y_pred_filtered else [])

    if len(y_true_f) == 0:
        print(f"  SKIP {tool_name}/{dataset_name}: no evaluable cells")
        return
    cm = confusion_matrix(y_true_f, y_pred_filtered, labels=display_labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(len(display_labels)+2, len(labels)+2))
    im = ax.imshow(cm_norm[:len(labels)], cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True (Published GT)", fontsize=11)
    ax.set_title(f"{tool_name} — {dataset_name}",
                 fontsize=12, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(display_labels)):
            val = cm_norm[i, j]
            n   = cm[i, j]
            txt_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}\n({n})",
                    ha="center", va="center", fontsize=7, color=txt_color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Row proportion")
    plt.tight_layout()
    out = f"{FIG_DIR}/Fig4_confusion_{dataset_name}_{tool_name}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Radar chart — average enrichR GT performance
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar_summary():
    bench_df = df_enrichr[
        df_enrichr["Dataset"].isin(["GSE149614", "GSE125449"])].copy()
    tools = [t for t in TOOL_ORDER if t in bench_df["Tool"].values]

    avg = bench_df.groupby("Tool")[["Accuracy", "Macro_F1", "Kappa"]].mean()
    avg["Completeness"] = 1 - bench_df.groupby("Tool")["Unknown_rate"].mean()
    avg = avg.reindex([t for t in tools if t in avg.index])

    categories = ["Accuracy", "Macro F1", "Kappa", "Completeness"]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for tool, row in avg.iterrows():
        values = [row["Accuracy"], row["Macro_F1"],
                  row["Kappa"], row["Completeness"]] + [row["Accuracy"]]
        ax.plot(angles, values, linewidth=2, label=tool,
                color=TOOL_COLORS.get(tool, "#999999"))
        ax.fill(angles, values, alpha=0.1, color=TOOL_COLORS.get(tool, "#999999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.set_title("Average Performance — Benchmark Datasets\n(enrichR GT, GSE125449 + GSE149614)",
                 fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    out = f"{FIG_DIR}/Fig5_radar_summary.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Generating figures ===\n")

    # Main: enrichR GT
    plot_heatmap(df_enrichr,  "enrichR",  suffix="")
    plot_bar_all(df_enrichr,  "enrichR",  suffix="")
    plot_unknown_rate()
    plot_radar_summary()

    # Supplementary: published GT
    plot_heatmap(df_published, "Published", suffix="_published")
    plot_bar_all(df_published, "Published", suffix="_published")

    # Confusion matrices (published GT)
    broad6  = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
    broad7  = ["T cell", "CAF", "Malignant cell", "TAM", "B cell", "TEC", "HPC-like"]

    # label remapping needed before confusion matrix for some tool/dataset combos
    BROAD7_TO_BROAD6 = {
        "T cell": "T/NK", "TAM": "Myeloid", "B cell": "B",
        "Malignant cell": "Hepatocyte", "TEC": "Endothelial", "CAF": "Fibroblast",
    }
    BROAD6_TO_BROAD7 = {v: k for k, v in BROAD7_TO_BROAD6.items()}

    def remap_pred(pred_series, remap):
        return pred_series.map(lambda x: remap.get(x, x))

    for tool in TOOL_ORDER:
        fname_map = {
            "signacX":    lambda ds: f"{ds}_signacX_pred.csv",
            "CyteType":   lambda ds: f"{ds}_CyteType_pred.csv",
        }
        get_fname = fname_map.get(tool, lambda ds: f"{ds}_{tool}_pred.csv")

        for ds, labels in [("GSE149614", broad6), ("GSE125449", broad7)]:
            path = f"{ANNOT_DIR}/{get_fname(ds)}"
            if not os.path.exists(path):
                continue
            # For GSE125449 (broad7 GT), remap preds that are in broad6 format
            pred_df_raw = pd.read_csv(path, index_col=0)
            if ds == "GSE125449" and tool in ("scGPT", "SingleR"):
                # scGPT/SingleR output broad6 → map to broad7 for published GT comparison
                pred_df_raw["pred_celltype"] = remap_pred(
                    pred_df_raw["pred_celltype"], BROAD6_TO_BROAD7)
            elif ds == "GSE125449" and tool == "ScType" and "pred_celltype_broad7" in pred_df_raw.columns:
                pred_df_raw["pred_celltype"] = pred_df_raw["pred_celltype_broad7"]
            elif ds == "GSE149614" and tool in ("CellTypist", "CellAssign"):
                # CellTypist/CellAssign on GSE149614 output broad6 directly — OK
                pass
            # Save temp file for the plot function to read
            tmp_path = f"/tmp/_cm_tmp_{tool}_{ds}.csv"
            pred_df_raw.to_csv(tmp_path)
            plot_confusion_matrix(tmp_path, ds, tool, labels)

    print(f"\nAll figures saved to: {FIG_DIR}")
