"""
Generate reviewer-facing robustness figures for the HCC annotation benchmark.

Inputs are produced by 12_coverage_robustness_analysis.py:
  - enrichR ground-truth coverage per dataset
  - coverage-adjusted metrics
  - enrichR-vs-published ground-truth concordance

Output:
  results/figures/FigS3_ground_truth_robustness.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_DIR = Path("/mnt/10t/holiday/hcc-annotation-benchmark")
RESULT_DIR = REPO_DIR / "results"
FIG_DIR = RESULT_DIR / "figures"

TOOL_ORDER = ["CellTypist", "CyteType", "scGPT", "SingleR", "signacX", "CellAssign"]
DATASET_ORDER = [
    "GSE125449",
    "GSE149614",
    "GSE156625",
    "GSE162616",
    "GSE202642",
    "GSE223204",
]
TOOL_COLORS = {
    "CellTypist": "#F39B7F",
    "CyteType": "#4DBBD5",
    "scGPT": "#8B6FAE",
    "SingleR": "#3C5488",
    "signacX": "#E64B35",
    "CellAssign": "#00A087",
}


def build_heatmap_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    enrichr = metrics[metrics["GT_type"] == "enrichR"].copy()
    heat = enrichr.pivot_table(
        index="Tool",
        columns="Dataset",
        values="Coverage_adjusted_Macro_F1",
        aggfunc="mean",
    )
    heat = heat.reindex(index=[t for t in TOOL_ORDER if t in heat.index])
    heat = heat.reindex(columns=[d for d in DATASET_ORDER if d in heat.columns])
    return heat


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    coverage = pd.read_csv(RESULT_DIR / "enrichr_ground_truth_coverage.csv")
    metrics = pd.read_csv(RESULT_DIR / "coverage_adjusted_metrics.csv")
    concord = pd.read_csv(RESULT_DIR / "enrichr_vs_published_gt_concordance.csv")

    coverage["Dataset"] = pd.Categorical(
        coverage["Dataset"], categories=DATASET_ORDER, ordered=True
    )
    coverage = coverage.sort_values("Dataset")
    concord["Dataset"] = pd.Categorical(
        concord["Dataset"], categories=["GSE125449", "GSE149614"], ordered=True
    )
    concord = concord.sort_values("Dataset")

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.35, wspace=0.28)

    ax_cov = fig.add_subplot(grid[0, 0])
    ax_concord = fig.add_subplot(grid[0, 1])
    ax_heat = fig.add_subplot(grid[1, :])

    bars = ax_cov.bar(
        coverage["Dataset"],
        coverage["enrichR_broad6_coverage"] * 100,
        color="#6A8A82",
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, coverage["enrichR_broad6_coverage"]):
        ax_cov.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_cov.set_ylim(0, 100)
    ax_cov.set_ylabel("Cells with broad6 enrichR label (%)")
    ax_cov.set_title("A. enrichR ground-truth coverage", fontweight="bold", loc="left")
    ax_cov.spines["top"].set_visible(False)
    ax_cov.spines["right"].set_visible(False)

    x = range(len(concord))
    ax_concord.bar(
        x,
        concord["Agreement_on_both_known"] * 100,
        color="#C17C74",
        edgecolor="white",
        linewidth=0.8,
        label="Agreement",
    )
    ax_concord.scatter(
        x,
        concord["Macro_F1_on_both_known"] * 100,
        color="#2F4858",
        s=70,
        zorder=3,
        label="Macro F1",
    )
    for i, row in concord.reset_index(drop=True).iterrows():
        ax_concord.text(
            i,
            row["Agreement_on_both_known"] * 100 + 1,
            f"{row['Agreement_on_both_known']:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_concord.set_xticks(list(x))
    ax_concord.set_xticklabels(concord["Dataset"].astype(str).tolist())
    ax_concord.set_ylim(0, 105)
    ax_concord.set_ylabel("Performance on cells known in both GTs (%)")
    ax_concord.set_title(
        "B. enrichR vs published GT concordance", fontweight="bold", loc="left"
    )
    ax_concord.legend(frameon=False, loc="lower right")
    ax_concord.spines["top"].set_visible(False)
    ax_concord.spines["right"].set_visible(False)

    heat = build_heatmap_matrix(metrics)
    sns.heatmap(
        heat,
        ax=ax_heat,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Coverage-adjusted Macro F1"},
    )
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_title(
        "C. Coverage-adjusted Macro F1 under enrichR GT",
        fontweight="bold",
        loc="left",
    )
    ax_heat.tick_params(axis="x", rotation=30)
    ax_heat.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Ground-truth robustness and coverage-aware benchmark interpretation",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    out_pdf = FIG_DIR / "FigS3_ground_truth_robustness.pdf"
    out_png = FIG_DIR / "FigS3_ground_truth_robustness.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
