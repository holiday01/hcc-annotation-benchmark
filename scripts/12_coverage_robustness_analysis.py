"""
Coverage-aware robustness analyses for the HCC annotation benchmark.

This script addresses two reviewer/editorial concerns:
1. enrichR-derived reference labels cover only a subset of cells.
2. database-derived labels may not capture disease-specific populations.

Outputs:
    results/coverage_adjusted_metrics.csv
    results/enrichr_ground_truth_coverage.csv
    results/enrichr_vs_published_gt_concordance.csv
    results/enrichr_unknown_published_label_distribution.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


REPO_DIR = Path("/mnt/10t/holiday/hcc-annotation-benchmark")
RESULT_DIR = REPO_DIR / "results"
GT_DIR = REPO_DIR / "ground_truth"

BROAD_6 = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
BROAD_7_TO_6 = {
    "T cell": "T/NK",
    "TAM": "Myeloid",
    "B cell": "B",
    "Malignant cell": "Hepatocyte",
    "TEC": "Endothelial",
    "CAF": "Fibroblast",
    "HPC-like": "Hepatocyte",
    "Unknown": "Unknown",
    "unclassified": "Unknown",
}


def to_broad6_published(dataset: str, labels: pd.Series) -> pd.Series:
    if dataset == "GSE125449":
        return labels.map(lambda x: BROAD_7_TO_6.get(str(x), "Unknown"))
    return labels.where(labels.isin(BROAD_6), "Unknown")


def to_broad6_enrichr(labels: pd.Series) -> pd.Series:
    return labels.where(labels.isin(BROAD_6), "Unknown")


def write_coverage_adjusted_metrics() -> None:
    summary = pd.read_csv(RESULT_DIR / "benchmark_summary.csv")
    summary["Evaluated_fraction"] = summary["N_evaluated"] / summary["N_cells"]
    summary["Coverage_adjusted_Macro_F1"] = (
        summary["Macro_F1"] * summary["Evaluated_fraction"]
    )
    summary["Coverage_adjusted_Accuracy"] = (
        summary["Accuracy"] * summary["Evaluated_fraction"]
    )

    out = RESULT_DIR / "coverage_adjusted_metrics.csv"
    summary.to_csv(out, index=False)
    print(f"Wrote {out}")


def write_enrichr_gt_coverage() -> None:
    rows = []
    for path in sorted(GT_DIR.glob("*_enrichr_ground_truth.tsv")):
        dataset = path.name.replace("_enrichr_ground_truth.tsv", "")
        gt = pd.read_csv(path, sep="\t", index_col=0)
        labels = to_broad6_enrichr(gt["ground_truth_enrichr_broad"])

        row = {
            "Dataset": dataset,
            "N_cells": len(labels),
            "N_enrichR_broad6": int((labels != "Unknown").sum()),
            "enrichR_broad6_coverage": round((labels != "Unknown").mean(), 4),
        }
        for label in BROAD_6 + ["Unknown"]:
            row[f"N_{label}"] = int((labels == label).sum())
        rows.append(row)

    out = RESULT_DIR / "enrichr_ground_truth_coverage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


def write_enrichr_published_concordance() -> None:
    concordance_rows = []
    unknown_rows = []

    for dataset in ["GSE125449", "GSE149614"]:
        pub = pd.read_csv(
            GT_DIR / f"{dataset}_ground_truth.tsv", sep="\t", index_col=0
        )["celltype"]
        enr = pd.read_csv(
            GT_DIR / f"{dataset}_enrichr_ground_truth.tsv", sep="\t", index_col=0
        )["ground_truth_enrichr_broad"]

        pub6 = to_broad6_published(dataset, pub)
        enr6 = to_broad6_enrichr(enr)
        merged = pub6.to_frame("published_broad6").join(
            enr6.rename("enrichr_broad6"), how="inner"
        )

        both_known = (merged["published_broad6"] != "Unknown") & (
            merged["enrichr_broad6"] != "Unknown"
        )
        yt = merged.loc[both_known, "published_broad6"]
        yp = merged.loc[both_known, "enrichr_broad6"]

        concordance_rows.append(
            {
                "Dataset": dataset,
                "N_cells": len(merged),
                "N_published_broad6": int(
                    (merged["published_broad6"] != "Unknown").sum()
                ),
                "N_enrichR_broad6": int((merged["enrichr_broad6"] != "Unknown").sum()),
                "N_both_known": int(both_known.sum()),
                "enrichR_broad6_coverage": round(
                    (merged["enrichr_broad6"] != "Unknown").mean(), 4
                ),
                "Agreement_on_both_known": round(accuracy_score(yt, yp), 4),
                "Macro_F1_on_both_known": round(
                    f1_score(yt, yp, labels=BROAD_6, average="macro", zero_division=0),
                    4,
                ),
                "Kappa_on_both_known": round(cohen_kappa_score(yt, yp), 4),
            }
        )

        unknown_counts = (
            merged.loc[merged["enrichr_broad6"] == "Unknown", "published_broad6"]
            .value_counts()
            .rename_axis("Published_broad6")
            .reset_index(name="N_cells")
        )
        unknown_counts.insert(0, "Dataset", dataset)
        unknown_rows.append(unknown_counts)

    out1 = RESULT_DIR / "enrichr_vs_published_gt_concordance.csv"
    out2 = RESULT_DIR / "enrichr_unknown_published_label_distribution.csv"
    pd.DataFrame(concordance_rows).to_csv(out1, index=False)
    pd.concat(unknown_rows, ignore_index=True).to_csv(out2, index=False)
    print(f"Wrote {out1}")
    print(f"Wrote {out2}")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_coverage_adjusted_metrics()
    write_enrichr_gt_coverage()
    write_enrichr_published_concordance()


if __name__ == "__main__":
    main()
