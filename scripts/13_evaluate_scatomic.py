"""
Evaluate scATOMIC as a cancer-specialised supplementary annotation baseline.

scATOMIC predictions are available from prior HCC analyses and overlap the
three validation datasets much more strongly than the two benchmark datasets.
For that reason, this script treats scATOMIC as a supplementary analysis rather
than as a direct replacement for the main five-tool benchmark.

Outputs:
    results/scATOMIC_validation_metrics.csv
    results/scATOMIC_label_mapping_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


REPO_DIR = Path("/mnt/10t/holiday/hcc-annotation-benchmark")
RESULT_DIR = REPO_DIR / "results"
GT_DIR = REPO_DIR / "ground_truth"
SCATOMIC_PATH = Path(
    "/mnt/10t/assi_result/HCC/experimental_scripts/model/scATOMIC/"
    "results_scATOMIC_summary.csv"
)

BROAD_6 = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]


def map_scatomic_to_broad6(label: str) -> str:
    lo = str(label).lower()
    if "unclassified" in lo or lo in {
        "any cell",
        "blood cell",
        "non blood cell",
        "stromal cell",
        "non stromal cell",
    }:
        return "Unknown"
    if any(
        key in lo
        for key in [
            "t cell",
            "natural killer",
            "nk",
            "mait",
            "t regulatory",
            "tfh",
            "cd4",
            "cd8",
            "t or nk",
        ]
    ):
        return "T/NK"
    if any(
        key in lo
        for key in ["macrophage", "monocyte", "dendritic", "cdc", "pdc", "mast"]
    ):
        return "Myeloid"
    if any(key in lo for key in ["b cell", "plasmablast", "plasma"]):
        return "B"
    if "endothelial" in lo:
        return "Endothelial"
    if any(
        key in lo
        for key in ["fibroblast", "caf", "smooth muscle", "pericyte", "nf"]
    ):
        return "Fibroblast"
    if any(
        key in lo
        for key in [
            "liver cancer",
            "billiary",
            "biliary",
            "epithelial",
            "gi epithelial",
            "cancer cell",
            "hepat",
        ]
    ):
        return "Hepatocyte"
    return "Unknown"


def load_scatomic_predictions() -> pd.DataFrame:
    df = pd.read_csv(SCATOMIC_PATH)
    df = df.set_index("cell_names")
    df["pred_broad6"] = df["layer_6"].map(map_scatomic_to_broad6)
    return df


def load_enrichr_gt(dataset: str) -> pd.Series:
    gt = pd.read_csv(
        GT_DIR / f"{dataset}_enrichr_ground_truth.tsv", sep="\t", index_col=0
    )["ground_truth_enrichr_broad"]
    return gt.where(gt.isin(BROAD_6), "Unknown")


def evaluate_dataset(dataset: str, pred: pd.Series) -> dict:
    gt = load_enrichr_gt(dataset)
    merged = gt.to_frame("gt").join(pred.rename("pred"), how="inner")
    known_mask = (merged["gt"] != "Unknown") & (merged["pred"] != "Unknown")

    yt = merged.loc[known_mask, "gt"]
    yp = merged.loc[known_mask, "pred"]
    macro_f1 = f1_score(yt, yp, labels=BROAD_6, average="macro", zero_division=0)

    row = {
        "Tool": "scATOMIC",
        "Dataset": dataset,
        "GT_type": "enrichR",
        "N_cells": len(gt),
        "N_prediction_overlap": len(merged),
        "Prediction_overlap_fraction": round(len(merged) / len(gt), 4),
        "N_evaluated": int(known_mask.sum()),
        "Evaluated_fraction_total": round(known_mask.sum() / len(gt), 4),
        "Unknown_rate_in_overlap": round((merged["pred"] == "Unknown").mean(), 4),
        "Accuracy": round(accuracy_score(yt, yp), 4),
        "Macro_F1": round(macro_f1, 4),
        "Kappa": round(cohen_kappa_score(yt, yp), 4),
        "Coverage_adjusted_Macro_F1": round(macro_f1 * known_mask.sum() / len(gt), 4),
    }
    return row


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scatomic = load_scatomic_predictions()

    mapping_summary = (
        scatomic.groupby(["layer_6", "pred_broad6"])
        .size()
        .reset_index(name="N_cells")
        .sort_values(["pred_broad6", "N_cells"], ascending=[True, False])
    )
    mapping_summary.to_csv(RESULT_DIR / "scATOMIC_label_mapping_summary.csv", index=False)

    pred = scatomic["pred_broad6"]
    rows = [evaluate_dataset(ds, pred) for ds in ["GSE162616", "GSE202642", "GSE223204"]]
    out = RESULT_DIR / "scATOMIC_validation_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
