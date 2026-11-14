"""
45_cnv_simulation_rev2.py — CNV and Splatter analyses re-scored with the revision-2 predictions and rules.

The CNV procedure is unchanged from 22/26/29_*.py (infercnvpy 0.6.1, GENCODE v38
positions, window 100, step 10, dynamic threshold 1.5, CNV score = mean |X_cnv|,
CNV-malignant = top quartile). What changes:
  * predictions: all twelve tools, revision-2 set (31_rescore_common_cells.load_raw_tools
    with "rev2"), mapped with _canonical_broad. A tool calls a cell malignant-like when its
    broad-6 class is Hepatocyte (hepatic parenchymal/epithelial, incl. malignant and HPC-like).
    Previously raw labels were matched literally against "Malignant cell"/"HPC-like"/
    "Hepatocyte", so tools whose labels are worded differently (SignacX "Epithelial",
    CyteType "Hepatocyte (...)", "... HCC") were scored as never calling malignant cells.
  * per-cell CNV scores are saved (cnv/<GSE>_<panel>_cnv_scores.csv.gz).
Splatter: CellTypist, ScType, SingleR predictions on the simulated data re-scored with the
revision-2 rules (fixed class set, Unknown = error, coverage reported).

Panels: GSE125449 baseline T+B+TAM normals and T+B-only sensitivity; GSE149614 T/NK+B normals.
Output (expected_results_rev2/): rev2_cnv_GSE125449.csv, rev2_cnv_GSE149614.csv,
rev2_simulated.csv, cnv/*.csv.gz
Environment: ~/envs/scgpt_rev2 (scanpy 1.11.5, infercnvpy 0.6.1).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r31", os.path.join(HERE, "31_rescore_common_cells.py"))
r31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r31)
import infercnvpy as cnv  # noqa: E402

BENCH = "/mnt/10t/assi_result/HCC/benchmark_datasets"
GTF = "/mnt/10t/holiday/hnsc_analysis/gencode.v38.annotation.gtf"
OUT = os.path.join(os.path.dirname(HERE), "expected_results_rev2")
os.makedirs(f"{OUT}/cnv", exist_ok=True)
np.random.seed(42)


def prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def tool_hepatocyte_calls(ds: str, cells: pd.Index) -> dict:
    raw, _ = r31.load_raw_tools(ds, "broad6", "rev2")
    out = {}
    for t in r31.TOOLS:
        if t in raw:
            lab = r31.map_raw(t, raw[t], "broad6").reindex(cells).fillna("NotRun")
            out[t] = (lab == "Hepatocyte").astype(int).values
    return out


def run_cnv(ds: str, panel: str, ref_cats: list) -> ad.AnnData:
    a = ad.read_h5ad(f"{BENCH}/{ds}_benchmark.h5ad")
    a.obs_names = a.obs_names.astype(str)
    cnv.io.genomic_position_from_gtf(GTF, a, gtf_gene_id="gene_name", inplace=True)
    cnv.tl.infercnv(a, reference_key="ground_truth", reference_cat=ref_cats,
                    window_size=100, step=10, dynamic_threshold=1.5, n_jobs=8)
    a.obs["cnv_score"] = np.asarray(np.abs(a.obsm["X_cnv"]).mean(axis=1)).ravel()
    thr = float(np.percentile(a.obs["cnv_score"], 75))
    a.obs["cnv_malignant"] = (a.obs["cnv_score"] >= thr).astype(int)
    a.obs[["ground_truth", "cnv_score", "cnv_malignant"]].rename_axis("barcode").to_csv(
        f"{OUT}/cnv/{ds}_{panel}_cnv_scores.csv.gz")
    print(f"  {ds} {panel}: threshold {thr:.4f}; CNV-malignant {int(a.obs['cnv_malignant'].sum())} cells", flush=True)
    return a


def gse125449():
    rows = []
    for panel, refs in [("TB_TAM", ["T cell", "B cell", "TAM"]), ("TB_only", ["T cell", "B cell"])]:
        a = run_cnv("GSE125449", panel, refs)
        cnv_lab = a.obs["cnv_malignant"].values
        pub = a.obs["ground_truth"].isin(["Malignant cell", "HPC-like"]).astype(int).values
        p, r, f = prf(cnv_lab, pub)
        rows.append({"panel": panel, "Tool": "CNV call vs published malignant", "N_cells": a.n_obs,
                     "N_CNV_malignant": int(cnv_lab.sum()), "CNV_Precision": round(p, 3),
                     "CNV_Recall": round(r, 3), "CNV_F1": round(f, 3)})
        for t, h in tool_hepatocyte_calls("GSE125449", a.obs_names).items():
            p, r, f = prf(h, cnv_lab)
            pp, pr, pf = prf(h, pub)
            rows.append({"panel": panel, "Tool": t, "N_cells": a.n_obs, "N_tool_malignant_like": int(h.sum()),
                         "CNV_Precision": round(p, 3), "CNV_Recall": round(r, 3), "CNV_F1": round(f, 3),
                         "PubGT_Precision": round(pp, 3), "PubGT_Recall": round(pr, 3), "PubGT_F1": round(pf, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/rev2_cnv_GSE125449.csv", index=False)
    print(df.to_string(index=False), flush=True)


def gse149614():
    a = run_cnv("GSE149614", "TNK_B", ["T/NK", "B"])
    cnv_lab = a.obs["cnv_malignant"].values
    hep = (a.obs["ground_truth"] == "Hepatocyte").values
    rows = [{"Tool": "published Hepatocyte", "N_tool_Hepatocyte": int(hep.sum()),
             "Frac_Hepatocyte_pred_that_is_CNV_Mal": round(cnv_lab[hep].mean(), 3)}]
    for t, h in tool_hepatocyte_calls("GSE149614", a.obs_names).items():
        p, r, f = prf(h, cnv_lab)
        rows.append({"Tool": t, "N_tool_Hepatocyte": int(h.sum()), "Precision_CNV_Mal": round(p, 3),
                     "Recall_CNV_Mal": round(r, 3), "F1_CNV_Mal": round(f, 3),
                     "Frac_Hepatocyte_pred_that_is_CNV_Mal": round(cnv_lab[h == 1].mean() if h.sum() else 0.0, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/rev2_cnv_GSE149614.csv", index=False)
    print(df.to_string(index=False), flush=True)


def simulated():
    # same ground-truth file the original evaluation read (02_benchmark_evaluation.evaluate_simulated)
    gt = r31.ev.load_enrichr_gt("simulated").astype(str)
    gt.index = gt.index.astype(str)
    gt = gt[gt != "Unknown"]
    present = np.array([(gt == c).any() for c in r31.BROAD6])
    yt, _ = r31.encode(gt, gt, r31.BROAD6)
    rows = []
    for t in ["CellTypist", "ScType", "SingleR"]:
        d = pd.read_csv(f"{BENCH}/annotation_results/simulated_{t}_pred.csv", index_col=0)
        d.index = d.index.astype(str)
        lab = d["pred_celltype"].fillna("Unknown").astype(str).map(r31.cb.broad6).reindex(gt.index).fillna("NotRun")
        _, yp = r31.encode(gt, lab, r31.BROAD6)
        m = r31.metrics_from_cm(r31.confmat(yt, yp, len(r31.BROAD6)), present)
        rows.append({"Tool": t, "N_cells": len(gt), "Coverage": round(m["coverage"], 4),
                     "Accuracy": round(m["acc"], 4), "MacroF1": round(m["macro_f1"], 4),
                     **{f"F1_{c}": round(v, 4) for c, v in zip(r31.BROAD6, m["f1"])}})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/rev2_simulated.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["simulated", "GSE125449", "GSE149614"]
    for p in parts:
        {"simulated": simulated, "GSE125449": gse125449, "GSE149614": gse149614}[p]()
