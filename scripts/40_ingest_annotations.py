"""
40_ingest_annotations.py — turn returned blinded annotation sheets into an independent ground truth.

Inputs: two (or more) filled copies of annotation_sheet_TEMPLATE.xlsx
(39_build_annotation_package.py / PROTOCOL.md) and, after adjudication, the
adjudication sheet this script writes in the first pass, filled in.

  python 40_ingest_annotations.py --sheets annot_XX.xlsx annot_YY.xlsx
      -> results/agreement.csv            Cohen's kappa per cohort (clusters, and cell-weighted)
         results/adjudication_TODO.xlsx   clusters where annotators disagree (for the third annotator)
         results/subclustering_requests.csv
  python 40_ingest_annotations.py --sheets ... --adjudication adjudication_DONE.xlsx
      -> ground_truth/<GSE>_curated_ground_truth.tsv  (barcode, cluster, curated_class, celltype)
         results/calibration_vs_published.csv          curated vs published labels on the cohorts that have them

Class -> broad-6: Hepatocyte/epithelial -> Hepatocyte; T/NK, B, Myeloid,
Endothelial, Fibroblast unchanged; Doublet/mixed, Low quality, Other -> Unknown
(not evaluated). Cluster labels are assigned to every cell of the cluster.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = "/mnt/10t/holiday/hcc_paper/annotation_package_rev2"
KEY = f"{ROOT}/key_DO_NOT_SHARE"
RES = f"{ROOT}/results"
GT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/ground_truth"
CLASSES = ["T/NK", "B", "Myeloid", "Endothelial", "Fibroblast", "Hepatocyte/epithelial",
           "Doublet/mixed", "Low quality", "Other"]
TO_B6 = {"Hepatocyte/epithelial": "Hepatocyte", "T/NK": "T/NK", "B": "B", "Myeloid": "Myeloid",
         "Endothelial": "Endothelial", "Fibroblast": "Fibroblast"}
BROAD6 = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
PUBLISHED = {"GSE125449": ("GSE125449_ground_truth.tsv", "celltype"),
             "GSE149614": ("GSE149614_ground_truth.tsv", "celltype"),
             "GSE156625": ("GSE156625_published_ground_truth.tsv", "celltype")}
B7_TO_B6 = {"T cell": "T/NK", "TAM": "Myeloid", "B cell": "B", "Malignant cell": "Hepatocyte",
            "HPC-like": "Hepatocyte", "TEC": "Endothelial", "CAF": "Fibroblast"}


def read_sheet(path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    who = pd.read_excel(xl, "README", header=None).iloc[0, 1] if "README" in xl.sheet_names else None
    who = str(who).strip() if who is not None and str(who) != "nan" else os.path.basename(path)
    rows = []
    for sh in xl.sheet_names:
        if sh == "README":
            continue
        d = pd.read_excel(xl, sh)
        d = d.rename(columns=lambda c: str(c).split(" ")[0])
        d["cohort_code"] = sh
        d["annotator"] = who
        rows.append(d[["cohort_code", "cluster", "n_cells", "class", "confidence", "needs_subclustering",
                       "annotator"]])
    out = pd.concat(rows)
    bad = out[~out["class"].isin(CLASSES)]
    if len(bad):
        raise SystemExit(f"{path}: {len(bad)} rows with missing/invalid class, e.g. "
                         f"{bad[['cohort_code', 'cluster', 'class']].head().to_dict('records')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--adjudication")
    ap.add_argument("--results", default=RES, help="output folder for agreement / adjudication files")
    ap.add_argument("--gt-out", default=GT_DIR, help="folder for <GSE>_curated_ground_truth.tsv")
    args = ap.parse_args()
    res, gt_out = args.results, args.gt_out
    os.makedirs(res, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)
    key = pd.read_csv(f"{KEY}/cohort_key.csv")
    code2ds = dict(zip(key["cohort_code"], key["dataset"]))
    sheets = [read_sheet(p) for p in args.sheets]
    if len(sheets) < 2:
        raise SystemExit("need at least two annotators")
    wide = sheets[0][["cohort_code", "cluster", "n_cells"]].copy()
    names = []
    for s in sheets:
        n = s["annotator"].iloc[0]
        names.append(n)
        wide = wide.merge(s[["cohort_code", "cluster", "class", "confidence", "needs_subclustering"]]
                          .rename(columns={"class": f"class_{n}", "confidence": f"conf_{n}",
                                           "needs_subclustering": f"sub_{n}"}),
                          on=["cohort_code", "cluster"], how="outer")
    a, b = names[0], names[1]

    agr = []
    for code, g in wide.groupby("cohort_code"):
        x, y, w = g[f"class_{a}"], g[f"class_{b}"], g["n_cells"]
        agr.append({"cohort_code": code, "dataset": code2ds.get(code), "n_clusters": len(g),
                    "pct_clusters_agree": round(100 * (x == y).mean(), 1),
                    "kappa_clusters": round(cohen_kappa_score(x, y), 3),
                    "kappa_cell_weighted": round(cohen_kappa_score(x, y, sample_weight=w), 3),
                    "pct_cells_agree": round(100 * w[x == y].sum() / w.sum(), 1)})
    pd.DataFrame(agr).to_csv(f"{res}/agreement.csv", index=False)
    print(pd.DataFrame(agr).to_string(index=False))

    sub = wide[(wide[[f"sub_{n}" for n in names]].astype(str).apply(lambda c: c.str.upper().eq("Y"))).any(axis=1)]
    sub[["cohort_code", "cluster", "n_cells"]].to_csv(f"{res}/subclustering_requests.csv", index=False)
    dis = wide[wide[f"class_{a}"] != wide[f"class_{b}"]]
    print(f"\n{len(dis)} clusters disagree; {len(sub)} flagged for sub-clustering")

    if not args.adjudication:
        todo = dis[["cohort_code", "cluster", "n_cells", f"class_{a}", f"class_{b}"]].copy()
        todo["final_class"] = ""
        todo["note"] = ""
        todo.to_excel(f"{res}/adjudication_TODO.xlsx", index=False)
        print(f"wrote {res}/adjudication_TODO.xlsx (third annotator fills final_class)")
        return

    adj = pd.read_excel(args.adjudication)
    final = wide[["cohort_code", "cluster", f"class_{a}", f"class_{b}"]].copy()
    final["final_class"] = np.where(final[f"class_{a}"] == final[f"class_{b}"], final[f"class_{a}"], None)
    final = final.merge(adj[["cohort_code", "cluster", "final_class"]].rename(columns={"final_class": "adj"}),
                        on=["cohort_code", "cluster"], how="left")
    final["final_class"] = final["final_class"].fillna(final["adj"])
    if final["final_class"].isna().any() or (~final["final_class"].isin(CLASSES)).any():
        raise SystemExit("adjudication incomplete or invalid for some clusters")

    calib = []
    for code, g in final.groupby("cohort_code"):
        ds = code2ds[code]
        asg = pd.read_csv(f"{KEY}/{ds}_cluster_assignments.csv.gz")
        lut = dict(zip(g["cluster"], g["final_class"]))
        asg["curated_class"] = asg["cluster"].map(lut)
        asg["celltype"] = asg["curated_class"].map(TO_B6).fillna("Unknown")
        out = f"{gt_out}/{ds}_curated_ground_truth.tsv"
        if os.path.exists(out):
            raise SystemExit(f"refusing to overwrite {out}")
        asg[["barcode", "cluster", "curated_class", "celltype"]].to_csv(out, sep="\t", index=False)
        print(f"wrote {out}: {asg['celltype'].value_counts().to_dict()}")
        if ds in PUBLISHED:
            f, col = PUBLISHED[ds]
            p = pd.read_csv(f"{GT_DIR}/{f}", sep="\t", index_col=0)[col].astype(str)
            p = p.map(lambda v: B7_TO_B6.get(v, v))
            p = p[p.isin(BROAD6)]
            c = asg.set_index("barcode")["celltype"].reindex(p.index)
            m = c.isin(BROAD6)
            labels = [x for x in BROAD6 if (p[m] == x).any()]
            calib.append({"cohort_code": code, "dataset": ds, "n_published": len(p),
                          "pct_curated_labelled": round(100 * m.mean(), 1),
                          "accuracy": round(accuracy_score(p[m], c[m]), 4),
                          "macro_F1": round(f1_score(p[m], c[m], labels=labels, average="macro", zero_division=0), 4),
                          "kappa": round(cohen_kappa_score(p[m], c[m]), 4)})
    if calib:
        pd.DataFrame(calib).to_csv(f"{res}/calibration_vs_published.csv", index=False)
        print(pd.DataFrame(calib).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
