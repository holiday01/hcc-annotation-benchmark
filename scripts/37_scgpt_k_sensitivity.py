"""
37_scgpt_k_sensitivity.py — scGPT k-NN classifier: macro F1 for k in {5, 10, 15, 20, 30}.

Uses the predictions saved by 35_scgpt_leave_cohort_out.py
(annotation_results_rev2/scgpt_k_sensitivity/<GSE>_scGPT_k.csv) and the
revision-2 scoring rules of 31_rescore_common_cells.py (canonical GT mapping,
macro F1 over the GT classes present, all GT-labelled cells; scGPT never
abstains). Writes expected_results_rev2/rev2_scgpt_k_sensitivity.csv.
"""
import importlib.util
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r31", os.path.join(HERE, "31_rescore_common_cells.py"))
r31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r31)
KDIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2/scgpt_k_sensitivity"
OUT = os.path.join(os.path.dirname(HERE), "expected_results_rev2", "rev2_scgpt_k_sensitivity.csv")

rows = []
evals = [("GSE125449", "published"), ("GSE149614", "published"), ("GSE156625", "published")] + \
        [(d, "enrichR") for d in r31.COHORTS]
for ds, gtt in evals:
    gt, _ = r31.load_gt_and_samples(ds, gtt, "broad6", "canonical")
    k = pd.read_csv(f"{KDIR}/{ds}_scGPT_k.csv", index_col=0)
    k.index = k.index.astype(str)
    k = k.reindex(gt.index)
    present = [c for c in r31.BROAD6 if (gt == c).any()]
    mask = pd.Series([c in present for c in r31.BROAD6]).to_numpy()
    yt, _ = r31.encode(gt, gt, r31.BROAD6)
    row = {"Dataset": ds, "GT": gtt, "N": len(gt), "Classes": "|".join(present)}
    for col in k.columns:
        _, yp = r31.encode(gt, k[col].fillna("Unknown").astype(str), r31.BROAD6)
        row[f"MacroF1_{col}"] = r31.metrics_from_cm(r31.confmat(yt, yp, len(r31.BROAD6)), mask)["macro_f1"]
    f = [row[c] for c in row if c.startswith("MacroF1_")]
    row["Range"] = max(f) - min(f)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, float_format="%.6f")
print(df.to_string(index=False))
