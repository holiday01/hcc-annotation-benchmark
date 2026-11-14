"""
41_sorted_ILC_check_GSE162616.py — lineage specificity on a flow-sorted cohort.

GSE162616 consists of innate lymphoid cells sorted by flow cytometry (GEO:
"scRNA-seq-based analysis of 9 ILC samples sorted by flow cytometry"); the
four samples in the benchmark are three HCC tumours and one tumour-adjacent
liver. By experimental design nearly every cell is an NK/ILC (broad-6 T/NK);
sort impurities can add a few immune cells, but non-immune lineages
(hepatocyte/epithelial, endothelial, fibroblast) should be essentially absent.
This gives a label-free check that does not depend on any annotation:
  * % of cells each tool assigns to T/NK (higher is better);
  * % assigned to a non-immune lineage (false-positive rate; should be ~0);
  * % Myeloid, % B, % Unknown.
Predictions: revision-2 set (31_rescore_common_cells.load_raw_tools(..., "rev2"))
with the canonical mapping; the enrichR GT composition is shown for comparison.
Writes expected_results_rev2/rev2_GSE162616_sorted_ILC_check.csv.
"""
import importlib.util
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r31", os.path.join(HERE, "31_rescore_common_cells.py"))
r31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r31)
OUT = os.path.join(os.path.dirname(HERE), "expected_results_rev2", "rev2_GSE162616_sorted_ILC_check.csv")
DS = "GSE162616"

cells = pd.read_csv(f"{r31.ev.ANNOT_DIR}/{DS}_CellTypist_pred.csv", index_col=0).index.astype(str)
raw, _ = r31.load_raw_tools(DS, "broad6", "rev2")
rows = []


def summarise(name, lab: pd.Series):
    lab = lab.reindex(cells).fillna("NotRun")
    f = lab.value_counts(normalize=True) * 100
    rows.append({"Source": name, "N_cells": len(lab),
                 "pct_T/NK": round(f.get("T/NK", 0), 2),
                 "pct_non_immune": round(sum(f.get(k, 0) for k in ("Hepatocyte", "Endothelial", "Fibroblast")), 2),
                 "pct_Hepatocyte": round(f.get("Hepatocyte", 0), 2),
                 "pct_Endothelial": round(f.get("Endothelial", 0), 2),
                 "pct_Fibroblast": round(f.get("Fibroblast", 0), 2),
                 "pct_Myeloid": round(f.get("Myeloid", 0), 2), "pct_B": round(f.get("B", 0), 2),
                 "pct_Unknown": round(f.get("Unknown", 0) + f.get("NotRun", 0), 2)})


for t in r31.TOOLS:
    if t in raw:                      # tools without predictions yet are skipped
        summarise(t, r31.map_raw(t, raw[t], "broad6"))
e = pd.read_csv(f"{r31.ev.GT_DIR}/{DS}_enrichr_ground_truth.tsv", sep="\t", index_col=0)
e.index = e.index.astype(str)
summarise("enrichR GT (reference only)",
          e["ground_truth_enrichr_fine"].astype(str).map(lambda t: r31.cb.gt_broad6(t, r31.bl.REJECT_KEYWORDS)))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.to_string(index=False))
