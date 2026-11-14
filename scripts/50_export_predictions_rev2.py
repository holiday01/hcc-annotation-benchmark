"""
50_export_predictions_rev2.py — per-cell predictions of all twelve configurations (revision 2).

Writes predictions_rev2/<GSE>_predictions_rev2.csv.gz: one row per benchmark cell, one column per
tool configuration, holding the tool's raw (pre-harmonisation) label exactly as scored by
31_rescore_common_cells.py (pred_set="rev2"). Map them to the evaluation classes with
_canonical_broad.py (or look them up in expected_results_rev2/rev2_label_mapping.csv).
For GSE125449 an extra column CellAssign_7class holds the seven-class CellAssign run.

Also copies the shared-reference cell list and the reference-exclusion table to reference_rev2/.
"""
import importlib.util
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("BENCHMARK_DATA_DIR", "/mnt/10t/assi_result/HCC/benchmark_datasets")
spec = importlib.util.spec_from_file_location("r31", os.path.join(HERE, "31_rescore_common_cells.py"))
r31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r31)

COHORTS = ["GSE125449", "GSE149614", "GSE156625", "GSE162616", "GSE202642", "GSE223204"]
ORDER = ["CellTypist", "SingleR", "SingleRdisco", "SeuratLT", "scmap", "scPred", "SingleCellNet",
         "ScType", "CellAssign", "SignacX", "scGPT", "CyteType"]
COLNAME = {"SingleR": "SingleR_HPCA", "SingleRdisco": "SingleR_liverref", "SeuratLT": "Seurat_label_transfer"}

out = os.path.join(ROOT, "predictions_rev2")
os.makedirs(out, exist_ok=True)
for ds in COHORTS:
    raw, not_run = r31.load_raw_tools(ds, "broad6", "rev2")
    assert not not_run, (ds, not_run)
    missing = [t for t in ORDER if t not in raw]
    assert not missing, (ds, missing)
    idx = raw[ORDER[0]].index
    for t in ORDER[1:]:
        idx = idx.union(raw[t].index)
    df = pd.DataFrame({COLNAME.get(t, t): raw[t].reindex(idx) for t in ORDER}, index=idx)
    if ds == "GSE125449":
        raw7, _ = r31.load_raw_tools(ds, "broad7", "rev2")
        df["CellAssign_7class"] = raw7["CellAssign"].reindex(idx)
    df.index.name = "barcode"
    n_na = int(df.isna().sum().sum())
    df.to_csv(os.path.join(out, f"{ds}_predictions_rev2.csv.gz"), compression="gzip")
    print(f"{ds}: {len(df):,} cells x {df.shape[1]} columns, missing values {n_na}")

ref = os.path.join(ROOT, "reference_rev2")
os.makedirs(ref, exist_ok=True)
shutil.copy(os.path.join(BASE, "rev2_inputs", "DISCO_reference", "cells.csv"),
            os.path.join(ref, "DISCO_shared_reference_cells.csv"))
shutil.copy(os.path.join(BASE, "annotation_results_rev2", "scgpt_reference_exclusion.csv"),
            os.path.join(ref, "benchmark_series_excluded_from_reference.csv"))
print("reference files copied")
