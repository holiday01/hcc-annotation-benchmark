"""
36_rerun_CellAssign_rev2.py — CellAssign on every benchmark cell with the documented model input.

Revision 2. The original CellAssign runs (05_run_CellAssign_on_benchmark.py,
27_run_CellAssign_GSE156625.py) called
    sc.pp.normalize_total(adata_sub, target_sum=None, inplace=True)
    adata_sub.obs["size_factor"] = adata_sub.obs["total_counts"] if "total_counts" in adata_sub.obs else 1.0
on the marker-gene subset. normalize_total rescales X in place and does not add
`total_counts` to obs, so CellAssign received non-integer, marker-normalised
values with a size factor of 1 for every cell (no library-size offset). The
three DISCO-derived cohorts (GSE162616, GSE202642, GSE223204) were run in a
notebook with a different, fine-grained marker panel and on part of the cells.

Here, for all six cohorts, as in the scvi-tools CellAssign tutorial:
  * raw UMI counts (rev2_inputs/<GSE>/counts.h5ad, 33_prepare_counts_rev2.py);
  * size_factor = total UMI per cell over all genes / mean over cells;
  * X restricted to the marker genes, left as integer counts;
  * scvi-tools defaults for training (CellAssign.train()), seed 42.
Marker panels are the ones used in the paper (05_run_CellAssign_on_benchmark.py):
MARKERS_BROAD for every cohort, plus MARKERS_125449 (7 classes) for GSE125449.

Output: annotation_results_rev2/<GSE>_CellAssign_pred.csv           (broad-6 panel)
        annotation_results_rev2/GSE125449_CellAssign7_pred.csv      (7-class panel)
        columns: barcode, pred_celltype, max_prob

Usage: ~/miniconda/envs/drug_sc/bin/python 36_rerun_CellAssign_rev2.py [GSE ...]
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import scvi  # noqa: E402
from scvi.external import CellAssign  # noqa: E402

scvi.settings.seed = 42
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s05", HERE / "05_run_CellAssign_on_benchmark.py")
s05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s05)

BENCH = Path("/mnt/10t/assi_result/HCC/benchmark_datasets")
IN = BENCH / "rev2_inputs"
OUT = BENCH / "annotation_results_rev2"
COHORTS = ["GSE125449", "GSE149614", "GSE156625", "GSE162616", "GSE202642", "GSE223204"]


def marker_matrix(var_names, panel: dict) -> pd.DataFrame:
    avail = set(var_names)
    genes = sorted({g for gs in panel.values() for g in gs if g in avail})
    m = pd.DataFrame(0, index=genes, columns=list(panel))
    for ct, gs in panel.items():
        found = [g for g in gs if g in avail]
        m.loc[found, ct] = 1
        print(f"    {ct}: {len(found)}/{len(gs)} markers present", flush=True)
    return m


def run(ds: str, panel: dict, out_name: str) -> None:
    out_path = OUT / out_name
    if out_path.exists():
        print(f"skip (exists): {out_path}")
        return
    print(f"\n=== {ds} -> {out_name}", flush=True)
    a = ad.read_h5ad(IN / ds / "counts.h5ad")
    lib = np.asarray(a.X.sum(axis=1)).ravel().astype(float)
    a.obs["size_factor"] = lib / lib.mean()
    mm = marker_matrix(a.var_names, panel)
    b = a[:, mm.index].copy()
    b.X = b.X.astype(np.float32)
    print(f"    {b.n_obs} cells, {b.n_vars} marker genes; cells with zero marker UMI: "
          f"{int((np.asarray(b.X.sum(axis=1)).ravel() == 0).sum())}", flush=True)
    CellAssign.setup_anndata(b, size_factor_key="size_factor")
    model = CellAssign(b, mm)
    model.train()
    prob = model.predict()
    pred = prob.idxmax(axis=1).values
    pd.DataFrame({"barcode": b.obs_names, "pred_celltype": pred,
                  "max_prob": prob.max(axis=1).values.round(4)}).to_csv(out_path, index=False)
    print(pd.Series(pred).value_counts().to_string(), flush=True)


def main(cohorts):
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in cohorts:
        run(ds, s05.MARKERS_BROAD, f"{ds}_CellAssign_pred.csv")
        if ds == "GSE125449":
            run(ds, s05.MARKERS_125449, "GSE125449_CellAssign7_pred.csv")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or COHORTS))
