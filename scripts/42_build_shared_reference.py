"""
42_build_shared_reference.py — one liver reference shared by every reference-based tool (revision 2).

The cells are exactly those of the scGPT k-NN reference built by
35_scgpt_leave_cohort_out.py: DISCO liver atlas, every sample of the six
benchmark series removed, 3,000 cells per broad-6 class (1,141 fibroblasts),
seed 42 (cell ids from scgpt_embeddings/reference_leave_cohorts_out.npz).
Raw counts are recovered exactly from DISCO's stored values
(33_prepare_counts_rev2.recover_counts). Training every reference-based tool
(Seurat label transfer, SingleR, scmap, scPred, SingleCellNet, scGPT k-NN) on
the same cells and labels separates the effect of the algorithm from the
effect of the reference.

Output: benchmark_datasets/rev2_inputs/DISCO_reference/
  counts.h5ad; matrix.mtx, genes.tsv, barcodes.tsv; cells.csv (barcode, label, disco_cell_type, sample)
"""
import importlib.util
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p33", os.path.join(HERE, "33_prepare_counts_rev2.py"))
p33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p33)
DISCO = "/mnt/10t/assi_result/DISCO_DATA/disco_adata_mt5_cell200gen5_no_combat_with_celltype_20240415.h5ad"
NPZ = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2/scgpt_embeddings/reference_leave_cohorts_out.npz"
OUT = "/mnt/10t/assi_result/HCC/benchmark_datasets/rev2_inputs/DISCO_reference"

ref = np.load(NPZ, allow_pickle=True)
cells = pd.Index(ref["cell"].astype(str))
labels = pd.Series(ref["label"].astype(str), index=cells)
a = ad.read_h5ad(DISCO, backed="r")
idx = a.obs_names.get_indexer(cells)
if (idx < 0).any():
    raise SystemExit(f"{int((idx < 0).sum())} reference cells not found in DISCO")
order = np.argsort(idx)
sub = a[idx[order]].to_memory()
C = p33.recover_counts(sub.X)
r = ad.AnnData(X=C, obs=pd.DataFrame(index=sub.obs_names.astype(str)),
               var=pd.DataFrame(index=sub.var_names.astype(str)))
r.var_names_make_unique()
r.obs["label"] = labels.reindex(r.obs_names).values
r.obs["disco_cell_type"] = sub.obs["cell_type"].astype(str).values
r.obs["sample"] = sub.obs["sample"].astype(str).values
os.makedirs(OUT, exist_ok=True)
r.write_h5ad(f"{OUT}/counts.h5ad")
sio.mmwrite(f"{OUT}/matrix.mtx", r.X.T.tocoo(), field="integer")
pd.Series(r.var_names).to_csv(f"{OUT}/genes.tsv", index=False, header=False)
pd.Series(r.obs_names).to_csv(f"{OUT}/barcodes.tsv", index=False, header=False)
r.obs.rename_axis("barcode").to_csv(f"{OUT}/cells.csv")
print(r.shape, r.obs["label"].value_counts().to_dict(), "samples:", r.obs["sample"].nunique())
