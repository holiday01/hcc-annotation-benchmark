"""
33_prepare_counts_rev2.py — raw UMI counts for every benchmark cell (revision 2).

SignacX and CellAssign model raw counts, but every benchmark AnnData stores
log1p(normalize_total(target_sum=1e4)) values only. For five cohorts the
counts are recovered exactly: in each cell the smallest non-zero value of
expm1(X) corresponds to one UMI, so counts = round(expm1(X) / min_nonzero);
every cell is checked (max |expm1(X)/min - round| must be < 0.05, else the
script stops). GSE156625's stored values are not recoverable, so its counts
come from the GEO matrix (GSE156625_HCCmatrix.mtx), restricted to the
benchmark cells.

Output, per cohort, in /mnt/10t/assi_result/HCC/benchmark_datasets/rev2_inputs/<GSE>/:
  counts.h5ad   cells x genes, integer counts, obs: sample   (CellAssign)
  matrix.mtx, genes.tsv, barcodes.tsv, cells.csv (barcode, sample, group)  (SignacX)
`group` = sample, except that samples with < 200 cells are pooled per cohort
("pooled_small") so every SignacX run has enough cells for a neighbour graph.

Usage: ~/.pyenv/versions/3.11.9/bin/python 33_prepare_counts_rev2.py [GSE ...]
"""
from __future__ import annotations

import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

BENCH = "/mnt/10t/assi_result/HCC/benchmark_datasets"
OPEN = "/mnt/10t/assi_result/HCC/open_source_model"
GEO = "/mnt/10t/assi_result/GEO_DATA"
OUT = f"{BENCH}/rev2_inputs"
MIN_GROUP = 200

H5AD = {
    "GSE125449": f"{BENCH}/GSE125449_benchmark.h5ad",
    "GSE149614": f"{BENCH}/GSE149614_benchmark.h5ad",
    "GSE162616": f"{OPEN}/cytetype/adata_CyteType_GSE162616.h5ad",
    "GSE202642": f"{OPEN}/cytetype/adata_CyteType_GSE202642.h5ad",
    "GSE223204": f"{OPEN}/cytetype/adata_CyteType_GSE223204.h5ad",
}


def recover_counts(X) -> sp.csr_matrix:
    X = sp.csr_matrix(X, dtype=np.float64)
    X.data = np.expm1(X.data)
    worst = 0.0
    out_data = np.empty_like(X.data)
    for i in range(X.shape[0]):
        s, e = X.indptr[i], X.indptr[i + 1]
        if s == e:
            continue
        v = X.data[s:e]
        c = v / v.min()
        r = np.rint(c)
        worst = max(worst, float(np.abs(c - r).max()))
        out_data[s:e] = r
    if worst >= 0.05:
        raise SystemExit(f"count recovery failed: max residual {worst:.4f}")
    print(f"    recovered counts; max residual over all cells = {worst:.2e}")
    C = sp.csr_matrix((out_data.astype(np.int32), X.indices, X.indptr), shape=X.shape)
    C.eliminate_zeros()
    return C


def samples_for(ds: str, obs_names: pd.Index, obs: pd.DataFrame) -> pd.Series:
    if ds in ("GSE125449", "GSE149614"):
        g = pd.read_csv(f"{BENCH}/ground_truth/{ds}_ground_truth.tsv", sep="\t", index_col=0)
        g.index = g.index.astype(str)
        return g["sample"].astype(str).reindex(obs_names)
    if ds == "GSE156625":
        g = pd.read_csv(f"{BENCH}/ground_truth/GSE156625_enrichr_ground_truth.tsv", sep="\t", index_col=0)
        g.index = g.index.astype(str)
        return g["patient_tumorsection"].astype(str).reindex(obs_names)
    return obs["sample"].astype(str)


def load(ds: str) -> ad.AnnData:
    if ds == "GSE156625":
        bench = ad.read_h5ad(f"{BENCH}/GSE156625_benchmark.h5ad", backed="r")
        cells = pd.Index(bench.obs_names.astype(str))
        sup = f"{GEO}/GSE156625/suppl"
        m = sio.mmread(f"{sup}/GSE156625_HCCmatrix.mtx").T.tocsr()
        bc = pd.read_csv(f"{sup}/GSE156625_HCCbarcodes.tsv", header=None, sep="\t")[0].astype(str)
        gdf = pd.read_csv(f"{sup}/GSE156625_HCCgenes.tsv", header=None, sep="\t")
        genes = (gdf[1] if gdf.shape[1] > 1 else gdf[0]).astype(str)
        raw = ad.AnnData(X=m)
        raw.obs_names = bc.values
        raw.var_names = genes.values
        raw.var_names_make_unique()
        if raw.obs_names.duplicated().any():
            raise SystemExit("GSE156625: duplicated barcodes in GEO matrix")
        missing = cells.difference(raw.obs_names)
        if len(missing):
            raise SystemExit(f"GSE156625: {len(missing)} benchmark cells absent from GEO matrix")
        raw = raw[cells].copy()
        vals = raw.X.data if sp.issparse(raw.X) else raw.X
        if not np.allclose(vals, np.rint(vals)):
            raise SystemExit("GSE156625: GEO matrix is not integer counts")
        raw.X = sp.csr_matrix(raw.X, dtype=np.int32)
        raw.obs = pd.DataFrame(index=raw.obs_names)
        return raw
    a = ad.read_h5ad(H5AD[ds])
    C = recover_counts(a.X)
    out = ad.AnnData(X=C, obs=pd.DataFrame(index=a.obs_names.astype(str)),
                     var=pd.DataFrame(index=a.var_names.astype(str)))
    out.obs["_src_sample"] = a.obs["sample"].astype(str).values if "sample" in a.obs else "NA"
    return out


def main(cohorts):
    for ds in cohorts:
        print(f"=== {ds}", flush=True)
        a = load(ds)
        a.var_names_make_unique()
        smp = samples_for(ds, a.obs_names, a.obs.rename(columns={"_src_sample": "sample"}))
        if smp.isna().any():
            raise SystemExit(f"{ds}: {int(smp.isna().sum())} cells without sample id")
        a.obs = pd.DataFrame({"sample": smp.values}, index=a.obs_names)
        sizes = a.obs["sample"].value_counts()
        small = set(sizes[sizes < MIN_GROUP].index)
        a.obs["group"] = [("pooled_small" if s in small else s) for s in a.obs["sample"]]
        print(f"    {a.n_obs} cells x {a.n_vars} genes; {sizes.size} samples "
              f"({len(small)} with < {MIN_GROUP} cells pooled); total UMI median "
              f"{int(np.median(np.asarray(a.X.sum(axis=1)).ravel()))}", flush=True)
        d = f"{OUT}/{ds}"
        os.makedirs(d, exist_ok=True)
        a.write_h5ad(f"{d}/counts.h5ad")
        sio.mmwrite(f"{d}/matrix.mtx", a.X.T.tocoo(), field="integer")
        pd.Series(a.var_names).to_csv(f"{d}/genes.tsv", index=False, header=False)
        pd.Series(a.obs_names).to_csv(f"{d}/barcodes.tsv", index=False, header=False)
        a.obs.rename_axis("barcode").to_csv(f"{d}/cells.csv")
        print(a.obs["group"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["GSE223204", "GSE125449", "GSE162616", "GSE156625", "GSE202642", "GSE149614"])
