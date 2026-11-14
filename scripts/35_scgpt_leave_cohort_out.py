"""
35_scgpt_leave_cohort_out.py — scGPT annotation with a reference free of benchmark cohorts.

Revision 2. The original reference for the scGPT k-NN classifier
(05_run_scGPT_annotation.py) was a 3,000-cells-per-class subsample of the DISCO
liver atlas, which contains samples of GSE162616 (2/4), GSE202642 (11/11) and
GSE223204 (2/2) — i.e. cells of the query cohorts could sit in the reference.
Here every DISCO cell whose `sample` or `sample_id` belongs to any of the six
benchmark series (GSM and SRX accessions from the local GEO series matrices,
plus the sample ids stored in the DISCO-derived query objects) is removed
before the per-class subsample. Everything else is unchanged: same checkpoint
(scGPT-human), same DISCO -> broad-6 map, 3,000 cells per class (seed 42),
embed_data(max_length=1200), cosine k-NN with k = 15. Embeddings are saved,
and k in {5, 10, 15, 20, 30} is reported as a sensitivity analysis.

Environment: ~/envs/scgpt_rev2 (venv on pyenv 3.11.9 with torch 2.10+cu128,
numpy 2.1.3, and a pure-Python torchtext.vocab stand-in; torchtext 0.18 cannot
load with torch builds that support sm_120).

Modes
  --validate  original (unfiltered) reference -> GSE125449; prints agreement
              with annotation_results/GSE125449_scGPT_pred.csv
  (default)   filtered reference -> all six cohorts

Output: annotation_results_rev2/<GSE>_scGPT_pred.csv (pred_celltype = k 15),
        annotation_results_rev2/scgpt_k_sensitivity/<GSE>_scGPT_k.csv,
        annotation_results_rev2/scgpt_embeddings/*.npz,
        annotation_results_rev2/scgpt_reference_exclusion.csv
"""
from __future__ import annotations

import argparse
import glob
import gzip
import importlib.util
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
import torch  # noqa: E402

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import scanpy as sc  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s05", HERE / "05_run_scGPT_annotation.py")
s05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s05)  # reuse DISCO_TO_BROAD6, BROAD6_LABELS, DATASETS, paths

BASE = Path("/mnt/10t/assi_result/HCC")
OUT = BASE / "benchmark_datasets" / "annotation_results_rev2"
GEO = Path("/mnt/10t/assi_result/GEO_DATA")
SERIES = ["GSE125449", "GSE149614", "GSE156625", "GSE162616", "GSE202642", "GSE223204"]
KS = [5, 10, 15, 20, 30]


def series_accessions() -> dict[str, set[str]]:
    """GSM and SRX accessions of each benchmark series."""
    acc = {s: set() for s in SERIES}
    for s in SERIES:
        for f in glob.glob(str(GEO / s / "matrix" / "*_series_matrix.txt")):
            with open(f, errors="ignore") as fh:
                for line in fh:
                    if line.startswith("!Sample_geo_accession"):
                        acc[s] |= set(re.findall(r"GSM\d+", line))
                    elif line.startswith("!Sample_relation"):
                        acc[s] |= set(re.findall(r"SRX\d+", line))
                    elif line.startswith("!series_matrix_table_begin"):
                        break
    for s in ("GSE162616", "GSE202642", "GSE223204"):
        a = sc.read_h5ad(s05.DATASETS[s], backed="r")
        acc[s] |= set(a.obs["sample"].astype(str))
    return acc


def build_reference(exclude: bool):
    print("Loading DISCO ...", flush=True)
    ref = sc.read_h5ad(s05.DISCO_H5AD)
    ref.obs["broad6"] = ref.obs["cell_type"].map(s05.DISCO_TO_BROAD6)
    rows = []
    if exclude:
        acc = series_accessions()
        smp = ref.obs["sample"].astype(str)
        sid = ref.obs["sample_id"].astype(str)
        drop = np.zeros(ref.n_obs, dtype=bool)
        for s, ids in acc.items():
            hit = smp.isin(ids) | sid.isin(ids)
            rows.append({"series": s, "n_accessions": len(ids), "n_DISCO_cells_removed": int(hit.sum()),
                         "n_DISCO_samples_removed": int(smp[hit].nunique())})
            drop |= hit.values
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        ref = ref[~drop].copy()
    ref = ref[ref.obs["broad6"].notna()].copy()
    idx = []
    for label in s05.BROAD6_LABELS:
        m = ref.obs["broad6"] == label
        take = min(int(m.sum()), 3000)
        idx.extend(ref.obs.index[m].to_series().sample(take, random_state=42).tolist())
        print(f"  reference {label}: {take} of {int(m.sum())}", flush=True)
    return ref[idx].copy(), pd.DataFrame(rows)


def embed(adata, batch_size):
    t0 = time.time()
    e = s05.run_scgpt_embed(adata, s05.MODEL_DIR, batch_size=batch_size)
    print(f"  embedded {e.shape[0]} cells in {time.time() - t0:.0f} s", flush=True)
    return np.asarray(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--cohorts", nargs="*", default=SERIES)
    args = ap.parse_args()
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    (OUT / "scgpt_embeddings").mkdir(parents=True, exist_ok=True)
    (OUT / "scgpt_k_sensitivity").mkdir(parents=True, exist_ok=True)

    ref, excl = build_reference(exclude=not args.validate)
    tag = "original_reference" if args.validate else "leave_cohorts_out"
    ref_emb = embed(ref, 128)
    ref_lab = ref.obs["broad6"].astype(str).values
    np.savez_compressed(OUT / "scgpt_embeddings" / f"reference_{tag}.npz",
                        emb=ref_emb, label=ref_lab, cell=ref.obs_names.values.astype(str),
                        sample=ref.obs["sample"].astype(str).values)
    if not args.validate:
        excl.to_csv(OUT / "scgpt_reference_exclusion.csv", index=False)

    cohorts = ["GSE125449"] if args.validate else args.cohorts
    for ds in cohorts:
        print(f"\n=== {ds} ({tag})", flush=True)
        q = s05.prepare_query(s05.DATASETS[ds])
        q_emb = embed(q, 64)
        np.savez_compressed(OUT / "scgpt_embeddings" / f"{ds}_query.npz",
                            emb=q_emb, cell=q.obs_names.values.astype(str))
        preds = {}
        for k in KS:
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
            knn.fit(ref_emb, ref_lab)
            preds[k] = knn.predict(q_emb)
        if args.validate:
            orig = pd.read_csv(BASE / "benchmark_datasets/annotation_results/GSE125449_scGPT_pred.csv",
                               index_col=0)["pred_celltype"].astype(str)
            new = pd.Series(preds[15], index=q.obs_names.astype(str))
            agree = (orig.reindex(new.index) == new).mean()
            print(f"VALIDATION agreement with original GSE125449 scGPT predictions (k=15): {agree:.4f}")
            print(pd.crosstab(orig.reindex(new.index), new).to_string())
            continue
        pd.DataFrame({"pred_celltype": preds[15]}, index=pd.Index(q.obs_names.astype(str), name="cell_id")) \
            .to_csv(OUT / f"{ds}_scGPT_pred.csv")
        pd.DataFrame({f"k{k}": v for k, v in preds.items()}, index=pd.Index(q.obs_names.astype(str), name="cell_id")) \
            .to_csv(OUT / "scgpt_k_sensitivity" / f"{ds}_scGPT_k.csv")
        print(pd.Series(preds[15]).value_counts().to_string(), flush=True)
    print("\nAll done.")


if __name__ == "__main__":
    sys.exit(main())
