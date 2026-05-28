"""
Run CyteType on GSE149614 and GSE125449 benchmark datasets.
CyteType is cluster-based: cluster → DE → API call → per-cluster label → map to cells.

Usage:
    python 04_run_CyteType_on_benchmark.py
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os, random, json

OUT_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
RESP_DIR = "/mnt/10t/holiday/hcc_paper/release/cytetype_responses"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESP_DIR, exist_ok=True)

# Deterministic seeding for Leiden / DE / preprocessing.
SEED = 42
random.seed(SEED); np.random.seed(SEED)


def run_cytetype(h5ad_path, dataset_name):
    print(f"\n=== {dataset_name} ===")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.shape}")

    # ── Preprocessing ────────────────────────────────────────────────────────
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata, random_state=SEED)
    sc.pp.neighbors(adata, random_state=SEED)

    # ── Leiden clustering ─────────────────────────────────────────────────────
    sc.tl.leiden(adata, key_added="clusters", resolution=0.5, random_state=SEED)
    n_clusters = adata.obs["clusters"].nunique()
    print(f"  Leiden clusters: {n_clusters}")

    # ── Rank genes per cluster (required by CyteType) ─────────────────────────
    sc.tl.rank_genes_groups(adata, groupby="clusters", method="t-test",
                            n_genes=50)

    # gene_symbols must be in adata.var
    adata.var["gene_symbols"] = adata.var_names

    # ── CyteType ──────────────────────────────────────────────────────────────
    from cytetype import CyteType

    print("  Calling CyteType API...")
    annotator = CyteType(adata, group_key="clusters")
    adata = annotator.run(
        study_context="Human liver hepatocellular carcinoma single-cell RNA-seq"
    )

    pred_col = "cytetype_annotation_clusters"
    print(f"  CyteType done. Cluster distribution:\n"
          f"{adata.obs[pred_col].value_counts()}")

    # ── Save raw per-cluster API response (for "frozen API responses" claim) ─
    try:
        jd = dict(adata.uns.get("cytetype_jobDetails", {}))
        rr = dict(adata.uns.get("cytetype_results", {}))
        jd.pop("auth_token", None)              # strip credentials
        result_field = rr.get("result")
        try:
            result_parsed = json.loads(result_field) if isinstance(result_field, str) else result_field
        except Exception as e:
            result_parsed = {"parse_error": str(e), "raw": result_field}
        payload = {"dataset": dataset_name, "job_id": rr.get("job_id"),
                   "jobDetails": jd, "result": result_parsed}
        resp_path = f"{RESP_DIR}/{dataset_name}_cytetype_response.json"
        with open(resp_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  Saved raw API response: {resp_path}")
    except Exception as e:
        print(f"  WARNING: could not save raw API response: {e}")

    # ── Save per-cell predictions ───────────────────────────────────────────
    out_df = pd.DataFrame({
        "barcode":       adata.obs_names,
        "pred_celltype": adata.obs[pred_col].values,
        "ground_truth":  adata.obs["ground_truth"].values,
    }, index=adata.obs_names)

    out_path = f"{OUT_DIR}/{dataset_name}_CyteType_pred.csv"
    out_df.to_csv(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Prediction distribution:\n{out_df['pred_celltype'].value_counts()}")


if __name__ == "__main__":
    BASE = "/mnt/10t/assi_result/HCC/benchmark_datasets"
    run_cytetype(f"{BASE}/GSE125449_benchmark.h5ad", "GSE125449")
    run_cytetype(f"{BASE}/GSE149614_benchmark.h5ad", "GSE149614")
    print("\nAll done.")
