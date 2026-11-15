"""
Build (or reuse) enrichR-based ground truth for benchmark datasets.

Default behaviour (deterministic / reproducible):
    Load the shipped snapshot TSV from ../ground_truth/<DATASET>_enrichr_ground_truth.tsv
    and write the broad label back onto the dataset's h5ad
    (column: ground_truth_enrichr_broad).

Refresh behaviour (re-queries the live EnrichR API; non-deterministic):
    pass --refresh on the command line. The pipeline is:
        1. Leiden clustering (resolution 0.5 for GSE125449, 0.3 for GSE149614).
        2. rank_genes_groups (t-test) to find the top 100 marker genes per cluster.
        3. enrichR against CellMarker_2024 per cluster (PanglaoDB_Augmented_2021,
           then Tabula_Sapiens, only if a library returns no result).
        4. The term with the smallest adjusted p-value becomes the cluster label.
        5. Broad-category mapping to standardised labels.
        6. Save 'ground_truth_enrichr_broad' to h5ad and overwrite the snapshot.

EnrichR snapshots shipped in this repository are dated 2026-04-04
(GSE125449, GSE149614, GSE162616, GSE202642, GSE223204) and 2026-04-14
(GSE156625). Every label comes from CellMarker_2024; the fallback libraries
were never reached. This script builds GSE125449 and GSE149614.
GSE162616, GSE202642 and GSE223204 were built by 10b_enrichr_gt_original3.py
from the clusters stored in the input objects, and GSE156625 by
16_prepare_gse156625_external.py from the authors' Louvain clusters and marker
ranking. Re-deriving the markers and the Enrichr query from the stored clusters
reproduces the snapshot term for all 115 clusters of the five cohorts.

Usage:
    # Re-attach shipped labels to h5ad (default)
    python 10_build_enrichr_groundtruth.py

    # Re-query EnrichR (results may differ if upstream databases changed)
    python 10_build_enrichr_groundtruth.py --refresh
"""

import argparse
import os
import warnings

import anndata as ad
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE   = os.environ.get("BENCHMARK_DATA_DIR",
                        "/mnt/10t/assi_result/HCC/benchmark_datasets")
GT_DIR = os.environ.get(
    "BENCHMARK_GT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "ground_truth"),
)
GT_DIR = os.path.abspath(GT_DIR)

# Databases to query (refresh mode only)
ENRICH_LIBS = ["CellMarker_2024", "PanglaoDB_Augmented_2021", "Tabula_Sapiens"]

# Number of top marker genes to send to enrichR per cluster
N_MARKERS = 100

# ─────────────────────────────────────────────────────────────────────────────
# Broad category mapping — collapses fine-grained enrichR hits to standard
# labels. Single canonical implementation is in _broad_label.py and is
# shared with 16_prepare_gse156625_external.py (previously these two
# scripts maintained drifted copies; cf. _broad_label.py header).
# ─────────────────────────────────────────────────────────────────────────────
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location(
    "_broad_label", _os.path.join(_os.path.dirname(__file__), "_broad_label.py"))
_bl = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_bl)
REJECT_KEYWORDS = _bl.REJECT_KEYWORDS
BROAD_MAP       = _bl.BROAD_MAP
broad_label     = _bl.broad_label


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot mode (default): re-attach shipped TSV to h5ad
# ─────────────────────────────────────────────────────────────────────────────
def attach_snapshot(h5ad_path: str, dataset_name: str):
    snapshot = os.path.join(GT_DIR, f"{dataset_name}_enrichr_ground_truth.tsv")
    if not os.path.exists(snapshot):
        raise FileNotFoundError(
            f"EnrichR snapshot missing: {snapshot}\n"
            f"Run with --refresh to regenerate, or restore from the repository."
        )
    print(f"\n{'='*60}\n{dataset_name}  (snapshot mode)\n{'='*60}")
    print(f"  Loading snapshot: {snapshot}")
    snap = pd.read_csv(snapshot, sep="\t", index_col="barcode")

    print(f"  Loading h5ad:    {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    common = adata.obs.index.intersection(snap.index)
    if len(common) == 0:
        raise ValueError(
            f"No overlapping barcodes between {h5ad_path} and {snapshot}"
        )
    coverage = len(common) / adata.n_obs
    print(f"  Barcode overlap: {len(common):,} / {adata.n_obs:,} "
          f"({coverage:.1%})")

    for col in ("ground_truth_enrichr_broad",
                "ground_truth_enrichr_fine",
                "leiden_enrichr"):
        if col in snap.columns:
            adata.obs[col] = snap[col].reindex(adata.obs.index)

    out_h5ad = h5ad_path.replace("_benchmark.h5ad",
                                 "_benchmark_enrichrGT.h5ad")
    adata.write_h5ad(out_h5ad)
    print(f"  Wrote h5ad with enrichR labels: {out_h5ad}")
    print(adata.obs["ground_truth_enrichr_broad"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Refresh mode: re-query EnrichR (non-deterministic over time)
# ─────────────────────────────────────────────────────────────────────────────
def enrichr_annotate_cluster(gene_list, cluster_id):
    import gseapy as gp

    if len(gene_list) == 0:
        return "Unknown"
    for lib in ENRICH_LIBS:
        try:
            enr = gp.enrichr(gene_list=gene_list, gene_sets=lib,
                             organism="human", outdir=None, verbose=False)
            res = enr.results
            if res is None or res.empty:
                continue
            sig = res[res["Adjusted P-value"] < 0.05].copy()
            if sig.empty:
                sig = res.copy()
            sig = sig.sort_values("Adjusted P-value")
            top_term = sig.iloc[0]["Term"]
            cell_type = top_term.split("_")[0].strip()
            print(f"    Cluster {cluster_id} ({lib}): {cell_type} "
                  f"(p={sig.iloc[0]['Adjusted P-value']:.3e})")
            return cell_type
        except Exception as e:
            print(f"    Cluster {cluster_id} ({lib}) error: {e}")
            continue
    return "Unknown"


def refresh_enrichr_gt(h5ad_path, dataset_name, leiden_resolution=0.5):
    import scanpy as sc

    sc.settings.verbosity = 1
    print(f"\n{'='*60}\n{dataset_name}  (refresh, resolution={leiden_resolution})\n{'='*60}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.shape}")

    sc.tl.leiden(adata, resolution=leiden_resolution,
                 key_added="leiden_enrichr",
                 flavor="igraph", n_iterations=2, directed=False)
    n_clusters = adata.obs["leiden_enrichr"].nunique()
    print(f"  Leiden clusters: {n_clusters}")
    sc.tl.rank_genes_groups(adata, groupby="leiden_enrichr",
                            method="t-test", n_genes=N_MARKERS)

    cluster_labels, cluster_broad = {}, {}
    for cluster in sorted(adata.obs["leiden_enrichr"].unique(),
                          key=lambda x: int(x)):
        genes = sc.get.rank_genes_groups_df(
            adata, group=cluster, key="rank_genes_groups"
        )["names"].tolist()
        genes = [g for g in genes if isinstance(g, str)][:N_MARKERS]
        fine  = enrichr_annotate_cluster(genes, cluster)
        broad = broad_label(fine)
        cluster_labels[cluster], cluster_broad[cluster] = fine, broad
        print(f"    Cluster {cluster}: fine={fine}  broad={broad}")

    adata.obs["ground_truth_enrichr_fine"]  = (
        adata.obs["leiden_enrichr"].map(cluster_labels)
    )
    adata.obs["ground_truth_enrichr_broad"] = (
        adata.obs["leiden_enrichr"].map(cluster_broad)
    )

    out_h5ad = h5ad_path.replace("_benchmark.h5ad",
                                 "_benchmark_enrichrGT.h5ad")
    adata.write_h5ad(out_h5ad)
    out_tsv = os.path.join(GT_DIR,
                           f"{dataset_name}_enrichr_ground_truth.tsv")
    gt_df = adata.obs[["leiden_enrichr",
                       "ground_truth_enrichr_fine",
                       "ground_truth_enrichr_broad"]].copy()
    if "ground_truth" in adata.obs.columns:
        gt_df["published_ground_truth"] = adata.obs["ground_truth"]
    gt_df.index.name = "barcode"
    gt_df.to_csv(out_tsv, sep="\t")
    print(f"  Wrote: {out_h5ad}\n         {out_tsv}")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = [
    # (dataset_name, leiden_resolution for refresh mode)
    ("GSE125449", 0.5),
    ("GSE149614", 0.3),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Re-query EnrichR instead of loading the shipped "
                             "snapshot. Results may differ from the snapshot "
                             "if EnrichR's underlying databases have changed.")
    parser.add_argument("--dataset", default=None,
                        help="Process only this dataset (e.g. GSE125449). "
                             "Default: all datasets in DATASETS.")
    args = parser.parse_args()

    targets = [d for d in DATASETS if args.dataset in (None, d[0])]
    for name, res in targets:
        h5ad = f"{BASE}/{name}_benchmark.h5ad"
        if args.refresh:
            refresh_enrichr_gt(h5ad, name, leiden_resolution=res)
        else:
            attach_snapshot(h5ad, name)

    print("\nDone.")


if __name__ == "__main__":
    main()
