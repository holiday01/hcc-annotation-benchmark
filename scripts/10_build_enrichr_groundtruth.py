"""
Build enrichR-based ground truth for benchmark datasets.

For each dataset:
  1. Leiden cluster at a resolution that approximates the number of known cell types
  2. rank_genes_groups (t-test) to find top marker genes per cluster
  3. enrichR against CellMarker_2024 + PanglaoDB_Augmented_2021 per cluster
  4. Top hit becomes the cluster label
  5. Broad-category mapping to standardised labels
  6. Save new column 'ground_truth_enrichr' to h5ad and CSV

Usage:
    python 10_build_enrichr_groundtruth.py
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import gseapy as gp
import os
import warnings
warnings.filterwarnings("ignore")

sc.settings.verbosity = 1

BASE    = "/mnt/10t/assi_result/HCC/benchmark_datasets"
GT_DIR  = f"{BASE}/ground_truth"
os.makedirs(GT_DIR, exist_ok=True)

# Databases to query (in priority order)
ENRICH_LIBS = ["CellMarker_2024", "PanglaoDB_Augmented_2021", "Tabula_Sapiens"]

# Number of top marker genes to send to enrichR per cluster
N_MARKERS = 100

# ─────────────────────────────────────────────────────────────────────────────
# Broad category mapping — collapses fine-grained enrichR hits to standard labels
# Adjust as needed for each dataset's known biology
# ─────────────────────────────────────────────────────────────────────────────
# ── Reject list: clearly non-liver / non-immune cell types ──────────────────
# Any enrichR hit containing these keywords → "Unknown"
REJECT_KEYWORDS = [
    "goblet", "intestinal crypt", "paneth", "microglia", "microglial",
    "mesangial", "medullary", "leydig", "spermatogon", "trophoblast",
    "villous", "pancreatic acinar", "pancreatic ductal", "pancreatic duct",
    "osteoblast", "osteoclast", "chondrocyte", "adipocyte", "keratinocyte",
    "melanocyte", "secretory cell lung", "transit amplifying",
    "cardiac", "cardiomyocyte", "podocyte", "proximal tubule", "distal tubule",
    "collecting duct", "loop of henle", "bowman", "glomerular",
    "enterocyte", "colonocyte", "crypt", "epithelial cell lung",
    "epithelial cell ovary", "fetal germ", "germinal center",
    "multilymphoid progenitor", "megakaryocyte progenitor",
    "stromal cell bone", "schwann", "oligodendrocyte", "astrocyte",
    "dopaminergic", "purkinje", "granule cell", "cortical neuron",
    "leydig", "sertoli", "granulocyte progenitor", "erythroid progenitor",
    "common myeloid progenitor", "common lymphoid progenitor",
]

# ── Accept map: keyword → BROAD6 label ───────────────────────────────────────
BROAD_MAP = {
    # ── T cells (all subtypes) ────────────────────────────────────────────────
    "t cell":                        "T/NK",
    "cd4":                           "T/NK",
    "cd8":                           "T/NK",
    "treg":                          "T/NK",
    "regulatory t":                  "T/NK",
    "mait":                          "T/NK",
    "nkt":                           "T/NK",
    "exhausted":                     "T/NK",   # exhausted CD8+ T
    "cytotoxic t":                   "T/NK",
    "gamma delta":                   "T/NK",
    "gdelta":                        "T/NK",
    "intraepithelial lymphocyte":    "T/NK",
    "mucosal":                       "T/NK",
    # ── NK / ILC ──────────────────────────────────────────────────────────────
    "natural killer":                "T/NK",
    "nk cell":                       "T/NK",
    "nk-cell":                       "T/NK",
    "innate lymphoid":               "T/NK",
    "ilc":                           "T/NK",
    # ── Myeloid ───────────────────────────────────────────────────────────────
    "macrophage":                    "Myeloid",
    "kupffer":                       "Myeloid",
    "tumor-associated macrophage":   "Myeloid",
    "monocyte":                      "Myeloid",
    "dendritic cell":                "Myeloid",
    "dendritic":                     "Myeloid",
    "plasmacytoid dendritic":        "Myeloid",
    "conventional dc":               "Myeloid",
    " dc":                           "Myeloid",   # space before DC to avoid false positives
    "neutrophil":                    "Myeloid",
    "mast cell":                     "Myeloid",
    "basophil":                      "Myeloid",
    "eosinophil":                    "Myeloid",
    # ── B / Plasma ────────────────────────────────────────────────────────────
    "b cell":                        "B",
    "b-cell":                        "B",
    "plasma cell":                   "B",
    "plasmablast":                   "B",
    "memory b":                      "B",
    "naive b":                       "B",
    "germinal centre b":             "B",
    # ── Hepatocyte / liver parenchyma ─────────────────────────────────────────
    "hepatocyte":                    "Hepatocyte",
    "liver bud hepatic":             "Hepatocyte",
    "hepatic progenitor":            "Hepatocyte",
    # ── Endothelial ───────────────────────────────────────────────────────────
    "endothelial":                   "Endothelial",
    "vascular smooth muscle":        "Endothelial",  # lumped with vascular
    "lymphatic":                     "Endothelial",
    # ── Fibroblast / stromal ──────────────────────────────────────────────────
    "fibroblast":                    "Fibroblast",
    "stellate cell":                 "Fibroblast",
    "myofibroblast":                 "Fibroblast",
    "cancer-associated fibroblast":  "Fibroblast",
    "caf":                           "Fibroblast",
    "pericyte":                      "Fibroblast",
    "smooth muscle cell":            "Fibroblast",
    # ── Liver-specific others ─────────────────────────────────────────────────
    "cholangiocyte":                 "Cholangiocyte",
    "biliary":                       "Cholangiocyte",
    "hpc":                           "HPC-like",
    # ── Erythroid (present in some HCC datasets) ──────────────────────────────
    "erythroid":                     "Erythroid",
    "erythrocyte":                   "Erythroid",
    "red blood cell":                "Erythroid",
}


def broad_label(cell_type_str: str) -> str:
    """Map enrichR cell type hit to broad category.
    Returns 'Unknown' for non-liver / non-immune cell types.
    """
    ct_lower = cell_type_str.lower()
    # First: reject clearly non-relevant cell types
    for rej in REJECT_KEYWORDS:
        if rej in ct_lower:
            return "Unknown"
    # Then: accept known liver/immune types
    for key, broad in BROAD_MAP.items():
        if key.lower() in ct_lower:
            return broad
    # Fallback: not in accept list → Unknown
    return "Unknown"


def enrichr_annotate_cluster(gene_list: list, cluster_id: str) -> str:
    """
    Run enrichR on gene_list against ENRICH_LIBS.
    Returns the top cell type label (fine-grained string from best hit).
    """
    if len(gene_list) == 0:
        return "Unknown"

    for lib in ENRICH_LIBS:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=lib,
                organism="human",
                outdir=None,
                verbose=False,
            )
            res = enr.results
            if res is None or res.empty:
                continue
            # Filter by adjusted p-value
            sig = res[res["Adjusted P-value"] < 0.05].copy()
            if sig.empty:
                sig = res.copy()   # take best hit even if not significant
            sig = sig.sort_values("Adjusted P-value")
            top_term = sig.iloc[0]["Term"]
            # CellMarker terms look like "Hepatocyte_Human" or "T Cell_Human"
            # PanglaoDB terms look like "Hepatocyte" or "T cell"
            cell_type = top_term.split("_")[0].strip()
            print(f"    Cluster {cluster_id} ({lib}): {cell_type} (p={sig.iloc[0]['Adjusted P-value']:.3e})")
            return cell_type
        except Exception as e:
            print(f"    Cluster {cluster_id} ({lib}) error: {e}")
            continue

    return "Unknown"


def build_enrichr_gt(h5ad_path: str, dataset_name: str,
                     leiden_resolution: float = 0.5):
    print(f"\n{'='*60}")
    print(f"{dataset_name}  (resolution={leiden_resolution})")
    print(f"{'='*60}")

    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.shape}")

    # ── Leiden clustering ─────────────────────────────────────────────────────
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added="leiden_enrichr",
                 flavor="igraph", n_iterations=2, directed=False)
    n_clusters = adata.obs["leiden_enrichr"].nunique()
    print(f"  Leiden clusters: {n_clusters}")

    # ── Marker genes ──────────────────────────────────────────────────────────
    sc.tl.rank_genes_groups(adata, groupby="leiden_enrichr",
                            method="t-test", n_genes=N_MARKERS)

    # ── enrichR per cluster ───────────────────────────────────────────────────
    cluster_labels = {}   # cluster_id -> fine label
    cluster_broad  = {}   # cluster_id -> broad label

    for cluster in sorted(adata.obs["leiden_enrichr"].unique(),
                          key=lambda x: int(x)):
        genes = sc.get.rank_genes_groups_df(
            adata, group=cluster, key="rank_genes_groups"
        )["names"].tolist()
        genes = [g for g in genes if isinstance(g, str)][:N_MARKERS]

        print(f"  Cluster {cluster}: {len(genes)} marker genes")
        fine  = enrichr_annotate_cluster(genes, cluster)
        broad = broad_label(fine)
        cluster_labels[cluster] = fine
        cluster_broad[cluster]  = broad
        print(f"    → fine: {fine}  broad: {broad}")

    # ── Map back to cells ─────────────────────────────────────────────────────
    adata.obs["ground_truth_enrichr_fine"]  = (
        adata.obs["leiden_enrichr"].map(cluster_labels)
    )
    adata.obs["ground_truth_enrichr_broad"] = (
        adata.obs["leiden_enrichr"].map(cluster_broad)
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  enrichR broad GT distribution:")
    print(adata.obs["ground_truth_enrichr_broad"].value_counts())
    print(f"\n  Cluster → label mapping:")
    for c in sorted(cluster_labels, key=int):
        n = (adata.obs["leiden_enrichr"] == c).sum()
        print(f"    Cluster {c:>2} ({n:>6} cells): "
              f"{cluster_labels[c]}  →  {cluster_broad[c]}")

    # ── Comparison with published GT ─────────────────────────────────────────
    if "ground_truth" in adata.obs.columns:
        print(f"\n  Overlap with published GT:")
        cross = pd.crosstab(
            adata.obs["ground_truth"],
            adata.obs["ground_truth_enrichr_broad"],
            margins=False
        )
        print(cross.to_string())

    # ── Save ─────────────────────────────────────────────────────────────────
    out_h5ad = h5ad_path.replace("_benchmark.h5ad", "_benchmark_enrichrGT.h5ad")
    adata.write_h5ad(out_h5ad)
    print(f"\n  Saved h5ad: {out_h5ad}")

    # CSV for downstream use
    out_csv = f"{GT_DIR}/{dataset_name}_enrichr_ground_truth.tsv"
    gt_df = adata.obs[["leiden_enrichr",
                        "ground_truth_enrichr_fine",
                        "ground_truth_enrichr_broad"]].copy()
    if "ground_truth" in adata.obs.columns:
        gt_df["published_ground_truth"] = adata.obs["ground_truth"]
    gt_df.index.name = "barcode"
    gt_df.to_csv(out_csv, sep="\t")
    print(f"  Saved GT TSV: {out_csv}")

    return adata


if __name__ == "__main__":
    # GSE125449: 7 published cell types → resolution ~0.5 gives ~15-20 clusters,
    # which we collapse to broad categories
    build_enrichr_gt(
        f"{BASE}/GSE125449_benchmark.h5ad",
        "GSE125449",
        leiden_resolution=0.5,
    )

    # GSE149614: 6 published cell types → similar resolution
    build_enrichr_gt(
        f"{BASE}/GSE149614_benchmark.h5ad",
        "GSE149614",
        leiden_resolution=0.3,
    )

    print("\nAll done.")
