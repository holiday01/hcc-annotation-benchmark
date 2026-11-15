"""
Build enrichR-based ground truth for the original 3 HCC datasets:
GSE162616, GSE202642, GSE223204.

These h5ad files already have Leiden clusters (column 'clusters').
Skip re-clustering; use existing clusters directly for marker gene extraction.

Output:
  ground_truth/{dataset}_enrichr_ground_truth.tsv
  (adds ground_truth_enrichr_fine / ground_truth_enrichr_broad columns)
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import gseapy as gp
import os
import warnings
warnings.filterwarnings("ignore")

sc.settings.verbosity = 0

OM_DIR  = "/mnt/10t/assi_result/HCC/open_source_model/cytetype"
GT_DIR  = "/mnt/10t/assi_result/HCC/benchmark_datasets/ground_truth"
os.makedirs(GT_DIR, exist_ok=True)

ENRICH_LIBS = ["CellMarker_2024", "PanglaoDB_Augmented_2021", "Tabula_Sapiens"]
N_MARKERS   = 100

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
    "sertoli", "granulocyte progenitor", "erythroid progenitor",
    "common myeloid progenitor", "common lymphoid progenitor",
]

BROAD_MAP = {
    # T cells
    "t cell":                        "T/NK",
    "cd4":                           "T/NK",
    "cd8":                           "T/NK",
    "treg":                          "T/NK",
    "regulatory t":                  "T/NK",
    "mait":                          "T/NK",
    "nkt":                           "T/NK",
    "exhausted":                     "T/NK",
    "cytotoxic t":                   "T/NK",
    "gamma delta":                   "T/NK",
    "mucosal":                       "T/NK",
    "intraepithelial lymphocyte":    "T/NK",
    # NK / ILC
    "natural killer":                "T/NK",
    "nk cell":                       "T/NK",
    "nk-cell":                       "T/NK",
    "innate lymphoid":               "T/NK",
    "ilc":                           "T/NK",
    # Myeloid
    "macrophage":                    "Myeloid",
    "kupffer":                       "Myeloid",
    "tumor-associated macrophage":   "Myeloid",
    "monocyte":                      "Myeloid",
    "dendritic cell":                "Myeloid",
    "dendritic":                     "Myeloid",
    "plasmacytoid dendritic":        "Myeloid",
    " dc":                           "Myeloid",
    "neutrophil":                    "Myeloid",
    "mast cell":                     "Myeloid",
    "basophil":                      "Myeloid",
    "eosinophil":                    "Myeloid",
    # B / Plasma
    "b cell":                        "B",
    "b-cell":                        "B",
    "plasma cell":                   "B",
    "plasmablast":                   "B",
    "memory b":                      "B",
    "naive b":                       "B",
    # Hepatocyte
    "hepatocyte":                    "Hepatocyte",
    "liver bud hepatic":             "Hepatocyte",
    "hepatic progenitor":            "Hepatocyte",
    # Endothelial
    "endothelial":                   "Endothelial",
    "lymphatic":                     "Endothelial",
    # Fibroblast / stromal
    "fibroblast":                    "Fibroblast",
    "stellate cell":                 "Fibroblast",
    "myofibroblast":                 "Fibroblast",
    "cancer-associated fibroblast":  "Fibroblast",
    "caf":                           "Fibroblast",
    "pericyte":                      "Fibroblast",
    "smooth muscle cell":            "Fibroblast",
    # Liver-specific
    "cholangiocyte":                 "Cholangiocyte",
    "biliary":                       "Cholangiocyte",
    "hpc":                           "HPC-like",
    # Erythroid
    "erythroid":                     "Erythroid",
    "erythrocyte":                   "Erythroid",
    "red blood cell":                "Erythroid",
}


def broad_label(fine: str) -> str:
    ct_lower = fine.lower()
    for rej in REJECT_KEYWORDS:
        if rej in ct_lower:
            return "Unknown"
    for key, broad in BROAD_MAP.items():
        if key.lower() in ct_lower:
            return broad
    return "Unknown"


def enrichr_annotate(gene_list, cluster_id):
    if not gene_list:
        return "Unknown"
    for lib in ENRICH_LIBS:
        try:
            enr = gp.enrichr(gene_list=gene_list, gene_sets=lib,
                             organism="human", outdir=None, verbose=False)
            res = enr.results
            if res is None or res.empty:
                continue
            sig = res.sort_values("Adjusted P-value")
            top = sig.iloc[0]["Term"].split("_")[0].strip()
            p   = sig.iloc[0]["Adjusted P-value"]
            print(f"    Cluster {cluster_id} ({lib}): {top} (p={p:.2e})")
            return top
        except Exception as e:
            print(f"    Cluster {cluster_id} ({lib}) error: {e}")
    return "Unknown"


for ds in ["GSE162616", "GSE202642", "GSE223204"]:
    print(f"\n{'='*60}")
    print(f"{ds}")
    print(f"{'='*60}")

    h5ad = f"{OM_DIR}/adata_CyteType_{ds}.h5ad"
    adata = ad.read_h5ad(h5ad)
    print(f"  Shape: {adata.shape}")

    cluster_col = "clusters"
    if cluster_col not in adata.obs.columns:
        print(f"  No '{cluster_col}' column — running Leiden at resolution 0.3")
        sc.tl.leiden(adata, resolution=0.3, key_added=cluster_col,
                     flavor="igraph", n_iterations=2, directed=False)

    n_clusters = adata.obs[cluster_col].nunique()
    print(f"  Clusters: {n_clusters}")

    # rank genes per cluster
    sc.tl.rank_genes_groups(adata, groupby=cluster_col,
                            method="t-test", n_genes=N_MARKERS)

    cluster_fine  = {}
    cluster_broad = {}

    for cl in sorted(adata.obs[cluster_col].unique(), key=lambda x: int(str(x))):
        genes = sc.get.rank_genes_groups_df(
            adata, group=str(cl), key="rank_genes_groups"
        )["names"].tolist()
        genes = [g for g in genes if isinstance(g, str)][:N_MARKERS]
        print(f"  Cluster {cl}: {len(genes)} marker genes")
        fine  = enrichr_annotate(genes, cl)
        broad = broad_label(fine)
        cluster_fine[str(cl)]  = fine
        cluster_broad[str(cl)] = broad
        print(f"    → {fine}  →  {broad}")

    adata.obs["ground_truth_enrichr_fine"]  = (
        adata.obs[cluster_col].astype(str).map(cluster_fine)
    )
    adata.obs["ground_truth_enrichr_broad"] = (
        adata.obs[cluster_col].astype(str).map(cluster_broad)
    )

    print(f"\n  enrichR broad distribution:")
    print(adata.obs["ground_truth_enrichr_broad"].value_counts())

    out_tsv = f"{GT_DIR}/{ds}_enrichr_ground_truth.tsv"
    gt_df = adata.obs[[cluster_col,
                        "ground_truth_enrichr_fine",
                        "ground_truth_enrichr_broad"]].copy()
    if "own_assign_cell_type" in adata.obs.columns:
        gt_df["own_assign_cell_type"] = adata.obs["own_assign_cell_type"]
    if "cell_type" in adata.obs.columns:
        gt_df["disco_cell_type"] = adata.obs["cell_type"]
    gt_df.index.name = "barcode"
    gt_df.to_csv(out_tsv, sep="\t")
    print(f"  Saved: {out_tsv}")

print("\nAll done.")
