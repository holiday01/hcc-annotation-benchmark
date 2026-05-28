"""
Prepare GSE156625_HCC as an external HCC benchmark dataset.

Rationale:
  - The public h5ad contains a low-dimensional/HVG expression matrix, but its
    .raw slot contains a full gene-space matrix.
  - The public h5ad also contains Louvain clusters and precomputed marker genes.
  - We reuse those marker genes to build an enrichR-derived broad6 ground truth,
    matching the existing benchmark framework as closely as possible.

Outputs:
  /mnt/10t/assi_result/HCC/benchmark_datasets/GSE156625_benchmark.h5ad
  /mnt/10t/assi_result/HCC/benchmark_datasets/ground_truth/GSE156625_enrichr_ground_truth.tsv
  /mnt/10t/holiday/hcc-annotation-benchmark/ground_truth/GSE156625_enrichr_ground_truth.tsv
"""

from __future__ import annotations

from pathlib import Path
import shutil
import warnings

import anndata as ad
import gseapy as gp
import pandas as pd
import scanpy as sc


warnings.filterwarnings("ignore")

SOURCE_H5AD = Path(
    "/mnt/10t/assi_result/GEO_DATA/GSE156625/suppl/GSE156625_HCCscanpyobj.h5ad"
)
BENCHMARK_DIR = Path("/mnt/10t/assi_result/HCC/benchmark_datasets")
WORK_GT_DIR = BENCHMARK_DIR / "ground_truth"
REPO_GT_DIR = Path("/mnt/10t/holiday/hcc-annotation-benchmark/ground_truth")
OUT_H5AD = BENCHMARK_DIR / "GSE156625_benchmark.h5ad"
OUT_GT = WORK_GT_DIR / "GSE156625_enrichr_ground_truth.tsv"

ENRICH_LIBS = ["CellMarker_2024", "PanglaoDB_Augmented_2021", "Tabula_Sapiens"]
N_MARKERS = 100

BROAD_6 = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]

# REJECT_KEYWORDS / BROAD_MAP / broad_label are imported from the shared
# _broad_label.py module to keep this script in sync with
# 10_build_enrichr_groundtruth.py. Previously these two scripts had
# drifted copies that mapped cholangiocyte/biliary/hpc to different
# broad classes between cohorts, producing inconsistent ground truth.
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location(
    "_broad_label", _os.path.join(_os.path.dirname(__file__), "_broad_label.py"))
_bl = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_bl)
REJECT_KEYWORDS = _bl.REJECT_KEYWORDS
BROAD_MAP       = _bl.BROAD_MAP
broad_label     = _bl.broad_label

MARKER_FALLBACK = {
    "T/NK": {"CD3D", "CD3E", "CD3G", "TRAC", "IL7R", "NKG7", "GNLY", "KLRD1"},
    "Myeloid": {"LYZ", "CD68", "C1QA", "C1QB", "LST1", "S100A8", "S100A9"},
    "B": {"MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "IGHG1"},
    "Hepatocyte": {"ALB", "APOA1", "APOA2", "TTR", "HP", "AFP", "GPC3", "KRT8"},
    "Endothelial": {"PECAM1", "VWF", "KDR", "CDH5", "ENG", "PLVAP"},
    "Fibroblast": {"COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "ACTA2", "RGS5"},
}


def fallback_label(marker_genes: list[str]) -> str:
    genes = {g.upper() for g in marker_genes[:50]}
    scores = {
        label: len(genes & markers)
        for label, markers in MARKER_FALLBACK.items()
    }
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    return best_label if best_score >= 2 else "Unknown"


def enrichr_cluster(marker_genes: list[str], cluster: str) -> tuple[str, str, str]:
    for lib in ENRICH_LIBS:
        try:
            enr = gp.enrichr(
                gene_list=marker_genes[:N_MARKERS],
                gene_sets=lib,
                organism="human",
                outdir=None,
                verbose=False,
            )
            res = enr.results
            if res is None or res.empty:
                continue
            sig = res[res["Adjusted P-value"] < 0.05].copy()
            if sig.empty:
                sig = res.copy()
            sig = sig.sort_values(["Adjusted P-value", "P-value"])
            top_term = str(sig.iloc[0]["Term"])
            fine = top_term.split("_")[0].strip()
            broad = broad_label(fine)
            print(f"Cluster {cluster:>2s} {lib}: {fine} -> {broad}")
            return fine, broad, lib
        except Exception as exc:
            print(f"Cluster {cluster:>2s} {lib} failed: {exc}")
    fallback = fallback_label(marker_genes)
    print(f"Cluster {cluster:>2s} fallback marker map -> {fallback}")
    return "MarkerFallback", fallback, "marker_fallback"


def extract_marker_genes(source: ad.AnnData) -> dict[str, list[str]]:
    names = source.uns["rank_genes_groups"]["names"]
    if not names.dtype.names:
        raise ValueError("rank_genes_groups['names'] does not have named groups")
    markers = {}
    for cluster in names.dtype.names:
        genes = [str(g) for g in names[cluster][:N_MARKERS] if isinstance(str(g), str)]
        markers[str(cluster)] = genes
    return markers


def main() -> None:
    WORK_GT_DIR.mkdir(parents=True, exist_ok=True)
    REPO_GT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading source: {SOURCE_H5AD}")
    source = ad.read_h5ad(SOURCE_H5AD)
    print(f"Source shape: {source.shape}; raw shape: {source.raw.shape if source.raw else None}")
    if source.raw is None:
        raise ValueError("GSE156625 source h5ad has no .raw matrix")

    print("Preparing full-gene log-normalized AnnData from .raw")
    full = source.raw.to_adata()
    full.obs = source.obs.copy()
    full.var_names_make_unique()
    sc.pp.normalize_total(full, target_sum=1e4)
    sc.pp.log1p(full)

    markers = extract_marker_genes(source)
    cluster_rows = []
    cluster_to_fine = {}
    cluster_to_broad = {}

    print(f"Annotating {len(markers)} Louvain clusters with enrichR")
    for cluster in sorted(markers.keys(), key=lambda x: int(x) if x.isdigit() else x):
        fine, broad, lib = enrichr_cluster(markers[cluster], cluster)
        cluster_to_fine[cluster] = fine
        cluster_to_broad[cluster] = broad
        cluster_rows.append(
            {
                "cluster": cluster,
                "ground_truth_enrichr_fine": fine,
                "ground_truth_enrichr_broad": broad,
                "source": lib,
                "top_markers": ";".join(markers[cluster][:20]),
            }
        )

    louvain = full.obs["louvain"].astype(str)
    full.obs["ground_truth_enrichr"] = louvain.map(cluster_to_fine).fillna("Unknown")
    full.obs["ground_truth_enrichr_broad"] = louvain.map(cluster_to_broad).fillna("Unknown")
    full.obs["ground_truth"] = full.obs["ground_truth_enrichr_broad"]

    gt = pd.DataFrame(
        {
            "barcode": full.obs_names,
            "clusters": louvain.values,
            "ground_truth_enrichr_fine": full.obs["ground_truth_enrichr"].values,
            "ground_truth_enrichr_broad": full.obs["ground_truth_enrichr_broad"].values,
            "NormalvsTumor": full.obs.get("NormalvsTumor", pd.Series(index=full.obs_names, dtype=str)).values,
            "patientno": full.obs.get("patientno", pd.Series(index=full.obs_names, dtype=str)).values,
            "patient_tumorsection": full.obs.get("patient_tumorsection", pd.Series(index=full.obs_names, dtype=str)).values,
        }
    ).set_index("barcode")

    gt.to_csv(OUT_GT, sep="\t")
    shutil.copy2(OUT_GT, REPO_GT_DIR / OUT_GT.name)
    print(f"Wrote {OUT_GT}")
    print(f"Wrote {REPO_GT_DIR / OUT_GT.name}")
    print(gt["ground_truth_enrichr_broad"].value_counts(dropna=False).to_string())

    cluster_table = pd.DataFrame(cluster_rows)
    cluster_out = WORK_GT_DIR / "GSE156625_enrichr_cluster_labels.tsv"
    cluster_table.to_csv(cluster_out, sep="\t", index=False)
    shutil.copy2(cluster_out, REPO_GT_DIR / cluster_out.name)
    print(f"Wrote {cluster_out}")

    print(f"Writing benchmark h5ad: {OUT_H5AD}")
    full.write_h5ad(OUT_H5AD, compression="gzip")
    print(f"Done: {OUT_H5AD}")


if __name__ == "__main__":
    main()
