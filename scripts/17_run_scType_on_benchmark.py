"""
Run a lightweight ScType marker-scoring baseline on HCC benchmark datasets.

This script uses the official ScType marker database and mirrors the core
ScType scoring idea: marker-sensitivity-weighted positive markers minus
negative markers, aggregated at cluster level. It avoids materialising a full
dense Seurat scale.data matrix, which is important for the 70k--120k-cell HCC
cohorts in this benchmark.

Inputs:
  - Official ScType marker DB:
    external/sc-type/ScTypeDB_full.xlsx
  - Existing benchmark/CyteType h5ad files

Outputs:
  - annotation_results/{dataset}_ScType_pred.csv
  - results/ScType_cluster_assignments.csv
  - results/ScType_marker_coverage.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

warnings.filterwarnings("ignore")


PROJECT = Path("/mnt/10t/holiday/hcc-annotation-benchmark")
DATA_ROOT = Path("/mnt/10t/assi_result/HCC")
BENCH_ROOT = DATA_ROOT / "benchmark_datasets"
ANNOT_DIR = BENCH_ROOT / "annotation_results"
RESULT_DIR = PROJECT / "results"
SCTYPE_DB = PROJECT / "external" / "sc-type" / "ScTypeDB_full.xlsx"

ANNOT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    h5ad: Path
    cluster_key: str | None = None
    leiden_resolution: float = 0.5


DATASETS = [
    DatasetConfig(
        "GSE125449",
        BENCH_ROOT / "GSE125449_benchmark.h5ad",
        cluster_key=None,
        leiden_resolution=0.5,
    ),
    DatasetConfig(
        "GSE149614",
        BENCH_ROOT / "GSE149614_benchmark.h5ad",
        cluster_key=None,
        leiden_resolution=0.5,
    ),
    DatasetConfig(
        "GSE156625",
        BENCH_ROOT / "GSE156625_benchmark.h5ad",
        cluster_key="louvain",
    ),
    DatasetConfig(
        "GSE162616",
        DATA_ROOT / "open_source_model" / "cytetype" / "adata_CyteType_GSE162616.h5ad",
        cluster_key="clusters",
    ),
    DatasetConfig(
        "GSE202642",
        DATA_ROOT / "open_source_model" / "cytetype" / "adata_CyteType_GSE202642.h5ad",
        cluster_key="clusters",
    ),
    DatasetConfig(
        "GSE223204",
        DATA_ROOT / "open_source_model" / "cytetype" / "adata_CyteType_GSE223204.h5ad",
        cluster_key="clusters",
    ),
]


GENE_ALIASES = {
    "CD1C": "CD1C",
    "CD2": "CD2",
    "CD3": "CD3D",
    "CD3D": "CD3D",
    "CD3E": "CD3E",
    "CD3G": "CD3G",
    "CD4": "CD4",
    "CD8": "CD8A",
    "CD8A": "CD8A",
    "CD8B": "CD8B",
    "CD9": "CD9",
    "CD11B": "ITGAM",
    "CD11C": "ITGAX",
    "CD14": "CD14",
    "CD15": "FUT4",
    "CD16": "FCGR3A",
    "CD19": "CD19",
    "CD20": "MS4A1",
    "CD24": "CD24",
    "CD25": "IL2RA",
    "CD27": "CD27",
    "CD31": "PECAM1",
    "CD33": "CD33",
    "CD34": "CD34",
    "CD36": "CD36",
    "CD38": "CD38",
    "CD41": "ITGA2B",
    "CD42A": "GP9",
    "CD42B": "GP1BA",
    "CD44": "CD44",
    "CD45": "PTPRC",
    "CD45RA": "PTPRC",
    "CD45RO": "PTPRC",
    "CD56": "NCAM1",
    "CD61": "ITGB3",
    "CD62L": "SELL",
    "CD63": "CD63",
    "CD64": "FCGR1A",
    "CD66B": "CEACAM8",
    "CD68": "CD68",
    "CD69": "CD69",
    "CD73": "NT5E",
    "CD80": "CD80",
    "CD83": "CD83",
    "CD86": "CD86",
    "CD94": "KLRD1",
    "CD105": "ENG",
    "CD110": "MPL",
    "CD115": "CSF1R",
    "CD117": "KIT",
    "CD122": "IL2RB",
    "CD123": "IL3RA",
    "CD125": "IL5RA",
    "CD127": "IL7R",
    "CD133": "PROM1",
    "CD138": "SDC1",
    "CD146": "MCAM",
    "CD161": "KLRB1",
    "CD166": "ALCAM",
    "CD193": "CCR3",
    "CD203C": "ENPP3",
    "CD206": "MRC1",
    "CD314": "KLRK1",
    "CD235A": "GYPA",
    "HLA-DR": "HLA-DRA",
    "HLA_DR": "HLA-DRA",
    "NKP46": "NCR1",
    "FC-EPSILON RI-ALPHA": "FCER1A",
}


def normalise_gene(gene: str) -> str | None:
    gene = str(gene).strip()
    if not gene or gene.upper() in {"NA", "NAN", "NONE"}:
        return None
    gene = gene.replace("///", ",")
    gene = gene.strip()
    key = gene.upper()
    return GENE_ALIASES.get(key, key)


def parse_gene_list(value) -> list[str]:
    if pd.isna(value):
        return []
    genes: list[str] = []
    for token in re.split(r"[,;/]+", str(value)):
        gene = normalise_gene(token)
        if gene:
            genes.append(gene)
    return sorted(set(genes))


def load_sctype_marker_sets() -> tuple[dict[str, list[str]], dict[str, list[str]], pd.DataFrame]:
    db = pd.read_excel(SCTYPE_DB)
    db = db[db["tissueType"].isin(["Liver", "Immune system"])].copy()
    db["cellName"] = db["cellName"].astype(str)

    positive: dict[str, list[str]] = {}
    negative: dict[str, list[str]] = {}
    rows = []

    for _, row in db.iterrows():
        cell_name = row["cellName"]
        label = f"{row['tissueType']}::{cell_name}"
        pos = parse_gene_list(row["geneSymbolmore1"])
        neg = parse_gene_list(row["geneSymbolmore2"])
        positive[label] = pos
        negative[label] = neg
        rows.append(
            {
                "sctype_label": label,
                "tissueType": row["tissueType"],
                "cellName": cell_name,
                "shortName": row.get("shortName", ""),
                "n_positive_markers": len(pos),
                "n_negative_markers": len(neg),
            }
        )

    return positive, negative, pd.DataFrame(rows)


def marker_sensitivity(positive: dict[str, list[str]]) -> dict[str, float]:
    n_types = max(len(positive), 2)
    counts: dict[str, int] = {}
    for genes in positive.values():
        for gene in set(genes):
            counts[gene] = counts.get(gene, 0) + 1
    return {
        gene: max(0.0, min(1.0, (n_types - count) / (n_types - 1)))
        for gene, count in counts.items()
    }


def to_memory_matrix(x):
    if hasattr(x, "to_memory"):
        x = x.to_memory()
    if sparse.issparse(x):
        return x.astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def mean_std_by_gene(x) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel().astype(np.float32)
        sq_mean = np.asarray(x.power(2).mean(axis=0)).ravel().astype(np.float32)
        var = np.maximum(sq_mean - mean**2, 1e-6)
    else:
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        var = x.var(axis=0, dtype=np.float64).astype(np.float32)
        var = np.maximum(var, 1e-6)
    return mean, np.sqrt(var).astype(np.float32)


def ensure_clusters(adata, cfg: DatasetConfig) -> pd.Series:
    if cfg.cluster_key and cfg.cluster_key in adata.obs:
        return adata.obs[cfg.cluster_key].astype(str).copy()

    key = "sctype_cluster"
    if "neighbors" not in adata.uns:
        raise ValueError(f"{cfg.name}: no cluster_key and no precomputed neighbors")

    print(f"  Running Leiden clustering for ScType (resolution={cfg.leiden_resolution})")
    sc.tl.leiden(adata, resolution=cfg.leiden_resolution, key_added=key)
    return adata.obs[key].astype(str).copy()


def sctype_to_broad6(label: str) -> str:
    if label == "Unknown":
        return "Unknown"
    lo = label.lower()
    if any(k in lo for k in ["t cells", "t cell", "nkt", "natural killer", "γδ", "nk"]):
        return "T/NK"
    if any(k in lo for k in ["b cells", "b cell", "plasma"]):
        return "B"
    if any(
        k in lo
        for k in [
            "monocyte",
            "macrophage",
            "kupffer",
            "dendritic",
            "neutrophil",
            "eosinophil",
            "basophil",
            "mast",
            "granulocyte",
            "myeloid",
        ]
    ):
        return "Myeloid"
    if "endothelial" in lo:
        return "Endothelial"
    if any(k in lo for k in ["hepatic stellate", "mesenchymal stem"]):
        return "Fibroblast"
    if any(
        k in lo
        for k in [
            "hepatocyte",
            "hepatoblast",
            "cholangiocyte",
            "liver progenitor",
            "cancer stem",
            "cancer cells",
        ]
    ):
        return "Hepatocyte"
    return "Unknown"


def sctype_to_broad7(label: str) -> str:
    broad6 = sctype_to_broad6(label)
    if broad6 == "T/NK":
        return "T cell"
    if broad6 == "B":
        return "B cell"
    if broad6 == "Myeloid":
        return "TAM"
    if broad6 == "Endothelial":
        return "TEC"
    if broad6 == "Fibroblast":
        return "CAF"
    lo = label.lower()
    if any(k in lo for k in ["hepatoblast", "cholangiocyte", "liver progenitor"]):
        return "HPC-like"
    if broad6 == "Hepatocyte":
        return "Malignant cell"
    return "Unknown"


def run_dataset(
    cfg: DatasetConfig,
    positive: dict[str, list[str]],
    negative: dict[str, list[str]],
    sensitivity: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== {cfg.name} ===")
    adata = ad.read_h5ad(cfg.h5ad, backed="r")
    clusters = ensure_clusters(adata, cfg)
    print(f"  Shape: {adata.shape}; clusters: {clusters.nunique()}")

    var_upper = pd.Index([str(v).upper() for v in adata.var_names])
    gene_to_pos = {}
    for i, gene in enumerate(var_upper):
        gene_to_pos.setdefault(gene, i)

    all_markers = sorted(
        {
            gene
            for genes in list(positive.values()) + list(negative.values())
            for gene in genes
            if gene in gene_to_pos
        }
    )
    if not all_markers:
        raise ValueError(f"{cfg.name}: no ScType markers overlap the expression matrix")

    marker_indices = [gene_to_pos[gene] for gene in all_markers]
    marker_pos = {gene: i for i, gene in enumerate(all_markers)}
    x = to_memory_matrix(adata[:, marker_indices].X)
    mean, std = mean_std_by_gene(x)

    labels = list(positive.keys())
    coeff = np.zeros((len(all_markers), len(labels)), dtype=np.float32)
    coverage_rows = []

    for j, label in enumerate(labels):
        pos = [g for g in positive[label] if g in marker_pos]
        neg = [g for g in negative[label] if g in marker_pos]
        if pos:
            denom = math.sqrt(len(pos))
            for gene in pos:
                coeff[marker_pos[gene], j] += sensitivity.get(gene, 1.0) / denom
        if neg:
            denom = math.sqrt(len(neg))
            for gene in neg:
                coeff[marker_pos[gene], j] -= sensitivity.get(gene, 1.0) / denom
        coverage_rows.append(
            {
                "Dataset": cfg.name,
                "ScType_label": label,
                "positive_markers_total": len(positive[label]),
                "positive_markers_present": len(pos),
                "negative_markers_total": len(negative[label]),
                "negative_markers_present": len(neg),
            }
        )

    scaled_coeff = coeff / std[:, None]
    offset = -(mean / std) @ coeff
    scores = x @ scaled_coeff
    if sparse.issparse(scores):
        scores = scores.toarray()
    scores = np.asarray(scores, dtype=np.float32) + offset.astype(np.float32)
    score_df = pd.DataFrame(scores, index=adata.obs_names, columns=labels)
    score_df["cluster"] = clusters.values

    cluster_sum = score_df.groupby("cluster", sort=True)[labels].sum()
    cluster_size = score_df.groupby("cluster", sort=True).size()

    cluster_rows = []
    cluster_to_label = {}
    cluster_to_score = {}
    for cluster, row in cluster_sum.iterrows():
        top_label = str(row.idxmax())
        top_score = float(row.max())
        ncells = int(cluster_size.loc[cluster])
        assigned = top_label if top_score >= (ncells / 4.0) else "Unknown"
        cluster_to_label[str(cluster)] = assigned
        cluster_to_score[str(cluster)] = top_score
        cluster_rows.append(
            {
                "Dataset": cfg.name,
                "cluster": str(cluster),
                "n_cells": ncells,
                "sctype_raw_label": top_label,
                "sctype_raw_score": round(top_score, 4),
                "sctype_label": assigned,
                "pred_celltype": sctype_to_broad6(assigned),
                "pred_celltype_broad7": sctype_to_broad7(assigned),
            }
        )

    pred = pd.DataFrame(index=adata.obs_names)
    pred["cluster"] = clusters.values
    pred["sctype_label"] = pred["cluster"].map(cluster_to_label).fillna("Unknown")
    pred["sctype_score"] = pred["cluster"].map(cluster_to_score).fillna(0.0)
    pred["pred_celltype"] = pred["sctype_label"].map(sctype_to_broad6).fillna("Unknown")
    pred["pred_celltype_broad7"] = pred["sctype_label"].map(sctype_to_broad7).fillna("Unknown")

    out = ANNOT_DIR / f"{cfg.name}_ScType_pred.csv"
    pred.to_csv(out)
    print(f"  Saved: {out}")
    print("  Broad6 prediction distribution:")
    print(pred["pred_celltype"].value_counts().to_string())

    return pd.DataFrame(cluster_rows), pd.DataFrame(coverage_rows)


def main() -> None:
    positive, negative, marker_info = load_sctype_marker_sets()
    sensitivity = marker_sensitivity(positive)
    print(f"Loaded ScType DB: {len(positive)} Liver/Immune marker signatures")
    print(marker_info[["tissueType", "cellName", "n_positive_markers", "n_negative_markers"]].to_string(index=False))

    all_clusters = []
    all_coverage = []
    for cfg in DATASETS:
        cluster_df, coverage_df = run_dataset(cfg, positive, negative, sensitivity)
        all_clusters.append(cluster_df)
        all_coverage.append(coverage_df)

    cluster_out = RESULT_DIR / "ScType_cluster_assignments.csv"
    coverage_out = RESULT_DIR / "ScType_marker_coverage.csv"
    pd.concat(all_clusters, ignore_index=True).to_csv(cluster_out, index=False)
    pd.concat(all_coverage, ignore_index=True).to_csv(coverage_out, index=False)
    print(f"\nSaved: {cluster_out}")
    print(f"Saved: {coverage_out}")


if __name__ == "__main__":
    main()
