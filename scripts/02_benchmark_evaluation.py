"""
Benchmark Evaluation Framework for HCC scRNA-seq Cell Type Annotation Tools.

Usage:
    python 02_benchmark_evaluation.py

Evaluates: CellTypist, CellAssign, signacX, CyteType, scGPT, SingleR, ScType
Datasets:  GSE125449, GSE149614 (enrichR GT + published GT for comparison)
           GSE156625, GSE162616, GSE202642, GSE223204 (enrichR GT)

Ground truth strategy:
  - Primary GT: enrichR-based (cluster marker → CellMarker_2024/PanglaoDB enrichment)
  - Secondary GT: published author annotations (GSE125449, GSE149614 only)
  - Non-standard enrichR labels → Unknown / excluded from evaluation

Prediction column notes:
  - CellTypist:  pred_celltype already mapped to broad; NO further mapping
  - CellAssign:  pred_celltype already mapped to broad; NO further mapping
  - signacX new: use pred_broad col + SIGNACX_TO_BROAD map
  - signacX ori3: use Pred.cell.type col + SIGNACX_TO_BROAD map
  - CyteType new: pred_celltype has LLM fine labels → fuzzy keyword map
  - CyteType ori3: Pred cell type col + CYTETYPE_TO_BROAD map
  - scGPT:       pred_celltype already in broad6; for broad7 GT → BROAD6_TO_BROAD7
  - SingleR:     pred_celltype already in broad6; for broad7 GT → BROAD6_TO_BROAD7
  - ScType:      pred_celltype broad6 + pred_celltype_broad7 in file

Metrics: Accuracy, Macro F1, Cohen's Kappa, Unknown rate, Per-class F1
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, classification_report
)
import warnings
warnings.filterwarnings("ignore")

OUT_DIR   = "/mnt/10t/assi_result/HCC/benchmark_datasets"
EVAL_DIR  = f"{OUT_DIR}/evaluation"
ANNOT_DIR = f"{OUT_DIR}/annotation_results"
GT_DIR    = f"{OUT_DIR}/ground_truth"
TOOLS_DIR = "/mnt/10t/assi_result/HCC/open_source_model"
os.makedirs(EVAL_DIR, exist_ok=True)

BROAD_6 = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
BROAD_7 = ["T cell", "CAF", "Malignant cell", "TAM", "B cell", "TEC", "HPC-like"]
VALID_ENRICHR_BROAD = set(BROAD_6) | set(BROAD_7)

# ─────────────────────────────────────────────────────────────────────────────
# Label maps — used only where prediction is NOT already in broad format
# ─────────────────────────────────────────────────────────────────────────────

# signacX pred_broad values → broad6
SIGNACX_TO_BROAD6 = {
    "TNK": "T/NK", "MPh": "Myeloid", "NonImmune": "Hepatocyte",
    "B": "B", "Plasma.cells": "B", "DC": "Myeloid",
    "Mast": "Myeloid", "Neutrophils": "Myeloid", "pDC": "Myeloid",
    "Endothelial": "Endothelial", "Fibroblasts": "Fibroblast",
    "Unclassified": "Unknown",
}

# CyteType original 3 (Pred cell type) → broad6
CYTETYPE_TO_BROAD6 = {
    "T cell": "T/NK", "NK cell": "T/NK", "NKT cell": "T/NK",
    "MAIT cell": "T/NK", "Treg cell": "T/NK", "Memory T cell": "T/NK",
    "Exhausted CD8 T cell": "T/NK", "Naive T cell": "T/NK",
    "Macrophage": "Myeloid", "Monocyte": "Myeloid",
    "DC": "Myeloid", "Dendritic cell": "Myeloid",
    "B cell": "B", "Plasma cell": "B",
    "Hepatocyte": "Hepatocyte",
    "Endothelial cell": "Endothelial", "Endothelial Cell": "Endothelial",
    "Fibroblast": "Fibroblast", "Stellate cell": "Fibroblast",
    "Unknown": "Unknown",
}

# signacX pred_broad → broad7 (for GSE125449 published GT)
SIGNACX_TO_BROAD7 = {
    "TNK": "T cell", "MPh": "TAM", "NonImmune": "Malignant cell",
    "B": "B cell", "Plasma.cells": "B cell",
    "Endothelial": "TEC", "Fibroblasts": "CAF",
    "Unclassified": "Unknown",
}

# broad7 → broad6 (for enrichR GT comparison when pred is in broad7 format)
BROAD7_TO_BROAD6 = {
    "T cell":        "T/NK",
    "TAM":           "Myeloid",
    "B cell":        "B",
    "Malignant cell":"Hepatocyte",
    "TEC":           "Endothelial",
    "CAF":           "Fibroblast",
    "HPC-like":      "Hepatocyte",
    "Unknown":       "Unknown",
}

# scGPT broad6 → broad7 (for GSE125449 published GT comparison)
BROAD6_TO_BROAD7 = {
    "T/NK":        "T cell",
    "Myeloid":     "TAM",
    "Hepatocyte":  "Malignant cell",
    "B":           "B cell",
    "Endothelial": "TEC",
    "Fibroblast":  "CAF",
    "Unknown":     "Unknown",
}

ORI_TO_BROAD6 = {
    "CD56 NK cell": "T/NK", "CD16 NK cell": "T/NK", "NK cell": "T/NK",
    "ILC": "T/NK", "INF-activated T cell": "T/NK",
    "GZMB CD8 T cell": "T/NK", "GZMK CD8 T cell": "T/NK",
    "CD4 T cell": "T/NK", "CD8 T cell": "T/NK",
    "Memory CD4 T cell": "T/NK", "Naive CD4 T cell": "T/NK",
    "MAIT cell": "T/NK", "Tfh cell": "T/NK",
    "Cycling CXCL13 exhausted CD8 T cell": "T/NK",
    "NK cell / NKT cell": "T/NK",
    "Monocyte": "Myeloid", "CD14 monocyte": "Myeloid",
    "CD16 monocyte": "Myeloid", "Macrophage": "Myeloid",
    "SPP1 macrophage": "Myeloid", "M1 macrophage": "Myeloid",
    "LYVE1 macrophage": "Myeloid", "MMP19 macrophage": "Myeloid",
    "Cycling macrophage": "Myeloid", "Neutrophil": "Myeloid",
    "pDC": "Myeloid", "Myeloid pre-pDC": "Myeloid",
    "cDC1": "Myeloid", "cDC2": "Myeloid", "Dendritic cell": "Myeloid",
    "M2 macrophage": "Myeloid", "CD16 macrophage": "Myeloid",
    "B cell": "B", "Plasma cell": "B",
    "Hepatocyte": "Hepatocyte", "AFP+ fetal hepatocyte": "Hepatocyte",
    "CCL2+ pancreatic ductal cell": "Hepatocyte",
    "CXCL6+ cholangiocyte": "Hepatocyte",
    "Epithelial Cells": "Hepatocyte",
    "Capillary EC": "Endothelial", "Lymphatic EC": "Endothelial",
    "Venous EC": "Endothelial", "Arterial EC": "Endothelial",
    "Endothelial Cells": "Endothelial",
    "Pericyte": "Fibroblast",
    "BNC2+ZFPM2+ fibroblast": "Fibroblast",
    "Vascular smooth muscle cell": "Fibroblast",
}


# ─────────────────────────────────────────────────────────────────────────────
# CyteType fuzzy mapper (for LLM-generated fine-grained labels in new 2)
# ─────────────────────────────────────────────────────────────────────────────

def cytetype_fuzzy_broad(label: str) -> str:
    """Map CyteType LLM annotation to broad6 via keyword priority."""
    lo = label.lower()
    # T/NK  (check before hepatocyte to handle "hepatocyte-contaminated T cell")
    if any(k in lo for k in ['t cell', 'cd4', 'cd8', 't-cell', 'cytotoxic t',
                              'nk cell', 'nkt', 'natural killer', 'treg',
                              'regulatory t', 'mait', 'memory t', 'naive t',
                              'exhausted t', 'gamma delta t', 'innate t']):
        return "T/NK"
    # B / Plasma
    if any(k in lo for k in ['b cell', 'b-cell', 'plasma cell', 'plasmablast',
                              'b lymphocyte', 'antibody-secreting']):
        return "B"
    # Myeloid
    if any(k in lo for k in ['macrophage', 'kupffer', 'monocyte', 'dendritic',
                              'neutrophil', 'mast cell', 'myeloid', ' dc',
                              'foam cell', 'inflammatory myeloid']):
        return "Myeloid"
    # Endothelial
    if any(k in lo for k in ['endotheli', 'sinusoidal', 'lsec',
                              'vascular endotheli', 'tumor vessel']):
        return "Endothelial"
    # Fibroblast / stromal
    if any(k in lo for k in ['fibroblast', 'stellate', 'pericyte',
                              'smooth muscle', 'myofibroblast', 'caf',
                              'perivascular', 'cancer-associated fibroblast']):
        return "Fibroblast"
    # Hepatocyte / liver parenchyma
    if any(k in lo for k in ['hepatocyte', 'hepato', 'hcc', 'liver progenitor',
                              'cholangio', 'biliary', 'malignant liver',
                              'tumor hepato']):
        return "Hepatocyte"
    return "Unknown"


def cytetype_fuzzy_broad7(label: str) -> str:
    """Map CyteType LLM label to broad7 (for GSE125449 published GT)."""
    lo = label.lower()
    if any(k in lo for k in ['t cell', 'cd4', 'cd8', 't-cell', 'cytotoxic t',
                              'nk cell', 'nkt', 'natural killer', 'treg',
                              'regulatory t', 'mait', 'memory t', 'naive t',
                              'exhausted t', 'gamma delta t', 'innate t']):
        return "T cell"
    if any(k in lo for k in ['b cell', 'b-cell', 'plasma cell', 'plasmablast',
                              'b lymphocyte', 'antibody-secreting']):
        return "B cell"
    if any(k in lo for k in ['kupffer', 'macrophage', 'monocyte', 'dendritic',
                              'neutrophil', 'mast cell', 'myeloid', ' dc',
                              'foam cell', 'inflammatory myeloid', 'tam']):
        return "TAM"
    if any(k in lo for k in ['tumor vessel', 'tumor endotheli', 'tec',
                              'endotheli', 'sinusoidal', 'lsec']):
        return "TEC"
    if any(k in lo for k in ['fibroblast', 'stellate', 'pericyte',
                              'smooth muscle', 'myofibroblast', 'caf',
                              'perivascular', 'cancer-associated fibroblast']):
        return "CAF"
    if any(k in lo for k in ['progenitor', 'hpc', 'cholangio', 'biliary',
                              'oval cell', 'ductal']):
        return "HPC-like"
    if any(k in lo for k in ['hepatocyte', 'hepato', 'hcc',
                              'malignant', 'carcinoma', 'tumor cell']):
        return "Malignant cell"
    return "Unknown"


def apply_cytetype_fuzzy(series: pd.Series) -> pd.Series:
    return series.map(cytetype_fuzzy_broad)


def apply_cytetype_fuzzy7(series: pd.Series) -> pd.Series:
    return series.map(cytetype_fuzzy_broad7)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def harmonise(series: pd.Series, mapping: dict, default: str = "Unknown") -> pd.Series:
    return series.map(lambda x: mapping.get(str(x).strip(), default))


def clean_enrichr_broad(series: pd.Series) -> pd.Series:
    return series.map(lambda x: x if x in VALID_ENRICHR_BROAD else "Unknown")


def compute_metrics(y_true, y_pred, tool_name, dataset_name, gt_type, label_set=None):
    mask_pred = y_pred != "Unknown"
    mask_gt   = y_true != "Unknown"
    mask      = mask_pred & mask_gt
    unknown_rate = (~mask_pred).mean()

    if mask.sum() == 0:
        print(f"    WARNING: {tool_name}/{dataset_name} — no evaluable cells after masking")
        return None

    yt, yp = y_true[mask], y_pred[mask]
    acc    = accuracy_score(yt, yp)
    f1     = f1_score(yt, yp, average="macro", zero_division=0)
    kappa  = cohen_kappa_score(yt, yp) if len(set(yt)) > 1 else 0.0

    result = {
        "Tool": tool_name, "Dataset": dataset_name, "GT_type": gt_type,
        "N_cells": len(y_true), "N_evaluated": int(mask.sum()),
        "Unknown_rate": round(unknown_rate, 4),
        "Accuracy": round(acc, 4), "Macro_F1": round(f1, 4),
        "Kappa": round(kappa, 4),
    }
    if label_set:
        report = classification_report(yt, yp, labels=label_set,
                                       output_dict=True, zero_division=0)
        for lbl in label_set:
            result[f"F1_{lbl}"] = round(report.get(lbl, {}).get("f1-score", 0.0), 4)

    print(f"    {tool_name:12s} | Acc={acc:.3f} | F1={f1:.3f} | "
          f"Kappa={kappa:.3f} | Unk={unknown_rate:.2%} | N={mask.sum()}")
    return result


def load_enrichr_gt(dataset_name):
    df = pd.read_csv(f"{GT_DIR}/{dataset_name}_enrichr_ground_truth.tsv",
                     sep="\t", index_col=0)
    return clean_enrichr_broad(df["ground_truth_enrichr_broad"])


# ─────────────────────────────────────────────────────────────────────────────
# Load tool predictions — each returns a Series indexed by barcode
# ─────────────────────────────────────────────────────────────────────────────

def load_celltypist(ds):
    """pred_celltype already broad-mapped in file."""
    path = f"{ANNOT_DIR}/{ds}_CellTypist_pred.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, index_col=0)
    return df["pred_celltype"].fillna("Unknown").astype(str)


def load_cellassign(ds):
    """pred_celltype already broad-mapped in file."""
    path = f"{ANNOT_DIR}/{ds}_CellAssign_pred.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, index_col=0)
    return df["pred_celltype"].fillna("Unknown").astype(str)


def load_signacx_new(ds):
    """Use pred_broad col → SIGNACX_TO_BROAD6."""
    path = f"{ANNOT_DIR}/{ds}_signacX_pred.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, index_col=0)
    col = "pred_broad" if "pred_broad" in df.columns else "pred_celltype"
    return harmonise(df[col].fillna("Unknown").astype(str), SIGNACX_TO_BROAD6)


def load_cytetype_new(ds, broad7=False):
    """LLM fine-grained labels → fuzzy mapper (broad6 or broad7)."""
    path = f"{ANNOT_DIR}/{ds}_CyteType_pred.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, index_col=0)
    raw = df["pred_celltype"].fillna("Unknown").astype(str)
    return apply_cytetype_fuzzy7(raw) if broad7 else apply_cytetype_fuzzy(raw)


def load_celltypist_broad6(ds):
    """CellTypist pred is broad7 for GSE125449 → convert to broad6."""
    pred = load_celltypist(ds)
    if pred is None: return None
    return harmonise(pred, BROAD7_TO_BROAD6)


def load_cellassign_broad6(ds):
    """CellAssign pred is broad7 for GSE125449 → convert to broad6."""
    pred = load_cellassign(ds)
    if pred is None: return None
    return harmonise(pred, BROAD7_TO_BROAD6)


def load_signacx_ori(ds):
    """Original 3: Pred.cell.type → SIGNACX_TO_BROAD6."""
    path = f"{TOOLS_DIR}/signacX/signacX_cell_prediction_summar_{ds}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df.index = df["barcode"].astype(str)
    return harmonise(df["Pred.cell.type"].fillna("Unknown").astype(str), SIGNACX_TO_BROAD6)


def load_cytetype_ori(ds):
    """Original 3: Pred cell type → CYTETYPE_TO_BROAD6."""
    path = f"{TOOLS_DIR}/cytetype/CyteType_cell_prediction_summary_{ds}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df.index = df["barcode"].astype(str)
    return harmonise(df["Pred cell type"].fillna("Unknown").astype(str), CYTETYPE_TO_BROAD6)


def load_cellassign_ori(ds):
    """Original 3: Pred cell type → CELLASSIGN_TO_BROAD6 (or already broad)."""
    path = f"{TOOLS_DIR}/cellassign/{ds}_CellAssign_cell_prediction_summary.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df.index = df["barcode"].astype(str)
    # Values may already be broad — pass through ORI_TO_BROAD6 first, then BROAD6 passthrough
    raw = df["Pred cell type"].fillna("Unknown").astype(str)
    mapped = harmonise(raw, ORI_TO_BROAD6)
    # Anything already in BROAD_6 should pass through directly
    mapped2 = pd.Series([
        v if v in VALID_ENRICHR_BROAD else (raw.iloc[i] if raw.iloc[i] in VALID_ENRICHR_BROAD else v)
        for i, v in enumerate(mapped)
    ], index=mapped.index)
    return mapped2


def load_scgpt(ds, broad7=False):
    """scGPT: pred_celltype already in broad6. broad7=True → BROAD6_TO_BROAD7."""
    path = f"{ANNOT_DIR}/{ds}_scGPT_pred.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, index_col=0)
    pred = df["pred_celltype"].fillna("Unknown").astype(str)
    if broad7:
        return harmonise(pred, BROAD6_TO_BROAD7)
    return pred


def load_singler(ds, broad7=False):
    """SingleR HPCA baseline: pred_celltype is broad6. broad7=True → BROAD6_TO_BROAD7."""
    path = f"{ANNOT_DIR}/{ds}_SingleR_pred.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, index_col=0)
    pred = df["pred_celltype"].fillna("Unknown").astype(str)
    if broad7:
        return harmonise(pred, BROAD6_TO_BROAD7)
    return pred


def load_sctype(ds, broad7=False):
    """ScType marker-scoring baseline: broad6 and broad7 predictions are saved."""
    path = f"{ANNOT_DIR}/{ds}_ScType_pred.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, index_col=0)
    col = "pred_celltype_broad7" if broad7 else "pred_celltype"
    if col not in df.columns:
        return None
    return df[col].fillna("Unknown").astype(str)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(ds, gt, tools_dict, gt_type, label_set):
    results = []
    print(f"\n  [{gt_type}] {ds}  (evaluable GT cells: {(gt!='Unknown').sum()})")
    for tool_name, pred in tools_dict.items():
        if pred is None:
            print(f"    SKIP {tool_name}: no prediction file")
            continue
        merged = gt.to_frame(name="gt").join(pred.rename("pred"), how="inner")
        r = compute_metrics(merged["gt"], merged["pred"],
                            tool_name, ds, gt_type, label_set)
        if r: results.append(r)
    return results


def evaluate_published_gt():
    print("\n" + "=" * 70)
    print("Published author GT: GSE149614 + GSE125449")
    print("=" * 70)
    results = []

    # GSE149614
    gt_df = pd.read_csv(f"{GT_DIR}/GSE149614_ground_truth.tsv", sep="\t", index_col=0)
    gt    = gt_df["celltype"]
    tools = {
        "CellTypist": load_celltypist("GSE149614"),
        "CellAssign":  load_cellassign("GSE149614"),
        "signacX":     load_signacx_new("GSE149614"),
        "CyteType":    load_cytetype_new("GSE149614"),
        "scGPT":       load_scgpt("GSE149614"),
        "SingleR":     load_singler("GSE149614"),
        "ScType":      load_sctype("GSE149614"),
    }
    results += evaluate_dataset("GSE149614", gt, tools, "published", BROAD_6)

    # GSE125449 (7 classes) — signacX uses BROAD7 map; CyteType uses broad7 fuzzy
    gt_df2 = pd.read_csv(f"{GT_DIR}/GSE125449_ground_truth.tsv", sep="\t", index_col=0)
    gt2    = gt_df2.loc[gt_df2["celltype"] != "unclassified", "celltype"]

    # signacX: override with broad7 map
    sx125_path = f"{ANNOT_DIR}/GSE125449_signacX_pred.csv"
    sx125_pred = None
    if os.path.exists(sx125_path):
        df_sx = pd.read_csv(sx125_path, index_col=0)
        col = "pred_broad" if "pred_broad" in df_sx.columns else "pred_celltype"
        sx125_pred = harmonise(df_sx[col].fillna("Unknown").astype(str), SIGNACX_TO_BROAD7)

    tools2 = {
        "CellTypist": load_celltypist("GSE125449"),      # already broad7
        "CellAssign":  load_cellassign("GSE125449"),     # already broad7
        "signacX":     sx125_pred,
        "CyteType":    load_cytetype_new("GSE125449", broad7=True),
        "scGPT":       load_scgpt("GSE125449", broad7=True),  # broad6 → broad7
        "SingleR":     load_singler("GSE125449", broad7=True), # broad6 → broad7
        "ScType":      load_sctype("GSE125449", broad7=True),
    }
    results += evaluate_dataset("GSE125449", gt2, tools2, "published", BROAD_7)

    return results


def evaluate_enrichr_new2():
    print("\n" + "=" * 70)
    print("enrichR GT: GSE149614 + GSE125449")
    print("=" * 70)
    results = []

    # GSE149614 — pred already broad6
    gt149 = load_enrichr_gt("GSE149614")
    tools149 = {
        "CellTypist": load_celltypist("GSE149614"),
        "CellAssign":  load_cellassign("GSE149614"),
        "signacX":     load_signacx_new("GSE149614"),
        "CyteType":    load_cytetype_new("GSE149614"),
        "scGPT":       load_scgpt("GSE149614"),
        "SingleR":     load_singler("GSE149614"),
        "ScType":      load_sctype("GSE149614"),
    }
    results += evaluate_dataset("GSE149614", gt149, tools149, "enrichR", BROAD_6)

    # GSE125449 — CellTypist/CellAssign pred is broad7 → convert to broad6
    gt125 = load_enrichr_gt("GSE125449")
    sx125_path = f"{ANNOT_DIR}/GSE125449_signacX_pred.csv"
    sx125_pred6 = None
    if os.path.exists(sx125_path):
        df_sx = pd.read_csv(sx125_path, index_col=0)
        col = "pred_broad" if "pred_broad" in df_sx.columns else "pred_celltype"
        sx125_pred6 = harmonise(df_sx[col].fillna("Unknown").astype(str), SIGNACX_TO_BROAD6)
    tools125 = {
        "CellTypist": load_celltypist_broad6("GSE125449"),  # broad7 → broad6
        "CellAssign":  load_cellassign_broad6("GSE125449"), # broad7 → broad6
        "signacX":     sx125_pred6,
        "CyteType":    load_cytetype_new("GSE125449"),       # fuzzy → broad6
        "scGPT":       load_scgpt("GSE125449"),              # already broad6
        "SingleR":     load_singler("GSE125449"),            # already broad6
        "ScType":      load_sctype("GSE125449"),             # already broad6
    }
    results += evaluate_dataset("GSE125449", gt125, tools125, "enrichR", BROAD_6)

    return results


def evaluate_enrichr_original3():
    print("\n" + "=" * 70)
    print("enrichR GT: GSE156625 + GSE162616 + GSE202642 + GSE223204")
    print("=" * 70)
    results = []

    # GSE156625 — use new prediction files (run in this benchmark expansion)
    gt156 = load_enrichr_gt("GSE156625")
    tools156 = {
        "CellTypist": load_celltypist("GSE156625"),
        "signacX":     load_signacx_new("GSE156625"),
        "CyteType":    load_cytetype_new("GSE156625"),
        "CellAssign":  None,   # not available (scvi-tools env incompatibility)
        "scGPT":       load_scgpt("GSE156625"),
        "SingleR":     load_singler("GSE156625"),
        "ScType":      load_sctype("GSE156625"),
    }
    results += evaluate_dataset("GSE156625", gt156, tools156, "enrichR", BROAD_6)

    for ds in ["GSE162616", "GSE202642", "GSE223204"]:
        gt = load_enrichr_gt(ds)
        tools = {
            "CellTypist": load_celltypist(ds),
            "signacX":     load_signacx_ori(ds),
            "CyteType":    load_cytetype_ori(ds),
            "CellAssign":  load_cellassign_ori(ds),
            "scGPT":       load_scgpt(ds),
            "SingleR":     load_singler(ds),
            "ScType":      load_sctype(ds),
        }
        results += evaluate_dataset(ds, gt, tools, "enrichR", BROAD_6)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_simulated():
    """Evaluate tools on Splatter-simulated data with 100% known ground truth."""
    print("\n" + "=" * 70)
    print("Simulated data (Splatter, known GT = BROAD_6)")
    print("=" * 70)
    gt = load_enrichr_gt("simulated")
    tools = {
        "CellTypist": load_celltypist("simulated"),
        "ScType":     load_sctype("simulated"),
        "SingleR":    load_singler("simulated"),
        "signacX":    None,   # requires network for model download
        "CyteType":   None,   # requires cluster API
        "CellAssign": None,   # scvi-tools incompatible
        "scGPT":      None,   # GPU required
    }
    return evaluate_dataset("simulated", gt, tools, "simulated_GT", BROAD_6)


if __name__ == "__main__":
    all_results = []
    all_results += evaluate_published_gt()
    all_results += evaluate_enrichr_new2()
    all_results += evaluate_enrichr_original3()
    all_results += evaluate_simulated()

    if not all_results:
        print("No results produced.")
    else:
        summary = pd.DataFrame(all_results)
        out = f"{EVAL_DIR}/benchmark_summary.csv"
        summary.to_csv(out, index=False)
        print(f"\n{'=' * 70}")
        print(f"Full results → {out}")

        cols = ["Tool", "Dataset", "GT_type", "Accuracy", "Macro_F1", "Kappa",
                "Unknown_rate", "N_evaluated"]
        print("\n" + summary[cols].to_string(index=False))

        # Pivot: enrichR GT macro F1
        e = summary[summary["GT_type"] == "enrichR"]
        if not e.empty:
            pivot = e.pivot_table(index="Tool", columns="Dataset",
                                  values="Macro_F1", aggfunc="mean")
            print("\nMacro F1 — enrichR GT:")
            print(pivot.round(3).to_string())

        # Pivot: published GT macro F1
        p = summary[summary["GT_type"] == "published"]
        if not p.empty:
            pivot2 = p.pivot_table(index="Tool", columns="Dataset",
                                   values="Macro_F1", aggfunc="mean")
            print("\nMacro F1 — published GT:")
            print(pivot2.round(3).to_string())
