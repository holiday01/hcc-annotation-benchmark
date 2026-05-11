"""
scGPT-based cell type annotation for all 5 HCC benchmark datasets.

Approach:
  1. Use DISCO liver data as reference (with fine-grained cell_type labels)
  2. Map DISCO cell_type → BROAD6 labels
  3. Compute scGPT embeddings for reference (sampled) and query datasets
  4. KNN classification in embedding space
  5. Save predictions to annotation_results/

Model: wanglab/scGPT-human (downloaded from HuggingFace)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/mnt/10t/assi_result/HCC")
BENCHMARK = BASE / "benchmark_datasets"
MODEL_DIR = BENCHMARK / "scgpt_model"
RESULTS_DIR = BENCHMARK / "annotation_results"
RESULTS_DIR.mkdir(exist_ok=True)

DISCO_H5AD = Path(
    "/mnt/10t/assi_result/DISCO_DATA/"
    "disco_adata_mt5_cell200gen5_no_combat_with_celltype_20240415.h5ad"
)

# ── dataset configs ────────────────────────────────────────────────────────────
DATASETS = {
    "GSE125449": BENCHMARK / "GSE125449_benchmark.h5ad",
    "GSE149614": BENCHMARK / "GSE149614_benchmark.h5ad",
    "GSE156625": BENCHMARK / "GSE156625_benchmark.h5ad",
    "GSE162616": BASE / "open_source_model/cytetype/adata_CyteType_GSE162616.h5ad",
    "GSE202642": BASE / "open_source_model/cytetype/adata_CyteType_GSE202642.h5ad",
    "GSE223204": BASE / "open_source_model/cytetype/adata_CyteType_GSE223204.h5ad",
}

# ── DISCO cell_type → BROAD6 mapping ──────────────────────────────────────────
DISCO_TO_BROAD6 = {
    # T/NK
    "CD56 NK cell": "T/NK",
    "GZMK CD8 T cell": "T/NK",
    "NK cell": "T/NK",
    "Memory CD4 T cell": "T/NK",
    "CD4 T cell": "T/NK",
    "CD16 NK cell": "T/NK",
    "CD8 T cell": "T/NK",
    "Treg cell": "T/NK",
    "Naive CD4 T cell": "T/NK",
    "Naive CD8 T cell": "T/NK",
    "GZMB CD8 T cell": "T/NK",
    "MAIT cell": "T/NK",
    "ILC": "T/NK",
    "INF-activated T cell": "T/NK",
    "Tfh cell": "T/NK",
    "Cycling DN/DP T cell": "T/NK",
    "Cycling T/NK cell": "T/NK",
    "CXCL13 exhausted CD8 T cell": "T/NK",
    "Gamma delta T cell": "T/NK",
    "Cycling T cell": "T/NK",
    "GZMK+IL7R+ CD8 T cell": "T/NK",
    "GZMK+IL7R- CD8 T cell": "T/NK",
    "Cycling CXCL13 exhausted CD8 T cell": "T/NK",
    "Memory CD8 T cell": "T/NK",
    "HSP+ T cell": "T/NK",
    "XCL1 NK cell": "T/NK",
    "Tissue-resident NK cell": "T/NK",
    "CD8aa+ T cell": "T/NK",
    "Memory T cell": "T/NK",
    "T cell": "T/NK",
    "T/NK cell": "T/NK",
    "KLRB1 CD8 T cell": "T/NK",
    # Myeloid
    "cDC2": "Myeloid",
    "Macrophage": "Myeloid",
    "Neutrophil": "Myeloid",
    "CD14 monocyte": "Myeloid",
    "CD16 monocyte": "Myeloid",
    "Monocyte": "Myeloid",
    "M1 macrophage": "Myeloid",
    "cDC1": "Myeloid",
    "Cycling macrophage": "Myeloid",
    "LYVE1 macrophage": "Myeloid",
    "MMP19 macrophage": "Myeloid",
    "pDC": "Myeloid",
    "SPP1 macrophage": "Myeloid",
    "MHCII high CD14 monocyte": "Myeloid",
    "MHCII low CD14 monocyte": "Myeloid",
    "Dendritic cell": "Myeloid",
    "Mast cell": "Myeloid",
    "Cycling myeloid cell": "Myeloid",
    "Mast cell primed GMP": "Myeloid",
    "mregDC": "Myeloid",
    "Cycling LYVE1 macrophage": "Myeloid",
    "Basophil": "Myeloid",
    "Cycling basophil": "Myeloid",
    "Myeloid pre-pDC": "Myeloid",
    "Megakaryocyte": "Myeloid",
    # Hepatocyte
    "AFP+ fetal hepatocyte": "Hepatocyte",
    "Hepatocyte": "Hepatocyte",
    "AFP+ epithelial cell": "Hepatocyte",
    # B
    "B cell": "B",
    "Plasma cell": "B",
    "Memory B cell": "B",
    "Naive B cell": "B",
    "Cycling GC B cell": "B",
    # Endothelial
    "Capillary EC": "Endothelial",
    "Arterial EC": "Endothelial",
    "Venous EC": "Endothelial",
    "BGN+ EC": "Endothelial",
    "Lymphatic EC": "Endothelial",
    "Fetal MHC- capillary EC": "Endothelial",
    "Cycling EC": "Endothelial",
    "Endothelial cell": "Endothelial",
    # Fibroblast
    "Pericyte": "Fibroblast",
    "Vascular smooth muscle cell": "Fibroblast",
    "BNC2+ZFPM2+ fibroblast": "Fibroblast",
    "CYGB+ pericyte": "Fibroblast",
    "ADAM12+ fibroblast": "Fibroblast",
    "Fibroblast": "Fibroblast",
}

BROAD6_LABELS = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]

# cholangiocytes, ductal cells, etc. → drop from reference (ambiguous)
VALID_BROAD6 = set(BROAD6_LABELS)


def prepare_reference(max_cells_per_type: int = 3000) -> sc.AnnData:
    """Load DISCO, map labels, subsample for efficiency."""
    print("Loading DISCO reference data...")
    ref = sc.read_h5ad(DISCO_H5AD)

    # map to broad6
    ref.obs["broad6"] = ref.obs["cell_type"].map(DISCO_TO_BROAD6)
    ref = ref[ref.obs["broad6"].notna()].copy()
    print(f"Reference after label mapping: {ref.shape[0]} cells")
    print(ref.obs["broad6"].value_counts().to_string())

    # subsample per class for speed
    idx = []
    for label in BROAD6_LABELS:
        mask = ref.obs["broad6"] == label
        n = mask.sum()
        take = min(n, max_cells_per_type)
        idx.extend(
            ref.obs.index[mask]
            .to_series()
            .sample(take, random_state=42)
            .tolist()
        )
    ref = ref[idx].copy()
    print(f"Subsampled reference: {ref.shape[0]} cells")
    return ref


def prepare_query(h5ad_path: Path) -> sc.AnnData:
    """Load query dataset, standardise gene col name."""
    adata = sc.read_h5ad(h5ad_path)
    # Ensure gene names are in index (not a separate column)
    if "gene_symbols" in adata.var.columns and adata.var.index[0].startswith("ENS"):
        adata.var.index = adata.var["gene_symbols"].values
        adata.var_names_make_unique()
    return adata


def run_scgpt_embed(adata: sc.AnnData, model_dir: Path, batch_size: int = 64) -> np.ndarray:
    """Compute scGPT embeddings. Returns numpy array (n_cells, 512)."""
    from scgpt.tasks import embed_data

    embedded = embed_data(
        adata,
        model_dir=str(model_dir),
        gene_col="index",
        max_length=1200,
        batch_size=batch_size,
        device="cuda",
        use_fast_transformer=False,  # no flash_attn installed
        return_new_adata=True,
    )
    # embed_data with return_new_adata=True stores embeddings in result.X (n_cells, 512)
    return embedded.X


def knn_annotate(
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    query_emb: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """KNN classifier in embedding space."""
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(ref_emb, ref_labels)
    return knn.predict(query_emb)


def main():
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── prepare reference ─────────────────────────────────────────────────────
    ref = prepare_reference(max_cells_per_type=3000)

    print("\n=== Embedding reference data with scGPT ===")
    ref_emb = run_scgpt_embed(ref, MODEL_DIR, batch_size=128)
    ref_labels = ref.obs["broad6"].values
    print(f"Reference embeddings: {ref_emb.shape}")

    # ── annotate each dataset ─────────────────────────────────────────────────
    for ds_name, h5ad_path in DATASETS.items():
        out_path = RESULTS_DIR / f"{ds_name}_scGPT_pred.csv"
        if out_path.exists():
            print(f"\n[{ds_name}] Already done, skipping.")
            continue

        print(f"\n=== {ds_name} ===")
        query = prepare_query(h5ad_path)
        print(f"Query shape: {query.shape}")

        print(f"Embedding {ds_name} with scGPT...")
        query_emb = run_scgpt_embed(query, MODEL_DIR, batch_size=64)
        print(f"Query embeddings: {query_emb.shape}")

        print("KNN annotation (k=15)...")
        preds = knn_annotate(ref_emb, ref_labels, query_emb, k=15)

        result_df = pd.DataFrame(
            {"cell_id": query.obs.index, "pred_celltype": preds}
        ).set_index("cell_id")
        result_df.to_csv(out_path)
        print(f"Saved: {out_path}")
        print(pd.Series(preds).value_counts().to_string())

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
