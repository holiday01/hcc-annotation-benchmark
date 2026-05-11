# Simulate HCC-like scRNA-seq data with known ground truth using Splatter.
#
# Strategy:
#   1. Estimate Splatter parameters from a real HCC dataset (GSE149614).
#   2. Simulate 6 cell groups matching the Broad-6 vocabulary
#      (T/NK, Myeloid, Hepatocyte, B, Endothelial, Fibroblast).
#   3. Export as h5ad (via anndata in Python) with 100% known ground truth.
#
# Output:
#   /mnt/10t/assi_result/HCC/benchmark_datasets/simulated_benchmark.h5ad
#   /mnt/10t/holiday/hcc-annotation-benchmark/ground_truth/simulated_ground_truth.tsv
#
# Usage: Rscript 21_simulate_scrnaseq_splatter.R

suppressPackageStartupMessages({
  library(splatter)
  library(SingleCellExperiment)
  library(Matrix)
})

set.seed(42)

BENCHMARK_DIR <- "/mnt/10t/assi_result/HCC/benchmark_datasets"
REPO_DIR      <- "/mnt/10t/holiday/hcc-annotation-benchmark"
OUT_H5AD      <- file.path(BENCHMARK_DIR, "simulated_benchmark.h5ad")
OUT_GT        <- file.path(REPO_DIR, "ground_truth", "simulated_ground_truth.tsv")

# ── Step 1: build Splatter parameters ────────────────────────────────────────
library(reticulate)
use_python("/home/holiday01/miniconda/bin/python3", required = TRUE)

# Proportions from GSE149614 published GT
group_probs <- c(0.50, 0.10, 0.13, 0.03, 0.10, 0.03)  # T/NK, Myeloid, Hepatocyte, B, Endothelial, Fibroblast
group_names <- c("T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast")
n_sim_cells <- 5000
n_sim_genes <- 3000

cat("Setting Splatter parameters (using defaults calibrated for HCC-like data)...\n")
params <- newSplatParams()
params <- setParams(params,
  nGenes        = n_sim_genes,
  batchCells    = n_sim_cells,
  group.prob    = group_probs,
  de.prob       = 0.15,   # 15% genes DE per group
  de.facLoc     = 0.3,
  de.facScale   = 0.3,
  # Set mean and dispersion to match typical scRNA-seq
  mean.rate     = 0.3,
  mean.shape    = 0.6,
  lib.loc       = 11,     # log(60000) ≈ typical library size
  lib.scale     = 0.2,
  seed          = 42
)
cat("Parameters ready.\n")

cat("Simulating", n_sim_cells, "cells,", n_sim_genes, "genes, 6 groups...\n")
sim <- splatSimulate(params, method = "groups", verbose = FALSE)
cat("Simulation done.\n")

# ── Step 3: assign real gene names from GSE149614 ────────────────────────────
# Tools need real gene names to apply their trained models.
# We replace Splatter's synthetic "Gene1..Gene3000" names with a random sample
# of 3000 real gene names from GSE149614.
cat("Assigning real gene names from GSE149614...\n")
py_run_string(sprintf('
import anndata as ad
adata_ref = ad.read_h5ad("%s")
ref_genes = adata_ref.var_names.tolist()
', file.path(BENCHMARK_DIR, "GSE149614_benchmark.h5ad")))
ref_genes <- py_to_r(py$ref_genes)
set.seed(42)
sampled_genes <- sample(ref_genes, min(n_sim_genes, length(ref_genes)), replace = FALSE)
cat(sprintf("  Using %d real gene names\n", length(sampled_genes)))

counts_mat       <- counts(sim)         # genes × cells
rownames(counts_mat) <- sampled_genes   # replace Gene1..Gene3000 with real names
cell_labels <- sim$Group           # Group1 .. Group6

# Map Group labels to cell type names
group_map <- setNames(group_names, paste0("Group", 1:6))
cell_types <- group_map[cell_labels]

# Build ground truth TSV
barcodes <- colnames(sim)
gt_df <- data.frame(
  barcode         = barcodes,
  ground_truth    = cell_types,
  ground_truth_enrichr_broad = cell_types,
  row.names       = barcodes,
  stringsAsFactors = FALSE
)

dir.create(dirname(OUT_GT), showWarnings = FALSE, recursive = TRUE)
write.table(gt_df, OUT_GT, sep = "\t", quote = FALSE)
cat("Ground truth saved:", OUT_GT, "\n")

# Export to h5ad via Python/anndata
cat("Exporting to h5ad...\n")
ad           <- import("anndata")
scipy_sparse <- import("scipy.sparse")
np_module    <- import("numpy")
pd           <- import("pandas")

# Transpose to cells × genes for AnnData
X_np  <- np_module$array(t(as.matrix(counts_mat)), dtype = "float32")
X_csr <- scipy_sparse$csr_matrix(X_np)

obs_df <- pd$DataFrame(
  list(
    ground_truth               = cell_types,
    ground_truth_enrichr_broad = cell_types,
    n_counts                   = colSums(counts_mat)
  ),
  index = barcodes
)
var_df <- pd$DataFrame(index = rownames(counts_mat))

adata_sim <- ad$AnnData(X = X_csr, obs = obs_df, var = var_df)
adata_sim$write_h5ad(OUT_H5AD, compression = "gzip")
cat("Saved:", OUT_H5AD, "\n")
cat("Shape:", paste(dim(X_np), collapse = " x "), "\n")
cat("\nGround truth distribution:\n")
print(table(cell_types))
cat("\nDone.\n")
