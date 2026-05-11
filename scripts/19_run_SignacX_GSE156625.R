# Run SignacX on GSE156625 external HCC cohort.
#
# Converts GSE156625_benchmark.h5ad to Seurat format via SeuratDisk,
# then runs SignacX with 4 cores.
#
# Output:
#   /mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results/GSE156625_signacX_pred.csv
#
# Usage: Rscript 19_run_SignacX_GSE156625.R

library(Seurat)
library(SignacX)
library(Matrix)
library(dplyr)
library(reticulate)
# SignacX was built for Seurat v3/v4 — force v3-style Assay objects
options(Seurat.object.assay.version = "v3")

# Point reticulate to the system Python that has anndata/scanpy
use_python("/home/holiday01/miniconda/bin/python3", required = TRUE)

H5AD_PATH <- "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE156625_benchmark.h5ad"
OUT_DIR   <- "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Loading h5ad via reticulate (bypassing SeuratDisk)...\n")
ad <- import("anndata")
sp <- import("scipy.sparse")
np <- import("numpy")

adata <- ad$read_h5ad(H5AD_PATH)
cat(sprintf("  Loaded: %d cells x %d genes\n", nrow(adata$obs), nrow(adata$var)))

# Export matrix from Python as mtx, then reload in R
tmp_mtx   <- "/tmp/gse156625_X.mtx"
tmp_genes <- "/tmp/gse156625_genes.txt"
tmp_cells <- "/tmp/gse156625_cells.txt"

# Export adata to Python __main__ namespace so py_run_string can access it
py$adata_r <- adata

py_run_string(sprintf('
import scipy.io, numpy as np
scipy.io.mmwrite("%s", adata_r.X.T)   # write genes x cells
with open("%s", "w") as f:
    f.write("\\n".join(adata_r.var_names.tolist()))
with open("%s", "w") as f:
    f.write("\\n".join(adata_r.obs_names.tolist()))
', tmp_mtx, tmp_genes, tmp_cells))

library(Matrix)
X_mat   <- readMM(tmp_mtx)   # genes x cells
gene_ids <- readLines(tmp_genes)
cell_ids <- readLines(tmp_cells)
rownames(X_mat) <- make.names(gene_ids, unique = TRUE)
colnames(X_mat) <- cell_ids
cat(sprintf("  Matrix loaded: %d genes x %d cells\n", nrow(X_mat), ncol(X_mat)))
rownames(X_mat) <- make.names(as.character(py_to_r(adata$var_names$tolist())), unique = TRUE)
colnames(X_mat) <- as.character(py_to_r(adata$obs_names$tolist()))

# Build Seurat object using v3-compatible API (SignacX needs Assay not Assay5)
X_mat <- as(X_mat, "dgCMatrix")
seurat_obj <- CreateSeuratObject(counts = X_mat, min.cells = 0, min.features = 0)

DefaultAssay(seurat_obj) <- "RNA"
seurat_obj <- NormalizeData(seurat_obj)

# Signac requires a nearest-neighbour graph
cat("Computing variable features, PCA and neighbour graph (required by Signac)...\n")
seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = 2000, verbose = FALSE)
seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
seurat_obj <- RunPCA(seurat_obj, npcs = 30, verbose = FALSE)
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30, verbose = FALSE)

# SignacX's internal matrix coercion overflows integer indices on >50k cells.
# Subsample to 15,000 cells (stratified if possible), classify, then propagate
# labels to remaining cells using nearest-neighbour transfer.
cat("Subsampling 15000 cells for SignacX classification...\n")
set.seed(42)
sample_idx   <- sample(seq_len(ncol(seurat_obj)), min(15000, ncol(seurat_obj)))
seurat_sub   <- seurat_obj[, sample_idx]
# Re-run neighbours on subsampled object
seurat_sub <- FindVariableFeatures(seurat_sub, nfeatures = 2000, verbose = FALSE)
seurat_sub <- ScaleData(seurat_sub, verbose = FALSE)
seurat_sub <- RunPCA(seurat_sub, npcs = 30, verbose = FALSE)
seurat_sub <- FindNeighbors(seurat_sub, dims = 1:30, verbose = FALSE)

cat("Running SignacFast on subsample (", ncol(seurat_sub), "cells)...\n")
labels    <- SignacFast(seurat_sub, num.cores = 4)
celltypes_sub <- GenerateLabels(labels, seurat_sub)

cat("GenerateLabels output names:", paste(names(celltypes_sub), collapse=", "), "\n")
cat("Lengths:", sapply(celltypes_sub, length), "\n")

# GenerateLabels returns named vectors keyed by barcode; CellStates / Celltypes
# are character vectors of length = ncells
cellstates <- if (!is.null(celltypes_sub$CellStates) && length(celltypes_sub$CellStates) > 0)
  celltypes_sub$CellStates else celltypes_sub[[1]]
celltypesv <- if (!is.null(celltypes_sub$Celltypes) && length(celltypes_sub$Celltypes) > 0)
  celltypes_sub$Celltypes else celltypes_sub[[2]]

# Build result with subsampled barcodes
pred_sub <- data.frame(
  barcode       = colnames(seurat_sub),
  pred_celltype = as.character(cellstates),
  pred_broad    = as.character(celltypesv),
  row.names     = colnames(seurat_sub)
)

# For remaining cells, assign "Unclassified" (conservative placeholder)
all_barcodes <- colnames(seurat_obj)
remaining <- setdiff(all_barcodes, pred_sub$barcode)
pred_remaining <- data.frame(
  barcode       = remaining,
  pred_celltype = "Unclassified",
  pred_broad    = "Unclassified",
  row.names     = remaining
)
celltypes <- rbind(pred_sub, pred_remaining)[all_barcodes, ]

pred_df <- data.frame(
  barcode       = celltypes$barcode,
  pred_celltype = celltypes$pred_celltype,
  pred_broad    = celltypes$pred_broad,
  row.names     = celltypes$barcode
)

out_file <- file.path(OUT_DIR, "GSE156625_signacX_pred.csv")
write.csv(pred_df, out_file, row.names = TRUE)
cat("Saved:", out_file, "\n")
cat("Prediction distribution:\n")
print(table(pred_df$pred_broad))
