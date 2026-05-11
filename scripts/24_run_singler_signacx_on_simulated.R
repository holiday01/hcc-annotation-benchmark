# Run SingleR and SignacX on simulated_benchmark.h5ad.
#
# Output:
#   annotation_results/simulated_SingleR_pred.csv
#   annotation_results/simulated_signacX_pred.csv
#
# Usage: Rscript 24_run_singler_signacx_on_simulated.R

suppressPackageStartupMessages({
  library(Seurat)
  library(SingleR)
  library(celldex)
  library(SignacX)
  library(Matrix)
  library(reticulate)
})
options(Seurat.object.assay.version = "v3")
use_python("/home/holiday01/miniconda/bin/python3", required = TRUE)

H5AD   <- "/mnt/10t/assi_result/HCC/benchmark_datasets/simulated_benchmark.h5ad"
OUT_DIR <- "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Load simulated h5ad via reticulate ───────────────────────────────────────
cat("Loading simulated h5ad...\n")
ad <- import("anndata")
sp <- import("scipy.sparse")

adata <- ad$read_h5ad(H5AD)
cat(sprintf("  %d cells x %d genes\n", nrow(adata$obs), nrow(adata$var)))

# Export count matrix to mtx
tmp_mtx   <- "/tmp/sim_X.mtx"
tmp_genes <- "/tmp/sim_genes.txt"
tmp_cells <- "/tmp/sim_cells.txt"
py$adata_r <- adata
py_run_string(sprintf('
import scipy.io
scipy.io.mmwrite("%s", adata_r.X.T)  # genes x cells
with open("%s", "w") as f:
    f.write("\\n".join(adata_r.var_names.tolist()))
with open("%s", "w") as f:
    f.write("\\n".join(adata_r.obs_names.tolist()))
', tmp_mtx, tmp_genes, tmp_cells))

X_mat    <- readMM(tmp_mtx)
gene_ids <- readLines(tmp_genes)
cell_ids <- readLines(tmp_cells)
rownames(X_mat) <- make.names(gene_ids, unique = TRUE)
colnames(X_mat) <- cell_ids
X_mat <- as(X_mat, "dgCMatrix")
cat(sprintf("  Matrix: %d genes x %d cells\n", nrow(X_mat), ncol(X_mat)))

# ── SingleR ──────────────────────────────────────────────────────────────────
cat("\nRunning SingleR...\n")
ref <- celldex::HumanPrimaryCellAtlasData()

# SingleR needs a counts/logcounts SCE; use the raw count matrix
library(SingleCellExperiment)
sce <- SingleCellExperiment(assays = list(counts = X_mat))
sce <- scuttle::logNormCounts(sce)

sr_pred <- SingleR(
  test      = sce,
  ref       = ref,
  labels    = ref$label.main,
  BPPARAM   = BiocParallel::MulticoreParam(4)
)

# Map SingleR labels to broad6
SINGLER_TO_BROAD6 <- c(
  "T_cells" = "T/NK", "NK_cell" = "T/NK", "CMP" = "Myeloid",
  "Monocyte" = "Myeloid", "Macrophage" = "Myeloid", "DC" = "Myeloid",
  "Neutrophils" = "Myeloid", "B_cell" = "B", "Plasma_cells" = "B",
  "Hepatocytes" = "Hepatocyte", "Endothelial_cells" = "Endothelial",
  "Fibroblasts" = "Fibroblast", "Smooth_muscle_cells" = "Fibroblast"
)
raw_labels <- sr_pred$labels
broad <- ifelse(raw_labels %in% names(SINGLER_TO_BROAD6),
                SINGLER_TO_BROAD6[raw_labels], "Unknown")

sr_df <- data.frame(
  barcode       = cell_ids,
  pred_celltype = broad,
  pred_singler_raw = raw_labels,
  row.names     = cell_ids
)
out_sr <- file.path(OUT_DIR, "simulated_SingleR_pred.csv")
write.csv(sr_df, out_sr, row.names = TRUE)
cat("Saved:", out_sr, "\n")
cat("SingleR distribution:\n")
print(table(sr_df$pred_celltype))

# ── SignacX ───────────────────────────────────────────────────────────────────
cat("\nRunning SignacX on simulated data...\n")
seurat_obj <- CreateSeuratObject(counts = X_mat, min.cells = 0, min.features = 0)
seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = 2000, verbose = FALSE)
seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
seurat_obj <- RunPCA(seurat_obj, npcs = 30, verbose = FALSE)
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30, verbose = FALSE)

labels    <- SignacFast(seurat_obj, num.cores = 4)
celltypes <- GenerateLabels(labels, seurat_obj)

cat("GenerateLabels names:", paste(names(celltypes), collapse=", "), "\n")

cellstates <- if (!is.null(celltypes$CellStates) && length(celltypes$CellStates) > 0)
  celltypes$CellStates else celltypes[[grep("CellStates|Celltypes", names(celltypes))[1]]]
celltypesv <- if (!is.null(celltypes$Celltypes) && length(celltypes$Celltypes) > 0)
  celltypes$Celltypes else celltypes[[grep("CellTypes|Celltypes", names(celltypes))[1]]]

# Map SignacX broad labels to broad6
SIGNACX_TO_BROAD6 <- c(
  "TNK" = "T/NK", "MPh" = "Myeloid", "NonImmune" = "Hepatocyte",
  "B" = "B", "Plasma.cells" = "B", "DC" = "Myeloid",
  "Endothelial" = "Endothelial", "Fibroblasts" = "Fibroblast",
  "Lymphocytes" = "T/NK", "Myeloid" = "Myeloid",
  "Unclassified" = "Unknown"
)
broad_sx <- ifelse(as.character(celltypesv) %in% names(SIGNACX_TO_BROAD6),
                   SIGNACX_TO_BROAD6[as.character(celltypesv)], "Unknown")

sx_df <- data.frame(
  barcode       = cell_ids,
  pred_celltype = broad_sx,
  pred_broad    = as.character(celltypesv),
  row.names     = cell_ids
)
out_sx <- file.path(OUT_DIR, "simulated_signacX_pred.csv")
write.csv(sx_df, out_sx, row.names = TRUE)
cat("Saved:", out_sx, "\n")
cat("SignacX distribution:\n")
print(table(sx_df$pred_celltype))

cat("\nDone.\n")
