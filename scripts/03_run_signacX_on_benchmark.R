# Run signacX on GSE149614 and GSE125449 benchmark datasets
# Output: CSV with barcode + pred_celltype for each dataset
#
# Usage: Rscript 03_run_signacX_on_benchmark.R
#
# Requires: Seurat, SignacX, SeuratDisk

library(Seurat)
library(SignacX)
library(SeuratDisk)
library(Matrix)
library(dplyr)

OUT_DIR <- "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

run_signacx <- function(h5ad_path, dataset_name) {
  cat("\n===", dataset_name, "===\n")

  # Load h5ad via SeuratDisk
  Convert(h5ad_path, dest = "h5seurat", overwrite = TRUE)
  seurat_obj <- LoadH5Seurat(gsub(".h5ad", ".h5seurat", h5ad_path))

  # Ensure RNA assay is set
  DefaultAssay(seurat_obj) <- "RNA"

  # Normalize if not already done
  seurat_obj <- NormalizeData(seurat_obj)

  # Run SignacX
  cat("Running SignacX...\n")
  labels <- SignacX(seurat_obj, num.cores = 4)
  celltypes <- GenerateLabels(labels, seurat_obj)

  # Extract predictions
  pred_df <- data.frame(
    barcode      = colnames(seurat_obj),
    pred_celltype = celltypes$CellStates,
    pred_broad    = celltypes$Celltypes,
    row.names    = colnames(seurat_obj)
  )

  out_file <- file.path(OUT_DIR, paste0(dataset_name, "_signacX_pred.csv"))
  write.csv(pred_df, out_file, row.names = TRUE)
  cat("Saved:", out_file, "\n")
  cat("Prediction distribution:\n")
  print(table(pred_df$pred_broad))
}

# Run on both datasets (GSE149614 only after h5ad is ready)
run_signacx(
  "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE125449_benchmark.h5ad",
  "GSE125449"
)

# Uncomment after GSE149614_benchmark.h5ad is ready:
# run_signacx(
#   "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE149614_benchmark.h5ad",
#   "GSE149614"
# )
