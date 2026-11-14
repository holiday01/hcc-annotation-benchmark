#!/usr/bin/env Rscript
# 34_rerun_SignacX_rev2.R — SignacX on every benchmark cell, one procedure for all cohorts.
#
# Why (revision 2): the original SignacX runs differed between cohorts —
#   GSE125449 / GSE149614: log-normalised values read as counts (normalised twice),
#                          impute=TRUE on one cohort and impute=FALSE on the other;
#   GSE156625: 15,000-cell random subsample, the rest written as "Unclassified";
#   GSE162616 / GSE202642: only some samples run; GSE223204: barcodes did not match.
# Here every cohort is run from raw UMI counts (33_prepare_counts_rev2.py), per
# sample (samples with < 200 cells pooled), with identical settings:
#   NormalizeData -> 2,000 HVG -> ScaleData -> PCA (30, seed 42) -> FindNeighbors (dims 1:30)
#   -> SignacFast (patched SignacX 2.2.5, HPCA models, impute=TRUE) -> GenerateLabels.
# Running per sample keeps every neighbour graph within one library and below the
# ~46k-cell size at which SignacX's internal matrix indexing overflows.
#
# Output: annotation_results_rev2/<GSE>_SignacX_pred.csv
#   barcode, pred_celltype (CellStates), pred_broad (CellTypes), group
#
# Usage: Rscript 34_rerun_SignacX_rev2.R [GSE ...]      (env SIGNACX_CORES, default 6)

args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
HERE <- dirname(normalizePath(sub("^--file=", "", file_arg)))
source(file.path(HERE, "_signacx_patched.R"))

IN  <- "/mnt/10t/assi_result/HCC/benchmark_datasets/rev2_inputs"
OUT <- "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
NCORES <- as.integer(Sys.getenv("SIGNACX_CORES", "6"))
cohorts <- if (length(args)) args else
  c("GSE223204", "GSE125449", "GSE162616", "GSE156625", "GSE202642", "GSE149614")
set.seed(42)

run_group <- function(mat, cells) {
  obj <- CreateSeuratObject(counts = mat[, cells, drop = FALSE], assay = "RNA",
                            min.cells = 0, min.features = 0)
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE)
  obj <- ScaleData(obj, verbose = FALSE)
  obj <- RunPCA(obj, npcs = 30, seed.use = 42, verbose = FALSE)
  obj <- FindNeighbors(obj, dims = 1:30, verbose = FALSE)
  labels <- SignacFast_fixed(obj, num.cores = NCORES, impute = TRUE, verbose = FALSE)
  ct <- GenerateLabels(labels, E = obj)
  data.frame(barcode = colnames(obj),
             pred_celltype = as.character(ct$CellStates),
             pred_broad = as.character(ct$CellTypes),
             stringsAsFactors = FALSE)
}

for (ds in cohorts) {
  out_path <- file.path(OUT, paste0(ds, "_SignacX_pred.csv"))
  if (file.exists(out_path)) { cat("skip (exists):", out_path, "\n"); next }
  cat("\n===", ds, "===\n")
  d <- file.path(IN, ds)
  mat <- readMM(file.path(d, "matrix.mtx"))
  rownames(mat) <- readLines(file.path(d, "genes.tsv"))
  colnames(mat) <- readLines(file.path(d, "barcodes.tsv"))
  mat <- as(mat, "CsparseMatrix")
  meta <- read.csv(file.path(d, "cells.csv"), stringsAsFactors = FALSE)
  stopifnot(identical(meta$barcode, colnames(mat)))
  cat("  ", nrow(mat), "genes x", ncol(mat), "cells;", length(unique(meta$group)), "groups\n")
  res <- list()
  for (g in unique(meta$group)) {
    cells <- meta$barcode[meta$group == g]
    t0 <- proc.time()[3]
    r <- run_group(mat, cells)
    r$group <- g
    res[[g]] <- r
    cat(sprintf("   %-16s %6d cells  %6.0f s  %s\n", g, length(cells), proc.time()[3] - t0,
                paste(names(sort(table(r$pred_broad), decreasing = TRUE))[1:3], collapse = ",")))
  }
  out <- do.call(rbind, res)
  out <- out[match(meta$barcode, out$barcode), ]
  stopifnot(!any(is.na(out$barcode)))
  write.csv(out, out_path, row.names = FALSE)
  cat("  saved", out_path, "\n")
  print(sort(table(out$pred_broad), decreasing = TRUE))
}
cat("\nAll done.\n")
