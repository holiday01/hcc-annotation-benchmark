#!/usr/bin/env Rscript
# 43_run_reference_tools_rev2.R — reference-based tools trained on one shared reference (revision 2).
#
# Reviewer 1, comment 1 (method coverage). Every tool here is trained on the same
# cells and labels: the leave-benchmark-cohorts-out DISCO liver reference built by
# 42_build_shared_reference.py (16,141 cells, broad-6 labels; identical to the
# scGPT k-NN reference). Queries are the raw-count matrices of all benchmark cells
# (33_prepare_counts_rev2.py). Each tool runs with its documented defaults:
#
#   SeuratLT      Seurat label transfer (the algorithm behind Azimuth):
#                 LogNormalize, 2,000 HVG, PCA (30) on the reference;
#                 FindTransferAnchors(reference.reduction = "pca", dims 1:30) + TransferData.
#                 No abstention by default; prediction.score.max is saved.
#   SingleRdisco  SingleR with the shared reference, default marker detection
#                 (de.method = "classic", as in the HPCA-based SingleR run; "wilcox" needs the
#                 scrapper backend, not installed); pruned labels (NA -> Unknown).
#                 Isolates the reference effect for SingleR.
#
# Output: annotation_results_rev2/<GSE>_<Tool>_pred.csv (barcode, pred_celltype, score)
# Usage:  Rscript 43_run_reference_tools_rev2.R [--tools SeuratLT,SingleRdisco] [GSE ...]

suppressMessages({
  library(Matrix)
  library(Seurat)
  library(SingleR)
  library(BiocParallel)
})
args <- commandArgs(trailingOnly = TRUE)
tools <- c("SeuratLT", "SingleRdisco")
if (length(args) && args[1] == "--tools") { tools <- strsplit(args[2], ",")[[1]]; args <- args[-(1:2)] }
cohorts <- if (length(args)) args else
  c("GSE223204", "GSE125449", "GSE162616", "GSE156625", "GSE202642", "GSE149614")
IN  <- "/mnt/10t/assi_result/HCC/benchmark_datasets/rev2_inputs"
OUT <- "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2"
set.seed(42)
cat("Seurat", as.character(packageVersion("Seurat")), "| SingleR", as.character(packageVersion("SingleR")), "\n")

read_counts <- function(d) {
  m <- readMM(file.path(d, "matrix.mtx"))
  rownames(m) <- readLines(file.path(d, "genes.tsv"))
  colnames(m) <- readLines(file.path(d, "barcodes.tsv"))
  as(m, "CsparseMatrix")
}

ref_dir <- file.path(IN, "DISCO_reference")
ref_counts <- read_counts(ref_dir)
ref_meta <- read.csv(file.path(ref_dir, "cells.csv"), stringsAsFactors = FALSE)
stopifnot(identical(ref_meta$barcode, colnames(ref_counts)))
ref <- CreateSeuratObject(counts = ref_counts, min.cells = 0, min.features = 0)
ref$label <- ref_meta$label
ref <- NormalizeData(ref, verbose = FALSE)
ref <- FindVariableFeatures(ref, nfeatures = 2000, verbose = FALSE)
ref <- ScaleData(ref, verbose = FALSE)
ref <- RunPCA(ref, npcs = 30, seed.use = 42, verbose = FALSE)
ref_log <- GetAssayData(ref, layer = "data")
cat("reference:", ncol(ref), "cells;", paste(names(table(ref$label)), table(ref$label), collapse = ", "), "\n")

for (ds in cohorts) {
  cat("\n===", ds, "===\n")
  q_counts <- read_counts(file.path(IN, ds))
  q <- CreateSeuratObject(counts = q_counts, min.cells = 0, min.features = 0)
  q <- NormalizeData(q, verbose = FALSE)
  if ("SeuratLT" %in% tools) {
    out <- file.path(OUT, paste0(ds, "_SeuratLT_pred.csv"))
    if (!file.exists(out)) {
      t0 <- proc.time()[3]
      anchors <- FindTransferAnchors(reference = ref, query = q, dims = 1:30,
                                     reference.reduction = "pca", verbose = FALSE)
      p <- TransferData(anchorset = anchors, refdata = ref$label, dims = 1:30, verbose = FALSE)
      res <- data.frame(barcode = colnames(q), pred_celltype = p$predicted.id,
                        score = round(p$prediction.score.max, 4), stringsAsFactors = FALSE)
      write.csv(res, out, row.names = FALSE)
      cat(sprintf("  SeuratLT %.0f s: %s\n", proc.time()[3] - t0,
                  paste(names(table(res$pred_celltype)), table(res$pred_celltype), collapse = ", ")))
    }
  }
  if ("SingleRdisco" %in% tools) {
    out <- file.path(OUT, paste0(ds, "_SingleRdisco_pred.csv"))
    if (!file.exists(out)) {
      t0 <- proc.time()[3]
      s <- SingleR(test = GetAssayData(q, layer = "data"), ref = ref_log, labels = ref$label,
                   BPPARAM = MulticoreParam(8))
      lab <- s$pruned.labels
      lab[is.na(lab)] <- "Unknown"
      res <- data.frame(barcode = rownames(s), pred_celltype = lab, pred_unpruned = s$labels,
                        score = round(s$delta.next, 4), stringsAsFactors = FALSE)
      write.csv(res, out, row.names = FALSE)
      cat(sprintf("  SingleRdisco %.0f s: %s\n", proc.time()[3] - t0,
                  paste(names(table(res$pred_celltype)), table(res$pred_celltype), collapse = ", ")))
    }
  }
}
cat("\nAll done.\n")
