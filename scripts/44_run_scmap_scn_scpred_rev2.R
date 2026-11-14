#!/usr/bin/env Rscript
# 44_run_scmap_scn_scpred_rev2.R — scmap, SingleCellNet and scPred on the shared reference (revision 2).
#
# Reviewer 1, comment 1. Same reference and queries as 43_run_reference_tools_rev2.R
# (42_build_shared_reference.py: leave-benchmark-cohorts-out DISCO liver reference,
# 16,141 cells, broad-6 labels; queries = raw counts of all benchmark cells).
# Packages live in ~/R/rev2_lib (scmap 1.32.0, singleCellNet 0.1.0, scPred 1.9.2 with
# harmony 1.2.3) and Seurat 4.4 from ~/R/seurat4_lib (scPred targets the Seurat 4 API).
# Documented defaults:
#   scmap          scmap-cluster: logcounts = log2(counts per 10k + 1); selectFeatures(n = 500);
#                  indexCluster on the broad-6 label; scmapCluster(threshold = 0.7);
#                  "unassigned" = abstention.
#   SingleCellNet  scn_train(nTopGenes = 10, nRand = 70, nTrees = 1000, nTopGenePairs = 25)
#                  on raw counts; scn_predict on the classifier genes in chunks of 5,000 cells;
#                  label = highest-scoring class; "rand" = abstention.
#   scPred         NormalizeData -> 2,000 HVG -> ScaleData -> PCA (30) on the reference;
#                  getFeatureSpace(label) + trainModel() (svmRadial); scPredict(threshold 0.55);
#                  "unassigned" = abstention.
# Output: annotation_results_rev2/<GSE>_<Tool>_pred.csv (barcode, pred_celltype, score)
# Usage:  Rscript 44_run_scmap_scn_scpred_rev2.R [--tools scmap,SingleCellNet,scPred] [GSE ...]

# Load the system Matrix (1.7) before adding the Seurat 4 library, as in _signacx_patched.R;
# loading Matrix 1.6.5 from ~/R/seurat4_lib breaks Seurat/SeuratObject 4 ("generalMatrix_validate").
suppressMessages(library(Matrix))
.libPaths(c(path.expand("~/R/rev2_lib"), path.expand("~/R/seurat4_lib"), .libPaths()))
args <- commandArgs(trailingOnly = TRUE)
tools <- c("scmap", "SingleCellNet", "scPred")
if (length(args) && args[1] == "--tools") { tools <- strsplit(args[2], ",")[[1]]; args <- args[-(1:2)] }
cohorts <- if (length(args)) args else
  c("GSE223204", "GSE125449", "GSE162616", "GSE156625", "GSE202642", "GSE149614")
IN  <- "/mnt/10t/assi_result/HCC/benchmark_datasets/rev2_inputs"
OUT <- "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2"
set.seed(42)

read_counts <- function(d) {
  m <- readMM(file.path(d, "matrix.mtx"))
  rownames(m) <- readLines(file.path(d, "genes.tsv"))
  colnames(m) <- readLines(file.path(d, "barcodes.tsv"))
  as(m, "CsparseMatrix")
}
log2cp10k <- function(m) {
  m <- m %*% Diagonal(x = 1e4 / pmax(colSums(m), 1))
  m@x <- log2(m@x + 1)
  m
}
done <- function(ds, tool) file.exists(file.path(OUT, paste0(ds, "_", tool, "_pred.csv")))
save <- function(ds, tool, barcode, lab, score) {
  res <- data.frame(barcode = barcode, pred_celltype = lab, score = round(score, 4), stringsAsFactors = FALSE)
  write.csv(res, file.path(OUT, paste0(ds, "_", tool, "_pred.csv")), row.names = FALSE)
  cat(sprintf("  %s: %s\n", tool, paste(names(table(lab)), table(lab), collapse = ", ")))
}

ref_dir <- file.path(IN, "DISCO_reference")
ref_counts <- read_counts(ref_dir)
ref_meta <- read.csv(file.path(ref_dir, "cells.csv"), stringsAsFactors = FALSE)
stopifnot(identical(ref_meta$barcode, colnames(ref_counts)))
labels <- ref_meta$label
cat("reference:", ncol(ref_counts), "cells; tools:", paste(tools, collapse = ", "), "\n")

# ── train once ──
models <- list()
if ("scmap" %in% tools) {
  suppressMessages({ library(SingleCellExperiment); library(scmap) })
  rs <- SingleCellExperiment(assays = list(counts = ref_counts, logcounts = log2cp10k(ref_counts)),
                             colData = DataFrame(label = labels))
  rowData(rs)$feature_symbol <- rownames(rs)
  rs <- selectFeatures(rs, n_features = 500, suppress_plot = TRUE)
  rs <- indexCluster(rs, cluster_col = "label")
  models$scmap <- metadata(rs)$scmap_cluster_index
  cat("scmap index:", nrow(models$scmap), "features\n")
}
if ("SingleCellNet" %in% tools) {
  suppressMessages(library(singleCellNet))
  keep <- rowSums(ref_counts > 0) >= 10
  st <- data.frame(cell = colnames(ref_counts), label = labels, stringsAsFactors = FALSE)
  rownames(st) <- st$cell
  t0 <- proc.time()[3]
  ci <- scn_train(stTrain = st, expTrain = as.matrix(ref_counts[keep, ]), nTopGenes = 10, nRand = 70,
                  nTrees = 1000, nTopGenePairs = 25, dLevel = "label", colName_samp = "cell")
  models$SingleCellNet <- ci$cnProc
  cat(sprintf("SingleCellNet trained in %.0f s on %d genes\n", proc.time()[3] - t0, length(ci$cnProc$cgenes)))
}
if ("scPred" %in% tools) {
  suppressMessages({ library(Seurat); library(scPred) })
  ref <- CreateSeuratObject(counts = ref_counts, min.cells = 0, min.features = 0)
  ref$label <- labels
  ref <- NormalizeData(ref, verbose = FALSE)
  ref <- FindVariableFeatures(ref, nfeatures = 2000, verbose = FALSE)
  ref <- ScaleData(ref, verbose = FALSE)
  ref <- RunPCA(ref, npcs = 30, seed.use = 42, verbose = FALSE)
  ref <- getFeatureSpace(ref, "label")
  t0 <- proc.time()[3]
  ref <- trainModel(ref)
  models$scPred <- ref
  cat(sprintf("scPred trained in %.0f s\n", proc.time()[3] - t0))
}

# ── predict ──
for (ds in cohorts) {
  cat("\n===", ds, "===\n")
  q <- read_counts(file.path(IN, ds))
  if ("scmap" %in% tools && !done(ds, "scmap")) {
    qs <- SingleCellExperiment(assays = list(counts = q, logcounts = log2cp10k(q)))
    rowData(qs)$feature_symbol <- rownames(qs)
    r <- scmapCluster(projection = qs, index_list = list(ref = models$scmap), threshold = 0.7)
    save(ds, "scmap", colnames(q), r$scmap_cluster_labs[, "ref"], r$scmap_cluster_siml[, "ref"])
  }
  if ("SingleCellNet" %in% tools && !done(ds, "SingleCellNet")) {
    cp <- models$SingleCellNet
    g <- cp$cgenes
    qg <- q[intersect(g, rownames(q)), , drop = FALSE]
    miss <- setdiff(g, rownames(q))
    if (length(miss)) qg <- rbind(qg, Matrix(0, length(miss), ncol(q), sparse = TRUE,
                                             dimnames = list(miss, colnames(q))))
    qg <- qg[g, ]
    chunks <- split(seq_len(ncol(qg)), ceiling(seq_len(ncol(qg)) / 5000))
    cr <- do.call(cbind, lapply(chunks, function(i) scn_predict(cp, as.matrix(qg[, i, drop = FALSE]), nrand = 0)))
    lab <- rownames(cr)[apply(cr, 2, which.max)]
    save(ds, "SingleCellNet", colnames(q), lab, apply(cr, 2, max))
  }
  if ("scPred" %in% tools && !done(ds, "scPred")) {
    qo <- CreateSeuratObject(counts = q, min.cells = 0, min.features = 0)
    qo <- NormalizeData(qo, verbose = FALSE)
    qo <- scPredict(qo, models$scPred, threshold = 0.55)
    save(ds, "scPred", colnames(qo), as.character(qo$scpred_prediction), qo$scpred_max)
  }
}
cat("\nAll done.\n")
