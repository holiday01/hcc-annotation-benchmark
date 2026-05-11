#!/usr/bin/env Rscript

# Run SingleR with the celldex Human Primary Cell Atlas reference on the
# full-matrix benchmark datasets. The output schema matches the existing
# annotation_results/*_pred.csv files used by 02_benchmark_evaluation.py.

suppressPackageStartupMessages({
  library(Matrix)
  library(SingleR)
  library(celldex)
  library(BiocParallel)
  library(SummarizedExperiment)
  library(zellkonverter)
})

BASE <- "/mnt/10t/assi_result/HCC/benchmark_datasets"
OUT_DIR <- file.path(BASE, "annotation_results")
DATASETS <- c("GSE125449", "GSE149614", "GSE156625",
              "GSE162616", "GSE202642", "GSE223204")
H5AD_INPUTS <- c(
  "GSE156625" = "/mnt/10t/assi_result/HCC/benchmark_datasets/GSE156625_benchmark.h5ad",
  "GSE162616" = "/mnt/10t/assi_result/HCC/open_source_model/cytetype/adata_CyteType_GSE162616.h5ad",
  "GSE202642" = "/mnt/10t/assi_result/HCC/open_source_model/cytetype/adata_CyteType_GSE202642.h5ad",
  "GSE223204" = "/mnt/10t/assi_result/HCC/open_source_model/cytetype/adata_CyteType_GSE223204.h5ad"
)

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

hpca_to_broad6 <- function(label) {
  label <- ifelse(is.na(label), "Unknown", as.character(label))
  mapping <- c(
    "T_cells" = "T/NK",
    "NK_cell" = "T/NK",
    "B_cell" = "B",
    "Pre-B_cell_CD34-" = "B",
    "Pro-B_cell_CD34+" = "B",
    "DC" = "Myeloid",
    "Macrophage" = "Myeloid",
    "Monocyte" = "Myeloid",
    "Neutrophils" = "Myeloid",
    "Myelocyte" = "Myeloid",
    "Pro-Myelocyte" = "Myeloid",
    "Epithelial_cells" = "Hepatocyte",
    "Hepatocytes" = "Hepatocyte",
    "Endothelial_cells" = "Endothelial",
    "Fibroblasts" = "Fibroblast",
    "MSC" = "Fibroblast",
    "Smooth_muscle_cells" = "Fibroblast"
  )
  out <- unname(mapping[label])
  out[is.na(out)] <- "Unknown"
  out
}

read_benchmark_matrix <- function(dataset) {
  ds_dir <- file.path(BASE, paste0(dataset, "_mtx"))
  matrix_path <- file.path(ds_dir, "matrix.mtx")
  features_path <- file.path(ds_dir, "features.tsv")
  barcodes_path <- file.path(ds_dir, "barcodes.tsv")

  counts <- readMM(matrix_path)
  features <- read.delim(features_path, header = FALSE, stringsAsFactors = FALSE)
  barcodes <- readLines(barcodes_path)

  if (nrow(features) != nrow(counts) || length(barcodes) != ncol(counts)) {
    stop(sprintf(
      "%s dimension mismatch: matrix=%sx%s, features=%s, barcodes=%s",
      dataset, nrow(counts), ncol(counts), nrow(features), length(barcodes)
    ))
  }

  gene_symbol <- if (ncol(features) >= 2) features[[2]] else features[[1]]
  gene_symbol <- make.unique(as.character(gene_symbol))
  rownames(counts) <- gene_symbol
  colnames(counts) <- barcodes
  as(counts, "dgCMatrix")
}

read_h5ad_log_matrix <- function(dataset) {
  h5ad_path <- H5AD_INPUTS[[dataset]]
  if (is.null(h5ad_path) || !file.exists(h5ad_path)) {
    stop(sprintf("No h5ad input configured for %s", dataset))
  }

  sce <- readH5AD(h5ad_path, use_hdf5 = FALSE)
  expr <- assay(sce, "X")

  if (is.null(rownames(expr)) || is.null(colnames(expr))) {
    stop(sprintf("%s h5ad lacks gene or barcode names", dataset))
  }
  as(expr, "dgCMatrix")
}

log_normalize_sparse <- function(counts, scale_factor = 10000) {
  lib_size <- Matrix::colSums(counts)
  keep <- lib_size > 0
  if (!all(keep)) {
    counts <- counts[, keep, drop = FALSE]
    lib_size <- lib_size[keep]
  }

  norm <- t(t(counts) / lib_size) * scale_factor
  norm@x <- log1p(norm@x)
  norm
}

load_expression <- function(dataset) {
  if (dataset %in% names(H5AD_INPUTS)) {
    message("Reading log-normalized h5ad X from CyteType input")
    return(read_h5ad_log_matrix(dataset))
  }
  message("Reading raw MatrixMarket counts and applying log-normalization")
  log_normalize_sparse(read_benchmark_matrix(dataset))
}

run_singler_dataset <- function(dataset, ref) {
  message("\n== ", dataset, " ==")
  logcounts <- load_expression(dataset)
  message("Loaded matrix: ", nrow(logcounts), " genes x ", ncol(logcounts), " cells")

  common_genes <- intersect(rownames(logcounts), rownames(ref))
  message("Common genes with HPCA: ", length(common_genes))
  if (length(common_genes) < 1000) {
    stop(sprintf("%s has too few common genes for SingleR: %s", dataset, length(common_genes)))
  }

  pred <- SingleR(
    test = logcounts[common_genes, , drop = FALSE],
    ref = ref[common_genes, ],
    labels = ref$label.main,
    assay.type.test = 1,
    assay.type.ref = "logcounts",
    BPPARAM = SerialParam()
  )

  raw_label <- as.character(pred$labels)
  pruned_label <- as.character(pred$pruned.labels)
  pred_label <- ifelse(is.na(pruned_label), "Unknown", pruned_label)
  pred_broad6 <- hpca_to_broad6(pred_label)

  out <- data.frame(
    barcode = colnames(logcounts),
    pred_raw = raw_label,
    pred_pruned = pred_label,
    pred_celltype = pred_broad6,
    singler_delta = pred$delta.next,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  rownames(out) <- out$barcode

  out_path <- file.path(OUT_DIR, paste0(dataset, "_SingleR_pred.csv"))
  write.csv(out, out_path, quote = TRUE)
  message("Wrote: ", out_path)
  message("Broad6 labels:")
  print(table(out$pred_celltype, useNA = "ifany"))
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  datasets <- if (length(args) > 0) args else DATASETS

  ref <- HumanPrimaryCellAtlasData()
  for (dataset in datasets) {
    run_singler_dataset(dataset, ref)
    gc()
  }
}

main()
