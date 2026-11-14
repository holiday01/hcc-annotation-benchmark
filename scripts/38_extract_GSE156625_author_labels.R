#!/usr/bin/env Rscript
# 38_extract_GSE156625_author_labels.R — published cell-type labels for GSE156625 (revision 2).
#
# GEO supplementary file GSE156625_HCCF_integrated.RData is a Seurat object written
# with saveRDS() (load() fails with "bad restore file magic number"; use readRDS()).
# Its meta.data holds the authors' CellType (and the myeloid split CT_mye) for the
# 73,589 HCC cells used in the benchmark (rownames "HCC_<barcode>") plus fetal cells.
#
# Broad-6 mapping (marker detection checked: "Immune" cells are CD3E+/NKG7+ and
# CD79A-/LYZ-):
#   Immune -> T/NK; Bcells, Fetal B -> B; Myeloid -> Myeloid; Endothelial -> Endothelial;
#   Fibroblast -> Fibroblast; Hepatocytes -> Hepatocyte;
#   Erythroids, Others -> Unknown (not evaluated; the HCC "Erythroids" express ALB, not HBB).
#
# Output: benchmark_datasets/ground_truth/GSE156625_published_ground_truth.tsv
#         (barcode, author_CellType, author_CT_mye, celltype), also copied to release/ground_truth/.

suppressMessages({
  library(Matrix)
  lib4 <- "/home/holiday01/R/seurat4_lib"
  .libPaths(c(lib4, .libPaths()))
  library(SeuratObject, lib.loc = lib4)
})
SRC <- "/mnt/10t/assi_result/GEO_DATA/GSE156625/suppl/GSE156625_HCCF_integrated.RData"
OUT <- "/mnt/10t/assi_result/HCC/benchmark_datasets/ground_truth/GSE156625_published_ground_truth.tsv"
REL <- "/mnt/10t/holiday/hcc_paper/release/ground_truth/GSE156625_published_ground_truth.tsv"
BENCH <- "/mnt/10t/assi_result/HCC/benchmark_datasets/rev2_inputs/GSE156625/barcodes.tsv"

x <- readRDS(SRC)
md <- x@meta.data[x@meta.data$tech == "HCC", ]
bc <- sub("^HCC_", "", rownames(md))
map <- c(Immune = "T/NK", Bcells = "B", "Fetal B" = "B", Myeloid = "Myeloid",
         Endothelial = "Endothelial", Fibroblast = "Fibroblast", Hepatocytes = "Hepatocyte")
ct <- unname(map[as.character(md$CellType)])
ct[is.na(ct)] <- "Unknown"
out <- data.frame(barcode = bc, author_CellType = as.character(md$CellType),
                  author_CT_mye = as.character(md$CT_mye), celltype = ct, stringsAsFactors = FALSE)
bench <- readLines(BENCH)
stopifnot(!any(duplicated(out$barcode)), all(bench %in% out$barcode))
out <- out[match(bench, out$barcode), ]
for (p in c(OUT, REL)) {
  if (file.exists(p)) stop("refusing to overwrite ", p)
  write.table(out, p, sep = "\t", quote = FALSE, row.names = FALSE)
  cat("wrote", p, "\n")
}
print(table(out$author_CellType, out$celltype))
