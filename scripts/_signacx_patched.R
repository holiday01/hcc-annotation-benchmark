# Patched SignacX 2.2.5 helpers (library setup, CID.Louvain_fixed, SignacFast_fixed),
# copied verbatim from benchmark_datasets/scripts/run_signacX_fixed.R lines 10-172
# (the script that produced the original GSE125449/GSE149614 SignacX runs).
# Sourced by 34_rerun_SignacX_rev2.R.

Sys.setenv(PATH = paste("/home/holiday01/.local/bin", Sys.getenv("PATH"), sep=":"))

library(Matrix)   # load system Matrix 1.7.x first
lib4 <- "/home/holiday01/R/seurat4_lib"
.libPaths(c(lib4, .libPaths()))
library(SeuratObject, lib.loc=lib4)
library(Seurat, lib.loc=lib4)
library(SignacX)

cat("Matrix:", as.character(packageVersion("Matrix")), "\n")
cat("Seurat:", as.character(packageVersion("Seurat")), "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Patched CID.Louvain — avoids integer overflow from `adjmatrix * 1` on
# large matrices (>~50k cells).  Root cause: linear-index col*nrow overflows
# 32-bit int when nrow >= ~46341 (sqrt(2^31)).  Fix: read @i/@p slots directly
# and build the igraph from an edge list, bypassing all matrix indexing.
# ─────────────────────────────────────────────────────────────────────────────
CID.Louvain_fixed <- function(edges, graph.used="nn") {
  if (nrow(edges) != ncol(edges)) {
    # Edge-list input (data.frame with V1, V2)
    el <- cbind(as.integer(edges$V1), as.integer(edges$V2))
    n  <- max(el)
    g  <- igraph::graph_from_edgelist(el, directed=FALSE)
    igraph::V(g)$name <- as.character(seq_len(n))
  } else {
    # Square adjacency matrix — extract @i/@p slots directly to avoid overflow
    # (Graph inherits from dgCMatrix so slots are already accessible)
    irow <- edges@i + 1L                         # 1-based row indices
    icol <- rep(seq_len(ncol(edges)), diff(edges@p)) # 1-based col indices
    keep <- irow < icol                         # upper triangle
    el   <- cbind(irow[keep], icol[keep])
    n    <- nrow(edges)
    g    <- igraph::make_empty_graph(n=n, directed=FALSE)
    if (nrow(el) > 0L)
      g  <- igraph::add_edges(g, as.vector(t(el)))
  }
  as.character(igraph::cluster_louvain(g)$membership)
}
assignInNamespace("CID.Louvain", CID.Louvain_fixed, ns="SignacX")

# ─────────────────────────────────────────────────────────────────────────────
# Fixed SignacFast — identical to SignacX 2.2.5 except for the dummy fix
# ─────────────────────────────────────────────────────────────────────────────
SignacFast_fixed <- function(
    E,
    Models="default",
    spring.dir=NULL,
    num.cores=1,
    threshold=0,
    smooth=TRUE,
    impute=TRUE,
    verbose=TRUE,
    do.normalize=TRUE,
    return.probability=FALSE,
    graph.used="nn"
) {
  if (identical(Models, "default")) {
    cached <- "/mnt/10t/assi_result/HCC/benchmark_datasets/hpca_models.rds"
    Models <- if (file.exists(cached)) readRDS(cached) else GetModels_HPCA()
  }
  flag <- class(E) == "Seurat"

  if (flag) {
    default.assay <- Seurat::DefaultAssay(E)
    edges <- E@graphs[[which(grepl(paste0(default.assay, "_", graph.used), names(E@graphs)))]]
  }

  if (verbose) {
    cat(" ..........  Entry in SignacFast \n")
    ta <- proc.time()[3]
    if (!flag) cat(" ..........  Running SignacFast on input data matrix :\n")
    else cat(" ..........  Running SignacFast on Seurat object :\n")
    cat("             nrow = ", nrow(E), "\n", sep="")
    cat("             ncol = ", ncol(E), "\n", sep="")
  }

  logik <- CID.IsUnique(rownames(E))
  if (sum(logik) != length(logik)) E <- E[logik, ]

  gns <- sort(unique(unlist(sapply(Models, function(x) x$genes))))
  V <- E[rownames(E) %in% gns, ]
  if (class(V) %in% "data.frame") V <- Matrix::Matrix(as.matrix(V), sparse=TRUE)

  if (do.normalize) {
    if (!flag) V <- CID.Normalize(V)
    else V <- CID.Normalize(V@assays[[default.assay]]@counts)
  }

  normalize <- function(x) (x - min(x)) / (max(x) - min(x))
  V <- t(apply(V, 1, function(x) normalize(x)))
  logik <- apply(V, 1, function(x) any(is.na(x)))
  V <- V[!logik, ]

  if (flag) {
    louvain <- CID.Louvain_fixed(edges=edges)
    if (nrow(edges) != ncol(edges)) edges <- CID.GetDistMat(edges, n=1)
    else edges <- list(edges)
  } else {
    edges <- CID.LoadEdges(data.dir=spring.dir)
    louvain <- CID.Louvain_fixed(edges=edges)
    edges <- CID.GetDistMat(edges, n=1)
  }

  res <- pbmcapply::pbmclapply(Models, FUN=function(x) {
    Z <- V[rownames(V) %in% x$genes, ]
    if (impute) {
      Z <- KSoftImpute(E=Z, dM=edges, verbose=FALSE)
      Z <- t(apply(Z, 1, function(x) normalize(x)))
    }
    # FIXED: use length(missing_genes) instead of length(x$genes) - nrow(Z)
    missing_genes <- x$genes[!x$genes %in% rownames(Z)]
    dummy <- Matrix::Matrix(0, nrow=length(missing_genes), ncol=ncol(Z))
    rownames(dummy) <- missing_genes
    Z <- rbind(Z, dummy)
    Z <- Z[order(rownames(Z)), ]
    res <- lapply(x$classifiers, function(y) {
      Predict <- stats::predict(y, Matrix::t(Z))
      colnames(Predict) <- sort(y$model.list$response)
      return(Predict)
    })
    res <- res[sapply(res, function(x) !is.null(x))]
    res.squared.mean <- Reduce("+", lapply(res, "^", 2)) / length(res)
    res <- Reduce(res, f="+") / length(res)
    res.variance <- res.squared.mean - res^2
    res.sd <- sqrt(res.variance)
    # FIXED: which.max returns integer(0) for all-NA rows → use safe version
    xx <- vapply(seq_len(nrow(res)), function(i) {
      m <- which.max(res[i, ])
      if (length(m) == 0L) NA_integer_ else as.integer(m)
    }, integer(1))
    celltypes <- colnames(res)[pmax(xx, 1L)]
    celltypes[is.na(xx)] <- "Unclassified"
    kmax <- apply(res, 1, function(row) { m <- suppressWarnings(max(row, na.rm=TRUE)); if (is.infinite(m)) 0 else m })
    celltypes[kmax < threshold] <- "Unclassified"
    errors <- round(vapply(seq_along(xx), function(i) {
      if (is.na(xx[i])) NA_real_ else res.sd[i, xx[i]]
    }, numeric(1)), digits=4)
    df <- data.frame(
      celltypes=celltypes,
      probability=round(kmax, digits=3),
      sd=errors,
      percent_features_detected=round(Matrix::colSums(Z != 0) / nrow(Z), digits=3) * 100
    )
    if (smooth & any(as.character(unique(x$celltypes)) %in%
        c("Immune", "Myeloid", "NonImmune", "Lymphocytes",
          "Monocytes.Neutrophils", "Monocytes", "Fibroblasts", "Epithelial")))
      df$celltypes <- CID.smooth(df$celltypes, edges[[1]])
    if (return.probability) return(df) else return(df$celltypes)
  }, mc.cores=num.cores)

  res$louvain <- louvain

  if (verbose) {
    tb <- proc.time()[3] - ta
    cat("\n ..........  Exit SignacFast.\n")
    cat("             Execution time = ", tb, " s.\n", sep="")
  }
  return(res)
}
