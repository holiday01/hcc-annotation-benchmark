# Install R dependencies for the HCC scRNA-seq cell-type annotation benchmark.
#
# Tested with R 4.5.2. Versions below pin the releases used to generate the
# results shipped under expected_results/.
#
# Run from this directory:
#     Rscript install_r_deps.R

# Use a stable CRAN snapshot for reproducibility.
options(repos = c(CRAN = "https://cran.r-project.org"))

# ── Package manager
if (!requireNamespace("remotes", quietly = TRUE))    install.packages("remotes")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(version = "3.22", ask = FALSE, update = FALSE)

# ── CRAN packages (pinned)
cran_pkgs <- list(
    Seurat       = "5.5.0",
    SeuratObject = "5.2.0",
    SeuratDisk   = NA,    # GitHub-only; installed below
    HGNChelper   = "0.8.1",
    openxlsx     = "4.2.5",
    dplyr        = "1.1.4",
    Matrix       = "1.7-3"
)
for (pkg in names(cran_pkgs)) {
    if (is.na(cran_pkgs[[pkg]])) next  # skip GitHub-only
    if (!requireNamespace(pkg, quietly = TRUE) ||
        utils::packageVersion(pkg) != cran_pkgs[[pkg]]) {
        remotes::install_version(pkg, version = cran_pkgs[[pkg]],
                                 upgrade = "never")
    }
}

# ── Bioconductor packages (Bioc 3.22)
bioc_pkgs <- c("SingleR", "celldex", "splatter", "SummarizedExperiment",
               "SingleCellExperiment")
BiocManager::install(bioc_pkgs, ask = FALSE, update = FALSE)

# ── GitHub packages
github_pkgs <- list(
    SignacX     = c("mathewchamberlain/SignacX", "v2.2.5"),
    SeuratDisk  = c("mojaveazure/seurat-disk",    NA)
)
for (pkg in names(github_pkgs)) {
    repo <- github_pkgs[[pkg]][1]
    ref  <- github_pkgs[[pkg]][2]
    if (!requireNamespace(pkg, quietly = TRUE)) {
        if (is.na(ref)) {
            remotes::install_github(repo, upgrade = "never")
        } else {
            remotes::install_github(repo, ref = ref, upgrade = "never")
        }
    }
}

cat("\nInstalled versions:\n")
for (pkg in c("Seurat", "SignacX", "SingleR", "celldex", "splatter",
              "SeuratDisk")) {
    if (requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("  %-12s %s\n", pkg,
                    as.character(packageVersion(pkg))))
    }
}
