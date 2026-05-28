#!/usr/bin/env bash
#
# Reproduce the full HCC scRNA-seq cell-type annotation benchmark.
#
# Order matches the README "Pipeline" section. Set environment variables to
# point at your Python / R / data:
#
#     PYTHON       — Python interpreter         (default: python)
#     RSCRIPT      — Rscript path               (default: Rscript)
#     BENCHMARK_DATA_DIR — input data directory (passed through to scripts)
#
# Examples:
#
#     bash run_all.sh                           # run everything
#     bash run_all.sh preprocess                # only stage 1 (preprocessing)
#     PYTHON=/path/to/env/bin/python bash run_all.sh
#
# Stages:
#     preprocess   — 01 (count matrices → h5ad)
#     groundtruth  — 10 (attach EnrichR snapshot to h5ad)
#     annotate     — 03–06, 17 (run 7 tools on benchmark datasets)
#     external     — 16, 18–20, 27 (external GSE156625 cohort; 27 adds CellAssign)
#     cnv          — 22, 26, 29 (CNV malignant validation + T+B-only sensitivity + GSE149614 extension)
#     simulate     — 21, 23, 24 (Splatter negative control)
#     evaluate     — 02, 09, 12 (metrics)
#     robustness   — 25 (cell-level bootstrap CI, label-permutation null, Friedman/Nemenyi)
#     figures      — 07, 15 (figures)
#     scgpt_sweep  — 28 (KNN k sensitivity on cached scGPT embeddings; needs scgpt env)
#     scgpt_ft     — 30 (scGPT fine-tune skeleton; needs scgpt + GPU env)

set -euo pipefail

PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
SCRIPTS="$(cd "$(dirname "$0")"/scripts && pwd)"

log() { printf "\n\033[1;34m[run_all]\033[0m %s\n" "$*"; }
run_py() { log "python ${1}"; "${PYTHON}" "${SCRIPTS}/${1}"; }
run_r()  { log "Rscript ${1}"; "${RSCRIPT}" "${SCRIPTS}/${1}"; }

stage_preprocess() {
    run_py 01_preprocess_benchmark_datasets.py
}

stage_groundtruth() {
    # Default: re-attach shipped EnrichR snapshots (deterministic).
    # Pass --refresh to re-query EnrichR (non-deterministic).
    run_py 10_build_enrichr_groundtruth.py
}

stage_annotate() {
    run_py 06_run_CellTypist_on_benchmark.py
    run_py 17_run_scType_on_benchmark.py
    run_py 05_run_CellAssign_on_benchmark.py
    run_py 05_run_scGPT_annotation.py
    run_py 04_run_CyteType_on_benchmark.py
    run_r  03_run_signacX_on_benchmark.R
    run_r  14_run_SingleR_on_benchmark.R
}

stage_external() {
    run_py 16_prepare_gse156625_external.py
    run_py 18_run_CellAssign_GSE156625.py
    run_r  19_run_SignacX_GSE156625.R
    run_py 20_run_CyteType_GSE156625.py
    # CellAssign GSE156625 re-run in a Python 3.11 + scvi-tools 1.4.2 env
    # (resolves the original Python 3.13 incompatibility); produces
    # annotation_results/GSE156625_CellAssign_pred.csv.
    run_py 27_run_CellAssign_GSE156625.py
}

stage_cnv() {
    run_py 22_cnv_malignant_validation.py
    # T+B-only normal-panel sensitivity check for GSE125449 inferCNV
    # (Tirosh/Patel convention; removes TAM ambient-RNA confounding).
    run_py 26_cnv_sensitivity_TBonly.py
    # CNV-derived Malignant labels extended to GSE149614 (the larger
    # benchmark cohort); produces FigS_cnv_malignant_GSE149614.pdf and
    # expected_results/cnv_malignant_validation_GSE149614.csv.
    run_py 29_cnv_malignant_GSE149614.py
}

stage_robustness() {
    # Cell-level non-parametric bootstrap 95% CIs (B=1000), within-dataset
    # label-permutation null (B=1000), paired-intersection means, and
    # Friedman + Nemenyi post-hoc tests on Macro F1. Outputs:
    #   expected_results/bootstrap_ci.csv
    #   expected_results/permutation_null.csv
    #   expected_results/paired_intersection_means.csv
    #   expected_results/friedman_test.csv
    #   expected_results/friedman_nemenyi.csv
    run_py 25_statistical_robustness.py
}

stage_scgpt_sweep() {
    # KNN k sensitivity sweep on cached scGPT embeddings.
    # k ∈ {5, 10, 15, 20, 30}; requires the scgpt Python env.
    run_py 28_scgpt_knn_k_sensitivity.py
}

stage_scgpt_ft() {
    # scGPT supervised fine-tuning skeleton (Cell-Type Annotation task).
    # Requires scgpt + working CUDA torch build. Currently a placeholder.
    run_py 30_scgpt_finetune.py
}

stage_simulate() {
    run_r  21_simulate_scrnaseq_splatter.R
    run_py 23_run_tools_on_simulated.py
    run_r  24_run_singler_signacx_on_simulated.R
}

stage_evaluate() {
    run_py 02_benchmark_evaluation.py
    run_py 09_per_class_analysis.py
    run_py 12_coverage_robustness_analysis.py
    # scATOMIC is evaluated separately (outputs are precomputed externally).
    run_py 13_evaluate_scatomic.py
}

stage_figures() {
    run_py 07_generate_figures.py
    run_py 15_generate_robustness_figures.py
}

run_stage() {
    case "$1" in
        preprocess)  stage_preprocess  ;;
        groundtruth) stage_groundtruth ;;
        annotate)    stage_annotate    ;;
        external)    stage_external    ;;
        cnv)         stage_cnv         ;;
        simulate)    stage_simulate    ;;
        evaluate)    stage_evaluate    ;;
        robustness)  stage_robustness  ;;
        figures)     stage_figures     ;;
        scgpt_sweep) stage_scgpt_sweep ;;
        scgpt_ft)    stage_scgpt_ft    ;;
        all)
            stage_preprocess
            stage_groundtruth
            stage_annotate
            stage_external
            stage_cnv
            stage_simulate
            stage_evaluate
            stage_robustness
            stage_figures
            ;;
        *)
            echo "Unknown stage: $1"
            echo "Valid stages: preprocess groundtruth annotate external cnv simulate evaluate robustness figures scgpt_sweep scgpt_ft all"
            echo "  (scgpt_sweep and scgpt_ft are excluded from 'all' because they require a working scgpt + CUDA env.)"
            exit 1
            ;;
    esac
}

if [[ $# -eq 0 ]]; then
    run_stage all
else
    for stage in "$@"; do
        run_stage "${stage}"
    done
fi

log "All requested stages completed."
