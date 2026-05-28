# Expected results

These CSV files are the outputs that the pipeline in `scripts/` produced when
generating the figures and tables of the manuscript. Use them to spot-check
a rerun: after `bash run_all.sh evaluate` and `bash run_all.sh figures`,
compare the regenerated CSVs against these.

| File | Produced by | Used in |
|---|---|---|
| `benchmark_summary.csv` | `02_benchmark_evaluation.py` | Figure 2 (radar), Figure 4 (abstention), main-text metrics |
| `cnv_malignant_validation.csv` | `22_cnv_malignant_validation.py` | Figure 6 (CNV validation, T+B+TAM normal panel) |
| `cnv_malignant_validation_TBonly.csv` | `26_cnv_sensitivity_TBonly.py` | Supplementary Table S10, Figure S10 (T+B-only sensitivity) |
| `cnv_TBonly_vs_TBTAM_diff.csv` | `26_cnv_sensitivity_TBonly.py` | Supplementary Table S10, Figure S10 (delta vs baseline) |
| `cnv_malignant_validation_GSE149614.csv` | `29_cnv_malignant_GSE149614.py` | Supplementary Table S11, Figure S11 (CNV-Malignant on GSE149614) |
| `FigS_cnv_TBonly_sensitivity.pdf` | `26_cnv_sensitivity_TBonly.py` | Supplementary Figure S10 |
| `FigS_cnv_malignant_GSE149614.pdf` | `29_cnv_malignant_GSE149614.py` | Supplementary Figure S11 |
| `bootstrap_ci.csv` | `25_statistical_robustness.py` | Discussion limitations (iii); Supplementary Table S7 CI columns |
| `permutation_null.csv` | `25_statistical_robustness.py` | Discussion limitations (iii); Supplementary Table S9 |
| `paired_intersection_means.csv` | `25_statistical_robustness.py` | Supplementary Table S9 (paired-intersection means) |
| `friedman_test.csv` | `25_statistical_robustness.py` | Supplementary Table S9 (χ²=20.62, p=0.0021) |
| `friedman_nemenyi.csv` | `25_statistical_robustness.py` | Supplementary Table S9 (pairwise p-values) |
| `coverage_adjusted_metrics.csv` | `12_coverage_robustness_analysis.py` | Supplementary Figure S8 |
| `enrichr_ground_truth_coverage.csv` | `02_benchmark_evaluation.py` | Table S of EnrichR GT coverage |
| `enrichr_unknown_published_label_distribution.csv` | `02_benchmark_evaluation.py` | Supplementary |
| `enrichr_vs_published_gt_concordance.csv` | `02_benchmark_evaluation.py` | Figure 3 (EnrichR heatmap) |
| `scATOMIC_label_mapping_summary.csv` | `13_evaluate_scatomic.py` | Supplementary |
| `scATOMIC_validation_metrics.csv` | `13_evaluate_scatomic.py` | Supplementary |
| `ScType_cluster_assignments.csv` | `17_run_scType_on_benchmark.py` | Supplementary |
| `ScType_marker_coverage.csv` | `17_run_scType_on_benchmark.py` | Supplementary |
| `TableS1_perclass_metrics.csv` | `09_per_class_analysis.py` | Table S1 |

## Tolerance

These results were produced on Linux 6.8.0 with the package versions pinned
in `../requirements.txt` and `../install_r_deps.R`. Small numerical
differences (≤ 1 % in Macro F1 / Accuracy / Cohen's κ) across re-runs are
expected because:

* `scanpy.tl.leiden(...)` and `scanpy.pp.pca(...)` are sensitive to BLAS /
  thread count.
* `gp.enrichr(...)` queries the live EnrichR API — re-run the snapshot mode
  (`python 10_build_enrichr_groundtruth.py`, no `--refresh`) to get bit-for-bit
  reproducible ground-truth labels.
* `scGPT` annotation involves stochastic batch sampling unless seeds are
  fixed identically (the script sets `torch.manual_seed(0)` and
  `numpy.random.seed(0)`).

Larger discrepancies almost always trace back to either:

1. A different version of `CellMarker_2024` or `PanglaoDB_Augmented_2021` on
   EnrichR. Use the shipped snapshot to control this.
2. A different version of an annotation tool (especially CellTypist's
   bundled model `Cells_Adult_Liver.pkl`).
