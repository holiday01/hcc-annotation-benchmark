# Changelog

## v1.1 — Revision (2026-05-28)

This release accompanies the revised manuscript prepared for Cancer Reports
(Wiley OA) after rejections from Cancer Science and GigaScience. The
revision adds statistical rigour, a CNV-based malignant-cell evaluation
axis, and a series of release-tree fixes.

### Added — new analysis scripts

| Script | Purpose |
|---|---|
| `scripts/_broad_label.py` | Single canonical fine-to-broad keyword mapping; imported by scripts 10 and 16 (resolves prior drift bug where cholangiocyte/biliary/hpc mapped to different broad classes between cohorts) |
| `scripts/25_statistical_robustness.py` | Cell-level bootstrap 95% CIs (B=1000), within-dataset label-permutation null (B=1000), paired-intersection means across the five datasets evaluable by all seven tools, Friedman + Nemenyi post-hoc tests. seed=42, vectorised confusion-matrix arithmetic |
| `scripts/26_cnv_sensitivity_TBonly.py` | inferCNV sensitivity panel: re-runs malignant-cell validation on GSE125449 with T+B-only normal reference (TAM excluded) |
| `scripts/27_run_CellAssign_GSE156625.py` | CellAssign re-run on the external atlas cohort GSE156625 in a Python 3.11 + scvi-tools 1.4.2 env (resolves the prior Python 3.13 incompatibility that excluded CellAssign from GSE156625) |
| `scripts/28_scgpt_knn_k_sensitivity.py` | scGPT KNN classifier k sweep skeleton (k ∈ {5, 10, 15, 20, 30}) on cached embeddings. *Skeleton; requires working scgpt + CUDA env to execute* |
| `scripts/29_cnv_malignant_GSE149614.py` | Extends the inferCNV malignant-cell evaluation from GSE125449 to GSE149614 (concordance, not validation: CNV-Malignant is itself a derived label) |
| `scripts/30_scgpt_finetune.py` | scGPT supervised fine-tuning skeleton (Cell-Type Annotation task, CLS-token classifier). *Skeleton; same env blocker as 28* |

### Added — new release artefacts

| Path | Purpose |
|---|---|
| `cytetype_responses/GSE{156625,162616,202642,223204}_cytetype_response.json` | Frozen per-cluster CyteType API responses for the four validation cohorts (auth_token stripped); `manifest.json` indexes the four files |
| `ground_truth/cellassign_markers.tsv` | CellAssign marker matrix used in the paper (Broad6 + GSE125449 panels), in long-form |
| `expected_results/bootstrap_ci.csv` | 54 (tool × dataset × GT) rows; per-row Macro F1 / Accuracy / Cohen's Kappa with 95% bootstrap percentile CIs |
| `expected_results/permutation_null.csv` | Per-cohort permutation null Macro F1 distribution (observed, null mean/sd, null 95p/99p, empirical p-value) |
| `expected_results/paired_intersection_means.csv` | Per-tool paired-intersection mean Macro F1 across the 5 datasets evaluable by every tool |
| `expected_results/friedman_test.csv` | Friedman test result (χ² = 20.62, p = 0.0021, k = 7 tools, n = 5 datasets) |
| `expected_results/friedman_nemenyi.csv` | Nemenyi post-hoc pairwise p-value matrix (CyteType vs CellAssign and ScType vs CellAssign at p = 0.004) |
| `expected_results/cnv_malignant_validation_TBonly.csv` | Per-tool CNV F1 on GSE125449 under the T+B-only normal-panel sensitivity |
| `expected_results/cnv_TBonly_vs_TBTAM_diff.csv` | Per-tool baseline (T+B+TAM) vs sensitivity (T+B-only) ΔF1 |
| `expected_results/cnv_malignant_validation_GSE149614.csv` | Per-tool CNV-Malignant concordance F1 on GSE149614 (extension to second benchmark cohort) |
| `expected_results/FigS_cnv_TBonly_sensitivity.pdf` | Supplementary Figure S9 in the revised manuscript |
| `expected_results/FigS_cnv_malignant_GSE149614.pdf` | Supplementary Figure S10 in the revised manuscript |

### Changed — modified scripts

| Script | Change |
|---|---|
| `scripts/02_benchmark_evaluation.py` | Enabled CellAssign on GSE156625 (uses the prediction now produced by script 27); added explicit seed declaration |
| `scripts/04_run_CyteType_on_benchmark.py` | Added explicit seed=42; saves the raw per-cluster CyteType API response JSON (with auth_token stripped) to `cytetype_responses/`; uses `random_state=SEED` consistently for scanpy clustering |
| `scripts/05_run_scGPT_annotation.py` | Added explicit deterministic seeding (`numpy`, `torch`, `PYTHONHASHSEED`, `random`); `cudnn.deterministic = True` |
| `scripts/10_build_enrichr_groundtruth.py` | Now imports `REJECT_KEYWORDS` / `BROAD_MAP` / `broad_label` from the shared `_broad_label.py` module (eliminates prior drift with script 16) |
| `scripts/16_prepare_gse156625_external.py` | Same: imports from `_broad_label.py` |
| `scripts/07_generate_figures.py`, `09_per_class_analysis.py`, `15_generate_robustness_figures.py` | Minor updates for consistency |
| `run_all.sh` | Added `stage_robustness` (25), `stage_scgpt_sweep` (28), `stage_scgpt_ft` (30); extended `stage_cnv` to call 22+26+29; extended `stage_external` to include 27 |
| `README.md` | Updated stage table; added new file listings |
| `expected_results/README.md` | Added 11 new file rows |
| `CITATION.cff` | (Pending) ORCID 0000-0002-2087-2266 to be added at acceptance-time citation update |

### Fixed

* **`_broad_label.py`**: `"hepatic progenitor"` now maps to `"HPC-like"` (was previously routed to Hepatocyte, contradicting the module's own docstring and Cell Ontology, in which hepatic stem cell `CL_0019029` is distinct from mature hepatocyte `CL_0000182`). Existing enrichR snapshots contain zero rows matching this term, so no downstream cascade was needed.
* **Cross-script BROAD_MAP drift**: scripts 10 and 16 previously maintained independent copies of the broad-mapping dict, in which cholangiocyte / biliary / hpc were routed to different broad classes between cohorts. Resolved by the shared `_broad_label.py` module.

### Notes

* Scripts 28 (KNN k sweep) and 30 (scGPT fine-tune) are released as skeletons. Their execution requires a Python environment with `scgpt >= 0.2.4` plus a working `torch` CUDA build matching the host driver. As of the revision date, the only env where `scgpt` installs cleanly carries `torch 2.11.0+cu130` which fails to initialise CUDA on the host driver `12.8`; the env with working CUDA (`torch 2.10.0+cu128`) has a `torchtext` ABI conflict with `scgpt`'s pinned deps. These analyses are deferred to the next planned scGPT evaluation step and are flagged accordingly in the manuscript Discussion (Limitation v).
* All other scripts have been re-run end-to-end and the prediction tables / expected_results CSVs were regenerated.
