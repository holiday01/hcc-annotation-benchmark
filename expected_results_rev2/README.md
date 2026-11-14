# expected_results_rev2 — revision-2 scoring

**Which reference answers which question (revision 2).**
Accuracy claims rest only on investigator-curated labels and experimental design:
* published labels, broad-6, on GSE125449, GSE149614 and GSE156625 (`GT == "published"`;
  GSE156625 labels are the authors' `CellType` from `GSE156625_HCCF_integrated.RData`,
  extracted by `scripts/38_extract_GSE156625_author_labels.R`); Friedman/Nemenyi
  analysis `D_published_3cohorts_7tools`;
* GSE162616, flow-sorted innate lymphoid cells: label-free lineage-specificity check
  (`rev2_GSE162616_sorted_ILC_check.csv`, `scripts/41_sorted_ILC_check_GSE162616.py`).
The enrichR-derived labels (`GT == "enrichR"`, analyses A–C) are kept as a secondary
database-concordance analysis only: they are produced by an automated pipeline,
share marker resources (and, on four cohorts, the cluster partition) with ScType and
CyteType, and omit populations the databases do not match (e.g. 1,884 of 1,967
author-labelled fibroblasts in GSE156625).

Produced by `scripts/31_rescore_common_cells.py` (seed 42, B = 1000) from the
same per-cell predictions as `expected_results/`. No annotation tool was
re-run; only the label mapping and the scoring changed. The scoring rules
(R1–R7) are in the script docstring; the mapping rules are in the docstring
of `scripts/_canonical_broad.py`.

Why the numbers differ from `expected_results/benchmark_summary.csv`:

1. Macro F1 used to average over y_true ∪ y_pred. A tool that labelled even a
   few cells with a class absent from the ground truth got an extra
   F1 = 0 class (e.g. scGPT on GSE162616: 0.481 → 0.962 on the same cells).
   Now the class set is the GT classes present in the evaluated cells, the
   same for every tool.
2. Cells a tool was never run on used to disappear silently (inner join) and
   abstentions were removed per tool, so tools were ranked on different
   cells. Now every tool in a tool set is scored on the same cells and
   "Unknown" counts as an error; coverage is reported next to each score.
3. SignacX on GSE156625 ran on a 15,000-cell subsample; the other cells were
   written as "Unclassified" and are now treated as not run, not as
   abstentions.
4. One fine-to-broad mapping (`scripts/_canonical_broad.py`) for every tool's
   raw output and for the enrichR terms. The four previous mappings disagreed
   and had gaps: CyteType labels such as "Capillary EC", "Memory CD8 T cell"
   or "SPP1 macrophage" were missing from the exact-match table and became
   "Unknown" (most of the reported 65–73% CyteType abstention on
   GSE202642/GSE223204); "TAM" was missing from the fuzzy mapper; "mast cell"
   matched "t cell" (ScType and the enrichR GT mapped mast cells to T/NK);
   ScType "Plasmacytoid dendritic cells" went to B; SignacX was scored at its
   coarse level only, with every "NonImmune" cell counted as Hepatocyte, so
   its Endothelial/Fibroblast/Epithelial calls were never used. Under the
   canonical mapping, quality-control flags (doublet, artifact, low quality,
   ambient RNA) and labels that do not determine one broad class
   ("NonImmune", "Lymphocytes") count as Unknown.

5. SignacX, CellAssign and scGPT were re-run (`annotation_results_rev2/`):
   * Inputs: raw UMI counts for every benchmark cell
     (`scripts/33_prepare_counts_rev2.py`; recovered exactly from the stored
     log-normalised values, GSE156625 from the GEO count matrix).
   * SignacX (`scripts/34_rerun_SignacX_rev2.R`): every cohort, every cell,
     per sample (samples < 200 cells pooled), identical settings. Before:
     log-normalised values read as counts on GSE125449/GSE149614, imputation
     on for one and off for the other, a 15,000-cell subsample on GSE156625,
     part of the samples on GSE162616/GSE202642, nothing on GSE223204.
   * CellAssign (`scripts/36_rerun_CellAssign_rev2.py`): integer counts and a
     library-size factor as in the scvi-tools tutorial, broad-6 marker panel
     on all six cohorts (plus the 7-class panel on GSE125449). Before:
     `normalize_total(inplace=True)` turned the marker counts into
     non-integer values and every cell got size factor 1; the three
     DISCO-derived cohorts used a different marker panel on part of the cells.
   * scGPT (`scripts/35_scgpt_leave_cohort_out.py`): k-NN reference rebuilt
     from DISCO after removing every sample of the six benchmark series
     (`annotation_results_rev2/scgpt_reference_exclusion.csv`); embeddings
     and a k = 5–30 sensitivity analysis are saved.

6. Tools added for Reviewer 1, comment 1 (`scripts/42_build_shared_reference.py`,
   `43_run_reference_tools_rev2.R`, `44_run_scmap_scn_scpred_rev2.R`): Seurat label
   transfer (Azimuth's algorithm), scmap-cluster, SingleCellNet, scPred 1.9.2 (its
   default scPredict aligns the query to the reference with harmony; harmony 1.2.3,
   because harmony 2.x removed the HarmonyMatrix function scPred imports) and SingleR
   re-trained on the shared reference ("SingleRdisco").
   All reference-based tools, including the scGPT k-NN, use the same 16,141-cell
   leave-benchmark-cohorts-out DISCO liver reference. Tool set `all` contains every
   tool with output; `all7` keeps the seven tools of the original submission;
   Friedman analysis `E_published_3cohorts_<k>tools` is the primary multi-tool test.

Folders: this folder = canonical mapping + re-run predictions (main results);
`original_predictions/` = canonical mapping + published predictions;
`original_mapping/` = published per-tool mappings + published predictions.
`rev2_decomposition.csv` lines up the reported score and the three steps.

| File | Content |
|---|---|
| `rev2_scores.csv` | One row per (cohort, GT, vocabulary, tool set, tool): N, class set, coverage, coverage-aware accuracy / macro F1 / kappa, per-class F1, selective accuracy / macro F1 on cells every tool labelled, cell- and sample-level bootstrap 95% CIs, permutation null and p |
| `rev2_cellsets.csv` | Cells entering each (cohort, GT, tool set) comparison and which tools are in the set |
| `rev2_pairwise_bootstrap.csv` | Paired bootstrap CIs of macro F1 differences between every tool pair |
| `rev2_friedman.csv` | Friedman tests across enrichR cohorts (A: 6 cohorts × 6 tools without SignacX; B: 5 cohorts × 7 tools; C: 6 cohorts × 4 non-circular tools) |
| `rev2_nemenyi_*.csv`, `rev2_friedman_input_*.csv` | Nemenyi p-value matrices and the score matrices each Friedman test used |
| `rev2_label_audit.csv` | Per tool and cohort: not-run cells, abstentions, labels outside the vocabulary |
| `rev2_vs_original.csv` | New all7 scores next to the originally reported ones |
| `rev2_label_mapping.csv` | Every raw tool label per cohort, the broad class it maps to, the rule that fired, and the number of cells (canonical run only) |
| `rev2_GSE162616_sorted_ILC_check.csv` | Flow-sorted ILC cohort: % of cells each tool calls T/NK, non-immune (false positives), Myeloid, B, Unknown |
| `rev2_scgpt_k_sensitivity.csv` | scGPT macro F1 for k = 5, 10, 15, 20, 30 |

Tool sets: `all7` = every tool with output on the cohort; `core` = tools that
ran on ≥ 99% of GT-labelled cells; `noSignacX`; `noncircular` = CellTypist,
SingleR, CellAssign, scGPT (enrichR GT only). ScType and CyteType are flagged
`Circular_under_enrichR` because they share marker resources with the enrichR
GT and, on the four validation cohorts, the same cluster partition.

Checks: every re-run covers 100% of the benchmark cells (`rev2_cellsets.csv`:
Frac_common = 1 for all cohorts). Three re-run scores were recomputed from the
prediction files with sklearn and match. The scGPT environment was validated
by re-running the original (unfiltered) reference on GSE125449: 96.5%
agreement with the published predictions, the same as the agreement between
two runs of the new environment with different seeds (96.9%), i.e. the
difference is scGPT's own run-to-run variability (random gene sampling and
tie-breaking in binning).

Notes for the manuscript: DISCO also held 57 samples (80,787 cells) of
GSE156625 and 3 of GSE162616, so four of the six cohorts leaked into the
original scGPT reference; after exclusion the reference has 1,141 fibroblasts
(other classes 3,000). scGPT macro F1 varies by ≤ 0.036 over k = 5–30
(`rev2_scgpt_k_sensitivity.csv`).

Still open: the enrichR ground truth shares its cluster partition with ScType
and CyteType on the four validation cohorts (their enrichR scores are not
independent); no independently curated labels yet for those cohorts; the
additional tools requested by Reviewer 1 have not been run.
