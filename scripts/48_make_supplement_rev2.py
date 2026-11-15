"""
48_make_supplement_rev2.py — supplementary tables (LaTeX) and data files of the revised manuscript.

Writes /mnt/10t/holiday/hcc_paper/rev2/supplementary_rev2.tex and
rev2/supp/SupplementaryData{1,2,3}.csv. Every value comes from expected_results_rev2/.
"""
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(os.path.dirname(HERE), "expected_results_rev2")
A2 = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2"
OUT = "/mnt/10t/holiday/hcc_paper/rev2"
os.makedirs(f"{OUT}/supp", exist_ok=True)
NAME = {"SeuratLT": "Seurat label transfer", "SingleRdisco": "SingleR (liver ref.)",
        "SingleR": "SingleR (HPCA)", "CellAssign": "CellAssign", "CellTypist": "CellTypist",
        "ScType": "ScType", "SignacX": "SignacX", "CyteType": "CyteType", "scGPT": "scGPT",
        "scmap": "scmap", "scPred": "scPred", "SingleCellNet": "SingleCellNet"}
COH = ["GSE125449", "GSE149614", "GSE156625"]
s = pd.read_csv(f"{E}/rev2_scores.csv")
pub = s[(s.GT == "published") & (s.Vocab == "broad6") & (s.ToolSet == "all")]
order = pub.groupby("Tool").MacroF1.mean().sort_values(ascending=False).index.tolist()


def esc(x):
    return str(x).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def sg(x):
    """3 decimals with a typographic minus."""
    return f"{x:.3f}".replace("-", "$-$")


def add(tex):
    key = tex.split("\\label{tab:", 1)[1].split("}", 1)[0]
    parts[key] = tex


def table(caption, label, colspec, header, rows, long=False):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    if long:
        return (f"{{\\footnotesize\n\\begin{{longtable}}{{{colspec}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n"
                f"\\toprule\n{header} \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n{header} \\\\\n\\midrule\n\\endhead\n"
                f"\\bottomrule\n\\endfoot\n{body}\n\\end{{longtable}}}}\n")
    return (f"\\begin{{table}}[H]\n\\centering\n\\footnotesize\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\resizebox{{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}}{{!}}{{%\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n"
            f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")


parts = {}

# S1 label harmonisation (summary here; full table as Supplementary Data 1)
m = pd.read_csv(f"{E}/rev2_label_mapping.csv")
shutil.copy(f"{E}/rev2_label_mapping.csv", f"{OUT}/supp/SupplementaryData1_label_harmonisation.csv")
summ = (m[m.Vocab == "broad6"].groupby(["Tool", "Mapped"]).N_cells.sum().unstack(fill_value=0))
cols = [c for c in ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast", "Unknown"] if c in summ.columns]
rows = [[NAME.get(t, t), str(int(m[(m.Tool == t) & (m.Vocab == 'broad6')].Raw_label.nunique()))]
        + [f"{int(summ.loc[t, c]):,}" for c in cols] for t in [x for x in order if x in summ.index]]
add(table("Label harmonisation summary: number of distinct raw labels per tool and cells mapped to each "
                   "evaluation class, summed over the six cohorts. Unknown = abstention (including doublet, artefact "
                   "and unresolved labels). The complete raw label $\\rightarrow$ class table with the rule applied "
                   "and cell counts per cohort is Supplementary Data 1.", "tab:harm",
                   "l" + "r" * (len(cols) + 1), "Tool & Raw labels & " + " & ".join(cols), rows))

# S2 shared reference
ex = pd.read_csv(f"{A2}/scgpt_reference_exclusion.csv")
ref = pd.read_csv("/mnt/10t/assi_result/HCC/benchmark_datasets/rev2_inputs/DISCO_reference/cells.csv")
rows = [[r.series, str(int(r.n_accessions)), f"{int(r.n_DISCO_samples_removed)}", f"{int(r.n_DISCO_cells_removed):,}"]
        for r in ex.itertuples()]
comp = ref.label.value_counts()
add(table("Shared liver reference. Top: DISCO samples and cells removed because they belong to a benchmark "
                   "series (GSM/SRX accessions from the GEO series matrices and the DISCO-derived objects). "
                   f"Bottom: composition of the {len(ref):,}-cell reference drawn from the remaining "
                   f"{ref['sample'].nunique()} DISCO samples (up to 3,000 cells per class, seed 42).", "tab:ref",
                   "lrrr", "Series & Accessions & DISCO samples removed & DISCO cells removed", rows +
                   [["\\midrule Reference class", "", "", "Cells"]] +
                   [[c, "", "", f"{int(comp[c]):,}"] for c in ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]]))

# S3 scGPT k sensitivity
k = pd.read_csv(f"{E}/rev2_scgpt_k_sensitivity.csv")
rows = [[r.Dataset, "Investigator" if r.GT == "published" else "Database-derived"] +
        [f3(getattr(r, f"MacroF1_k{x}")) for x in (5, 10, 15, 20, 30)] +
        [f3(max(round(getattr(r, f"MacroF1_k{x}"), 3) for x in (5, 10, 15, 20, 30)) -
            min(round(getattr(r, f"MacroF1_k{x}"), 3) for x in (5, 10, 15, 20, 30)))] for r in k.itertuples()]
add(table("scGPT $k$-nearest-neighbour classifier: macro F1 for $k$ = 5, 10, 15 (used), 20 and 30 "
                   "(shared reference; abstentions do not occur). Range: largest minus smallest of the displayed values.", "tab:k", "llrrrrrr",
                   "Cohort & Labels & $k=5$ & $k=10$ & $k=15$ & $k=20$ & $k=30$ & Range", rows))

# S4 coverage and selective metrics
rows = []
for t in order:
    row = [NAME[t]]
    for ds in COH:
        r = pub[(pub.Tool == t) & (pub.Dataset == ds)].iloc[0]
        row += [f"{r.Coverage:.3f}", f3(r.Accuracy), f3(r.Kappa), f3(r.MacroF1_selective)]
    rows.append(row)
hdr = "Tool & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{ds}}}" for ds in COH)
add(table("Coverage, coverage-aware accuracy and Cohen's kappa, and selective macro F1 (cells to which every "
                   "configuration assigned a class) against investigator-curated labels. Columns per cohort: Cov., Acc., "
                   "$\\kappa$, selective macro F1.", "tab:cov", "l" + "rrrr" * 3, hdr + r" \\ & " +
                   " & ".join(["Cov. & Acc. & $\\kappa$ & Sel. F1"] * 3), rows))

# S5 sorted ILC
so = pd.read_csv(f"{E}/rev2_GSE162616_sorted_ILC_check.csv")
rows = [[NAME.get(r.Source, "Database-derived labels"), f"{r._3:.1f}", f"{r.pct_Myeloid:.1f}", f"{r.pct_B:.1f}",
         f"{r.pct_non_immune:.2f}", f"{r.pct_Unknown:.1f}"] for r in so.itertuples()]
add(table("GSE162616 (flow-sorted innate lymphoid cells): percentage of all 57,261 cells assigned to each "
                   "class (Figure 4).", "tab:sorted", "lrrrrr", "Tool & T/NK & Myeloid & B & Non-immune & Abstention", rows))

# S6 per-class F1
cls = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
rows = []
for ds in COH:
    for t in order:
        r = pub[(pub.Tool == t) & (pub.Dataset == ds)].iloc[0]
        rows.append([ds, NAME[t]] + [f3(r[f"F1_{c}"]) for c in cls])
add(table("Per-class F1 against investigator-curated labels (six classes; abstentions count as errors).",
                   "tab:perclass", "ll" + "r" * 6, "Cohort & Tool & " + " & ".join(cls), rows, long=True))

# S7 seven-class GSE125449
b7 = s[(s.Dataset == "GSE125449") & (s.Vocab == "broad7") & (s.ToolSet == "all")]
c7 = ["T cell", "TAM", "B cell", "Malignant cell", "HPC-like", "TEC", "CAF"]
rows = [[NAME[t], f"{b7[b7.Tool == t].Coverage.iloc[0]:.3f}", f3(b7[b7.Tool == t].MacroF1.iloc[0])] +
        [f3(b7[b7.Tool == t][f"F1_{c}"].iloc[0]) for c in c7]
        for t in b7.sort_values("MacroF1", ascending=False).Tool]
add(table("GSE125449 scored with its seven published classes (CellAssign with its seven-class marker panel). "
                   "Tools trained on the shared reference cannot output HPC-like cells, which the reference does not "
                   "contain.", "tab:seven", "lrr" + "r" * 7, "Tool & Cov. & Macro F1 & " + " & ".join(c7), rows))

# S8 decomposition
d = pd.read_csv(f"{E}/rev2_decomposition.csv")
d = d[d.MacroF1_reported.notna() & d.MacroF1_reruns.notna()]
rows = [[r.Dataset, "Inv." if r.GT == "published" else "DB", "7" if r.Vocab == "broad7" else "6", NAME[r.Tool],
         f3(r.MacroF1_reported), f3(r.MacroF1_rules), f3(r.MacroF1_mapping), f3(r.MacroF1_reruns),
         f3(r.UnknownRate_reported), f3(1 - r.Coverage_reruns)] for r in d.itertuples()]
add(table("Effect of the revised evaluation on the seven tools of the original submission. Reported: original "
                   "submission. Rules: same predictions on common cells with a fixed class set and abstentions counted "
                   "as errors. Mapping: additionally one label harmonisation. Re-runs: additionally SignacX, CellAssign "
                   "and scGPT re-run (final). Abst.: abstention rate as originally reported (over evaluated cells) and "
                   "final. Labels: Inv., investigator; DB, database-derived; classes: 6 or 7.", "tab:decomp",
                   "lllcrrrrrr", "Cohort & Labels & Cl. & Tool & Reported & Rules & Mapping & Re-runs & Abst. orig. & Abst. final", rows, long=True))

# S9 Friedman / Nemenyi
fr = pd.read_csv(f"{E}/rev2_friedman.csv")
desc = {"A": "Database-derived, 6 cohorts, 6 original tools without SignacX",
        "B": "Database-derived, 5 cohorts, 7 original tools",
        "C": "Database-derived, 6 cohorts, configurations not sharing resources with these labels",
        "D": "Investigator labels, 3 cohorts, 7 original tools",
        "E": "Investigator labels, 3 cohorts, all configurations (primary)"}
rows = [[esc(r.Analysis), desc[r.Analysis[0]], str(r.n_datasets), str(r.k_tools), f"{r.chi2:.2f}", f"{r.p:.3g}"]
        for r in fr.itertuples()]
nem = [f for f in os.listdir(E) if f.startswith("rev2_nemenyi_E_")][0]
nm = pd.read_csv(f"{E}/{nem}", index_col=0)
add(table("Friedman tests on macro F1 across cohorts. The Nemenyi post-hoc $p$-values for the primary analysis "
                   f"(E) are in Supplementary Data 3b; the smallest is {np.nanmin(nm.where(~np.eye(len(nm), dtype=bool)).values):.3f}.",
                   "tab:friedman", "lp{6.5cm}rrrr", "Analysis & Description & Cohorts & Configurations & $\\chi^2$ & $p$", rows))

# S10 / S11 CNV
c = pd.read_csv(f"{E}/rev2_cnv_GSE125449.csv")
rows = []
for t in ["CNV call vs published malignant"] + order:
    a = c[(c.Tool == t) & (c.panel == "TB_TAM")].iloc[0]
    b = c[(c.Tool == t) & (c.panel == "TB_only")].iloc[0]
    rows.append([NAME.get(t, "CNV-high cells (vs published)"), "--" if pd.isna(a.get("N_tool_malignant_like")) else f"{int(a.N_tool_malignant_like):,}",
                 f3(a.CNV_Precision), f3(a.CNV_Recall), f3(a.CNV_F1), f3(b.CNV_F1), f3(a.get("PubGT_F1"))])
add(table("GSE125449 CNV check (exploratory). CNV-high: top quartile of the CNV score over all cells of the "
                   "cohort; this is a relative threshold, not a validated malignancy call. Hepatocyte-lineage calls of "
                   "each configuration are compared with the CNV-high cells (T, B and TAM normal reference, baseline; "
                   "T and B cells only, sensitivity) and with the published malignant and HPC-like labels (PubGT F1).",
                   "tab:cnv125", "lrrrrrr",
                   "Tool & Hepatocyte-lineage calls & Precision & Recall & CNV F1 & CNV F1 (T+B) & PubGT F1", rows))
c = pd.read_csv(f"{E}/rev2_cnv_GSE149614.csv").set_index("Tool")
rows = [["Published hepatocytes", f"{int(c.loc['published Hepatocyte'].N_tool_Hepatocyte):,}", "--", "--", "--",
         f"{c.loc['published Hepatocyte'].Frac_Hepatocyte_pred_that_is_CNV_Mal:.3f}"]]
rows += [[NAME[t], f"{int(c.loc[t].N_tool_Hepatocyte):,}", f3(c.loc[t].Precision_CNV_Mal), f3(c.loc[t].Recall_CNV_Mal),
          f3(c.loc[t].F1_CNV_Mal), f3(c.loc[t].Frac_Hepatocyte_pred_that_is_CNV_Mal)] for t in order]
add(table("GSE149614 CNV check (exploratory; T/NK and B normal reference). Hepatocyte calls of each "
                   "configuration against the CNV-high cells (top quartile of the CNV score over all 71,915 cells); "
                   "last column: fraction of hepatocyte calls that are CNV-high.", "tab:cnv149",
                   "lrrrrr", "Tool & Hepatocyte calls & Precision & Recall & F1 & Frac. CNV-high", rows))

# S12-S14
shutil.copy(f"{E}/rev2_scores.csv", f"{OUT}/supp/SupplementaryData2_all_scores.csv")
pw = pd.read_csv(f"{E}/rev2_pairwise_bootstrap.csv")
pw.to_csv(f"{OUT}/supp/SupplementaryData3a_pairwise_bootstrap.csv", index=False)
shutil.copy(f"{E}/{nem}", f"{OUT}/supp/SupplementaryData3b_nemenyi_primary.csv")
sim = pd.read_csv(f"{E}/rev2_simulated.csv")
rows = [[NAME.get(r.Tool, r.Tool), f"{r.Coverage:.3f}", f"{r.Accuracy:.3f}", f"{r.MacroF1:.3f}"] +
        [f3(r[f"F1_{c}"]) for c in cls] for _, r in sim.iterrows()]
add(table("Splatter negative control (5,000 cells, six groups; revised scoring rules).", "tab:sim",
                   "lrrr" + "r" * 6, "Tool & Cov. & Acc. & Macro F1 & " + " & ".join(cls), rows))

# Stromal-label mapping sensitivity
gp = pd.read_csv(f"{E}/rev2_sensitivity_reference_gap.csv").set_index(["Mapping", "Scope"])
sm = pd.read_csv(f"{E}/rev2_sensitivity_stromal_mapping.csv").set_index(["Dataset", "Mapping", "Tool"])
rows = []
for sc in COH + ["Mean"]:
    a, b = gp.loc[("canonical", sc)], gp.loc[("lenient", sc)]
    fib = (lambda mp: "" if sc == "Mean" else f" ({sm.loc[(sc, mp, 'SingleR'), 'F1_Fibroblast']:.3f})")
    rows.append([sc if sc != "Mean" else "Mean of three", f3(a.MacroF1_liver_ref),
                 f3(a.MacroF1_HPCA) + fib("canonical"), f"{sg(a.Difference)} ({sg(a.CI_lo)} to {sg(a.CI_hi)})",
                 f3(b.MacroF1_HPCA) + fib("lenient"), f"{sg(b.Difference)} ({sg(b.CI_lo)} to {sg(b.CI_hi)})"])
add(table("Sensitivity of the reference comparison to the mapping of the HPCA label Tissue\\_stem\\_cells "
          "(mesenchymal stem/stromal cells). Main analysis: mapped to Unknown (excluded lineage). Lenient: mapped to "
          "Fibroblast, like the HPCA label MSC. No other configuration outputs a stem- or stromal-type label, so only "
          "SingleR (HPCA) changes; cells, class set and scoring rules are unchanged. Fibroblast F1 of SingleR (HPCA) in "
          "parentheses. Difference: SingleR (liver ref.) minus SingleR (HPCA), with 95\\% patient-level paired bootstrap "
          "intervals (1,000 resamples).", "tab:stromal", "lrrrrr",
          "Cohort & SingleR (liver ref.) & SingleR (HPCA), main & Difference & SingleR (HPCA), lenient & Difference",
          rows))

# GSE149614 leave-one-patient-out
lo = pd.read_csv(f"{E}/rev2_lopo_GSE149614.csv")
w = lo.pivot(index="Left_out", columns="Tool", values="MacroF1")
full, rest = w.loc["none"], w.drop(index="none")
rows = [[NAME[t], f3(full[t]), f3(rest[t].min()), f3(rest[t].max())] for t in order]
d = rest["SingleRdisco"] - rest["SingleR"]
rows.append(["SingleR (liver ref.) $-$ SingleR (HPCA)", f3(full["SingleRdisco"] - full["SingleR"]), f3(d.min()), f3(d.max())])
add(table(f"GSE149614 leave-one-patient-out: macro F1 against investigator labels with all cells, and the smallest "
          f"and largest value when each of the {len(rest)} patients (all sampling sites of the patient) is removed in turn.",
          "tab:lopo", "lrrr", "Configuration & All patients & Minimum & Maximum", rows))

# GSE149614 CNV-high hepatocytes by sample
cb = pd.read_csv(f"{E}/rev2_cnv_GSE149614_by_sample.csv")
SITE = {"T": "tumour", "N": "non-tumour liver", "P": "portal-vein tumour thrombus", "L": "lymph-node metastasis"}
rows = [[r.Group, SITE[r.Group[-1]], f"{r.N_hepatocytes:,}", f"{r.N_CNV_high:,}", f"{100 * r.Fraction_CNV_high:.1f}",
         f"{r.Median_CNV_score:.4f}"] for r in cb[cb.Unit == "sample"].itertuples()]
rows += [["\\midrule All " + SITE[r.Group], "", f"{r.N_hepatocytes:,}", f"{r.N_CNV_high:,}",
          f"{100 * r.Fraction_CNV_high:.1f}", f"{r.Median_CNV_score:.4f}"]
         for r in cb[cb.Unit == "site"].set_index("Group").loc[["T", "P", "L", "N"]].reset_index().itertuples()]
a = cb[cb.Unit == "all"].iloc[0]
rows.append(["All samples", "", f"{a.N_hepatocytes:,}", f"{a.N_CNV_high:,}", f"{100 * a.Fraction_CNV_high:.1f}",
             f"{a.Median_CNV_score:.4f}"])
t = cb[cb.Unit == "threshold"].iloc[0]
add(table(f"GSE149614: CNV-high cells among investigator-labelled hepatocytes, by sample and sampling site "
          f"(exploratory). CNV-high: CNV score at or above the 75th percentile over all {t.N_hepatocytes:,} cells "
          f"({t.N_CNV_high:,} cells; threshold {t.Median_CNV_score:.4f}). Sampling site is not a cell-level malignancy "
          f"label, and the cohort has a single lymph-node sample.", "tab:cnvsample", "llrrrr",
          "Sample & Site & Hepatocytes & CNV-high & \\% & Median CNV score", rows))

# final order = order of first mention in the manuscript
ORDER = ["ref", "k", "harm", "seven", "lopo", "friedman", "stromal", "cov", "sorted", "perclass", "decomp",
         "cnv125", "cnv149", "cnvsample", "sim"]
assert sorted(ORDER) == sorted(parts), (sorted(parts), sorted(ORDER))

doc = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,graphicx,float,amsmath,lmodern,caption,longtable}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\title{Supplementary Information\\[4pt]\large Benchmarking Cell-Type Annotation Tools for Hepatocellular
Carcinoma Single-Cell RNA Sequencing Across Six Cohorts}
\date{}
\begin{document}
\maketitle
\noindent Supplementary Data files (CSV): \textbf{Data 1}, label harmonisation table (every raw label of every tool,
the class it maps to, the rule applied and the number of cells per cohort);
\textbf{Data 2}, all scores (every cohort, reference, vocabulary, tool set and tool, with bootstrap intervals and
permutation nulls); \textbf{Data 3}, two files: paired bootstrap differences between tools (Data 3a) and
Nemenyi $p$-values of the primary analysis (Data 3b).

""" + "\n".join(parts[k] for k in ORDER) + r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.75\linewidth]{figures/FigureS1.pdf}
\caption{Secondary analysis: macro F1 against database-derived labels on all six cohorts. $\ast$ScType shares marker
resources with these labels and CyteType their cluster partition on four cohorts; their scores measure concordance with
the same procedure rather than accuracy.}
\end{figure}
\end{document}
"""
with open(f"{OUT}/supplementary_rev2.tex", "w") as fh:
    fh.write(doc)
print("supplement written")
