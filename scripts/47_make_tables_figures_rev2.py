"""
47_make_tables_figures_rev2.py — main tables, figures and supplementary tables of the revised manuscript.

Every number comes from expected_results_rev2/ (31, 32, 37, 41, 45) or the per-cell
predictions; nothing is typed by hand. Output: /mnt/10t/holiday/hcc_paper/rev2/
  tables/Table{1..5}.tex, figures/Figure{1..6}.pdf, figures/FigureS1.pdf, supp/TableS*.tex
Environment: ~/envs/scgpt_rev2 (matplotlib, pandas).
"""
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r31", os.path.join(HERE, "31_rescore_common_cells.py"))
r31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r31)
E = os.path.join(os.path.dirname(HERE), "expected_results_rev2")
OUT = "/mnt/10t/holiday/hcc_paper/rev2"
for d in ["tables", "figures", "supp"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})

NAME = {"SeuratLT": "Seurat label transfer", "SingleRdisco": "SingleR (liver ref.)",
        "SingleR": "SingleR (HPCA)", "CellAssign": "CellAssign", "CellTypist": "CellTypist",
        "ScType": "ScType", "SignacX": "SignacX", "CyteType": "CyteType", "scGPT": "scGPT",
        "scmap": "scmap", "scPred": "scPred", "SingleCellNet": "SingleCellNet"}
FAMILY = {"CellTypist": "Supervised classifier", "scPred": "Supervised classifier",
          "SingleCellNet": "Supervised classifier", "SingleR": "Reference correlation",
          "SingleRdisco": "Reference correlation", "SeuratLT": "Reference projection",
          "scmap": "Reference projection", "ScType": "Marker-based", "CellAssign": "Marker-based",
          "SignacX": "Neural network", "scGPT": "Foundation model", "CyteType": "LLM agent"}
FCOL = {"Supervised classifier": "#1f77b4", "Reference correlation": "#2ca02c",
        "Reference projection": "#17becf", "Marker-based": "#ff7f0e", "Neural network": "#9467bd",
        "Foundation model": "#d62728", "LLM agent": "#8c564b"}
SHARED = {"SingleRdisco", "SeuratLT", "scmap", "scPred", "SingleCellNet", "scGPT"}
COH = ["GSE125449", "GSE149614", "GSE156625"]
s = pd.read_csv(f"{E}/rev2_scores.csv")
pub = s[(s.GT == "published") & (s.Vocab == "broad6") & (s.ToolSet == "all")]
order = pub.groupby("Tool").MacroF1.mean().sort_values(ascending=False).index.tolist()


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def save_tex(path, body):
    with open(path, "w") as fh:
        fh.write(body)


# ── Table 1: datasets ──
t1 = r"""\begin{tabular}{lrrll}
\toprule
Dataset & Cells & Samples & Reference used for accuracy & Source of the analysed object \\
\midrule
GSE125449 & 9{,}752 & 19 & Investigator labels (7 classes) & GEO counts, processed here \\
GSE149614 & 71{,}915 & 21 & Investigator labels (6 classes) & GEO counts, processed here \\
GSE156625 & 73{,}589 & 58 sections, 15 patients & Investigator labels (authors' object on GEO) & Authors' HCC AnnData object \\
GSE162616 & 57{,}261 & 4 & Flow-sorted innate lymphoid cells (design) & DISCO \\
GSE202642 & 122{,}092 & 11 & -- (database-derived labels, secondary) & DISCO \\
GSE223204 & 23{,}165 & 2 & -- (database-derived labels, secondary) & DISCO \\
Simulated & 5{,}000 & -- & Known simulated groups (negative control) & Splatter \\
\bottomrule
\end{tabular}"""
save_tex(f"{OUT}/tables/Table1.tex", t1)

# ── Table 2: tools ──
t2rows = [
    ("CellTypist", "Supervised classifier (logistic regression)", "Built-in Healthy Human Liver model", "No"),
    ("scPred", "Supervised classifier (SVM)", "Shared liver reference", "Yes"),
    ("SingleCellNet", "Supervised classifier (random forest)", "Shared liver reference", "Yes (rand; none observed)"),
    ("SingleR (HPCA)", "Reference correlation", "Human Primary Cell Atlas", "Yes (pruning)"),
    ("SingleR (liver ref.)", "Reference correlation", "Shared liver reference", "Yes (pruning)"),
    ("Seurat label transfer", "Reference projection (anchors)", "Shared liver reference", "No"),
    ("scmap", "Reference projection (cluster centroids)", "Shared liver reference", "Yes"),
    ("ScType", "Marker-database scoring (clusters)", "ScType liver + immune markers", "Yes"),
    ("CellAssign", "Probabilistic marker model", "Literature marker matrix", "No"),
    ("SignacX", "Neural network (immune-specialised)", "Built-in HPCA-trained models", "Yes"),
    ("scGPT", "Foundation-model embedding + $k$-NN", "Shared liver reference", "No"),
    ("CyteType", "Large-language-model agent (clusters)", "None (cluster marker genes)", "Yes (QC flags)"),
]
body = "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in t2rows)
save_tex(f"{OUT}/tables/Table2.tex", r"""\begin{tabular}{llll}
\toprule
Tool (configuration) & Method family & Reference or prior knowledge & Abstains \\
\midrule
""" + body + "\n\\bottomrule\n\\end{tabular}")

# ── Table 3: runtime on GSE149614 (measured in this revision where available) ──
t3rows = [
    ("CellTypist", "$\\sim$5 s$^a$", "CPU"), ("ScType", "$\\sim$10 s$^a$", "CPU"),
    ("SingleR (HPCA)", "$\\sim$4 min$^a$", "CPU"), ("SingleR (liver ref.)", "4.5 min", "CPU (8 cores)"),
    ("Seurat label transfer", "35 s", "CPU"), ("scmap", "n.m.", "CPU"),
    ("scPred", "2.3 min training$^b$", "CPU"), ("SingleCellNet", "3.0 min training$^b$", "CPU"),
    ("CellAssign", "n.m.", "GPU (optional)"), ("SignacX", "4.3 min$^c$", "CPU (6 cores)"),
    ("scGPT", "3.1 min embedding$^d$", "GPU (16 GB)"), ("CyteType", "$\\sim$30 min$^a$", "Remote API"),
]
body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in t3rows)
save_tex(f"{OUT}/tables/Table3.tex", r"""\begin{tabular}{lll}
\toprule
Tool (configuration) & Runtime, GSE149614 (71{,}915 cells) & Hardware \\
\midrule
""" + body + "\n\\bottomrule\n\\end{tabular}")

# ── Table 4: investigator labels ──
lines = []
for t in order:
    cells = []
    for ds in COH:
        r = pub[(pub.Tool == t) & (pub.Dataset == ds)].iloc[0]
        ci = "" if pd.isna(r.MacroF1_CI_sample_lo) else f" ({r.MacroF1_CI_sample_lo:.2f}--{r.MacroF1_CI_sample_hi:.2f})"
        cells.append(f"{r.MacroF1:.3f}{ci} & {r.Coverage:.2f}")
    m = pub[pub.Tool == t].MacroF1.mean()
    lines.append(f"{NAME[t]} & " + " & ".join(cells) + f" & {m:.3f} \\\\")
save_tex(f"{OUT}/tables/Table4.tex", r"""\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{2}{c}{GSE125449} & \multicolumn{2}{c}{GSE149614} & \multicolumn{2}{c}{GSE156625} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Tool & Macro F1 (95\% CI) & Cov. & Macro F1 (95\% CI) & Cov. & Macro F1 (95\% CI) & Cov. & Mean \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")

# ── Table 5: database-derived labels (secondary) ──
en = s[(s.GT == "enrichR") & (s.ToolSet == "all")]
ecoh = ["GSE125449", "GSE149614", "GSE156625", "GSE162616", "GSE202642", "GSE223204"]
eorder = en.groupby("Tool").MacroF1.mean().sort_values(ascending=False).index.tolist()
lines = []
for t in eorder:
    vals = [en[(en.Tool == t) & (en.Dataset == ds)].MacroF1 for ds in ecoh]
    vals = [v.iloc[0] if len(v) else np.nan for v in vals]
    flag = "$^\\ast$" if t in ("ScType", "CyteType") else ""
    lines.append(f"{NAME[t]}{flag} & " + " & ".join(f3(v) for v in vals) + f" & {np.nanmean(vals):.3f} \\\\")
covrow = " & ".join(f"{100 * en[en.Dataset == ds].N_GT_labelled.iloc[0] / {'GSE125449': 9752, 'GSE149614': 71915, 'GSE156625': 73589, 'GSE162616': 57261, 'GSE202642': 122092, 'GSE223204': 23165}[ds]:.1f}\\%" for ds in ecoh)
save_tex(f"{OUT}/tables/Table5.tex", r"""\begin{tabular}{lrrrrrrr}
\toprule
Tool & GSE125449 & GSE149614 & GSE156625 & GSE162616 & GSE202642 & GSE223204 & Mean \\
\midrule
Cells with a database-derived label & """ + covrow + r""" & \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")

# ── Figure 1: macro F1 with sample-level CI per published cohort ──
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
y = np.arange(len(order))[::-1]
for ax, ds in zip(axes, COH):
    for yi, t in zip(y, order):
        r = pub[(pub.Tool == t) & (pub.Dataset == ds)].iloc[0]
        c = FCOL[FAMILY[t]]
        ax.errorbar(r.MacroF1, yi, xerr=[[r.MacroF1 - r.MacroF1_CI_sample_lo], [r.MacroF1_CI_sample_hi - r.MacroF1]],
                    fmt="o", color=c, ms=5, capsize=2, lw=1)
        ax.plot(r.MacroF1_selective, yi, marker="|", color="grey", ms=8)
    ax.set_title(ds)
    ax.set_xlim(0.55, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Macro F1 (coverage-aware)")
axes[0].set_yticks(y)
axes[0].set_yticklabels([NAME[t] for t in order])
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FCOL.items()]
handles.append(plt.Line2D([], [], marker="|", ls="", color="grey", ms=8, label="Selective (cells all tools labelled)"))
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(f"{OUT}/figures/Figure1.pdf", bbox_inches="tight")
plt.close(fig)

# ── Figure 2: reference effect ──
cls = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
w = 0.12
for i, ds in enumerate(COH):
    for j, (t, col) in enumerate([("SingleR", "#98df8a"), ("SingleRdisco", "#2ca02c")]):
        r = pub[(pub.Tool == t) & (pub.Dataset == ds)].iloc[0]
        x = np.arange(len(cls)) + (i * 2 + j - 2.5) * w
        ax.bar(x, [r[f"F1_{c}"] for c in cls], width=w, color=col, edgecolor="white",
               label=(NAME[t] if i == 0 else None), alpha=0.6 + 0.13 * i)
ax.set_xticks(np.arange(len(cls)))
ax.set_xticklabels(cls)
ax.set_ylabel("Per-class F1")
ax.set_title("A  SingleR: HPCA vs liver reference (bars: GSE125449, GSE149614, GSE156625)", fontsize=8.5, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_ylim(0, 1.05)
ax = axes[1]
tl = [t for t in order if t in SHARED]
mean = [pub[pub.Tool == t].MacroF1.mean() for t in tl]
ax.barh(np.arange(len(tl))[::-1], mean, color=[FCOL[FAMILY[t]] for t in tl])
for yi, t in zip(np.arange(len(tl))[::-1], tl):
    for ds, mk in zip(COH, ["o", "s", "^"]):
        ax.plot(pub[(pub.Tool == t) & (pub.Dataset == ds)].MacroF1.iloc[0], yi, mk, color="k", ms=3.5)
ax.set_yticks(np.arange(len(tl))[::-1])
ax.set_yticklabels([NAME[t] for t in tl])
ax.set_xlim(0.6, 1.0)
ax.set_xlabel("Macro F1 (bar: mean; markers: cohorts)")
ax.set_title("B  Tools trained on the shared liver reference", fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/figures/Figure2.pdf", bbox_inches="tight")
plt.close(fig)

# ── Figure 3: coverage vs macro F1 ──
fig, ax = plt.subplots(figsize=(6.2, 4.6))
for t in order:
    d = pub[pub.Tool == t]
    c = FCOL[FAMILY[t]]
    ax.scatter(d.Coverage, d.MacroF1, color=c, s=22, zorder=3)
    for _, r in d.iterrows():
        if r.Coverage < 0.97:
            right = r.Coverage > 0.94
            ax.annotate(f"{NAME[t]}\n{r.Dataset}", (r.Coverage, r.MacroF1), fontsize=6,
                        xytext=(-4, 3) if right else (3, -3), ha="right" if right else "left",
                        va="bottom" if right else "baseline",
                        textcoords="offset points")
ax.set_xlabel("Coverage (fraction of cells assigned a class)")
ax.set_ylabel("Macro F1 (abstentions counted as errors)")
ax.grid(alpha=0.3)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FCOL.items()]
ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/figures/Figure3.pdf", bbox_inches="tight")
plt.close(fig)

# ── Figure 4: sorted ILC composition ──
so = pd.read_csv(f"{E}/rev2_GSE162616_sorted_ILC_check.csv")
so = so[~so.Source.str.startswith("enrichR")].set_index("Source").loc[[t for t in order if t in set(so.Source)]]
parts = [("pct_T/NK", "T/NK", "#4c72b0"), ("pct_Myeloid", "Myeloid", "#dd8452"), ("pct_B", "B", "#55a868"),
         ("pct_non_immune", "Non-immune", "#c44e52"), ("pct_Unknown", "Abstention", "#bbbbbb")]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
left = np.zeros(len(so))
yy = np.arange(len(so))[::-1]
for col, lab, c in parts:
    ax.barh(yy, so[col].values, left=left, color=c, label=lab)
    left += so[col].values
ax.set_yticks(yy)
ax.set_yticklabels([NAME[t] for t in so.index])
ax.set_xlabel("% of GSE162616 cells (flow-sorted innate lymphoid cells)")
ax.set_xlim(0, 100)
ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.27))
fig.tight_layout()
fig.savefig(f"{OUT}/figures/Figure4.pdf", bbox_inches="tight")
plt.close(fig)

# ── Figure 5: confusion matrices, GSE149614 investigator labels ──
gt, _ = r31.load_gt_and_samples("GSE149614", "published", "broad6", "canonical")
raw, _ = r31.load_raw_tools("GSE149614", "broad6", "rev2")
cols = cls + ["Unknown"]
SHORT = {"T/NK": "T/NK", "Myeloid": "Myeloid", "Hepatocyte": "Hepatocyte", "B": "B", "Endothelial": "Endothelial", "Fibroblast": "Fibroblast", "Unknown": "Abstention"}
fig, axes = plt.subplots(3, 4, figsize=(8.2, 7.4), sharex=True, sharey=True)
for ax, t in zip(axes.ravel(), order):
    p = r31.map_raw(t, raw[t], "broad6").reindex(gt.index).fillna("Unknown")
    ct = pd.crosstab(gt, p).reindex(index=cls, columns=cols, fill_value=0)
    m = ct.div(ct.sum(axis=1), axis=0).values
    ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] >= 0.05:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if m[i, j] > 0.6 else "black")
    ax.set_title(NAME[t], fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([SHORT[c] for c in cols], rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(cls)))
    ax.set_yticklabels(cls, fontsize=7.5)
fig.supxlabel("Predicted class")
fig.supylabel("Investigator label (GSE149614)")
fig.tight_layout()
fig.savefig(f"{OUT}/figures/Figure5.pdf", bbox_inches="tight")
plt.close(fig)

# ── Figure 6: CNV ──
cs = pd.read_csv(f"{E}/cnv/GSE125449_TB_TAM_cnv_scores.csv.gz", index_col=0)
thr = np.percentile(cs.cnv_score, 75)
c6 = pd.read_csv(f"{E}/rev2_cnv_GSE125449.csv")
c6 = c6[(c6.panel == "TB_TAM") & (c6.Tool.isin(order))].set_index("Tool").loc[[t for t in order]]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]
types = cs.groupby("ground_truth").cnv_score.median().sort_values().index.tolist()
ax.boxplot([cs[cs.ground_truth == g].cnv_score for g in types], labels=types, showfliers=False)
ax.axhline(thr, color="red", ls="--", lw=1)
ax.set_ylabel("Mean absolute CNV signal")
ax.tick_params(axis="x", rotation=45)
ax.set_title("A  GSE125449: CNV signal by investigator label", fontsize=8.5, loc="left")
ax = axes[1]
x = np.arange(len(c6))
ax.bar(x - 0.2, c6.CNV_F1, width=0.4, label="vs CNV-high call", color="#4c72b0")
ax.bar(x + 0.2, c6.PubGT_F1, width=0.4, label="vs published malignant + HPC-like", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels([NAME[t] for t in c6.index], rotation=60, ha="right", fontsize=7.5)
ax.set_ylabel("F1 of hepatocyte-lineage calls")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("B  Hepatocyte-lineage calls vs CNV-high and published labels", fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/figures/Figure6.pdf", bbox_inches="tight")
plt.close(fig)

# ── Figure S1: database-derived labels heatmap ──
hm = en.pivot(index="Tool", columns="Dataset", values="MacroF1").loc[eorder, ecoh]
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(hm.values, vmin=0.5, vmax=1, cmap="RdYlGn")
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        ax.text(j, i, f3(hm.values[i, j]), ha="center", va="center", fontsize=7)
ax.set_xticks(range(len(ecoh)))
ax.set_xticklabels(ecoh, rotation=45, ha="right")
ax.set_yticks(range(len(eorder)))
ax.set_yticklabels([NAME[t] + (" *" if t in ("ScType", "CyteType") else "") for t in eorder])
fig.colorbar(im, ax=ax, label="Macro F1 vs database-derived labels")
fig.tight_layout()
fig.savefig(f"{OUT}/figures/FigureS1.pdf", bbox_inches="tight")
plt.close(fig)
print("tables and figures written to", OUT)
