"""
46_manuscript_numbers.py — every number the revised manuscript quotes, from the deposited files.

Writes expected_results_rev2/rev2_manuscript_numbers.md (human-readable) and .json.
The manuscript, supplement and response letter must quote numbers only from this file;
re-run it after any re-analysis and diff the output.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(os.path.dirname(HERE), "expected_results_rev2")
A2 = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2"
GT = "/mnt/10t/assi_result/HCC/benchmark_datasets/ground_truth"
N = {}
md = []


def put(key, val, note=""):
    N[key] = val
    md.append(f"| `{key}` | {val} | {note} |")


s = pd.read_csv(f"{E}/rev2_scores.csv")
cells = pd.read_csv(f"{E}/rev2_cellsets.csv")
fr = pd.read_csv(f"{E}/rev2_friedman.csv")

# ── primary: published labels, broad-6, all tools ──
p = s[(s.GT == "published") & (s.Vocab == "broad6") & (s.ToolSet == "all")]
cohorts = sorted(p.Dataset.unique())
put("primary_cohorts", ", ".join(cohorts))
put("n_tools", int(p.Tool.nunique()))
for ds in cohorts:
    put(f"published_N_{ds}", int(p[p.Dataset == ds].N_common.iloc[0]), "GT-labelled cells scored")
piv = p.pivot(index="Tool", columns="Dataset", values="MacroF1")
cov = p.pivot(index="Tool", columns="Dataset", values="Coverage")
mean = piv.mean(axis=1).sort_values(ascending=False)
for t in mean.index:
    put(f"pub_meanF1_{t}", round(mean[t], 3), "mean macro F1 over 3 published cohorts (coverage-aware)")
    for ds in cohorts:
        r = p[(p.Tool == t) & (p.Dataset == ds)].iloc[0]
        put(f"pub_F1_{t}_{ds}", round(r.MacroF1, 3),
            f"sample CI {r.MacroF1_CI_sample_lo:.3f}-{r.MacroF1_CI_sample_hi:.3f}; cov {r.Coverage:.3f}; sel {r.MacroF1_selective:.3f}")
    put(f"pub_minCov_{t}", round(cov.loc[t].min(), 3))
put("pub_meanF1_range", f"{mean.min():.3f}-{mean.max():.3f}")
for _, r in fr.iterrows():
    put(f"friedman_{r.Analysis}", f"chi2={r.chi2:.2f}, p={r.p:.3g}, n={r.n_datasets}, k={r.k_tools}")
nem = [f for f in os.listdir(E) if f.startswith("rev2_nemenyi_E_")]
if nem:
    m = pd.read_csv(f"{E}/{nem[0]}", index_col=0)
    v = m.where(~np.eye(len(m), dtype=bool))
    i = np.unravel_index(np.nanargmin(v.values), v.shape)
    put("nemenyi_E_min_p", f"{v.values[i]:.3f} ({m.index[i[0]]} vs {m.columns[i[1]]})")
pw = pd.read_csv(f"{E}/rev2_pairwise_bootstrap.csv")
pw = pw[(pw.GT == "published") & (pw.Vocab == "broad6") & (pw.ToolSet == "all")]
for ds, g in pw.groupby("Dataset"):
    put(f"pairs_sampleCI_excl0_{ds}", f"{int(((g.CI_sample_lo > 0) | (g.CI_sample_hi < 0)).sum())} of {len(g)}")

# broad-7 GSE125449
b7 = s[(s.Dataset == "GSE125449") & (s.Vocab == "broad7") & (s.ToolSet == "all")].set_index("Tool")
for t in b7.index:
    put(f"b7_F1_{t}", round(b7.loc[t, "MacroF1"], 3),
        f"Malignant {b7.loc[t, 'F1_Malignant cell']:.3f}; HPC-like {b7.loc[t, 'F1_HPC-like']:.3f}")

# per-class fibroblast on published cohorts
for ds in cohorts:
    for t in p.Tool.unique():
        r = p[(p.Tool == t) & (p.Dataset == ds)].iloc[0]
        put(f"pubF1_Fibroblast_{t}_{ds}", round(r["F1_Fibroblast"], 3))

# ── enrichR (secondary) ──
e = s[(s.GT == "enrichR") & (s.ToolSet == "all")]
for ds in sorted(e.Dataset.unique()):
    tot = {"GSE125449": 9752, "GSE149614": 71915, "GSE156625": 73589, "GSE162616": 57261,
           "GSE202642": 122092, "GSE223204": 23165}[ds]
    n = int(e[e.Dataset == ds].N_GT_labelled.iloc[0])
    put(f"enrichR_coverage_{ds}", f"{n} of {tot} ({100 * n / tot:.1f}%)")
en = e.pivot(index="Tool", columns="Dataset", values="MacroF1")
for t in en.index:
    put(f"enrichR_meanF1_{t}", round(en.loc[t].mean(), 3), f"6-cohort mean; circular={t in ('ScType', 'CyteType')}")

# enrichR vs published on co-labelled cells
import importlib.util
spec = importlib.util.spec_from_file_location("r31", os.path.join(HERE, "31_rescore_common_cells.py"))
r31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r31)
for ds in cohorts:
    pubgt, _ = r31.load_gt_and_samples(ds, "published", "broad6", "canonical")
    eg, _ = r31.load_gt_and_samples(ds, "enrichR", "broad6", "canonical")
    both = pubgt.index.intersection(eg.index)
    put(f"enrichR_vs_pub_agree_{ds}", f"{(pubgt[both] == eg[both]).mean():.3f} on {len(both)} co-labelled cells")
    miss = pubgt[pubgt.index.difference(eg.index)]
    for c in ["Fibroblast", "Myeloid"]:
        put(f"enrichR_omits_{c}_{ds}", f"{int((miss == c).sum())} of {int((pubgt == c).sum())}")

# ── sorted ILC check ──
so = pd.read_csv(f"{E}/rev2_GSE162616_sorted_ILC_check.csv")
tools_so = so[~so.Source.str.startswith("enrichR")]
put("sorted_TNK_range", f"{tools_so['pct_T/NK'].min():.1f}-{tools_so['pct_T/NK'].max():.1f}%")
put("sorted_nonimmune_max", f"{tools_so.pct_non_immune.max():.2f}% ({tools_so.loc[tools_so.pct_non_immune.idxmax(), 'Source']})")
for _, r in so.iterrows():
    put(f"sorted_{r.Source}", f"T/NK {r['pct_T/NK']}%, Myeloid {r.pct_Myeloid}%, B {r.pct_B}%, non-immune {r.pct_non_immune}%, Unknown {r.pct_Unknown}%")

# ── scGPT, reference, k ──
ex = pd.read_csv(f"{A2}/scgpt_reference_exclusion.csv")
for _, r in ex.iterrows():
    put(f"disco_removed_{r.series}", f"{int(r.n_DISCO_cells_removed)} cells / {int(r.n_DISCO_samples_removed)} samples")
ref = pd.read_csv("/mnt/10t/assi_result/HCC/benchmark_datasets/rev2_inputs/DISCO_reference/cells.csv")
put("shared_ref_cells", len(ref), str(ref.label.value_counts().to_dict()))
put("shared_ref_samples", int(ref["sample"].nunique()))
k = pd.read_csv(f"{E}/rev2_scgpt_k_sensitivity.csv")
put("scgpt_k_max_range", round(k.Range.max(), 3))
put("scgpt_validation_agreement", "0.9645 (original reference, GSE125449, k=15) vs 0.969 between two seeds",
    "from 35_scgpt_leave_cohort_out.py --validate and scgpt_selfconsistency.py logs (2026-11-14)")

# ── GSE156625 published labels ──
g = pd.read_csv(f"{GT}/GSE156625_published_ground_truth.tsv", sep="\t")
put("GSE156625_published_usable", f"{int((g.celltype != 'Unknown').sum())} of {len(g)} ({100 * (g.celltype != 'Unknown').mean():.1f}%)")

# ── CNV / simulation (if present) ──
for f in ["rev2_cnv_GSE125449.csv", "rev2_cnv_GSE149614.csv", "rev2_simulated.csv"]:
    if os.path.exists(f"{E}/{f}"):
        d = pd.read_csv(f"{E}/{f}")
        for _, r in d.iterrows():
            put(f"{f[5:-4]}_{'_'.join(str(r[c]) for c in d.columns[:2])}",
                "; ".join(f"{c}={r[c]}" for c in d.columns[2:] if pd.notna(r[c])))

# ── 51_sensitivity_lopo_cnv.py: stromal-label mapping, leave-one-patient-out, CNV-high by sample ──
if os.path.exists(f"{E}/rev2_sensitivity_reference_gap.csv"):
    gp = pd.read_csv(f"{E}/rev2_sensitivity_reference_gap.csv")
    for _, r in gp.iterrows():
        fib = r.Contribution_Fibroblast / r.Difference if r.Difference else np.nan
        put(f"refgap_{r.Mapping}_{r.Scope}",
            f"liver {r.MacroF1_liver_ref:.3f}; HPCA {r.MacroF1_HPCA:.3f}; diff {r.Difference:.3f} "
            f"(95% CI {r.CI_lo:.3f} to {r.CI_hi:.3f}); fibroblast share {fib:.2f}",
            "SingleR liver reference - SingleR HPCA; patient-level paired bootstrap")
    sm = pd.read_csv(f"{E}/rev2_sensitivity_stromal_mapping.csv")
    for (ds, t), d in sm[sm.Tool == "SingleR"].groupby(["Dataset", "Tool"]):
        put(f"stromal_SingleR_HPCA_{ds}", "; ".join(f"{r.Mapping} {r.MacroF1:.3f} (fibroblast F1 {r.F1_Fibroblast:.3f})"
                                                    for r in d.itertuples()))
    changed = sm.pivot_table(index=["Dataset", "Tool"], columns="Mapping", values="MacroF1")
    changed = changed[(changed.canonical - changed.lenient).abs() > 1e-9]
    put("stromal_configs_changed", ", ".join(sorted({t for _, t in changed.index})), "configurations whose score changes")
if os.path.exists(f"{E}/rev2_lopo_GSE149614.csv"):
    lo = pd.read_csv(f"{E}/rev2_lopo_GSE149614.csv")
    w = lo.pivot(index="Left_out", columns="Tool", values="MacroF1")
    full, rest = w.loc["none"], w.drop(index="none")
    for t in w.columns:
        put(f"lopo_GSE149614_{t}", f"all {full[t]:.3f}; leave-one-patient-out {rest[t].min():.3f}-{rest[t].max():.3f}")
    d = rest["SingleRdisco"] - rest["SingleR"]
    put("lopo_GSE149614_gap_liver_minus_HPCA", f"all {full['SingleRdisco'] - full['SingleR']:.3f}; {d.min():.3f}-{d.max():.3f}")
    put("lopo_GSE149614_max_shift", f"{(rest - full).abs().max().max():.3f}", "largest change of any configuration")
if os.path.exists(f"{E}/rev2_cnv_GSE149614_by_sample.csv"):
    cb = pd.read_csv(f"{E}/rev2_cnv_GSE149614_by_sample.csv")
    for r in cb.itertuples():
        put(f"cnvhigh_GSE149614_{r.Unit}_{r.Group}",
            f"{r.N_CNV_high}/{r.N_hepatocytes} ({100 * r.Fraction_CNV_high:.1f}%); median score {r.Median_CNV_score:.4f}")
    smp = cb[cb.Unit == "sample"]
    for site in "TN":
        x = smp[smp.Group.str.endswith(site)].Fraction_CNV_high * 100
        put(f"cnvhigh_GSE149614_range_{site}_samples", f"{x.min():.1f}-{x.max():.1f}% ({len(x)} samples)")

if os.path.exists(f"{E}/rev2_cluster_purity_GSE156625.csv"):
    for i, r in enumerate(pd.read_csv(f"{E}/rev2_cluster_purity_GSE156625.csv").itertuples()):
        put(f"purity_GSE156625_{'authors' if i == 0 else 'independent'}",
            f"{100 * r.Majority_label_share:.1f}% ({r.N_clusters} clusters, {r.N_cells} cells)", r.Partition)

os.makedirs(E, exist_ok=True)
with open(f"{E}/rev2_manuscript_numbers.json", "w") as fh:
    json.dump(N, fh, indent=1, default=str)
with open(f"{E}/rev2_manuscript_numbers.md", "w") as fh:
    fh.write("# Numbers quoted in the revised manuscript\n\n| key | value | note |\n|---|---|---|\n")
    fh.write("\n".join(md) + "\n")
print(f"{len(N)} numbers written")
