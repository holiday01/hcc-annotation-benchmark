"""
51_sensitivity_lopo_cnv.py — three checks added in the second correction round (revision 2).

1. Stromal-label mapping sensitivity (investigator labels; GSE125449, GSE149614, GSE156625).
   Every configuration is re-scored with the HPCA label "Tissue_stem_cells" (mesenchymal
   stem/stromal cells) mapped to Fibroblast instead of Unknown. It is the only stem- or
   stromal-type label emitted by any tool (the HPCA label "MSC" is already mapped to
   Fibroblast), so only SingleR (HPCA) changes. Same cells, class set and scoring rules as
   31_rescore_common_cells.py. Patient-level paired bootstrap (B = 1000, seed 42; patients
   resampled within each cohort, the same draws for both configurations and both mappings) of
   the difference SingleR (liver reference) - SingleR (HPCA), per cohort and for the
   three-cohort mean, and the per-class contributions to that difference.
2. GSE149614 leave-one-patient-out: macro F1 of every configuration with each of the
   10 patients removed in turn (all sampling sites of the patient removed together).
3. GSE149614 CNV-high cells by sample: among author-labelled hepatocytes, the number and
   fraction above the pooled top-quartile CNV threshold of 45_cnv_simulation_rev2.py
   (75th percentile of the CNV score over all cells of the cohort), and the median CNV score,
   per sample and per sampling site (T tumour, N non-tumour liver, P portal-vein tumour
   thrombus, L lymph-node metastasis).
4. GSE156625 cluster purity: the share of investigator-labelled cells whose label equals the
   majority label of their cluster, for the authors' clustering (used by ScType and CyteType)
   and for the independent re-clustering of 39_build_annotation_package.py.

Writes (default expected_results_rev2/, or --out):
  rev2_sensitivity_stromal_mapping.csv   macro F1 / coverage / fibroblast F1 per configuration,
                                         cohort and mapping
  rev2_sensitivity_reference_gap.csv     SingleR liver reference - HPCA: difference, 95% CI,
                                         per-class contributions, per cohort and mean
  rev2_lopo_GSE149614.csv                macro F1 per configuration and left-out patient
  rev2_cnv_GSE149614_by_sample.csv       CNV-high hepatocytes per sample, per site and overall
  rev2_cluster_purity_GSE156625.csv      majority-label share of the two partitions
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r31", os.path.join(HERE, "31_rescore_common_cells.py"))
r31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r31)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "expected_results_rev2"))
ap.add_argument("--B", type=int, default=1000)
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)

PUB = ["GSE125449", "GSE149614", "GSE156625"]
V = r31.BROAD6
K = len(V)
LIVER, HPCA = "SingleRdisco", "SingleR"


def labels(tool, raw, lenient):
    lab = r31.map_raw(tool, raw[tool], "broad6")
    if lenient and tool == HPCA:
        lab = lab.copy()
        lab[raw[tool].astype(str).reindex(lab.index) == "Tissue_stem_cells"] = "Fibroblast"
    return lab


# ── 1. stromal-label mapping sensitivity ─────────────────────────────────────
rows, store = [], {}
for ds in PUB:
    gt, samples = r31.load_gt_and_samples(ds, "published", "broad6", "canonical")
    raw, _ = r31.load_raw_tools(ds, "broad6", "rev2")
    present = np.array([(gt == c).any() for c in V])
    yt, _ = r31.encode(gt, gt, V)
    s = samples.reindex(gt.index)
    assert s.notna().all(), ds
    s_codes, s_levels = pd.factorize(s)
    store[ds] = {"present": present, "n_s": len(s_levels)}
    for mapping in ("canonical", "lenient"):
        for t in r31.TOOLS:
            lab = labels(t, raw, mapping == "lenient").reindex(gt.index).fillna("Unknown").astype(str)
            _, yp = r31.encode(gt, lab, V)
            m = r31.metrics_from_cm(r31.confmat(yt, yp, K), present)
            rows.append({"Dataset": ds, "Mapping": mapping, "Tool": t, "N": len(gt),
                         "MacroF1": m["macro_f1"], "Coverage": m["coverage"],
                         **{f"F1_{c}": (m["f1"][i] if present[i] else np.nan) for i, c in enumerate(V)}})
            if t in (LIVER, HPCA):
                store[ds][(mapping, t)] = r31.per_sample_cms(yt, yp, s_codes, len(s_levels), K)
sens = pd.DataFrame(rows)
sens.to_csv(f"{OUT}/rev2_sensitivity_stromal_mapping.csv", index=False, float_format="%.6f")

# check against the main scores (canonical mapping must reproduce 31_rescore_common_cells.py)
main = pd.read_csv(os.path.join(os.path.dirname(HERE), "expected_results_rev2", "rev2_scores.csv"))
main = main[(main.GT == "published") & (main.Vocab == "broad6") & (main.ToolSet == "all")]
chk = sens[sens.Mapping == "canonical"].merge(main[["Dataset", "Tool", "MacroF1"]], on=["Dataset", "Tool"],
                                              suffixes=("", "_main"))
assert len(chk) == 36 and (chk.MacroF1 - chk.MacroF1_main).abs().max() < 1e-4, chk

# patient-level paired bootstrap of the reference gap
rng = np.random.default_rng(r31.SEED)
boot = {(mp, ds): np.empty(args.B) for mp in ("canonical", "lenient") for ds in PUB}
for b in range(args.B):
    for ds in PUB:
        n_s = store[ds]["n_s"]
        sidx = rng.integers(0, n_s, n_s)
        for mp in ("canonical", "lenient"):
            f = {}
            for t in (LIVER, HPCA):
                cm = store[ds][(mp, t)][sidx].sum(axis=0)
                pres_b = store[ds]["present"] & (cm.sum(axis=1) > 0)
                f[t] = r31.metrics_from_cm(cm, pres_b)["macro_f1"]
            boot[(mp, ds)][b] = f[LIVER] - f[HPCA]

gap = []
for mp in ("canonical", "lenient"):
    sub = sens[sens.Mapping == mp].set_index(["Dataset", "Tool"])
    contrib_mean = {c: 0.0 for c in V}
    for ds in PUB + ["Mean"]:
        if ds == "Mean":
            liver = np.mean([sub.loc[(d, LIVER), "MacroF1"] for d in PUB])
            hpca = np.mean([sub.loc[(d, HPCA), "MacroF1"] for d in PUB])
            dist = np.mean([boot[(mp, d)] for d in PUB], axis=0)
            contrib = contrib_mean
        else:
            liver, hpca = sub.loc[(ds, LIVER), "MacroF1"], sub.loc[(ds, HPCA), "MacroF1"]
            dist = boot[(mp, ds)]
            n_pres = int(store[ds]["present"].sum())
            contrib = {c: (sub.loc[(ds, LIVER), f"F1_{c}"] - sub.loc[(ds, HPCA), f"F1_{c}"]) / n_pres
                       if store[ds]["present"][i] else 0.0 for i, c in enumerate(V)}
            for c in V:
                contrib_mean[c] += contrib[c] / len(PUB)
        gap.append({"Mapping": mp, "Scope": ds, "MacroF1_liver_ref": liver, "MacroF1_HPCA": hpca,
                    "Difference": liver - hpca, "CI_lo": np.percentile(dist, 2.5),
                    "CI_hi": np.percentile(dist, 97.5),
                    **{f"Contribution_{c}": contrib[c] for c in V}})
gap = pd.DataFrame(gap)
gap.to_csv(f"{OUT}/rev2_sensitivity_reference_gap.csv", index=False, float_format="%.6f")
print(gap.round(4).to_string(index=False), flush=True)

# ── 2. GSE149614 leave-one-patient-out ───────────────────────────────────────
ds = "GSE149614"
gt, patients = r31.load_gt_and_samples(ds, "published", "broad6", "canonical")
patients = patients.reindex(gt.index)
raw, _ = r31.load_raw_tools(ds, "broad6", "rev2")
present_all = np.array([(gt == c).any() for c in V])
lab = {t: labels(t, raw, False).reindex(gt.index).fillna("Unknown").astype(str) for t in r31.TOOLS}
lo = []
for p in ["none"] + sorted(patients.unique()):
    keep = (patients != p).to_numpy()
    g = gt[keep]
    yt, _ = r31.encode(g, g, V)
    pres = present_all & np.array([(g == c).any() for c in V])
    for t in r31.TOOLS:
        _, yp = r31.encode(g, lab[t][keep], V)
        lo.append({"Left_out": p, "N_cells": int(keep.sum()), "Tool": t,
                   "MacroF1": r31.metrics_from_cm(r31.confmat(yt, yp, K), pres)["macro_f1"]})
lo = pd.DataFrame(lo)
lo.to_csv(f"{OUT}/rev2_lopo_GSE149614.csv", index=False, float_format="%.6f")
w = lo.pivot(index="Left_out", columns="Tool", values="MacroF1")
print("LOPO range per tool:\n", (w.drop(index="none").agg(["min", "max"]).T.round(3)), flush=True)

# ── 3. GSE149614 CNV-high hepatocytes by sample ──────────────────────────────
c = pd.read_csv(os.path.join(os.path.dirname(HERE), "expected_results_rev2", "cnv",
                             "GSE149614_TNK_B_cnv_scores.csv.gz"), index_col=0)
c.index = c.index.astype(str)
g = pd.read_csv(f"{r31.ev.GT_DIR}/{ds}_ground_truth.tsv", sep="\t", index_col=0)
g.index = g.index.astype(str)
c["sample"] = g["sample"].astype(str).reindex(c.index)
assert c["sample"].notna().all()
thr = float(np.percentile(c["cnv_score"], 75))
assert ((c["cnv_score"] >= thr).astype(int) == c["cnv_malignant"]).all()
hep = c[c["ground_truth"].astype(str).str.lower().str.contains("hepatocyte")]
out = []
for unit, key in [("sample", hep["sample"]), ("site", hep["sample"].str[-1]), ("all", pd.Series("all", index=hep.index))]:
    for k, d in hep.groupby(key):
        out.append({"Unit": unit, "Group": k, "N_hepatocytes": len(d),
                    "N_CNV_high": int(d["cnv_malignant"].sum()), "Fraction_CNV_high": d["cnv_malignant"].mean(),
                    "Median_CNV_score": d["cnv_score"].median()})
out.append({"Unit": "threshold", "Group": "all cells", "N_hepatocytes": len(c), "N_CNV_high": int(c["cnv_malignant"].sum()),
            "Fraction_CNV_high": c["cnv_malignant"].mean(), "Median_CNV_score": thr})
out = pd.DataFrame(out)
out.to_csv(f"{OUT}/rev2_cnv_GSE149614_by_sample.csv", index=False, float_format="%.6f")
print(out.round(4).to_string(index=False), flush=True)

# ── 4. GSE156625 cluster purity ─────────────────────────────────────────────
GTD = r31.ev.GT_DIR
lab156 = pd.read_csv(f"{GTD}/GSE156625_published_ground_truth.tsv", sep="\t", index_col=0)["celltype"].astype(str)
parts = [("authors' clustering (used by ScType and CyteType)",
          pd.read_csv(f"{GTD}/GSE156625_enrichr_ground_truth.tsv", sep="\t", index_col=0)["clusters"].astype(str)),
         ("independent re-clustering (39_build_annotation_package.py)",
          pd.read_csv("/mnt/10t/holiday/hcc_paper/annotation_package_rev2/key_DO_NOT_SHARE/"
                      "GSE156625_cluster_assignments.csv.gz", index_col="barcode")["cluster"].astype(str))]
pur = []
for name, part in parts:
    part.index = part.index.astype(str)
    df = pd.DataFrame({"p": part, "g": lab156}).dropna()
    df = df[df.g != "Unknown"]
    ct = pd.crosstab(df.p, df.g)
    pur.append({"Partition": name, "N_cells": int(ct.values.sum()), "N_clusters": ct.shape[0],
                "Majority_label_share": ct.max(axis=1).sum() / ct.values.sum()})
pur = pd.DataFrame(pur)
pur.to_csv(f"{OUT}/rev2_cluster_purity_GSE156625.csv", index=False, float_format="%.6f")
print(pur.to_string(index=False), flush=True)
