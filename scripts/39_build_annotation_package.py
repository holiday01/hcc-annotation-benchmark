"""
39_build_annotation_package.py — blinded cluster-annotation package for independent ground truth.

Revision 2 (Reviewer 1, comment 2). For every cohort, cells are re-clustered
from raw counts with a pipeline that shares nothing with the benchmarked tools
or with the enrichR ground truth, and each cluster is summarised for human
annotators who see no tool output, no enrichR label and no published label.

Per cohort (raw counts from 33_prepare_counts_rev2.py):
  QC metrics; Scrublet doublet scores per sample (samples < 200 cells pooled);
  normalize_total(1e4) + log1p; 2,000 HVG (seurat flavour); scale; PCA (50);
  kNN (k = 15, 30 PCs); Leiden (igraph) at the smallest resolution in
  0.4–3.0 giving >= TARGET clusters (30 / 40 / 50 for < 15k / < 80k / >= 80k
  cells); UMAP. Seed 42 throughout.
  DE: Wilcoxon, each cluster vs all other cells, on <= 2,000 random cells per
  cluster and genes detected in >= 1% of cells.
Cohorts are shown as letters A–F in a shuffled order; clusters get shuffled
codes (A01, A02, ...). Tissue context (tumour / adjacent / etc.) is shown;
sorting information and dataset accessions are not.

Output: /mnt/10t/holiday/hcc_paper/annotation_package_rev2/
  package/            -> give to annotators
    PROTOCOL.md (written separately), annotation_sheet_TEMPLATE.xlsx,
    cohort_<X>/report.html (self-contained), cohort_<X>/cluster_summary.csv
  key_DO_NOT_SHARE/   -> never give to annotators
    cohort_key.csv, <GSE>_cluster_assignments.csv.gz, partition_independence.csv,
    resolution_scan.csv

Environment: ~/envs/scgpt_rev2 (scanpy 1.11.5, leidenalg, scikit-image, openpyxl).
Usage: python 39_build_annotation_package.py [GSE ...]
"""
from __future__ import annotations

import base64
import glob
import html
import io
import json
import os
import re
import sys
import warnings
from collections import OrderedDict

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
BENCH = "/mnt/10t/assi_result/HCC/benchmark_datasets"
IN = f"{BENCH}/rev2_inputs"
GEO = "/mnt/10t/assi_result/GEO_DATA"
OPEN = "/mnt/10t/assi_result/HCC/open_source_model"
OUT = "/mnt/10t/holiday/hcc_paper/annotation_package_rev2"
PKG, KEY = f"{OUT}/package", f"{OUT}/key_DO_NOT_SHARE"
COHORTS = ["GSE125449", "GSE149614", "GSE156625", "GSE162616", "GSE202642", "GSE223204"]
CODE_ORDER = np.random.default_rng(20261114).permutation(len(COHORTS))
COHORT_CODE = {ds: "ABCDEF"[i] for ds, i in zip(COHORTS, CODE_ORDER)}

# Canonical lineage markers (liver / HCC scRNA-seq literature). Shown to annotators as a dot plot.
MARKERS = OrderedDict([
    ("Immune", ["PTPRC"]),
    ("T/NK/ILC", ["CD3D", "CD3E", "CD8A", "CD4", "IL7R", "FOXP3", "TRDC", "NKG7", "GNLY", "KLRD1", "KLRF1", "NCAM1"]),
    ("B/plasma", ["MS4A1", "CD79A", "CD19", "MZB1", "JCHAIN"]),
    ("Myeloid", ["LYZ", "CD14", "FCGR3A", "CD68", "CD163", "MARCO", "C1QA", "S100A8", "CSF3R",
                 "CD1C", "CLEC9A", "LILRA4", "TPSAB1", "CPA3"]),
    ("Endothelial", ["PECAM1", "VWF", "CDH5", "PLVAP", "CLEC4G", "FCN3", "STAB2", "ACKR1", "PROX1"]),
    ("Mesenchymal", ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB", "RGS5", "ACTA2", "MYH11"]),
    ("Hepatocyte/epithelial", ["ALB", "APOA1", "APOC3", "TTR", "SERPINA1", "CYP3A4", "HP", "GPC3", "AFP",
                               "EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "SOX9"]),
    ("Cycling", ["MKI67", "TOP2A"]),
    ("Erythroid/platelet", ["HBB", "HBA1", "PPBP"]),
])


def tissue_context(ds: str, cells: pd.Index, samples: pd.Series) -> pd.Series:
    """Per-cell tissue label shown to annotators."""
    if ds == "GSE125449":
        ctype = {}
        for f in glob.glob(f"{GEO}/GSE125449/matrix/*_series_matrix.txt"):
            titles, types = None, None
            for line in open(f, errors="ignore"):
                if line.startswith("!Sample_title"):
                    titles = re.findall(r'"([^"]*)"', line)
                elif line.startswith("!Sample_characteristics_ch1") and "cancer type" in line:
                    types = [v.split(": ", 1)[1] for v in re.findall(r'"([^"]*)"', line)]
                elif line.startswith("!series_matrix_table_begin"):
                    break
            ctype.update(dict(zip(titles, types)))
        short = {"Hepatocellular carcinoma": "HCC tumour", "Intrahepatic cholangiocarcinoma": "iCCA tumour"}
        return samples.map(lambda s: short.get(ctype.get(s, ""), "tumour"))
    if ds == "GSE149614":
        g = pd.read_csv(f"{BENCH}/ground_truth/GSE149614_ground_truth.tsv", sep="\t", index_col=0)
        g.index = g.index.astype(str)
        m = {"Tumor": "primary tumour", "Normal": "adjacent non-tumour", "PVTT": "portal vein tumour thrombus",
             "Lymph": "metastatic lymph node"}
        return g["site"].astype(str).reindex(cells).map(lambda v: m.get(v, v))
    if ds == "GSE156625":
        g = pd.read_csv(f"{BENCH}/ground_truth/GSE156625_enrichr_ground_truth.tsv", sep="\t", index_col=0)
        g.index = g.index.astype(str)
        return g["NormalvsTumor"].astype(str).reindex(cells).map({"T": "tumour", "N": "adjacent non-tumour"})
    if ds == "GSE162616":   # GSM4955419-21 HCC1-3 tumour, GSM4955425 tumour-adjacent liver (GEO)
        return samples.map(lambda s: "adjacent non-tumour" if s == "GSM4955425" else "tumour")
    if ds == "GSE202642":   # GEO sample titles (E-utilities): 499-505 HCC tissue, 506-509 adjacent liver
        return samples.map(lambda s: "tumour" if int(s[3:]) <= 6127505 else "adjacent non-tumour")
    if ds == "GSE223204":   # GSM6942879 non-tumour, GSM6942881 tumour (GEO)
        return samples.map({"GSM6942879": "adjacent non-tumour", "GSM6942881": "tumour"})
    raise ValueError(ds)


def other_partitions(ds: str, cells: pd.Index) -> dict:
    """Existing partitions the new clustering must be independent of (key file only)."""
    out = {}
    st = pd.read_csv(f"{BENCH}/annotation_results/{ds}_ScType_pred.csv", index_col=0)
    st.index = st.index.astype(str)
    out["ScType_cluster"] = st["cluster"].astype(str).reindex(cells)
    g = pd.read_csv(f"{BENCH}/ground_truth/{ds}_enrichr_ground_truth.tsv", sep="\t", index_col=0)
    g.index = g.index.astype(str)
    col = "leiden_enrichr" if "leiden_enrichr" in g.columns else "clusters"
    out["enrichR_GT_cluster"] = g[col].astype(str).reindex(cells)
    p = f"{BENCH}/annotation_results/{ds}_CyteType_pred.csv"
    if os.path.exists(p):
        d = pd.read_csv(p, index_col=0)
        d.index = d.index.astype(str)
        out["CyteType_label"] = d["pred_celltype"].astype(str).reindex(cells)
    else:
        d = pd.read_csv(f"{OPEN}/cytetype/CyteType_cell_prediction_summary_{ds}.csv")
        out["CyteType_label"] = pd.Series(d["Ori pred cell type"].astype(str).values,
                                          index=d["barcode"].astype(str)).reindex(cells)
    return out


def fig_b64(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def target_clusters(n: int) -> int:
    return 30 if n < 15000 else 40 if n < 80000 else 50


def build(ds: str):
    code = COHORT_CODE[ds]
    print(f"\n=== {ds} -> cohort {code}", flush=True)
    a = ad.read_h5ad(f"{IN}/{ds}/counts.h5ad")
    a.obs = pd.read_csv(f"{IN}/{ds}/cells.csv", index_col=0).reindex(a.obs_names)
    a.obs["tissue"] = tissue_context(ds, a.obs_names, a.obs["sample"].astype(str)).values
    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    sc.pp.scrublet(a, batch_key="group", random_state=SEED)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    h = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=50, random_state=SEED)
    a.obsm["X_pca"] = h.obsm["X_pca"]
    del h
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, random_state=SEED)
    target = target_clusters(a.n_obs)
    scan = []
    for res in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        sc.tl.leiden(a, resolution=res, key_added="leiden", random_state=SEED, flavor="igraph",
                     n_iterations=2, directed=False)
        k = a.obs["leiden"].nunique()
        scan.append({"Dataset": ds, "resolution": res, "n_clusters": k, "target": target})
        print(f"    resolution {res}: {k} clusters", flush=True)
        if k >= target:
            break
    res_used = scan[-1]["resolution"]
    ids = sorted(a.obs["leiden"].unique(), key=int)
    perm = np.random.default_rng(SEED + ord(code)).permutation(len(ids))
    lut = {cid: f"{code}{p + 1:02d}" for cid, p in zip(ids, perm)}
    a.obs["cluster"] = pd.Categorical(a.obs["leiden"].map(lut), categories=sorted(lut.values()))
    sc.tl.umap(a, random_state=SEED)

    # ── DE on <= 2,000 cells per cluster ──
    rng = np.random.default_rng(SEED)
    take = np.concatenate([rng.choice(np.where(a.obs["cluster"].values == c)[0],
                                      size=min(2000, int((a.obs["cluster"] == c).sum())), replace=False)
                           for c in a.obs["cluster"].cat.categories])
    detected = np.asarray((a.layers["counts"] > 0).mean(axis=0)).ravel() if "counts" in a.layers else None
    if detected is None:
        detected = np.asarray((a.X > 0).mean(axis=0)).ravel()
    de = a[np.sort(take), detected >= 0.01].copy()
    sc.tl.rank_genes_groups(de, "cluster", method="wilcoxon", pts=True, use_raw=False)

    # ── per-cluster summary ──
    rows, cards = [], []
    tissues = sorted(a.obs["tissue"].astype(str).unique())
    n_samples_total = a.obs["sample"].nunique()
    for c in a.obs["cluster"].cat.categories:
        m = (a.obs["cluster"] == c).values
        ob = a.obs[m]
        tis = ob["tissue"].astype(str).value_counts(normalize=True)
        smp = ob["sample"].astype(str).value_counts(normalize=True)
        r = sc.get.rank_genes_groups_df(de, group=c)
        r = r[(r["logfoldchanges"] > 0.5) & (r["pvals_adj"] < 0.05)].head(25)
        pts = de.uns["rank_genes_groups"]["pts"][c].reindex(r["names"]).values
        pts_rest = de.uns["rank_genes_groups"]["pts_rest"][c].reindex(r["names"]).values
        row = {"cluster": c, "n_cells": int(m.sum()), "pct_of_cohort": round(100 * m.mean(), 2),
               "tissue_composition": "; ".join(f"{t} {100 * v:.0f}%" for t, v in tis.items()),
               "n_samples_contributing": int((smp > 0.01).sum()), "n_samples_total": n_samples_total,
               "largest_sample_share_pct": round(100 * smp.iloc[0], 1),
               "median_genes": int(ob["n_genes_by_counts"].median()),
               "median_UMI": int(ob["total_counts"].median()),
               "median_pct_mt": round(float(ob["pct_counts_mt"].median()), 1),
               "median_doublet_score": round(float(ob["doublet_score"].median()), 3),
               "pct_predicted_doublet": round(100 * float(ob["predicted_doublet"].mean()), 1),
               "top_DE_genes": ", ".join(r["names"].astype(str).tolist())}
        rows.append(row)
        de_rows = "".join(
            f"<tr><td>{html.escape(str(g))}</td><td>{lf:.2f}</td><td>{100 * pi:.0f}%</td><td>{100 * po:.0f}%</td></tr>"
            for g, lf, pi, po in zip(r["names"], r["logfoldchanges"], pts, pts_rest))
        cards.append(f"""
<div class="card" id="{c}"><h3>{c}</h3>
<p><b>{row['n_cells']:,} cells</b> ({row['pct_of_cohort']}% of cohort) · tissue: {html.escape(row['tissue_composition'])}<br>
cells from {row['n_samples_contributing']} of {n_samples_total} samples (largest sample {row['largest_sample_share_pct']}%)<br>
median genes {row['median_genes']:,} · median UMI {row['median_UMI']:,} · median %mito {row['median_pct_mt']} ·
median doublet score {row['median_doublet_score']} · predicted doublets {row['pct_predicted_doublet']}%</p>
<table><tr><th>gene</th><th>log2FC</th><th>% in cluster</th><th>% in rest</th></tr>{de_rows}</table></div>""")
    summary = pd.DataFrame(rows)

    # ── figures ──
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    sc.pl.umap(a, color="cluster", legend_loc="on data", legend_fontsize=7, ax=axes[0], show=False,
               title=f"Cohort {code}: clusters", frameon=False)
    sc.pl.umap(a, color="tissue", ax=axes[1], show=False, title="tissue", frameon=False)
    sc.pl.umap(a, color="doublet_score", ax=axes[2], show=False, title="doublet score", frameon=False,
               vmax=0.5)
    umap_b64 = fig_b64(fig)
    present = OrderedDict((k, [g for g in v if g in a.var_names]) for k, v in MARKERS.items())
    present = OrderedDict((k, v) for k, v in present.items() if v)
    dp = sc.pl.dotplot(a, var_names=present, groupby="cluster", standard_scale="var", show=False,
                       return_fig=True, figsize=(26, 0.32 * len(summary) + 3))
    dp.make_figure()
    dot_b64 = fig_b64(dp.fig, dpi=100)

    tissue_counts = a.obs["tissue"].astype(str).value_counts()
    head = f"""<!doctype html><html><head><meta charset="utf-8"><title>Cohort {code} — cluster annotation</title>
<style>body{{font-family:Helvetica,Arial,sans-serif;margin:24px;max-width:1500px;color:#222}}
.card{{border:1px solid #ccc;border-radius:6px;padding:10px 14px;margin:12px 0;break-inside:avoid}}
table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ddd;padding:2px 8px;text-align:left}}
img{{max-width:100%}}h3{{margin:4px 0}}.nav a{{margin-right:8px;font-size:13px}}</style></head><body>
<h1>Cohort {code}</h1>
<p>{a.n_obs:,} cells · {n_samples_total} samples · {len(summary)} clusters ·
tissue: {html.escape('; '.join(f'{t} {n:,}' for t, n in tissue_counts.items()))}</p>
<p>Annotate every cluster in <code>annotation_sheet</code>, sheet <b>{code}</b>, following PROTOCOL.md.
Cluster codes are arbitrary; their order carries no meaning.</p>
<div class="nav">{''.join(f'<a href="#{c}">{c}</a>' for c in summary['cluster'])}</div>
<h2>UMAP</h2><img src="data:image/png;base64,{umap_b64}">
<h2>Canonical marker dot plot</h2><p>Dot size: fraction of cells expressing; colour: mean expression
scaled per gene (0–1) across clusters.</p><img src="data:image/png;base64,{dot_b64}">
<h2>Clusters</h2><p>Top differentially expressed genes: Wilcoxon test, cluster vs all other cells
(≤ 2,000 cells per cluster), log2 fold change &gt; 0.5, adjusted p &lt; 0.05, up to 25 genes.</p>"""
    os.makedirs(f"{PKG}/cohort_{code}", exist_ok=True)
    with open(f"{PKG}/cohort_{code}/report.html", "w") as fh:
        fh.write(head + "".join(cards) + "</body></html>")
    summary.to_csv(f"{PKG}/cohort_{code}/cluster_summary.csv", index=False)

    # ── key files (never shared) ──
    os.makedirs(KEY, exist_ok=True)
    pd.DataFrame({"barcode": a.obs_names, "cluster": a.obs["cluster"].astype(str).values,
                  "leiden_id": a.obs["leiden"].astype(str).values, "resolution": res_used,
                  "doublet_score": a.obs["doublet_score"].round(4).values,
                  "predicted_doublet": a.obs["predicted_doublet"].values}).to_csv(
        f"{KEY}/{ds}_cluster_assignments.csv.gz", index=False)
    indep = []
    for name, part in other_partitions(ds, a.obs_names).items():
        ok = part.notna().values
        newc = a.obs["cluster"].astype(str).values[ok]
        ct = pd.crosstab(newc, part[ok].values)
        indep.append({"Dataset": ds, "cohort_code": code, "compared_with": name,
                      "n_new_clusters": len(summary), "n_other_groups": part[ok].nunique(),
                      "ARI": round(adjusted_rand_score(newc, part[ok].values), 4),
                      # share of cells whose new cluster lies mostly inside one existing group
                      "nesting_purity": round(ct.max(axis=1).sum() / ct.values.sum(), 4)})
    print(f"    {len(summary)} clusters at resolution {res_used}; ARI vs existing partitions: "
          + ", ".join(f"{d['compared_with']} {d['ARI']}" for d in indep), flush=True)
    return summary, scan, indep


def write_sheet(summaries: dict):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    for line in ["Annotator ID (initials):", "", "Fill one row per cluster on every cohort sheet (A–F).",
                 "Follow PROTOCOL.md. Work independently; do not consult the other annotator, any tool output",
                 "or any published annotation of these data.", "",
                 "class: one of the listed options (drop-down).",
                 "confidence: 3 = clear, 2 = probable, 1 = uncertain.",
                 "key_markers: the genes that decided the call.",
                 "needs_subclustering: Y if the cluster looks like two or more distinct populations."]:
        ws.append([line])
    ws["B1"].fill = PatternFill("solid", fgColor="FFF2CC")
    classes = '"T/NK,B,Myeloid,Endothelial,Fibroblast,Hepatocyte/epithelial,Doublet/mixed,Low quality,Other"'
    for code in sorted(summaries):
        s = summaries[code]
        sh = wb.create_sheet(code)
        cols = ["cluster", "n_cells", "top_DE_genes (first 12)", "class", "other_specify", "confidence",
                "fine_label (optional)", "key_markers", "needs_subclustering (Y/N)", "notes"]
        sh.append(cols)
        for c in sh[1]:
            c.font = Font(bold=True)
        for _, r in s.iterrows():
            sh.append([r["cluster"], int(r["n_cells"]), ", ".join(str(r["top_DE_genes"]).split(", ")[:12]),
                       None, None, None, None, None, None, None])
        n = len(s) + 1
        for col, formula in [("D", classes), ("F", '"3,2,1"'), ("I", '"Y,N"')]:
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)
            sh.add_data_validation(dv)
            dv.add(f"{col}2:{col}{n}")
        for col, w in zip("ABCDEFGHIJ", [9, 9, 70, 22, 16, 11, 22, 28, 14, 30]):
            sh.column_dimensions[col].width = w
        sh.freeze_panes = "B2"
    wb.save(f"{PKG}/annotation_sheet_TEMPLATE.xlsx")


def main(cohorts):
    os.makedirs(PKG, exist_ok=True)
    os.makedirs(KEY, exist_ok=True)
    pd.DataFrame([{"cohort_code": COHORT_CODE[d], "dataset": d} for d in COHORTS]).sort_values(
        "cohort_code").to_csv(f"{KEY}/cohort_key.csv", index=False)
    scans, indeps = [], []
    for ds in cohorts:
        s, scan, indep = build(ds)
        scans += scan
        indeps += indep
    def merge(path, new):
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        if len(old):
            old = old[~old["Dataset"].isin(cohorts)]
        pd.concat([old, pd.DataFrame(new)]).to_csv(path, index=False)
    merge(f"{KEY}/resolution_scan.csv", scans)
    merge(f"{KEY}/partition_independence.csv", indeps)
    done = {COHORT_CODE[d]: pd.read_csv(f"{PKG}/cohort_{COHORT_CODE[d]}/cluster_summary.csv")
            for d in COHORTS if os.path.exists(f"{PKG}/cohort_{COHORT_CODE[d]}/cluster_summary.csv")}
    write_sheet(done)
    print("\nsheets written for cohorts:", sorted(done))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or COHORTS))
