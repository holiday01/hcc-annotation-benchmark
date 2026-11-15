"""
31_rescore_common_cells.py — revision-2 scoring of the HCC annotation benchmark.

Addresses Reviewer 1, comment 3 ("scores are not comparable across methods").
The per-cell predictions are NOT re-generated; only the scoring changes.

Scoring rules (all evaluations)
-------------------------------
R1  Fixed class set. Macro F1 averages over the ground-truth classes that have
    at least one cell in the evaluated cell set, identical for every tool.
    (02_benchmark_evaluation.py averaged over y_true ∪ y_pred, so a tool that
    predicted a class absent from the GT received an extra F1 = 0 class.)
R2  Three prediction states per cell: a vocabulary label, "Unknown" (the tool
    ran and abstained) and "NotRun" (no output for that cell, e.g. a sample
    the tool was not run on, or the SignacX 15,000-cell subsample on
    GSE156625). Labels outside the vocabulary are counted as "Unknown" and
    logged in rev2_label_audit.csv.
R3  Common input cells. For a tool set S the evaluated cells are the
    GT-labelled cells on which every tool in S has output (not "NotRun").
    Every tool in S is scored on exactly the same cells.
R4  Coverage-aware primary metrics. On the common cells "Unknown" counts as
    an error (accuracy and per-class recall); coverage = fraction of the
    common cells that received a vocabulary label.
R5  Selective metrics on identical cells. On the subset of common cells
    that every tool in S labelled (no tool abstained), accuracy and macro F1
    are reported; the class set is the GT classes present in that subset
    (identical for every tool in S, listed in Classes_selective).
R6  One label hierarchy for the cross-tool comparison: broad-6
    (T/NK, Myeloid, Hepatocyte, B, Endothelial, Fibroblast). GSE125449
    published labels are additionally scored in broad-7 (supplementary).
R7  One fine-to-broad mapping (--mapping canonical, default): every tool's
    raw output label and every enrichR term go through _canonical_broad.py
    (rules in its docstring); the full raw -> broad table is written to
    rev2_label_mapping.csv. --mapping original reproduces the per-tool
    mappings of 02_benchmark_evaluation.py, so the effect of R1-R5 and of
    R7 can be separated. scGPT outputs only broad labels and is identical
    under both.
R8  --predictions rev2 (default) takes SignacX, CellAssign and scGPT from the
    revision-2 re-runs (annotation_results_rev2/: every cell, raw counts,
    one procedure per tool, scGPT reference without benchmark cohorts);
    --predictions original keeps the published per-cell predictions.
    Output folders: expected_results_rev2/ (canonical + re-runs),
    original_predictions/ (canonical + published predictions),
    original_mapping/ (published mapping + published predictions).

Tool sets
---------
  all7    every tool with any output on the cohort (SignacX has none on
          GSE223204, so all7 there has six tools)
  core    tools that ran on >= 99% of the GT-labelled cells of the cohort
          (gives a near-complete cell set for the comparison of those tools)
  noSignacX   (enrichR GT) the original tools except SignacX (kept for comparison
              with the original submission, in which SignacX had no output
              on GSE223204)
  noncircular (enrichR GT) every configuration except ScType and CyteType, i.e.
              tools whose references are independent of CellMarker/PanglaoDB
              and of the GT cluster partition (SignacX included: after the
              revision-2 re-run it has output on every cohort)

Uncertainty
-----------
  Cell-level bootstrap (B = 1000) and, when the evaluated cells come from
  >= 5 samples/patients, sample-level bootstrap (B = 1000; samples resampled
  with replacement, per-sample confusion matrices summed). The resampling unit
  is the patient where patients are known: GSE125449 (one sample per patient),
  GSE156625 (patientno) and GSE149614 (the sites of each patient, e.g. HCC07T/N/P,
  are kept together; 10 patients). Paired
  differences between tools use the same resamples for both tools.
  Label-permutation null (B = 1000) per (evaluation, tool).
  Friedman + Nemenyi across cohorts on the primary macro F1.

Usage: python 31_rescore_common_cells.py [--B 1000] [--out DIR]
Seed 42.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

warnings.filterwarnings("ignore")
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "02_benchmark_evaluation.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
sys.path.insert(0, HERE)
import _broad_label as bl          # noqa: E402  (REJECT_KEYWORDS of the enrichR GT)
import _canonical_broad as cb      # noqa: E402

BROAD6 =["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
BROAD7 = ["T cell", "TAM", "B cell", "Malignant cell", "HPC-like", "TEC", "CAF"]
ORIGINAL7 = ["CellTypist", "SingleR", "ScType", "CellAssign", "SignacX", "CyteType", "scGPT"]
# Added in revision 2 (Reviewer 1, comment 1); all trained on the shared reference of
# 42_build_shared_reference.py (43_run_reference_tools_rev2.R). Included when present.
ADDED = ["SeuratLT", "SingleRdisco", "scmap", "scPred", "SingleCellNet"]
TOOLS = ORIGINAL7 + ADDED
# Tools whose enrichR-GT score is not independent: shared marker databases and,
# on the validation cohorts, the same cluster partition as the GT.
CIRCULAR_ENRICHR = {"ScType", "CyteType"}
COHORTS = ["GSE125449", "GSE149614", "GSE156625", "GSE162616", "GSE202642", "GSE223204"]
H5AD_VAL = "/mnt/10t/assi_result/HCC/open_source_model/cytetype/adata_CyteType_{}.h5ad"


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _clean_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index.astype(str)
    return s[~s.index.duplicated()]


def _signacx_125449(vocab_map):
    d = pd.read_csv(f"{ev.ANNOT_DIR}/GSE125449_signacX_pred.csv", index_col=0)
    col = "pred_broad" if "pred_broad" in d.columns else "pred_celltype"
    return ev.harmonise(d[col].fillna("Unknown").astype(str), vocab_map)


def load_tools(ds: str, vocab: str) -> tuple[dict, dict]:
    """Return ({tool: Series of labels}, {tool: Index of NotRun cells})."""
    not_run = {}
    if vocab == "broad7":  # GSE125449 published only
        tools = {
            "CellTypist": ev.load_celltypist(ds),
            "CellAssign": ev.load_cellassign(ds),
            "SignacX": _signacx_125449(ev.SIGNACX_TO_BROAD7),
            "CyteType": ev.load_cytetype_new(ds, broad7=True),
            "scGPT": ev.load_scgpt(ds, broad7=True),
            "SingleR": ev.load_singler(ds, broad7=True),
            "ScType": ev.load_sctype(ds, broad7=True),
        }
    elif ds == "GSE125449":
        tools = {
            "CellTypist": ev.load_celltypist_broad6(ds),
            "CellAssign": ev.load_cellassign_broad6(ds),
            "SignacX": _signacx_125449(ev.SIGNACX_TO_BROAD6),
            "CyteType": ev.load_cytetype_new(ds),
            "scGPT": ev.load_scgpt(ds),
            "SingleR": ev.load_singler(ds),
            "ScType": ev.load_sctype(ds),
        }
    elif ds in ("GSE149614", "GSE156625"):
        tools = {
            "CellTypist": ev.load_celltypist(ds),
            "CellAssign": ev.load_cellassign(ds),
            "SignacX": ev.load_signacx_new(ds),
            "CyteType": ev.load_cytetype_new(ds),
            "scGPT": ev.load_scgpt(ds),
            "SingleR": ev.load_singler(ds),
            "ScType": ev.load_sctype(ds),
        }
        if ds == "GSE156625":
            # 19_run_SignacX_GSE156625.R ran SignacX on a 15,000-cell subsample and
            # wrote "Unclassified" for the remaining cells: those are NotRun.
            raw = pd.read_csv(f"{ev.ANNOT_DIR}/GSE156625_signacX_pred.csv", index_col=0)
            raw.index = raw.index.astype(str)
            not_run["SignacX"] = raw.index[raw["pred_broad"].astype(str) == "Unclassified"]
    else:
        tools = {
            "CellTypist": ev.load_celltypist(ds),
            "CellAssign": ev.load_cellassign_ori(ds),
            "SignacX": ev.load_signacx_ori(ds),
            "CyteType": ev.load_cytetype_ori(ds),
            "scGPT": ev.load_scgpt(ds),
            "SingleR": ev.load_singler(ds),
            "ScType": ev.load_sctype(ds),
        }
    return {k: _clean_index(v) for k, v in tools.items() if v is not None}, not_run


def _read_ann(ds, tool, col):
    d = pd.read_csv(f"{ev.ANNOT_DIR}/{ds}_{tool}_pred.csv", index_col=0)
    return d[col].fillna("Unknown").astype(str)


def _read_ori(sub, fname, col):
    d = pd.read_csv(f"{ev.TOOLS_DIR}/{sub}/{fname}")
    return pd.Series(d[col].fillna("Unknown").astype(str).values, index=d["barcode"].astype(str))


REV2_DIR = "/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2"


def _signacx_fine_or_coarse(fine: pd.Series, coarse: pd.Series) -> pd.Series:
    # finest SignacX level that resolves to a broad-6 class, else the coarse label
    use_fine = fine.map(cb.broad6) != "Unknown"
    return fine.where(use_fine, coarse)


def load_raw_tools(ds: str, vocab: str = "broad6", pred_set: str = "original") -> tuple[dict, dict]:
    """Raw (pre-harmonisation) labels of every tool: {tool: Series}, {tool: NotRun index}.

    pred_set="rev2" replaces SignacX, CellAssign and scGPT by the re-runs
    (34_rerun_SignacX_rev2.R, 36_rerun_CellAssign_rev2.py,
    35_scgpt_leave_cohort_out.py) in annotation_results_rev2/.
    """
    if pred_set == "rev2":
        raw, not_run = load_raw_tools(ds, vocab, "original")
        not_run = {}
        d = pd.read_csv(f"{REV2_DIR}/{ds}_SignacX_pred.csv")
        d.index = d["barcode"].astype(str)
        raw["SignacX"] = _signacx_fine_or_coarse(d["pred_celltype"].fillna("Unknown").astype(str),
                                                 d["pred_broad"].fillna("Unknown").astype(str))
        ca = "GSE125449_CellAssign7_pred.csv" if (ds == "GSE125449" and vocab == "broad7") \
            else f"{ds}_CellAssign_pred.csv"
        d = pd.read_csv(f"{REV2_DIR}/{ca}")
        raw["CellAssign"] = pd.Series(d["pred_celltype"].astype(str).values, index=d["barcode"].astype(str))
        d = pd.read_csv(f"{REV2_DIR}/{ds}_scGPT_pred.csv", index_col=0)
        raw["scGPT"] = d["pred_celltype"].fillna("Unknown").astype(str)
        for t in ADDED:
            p = f"{REV2_DIR}/{ds}_{t}_pred.csv"
            if os.path.exists(p):
                d = pd.read_csv(p)
                raw[t] = pd.Series(d["pred_celltype"].fillna("Unknown").astype(str).values,
                                   index=d["barcode"].astype(str))
        return {k: _clean_index(v) for k, v in raw.items()}, not_run
    not_run = {}
    raw = {
        "CellTypist": _read_ann(ds, "CellTypist", "pred_raw"),
        "SingleR": _read_ann(ds, "SingleR", "pred_pruned"),
        "ScType": _read_ann(ds, "ScType", "sctype_label"),
        "scGPT": _read_ann(ds, "scGPT", "pred_celltype"),       # broad labels only
    }
    if ds in ("GSE125449", "GSE149614", "GSE156625"):
        raw["CellAssign"] = _read_ann(ds, "CellAssign", "pred_celltype")
        raw["CyteType"] = _read_ann(ds, "CyteType", "pred_celltype")
        d = pd.read_csv(f"{ev.ANNOT_DIR}/{ds}_signacX_pred.csv", index_col=0)
        fine = d["pred_celltype"].fillna("Unknown").astype(str)
        coarse = d["pred_broad"].fillna("Unknown").astype(str) if "pred_broad" in d.columns else fine
        raw["SignacX"] = _signacx_fine_or_coarse(fine, coarse)
        if ds == "GSE156625":
            not_run["SignacX"] = d.index.astype(str)[coarse.values == "Unclassified"]
    else:
        raw["CellAssign"] = _read_ori("cellassign", f"{ds}_CellAssign_cell_prediction_summary.csv", "Pred cell type")
        raw["CyteType"] = _read_ori("cytetype", f"CyteType_cell_prediction_summary_{ds}.csv", "Ori pred cell type")
        raw["SignacX"] = _read_ori("signacX", f"signacX_cell_prediction_summar_{ds}.csv", "Pred.cell.type")
    return {k: _clean_index(v) for k, v in raw.items()}, not_run


def map_raw(tool: str, raw: pd.Series, vocab: str) -> pd.Series:
    if tool == "scGPT":
        return raw if vocab == "broad6" else raw.map(cb.B6_TO_B7).fillna("Unknown")
    fn = cb.broad6 if vocab == "broad6" else cb.broad7
    lut = {u: fn(u) for u in raw.unique()}
    return raw.map(lut)


def load_gt_and_samples(ds: str, gt_type: str, vocab: str, mapping: str = "canonical") -> tuple[pd.Series, pd.Series]:
    if gt_type == "published":
        # GSE156625: authors' CellType from GSE156625_HCCF_integrated.RData
        # (38_extract_GSE156625_author_labels.R); Erythroids/Others -> Unknown.
        fname = "GSE156625_published_ground_truth.tsv" if ds == "GSE156625" else f"{ds}_ground_truth.tsv"
        d = pd.read_csv(f"{ev.GT_DIR}/{fname}", sep="\t", index_col=0)
        d.index = d.index.astype(str)
        gt = d["celltype"].astype(str)
        if ds == "GSE125449":
            gt = gt[gt != "unclassified"]
            if vocab == "broad6":
                gt = gt.map(ev.BROAD7_TO_BROAD6)
        if ds == "GSE156625":
            e = pd.read_csv(f"{ev.GT_DIR}/GSE156625_enrichr_ground_truth.tsv", sep="\t", index_col=0)
            e.index = e.index.astype(str)
            samples = e["patientno"].astype(str)
        else:
            samples = d["sample"].astype(str)
    else:
        d = pd.read_csv(f"{ev.GT_DIR}/{ds}_enrichr_ground_truth.tsv", sep="\t", index_col=0)
        d.index = d.index.astype(str)
        if mapping == "canonical":
            fine = d["ground_truth_enrichr_fine"].astype(str)
            lut = {u: cb.gt_broad6(u, bl.REJECT_KEYWORDS) for u in fine.unique()}
            gt = fine.map(lut)
        else:
            gt = ev.clean_enrichr_broad(d["ground_truth_enrichr_broad"].astype(str))
        if ds in ("GSE125449", "GSE149614"):
            p = pd.read_csv(f"{ev.GT_DIR}/{ds}_ground_truth.tsv", sep="\t", index_col=0)
            p.index = p.index.astype(str)
            samples = p["sample"].astype(str)
        elif ds == "GSE156625":
            samples = d["patientno"].astype(str)
        else:
            import anndata as ad
            a = ad.read_h5ad(H5AD_VAL.format(ds), backed="r")
            samples = pd.Series(a.obs["sample"].astype(str).values, index=a.obs_names.astype(str))
    if ds == "GSE149614":
        # 21 samples from 10 patients (tumour, non-tumour, portal-vein and lymph-node sites, e.g. HCC07T,
        # HCC07N, HCC07P): the resampling unit is the patient, so all sites of a patient stay together.
        samples = samples.str.extract(r"^(HCC\d+)", expand=False).fillna(samples)
    gt = gt[gt.notna() & (gt != "Unknown")]
    return _clean_index(gt), _clean_index(samples)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion-matrix arithmetic. Columns: K vocabulary classes + "Unknown".
# Rows: K classes (truth). Classes absent from the evaluated GT are dropped
# from the macro average but still count as wrong predictions.
# ─────────────────────────────────────────────────────────────────────────────

def encode(gt: pd.Series, pred: pd.Series, vocab: list[str]):
    lab = {c: i for i, c in enumerate(vocab)}
    unk = len(vocab)
    yt = gt.map(lab).to_numpy(dtype=np.int64)
    yp = pred.map(lambda v: lab.get(v, unk)).to_numpy(dtype=np.int64)
    return yt, yp


def confmat(yt, yp, K):
    return np.bincount(yt * (K + 1) + yp, minlength=K * (K + 1)).reshape(K, K + 1)


def metrics_from_cm(cm: np.ndarray, present: np.ndarray) -> dict:
    K = cm.shape[0]
    tp = np.diag(cm[:, :K]).astype(float)
    pred_tot = cm[:, :K].sum(axis=0).astype(float)
    true_tot = cm.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        rec = np.where(true_tot > 0, tp / true_tot, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    n = cm.sum()
    acc = tp.sum() / n if n else np.nan
    # Cohen's kappa on a square (K+1) table: truth never equals "Unknown".
    sq = np.zeros((K + 1, K + 1))
    sq[:K, :] = cm
    po = np.trace(sq) / n
    pe = (sq.sum(axis=1) * sq.sum(axis=0)).sum() / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return {"acc": acc, "macro_f1": f1[present].mean(), "kappa": kappa,
            "coverage": 1 - cm[:, K].sum() / n, "f1": f1}


def per_sample_cms(yt, yp, s_codes, n_s, K):
    out = np.zeros((n_s, K, K + 1), dtype=np.int64)
    flat = s_codes * (K * (K + 1)) + yt * (K + 1) + yp
    out[:] = np.bincount(flat, minlength=n_s * K * (K + 1)).reshape(n_s, K, K + 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# One evaluation = (cohort, GT, vocabulary, tool set)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(ds, gt_type, vocab_name, toolset_name, gt, samples, preds, B, rng_seed):
    vocab = BROAD6 if vocab_name == "broad6" else BROAD7
    K = len(vocab)
    tools = list(preds)
    ran_all = np.logical_and.reduce([preds[t] != "NotRun" for t in tools])
    cells = gt.index[ran_all]
    g = gt.loc[cells]
    present_labels = [c for c in vocab if (g == c).any()]
    present = np.array([c in present_labels for c in vocab])
    yt, _ = encode(g, g, vocab)
    smp = samples.reindex(cells).fillna("NA").to_numpy()
    s_uni, s_codes = np.unique(smp, return_inverse=True)
    n_s = len(s_uni)

    called_all = np.logical_and.reduce(
        [preds[t].loc[cells].isin(vocab).to_numpy() for t in tools])
    # Selective metrics: the class set is the GT classes present among the cells
    # every tool labelled (can be smaller than the common-cell class set).
    present_sel = present & (np.bincount(yt[called_all], minlength=K) > 0)
    sel_labels = [c for c, p in zip(vocab, present_sel) if p]

    rows, enc = [], {}
    for t in tools:
        _, yp = encode(g, preds[t].loc[cells], vocab)
        enc[t] = yp
        m = metrics_from_cm(confmat(yt, yp, K), present)
        if called_all.sum():
            ms = metrics_from_cm(confmat(yt[called_all], yp[called_all], K), present_sel)
        else:
            ms = {"acc": np.nan, "macro_f1": np.nan}
        r = {"Dataset": ds, "GT": gt_type, "Vocab": vocab_name, "ToolSet": toolset_name,
             "Tool": t, "Circular_under_enrichR": (gt_type == "enrichR" and t in CIRCULAR_ENRICHR),
             "N_GT_labelled": len(gt), "N_common": len(cells),
             "Frac_common": round(len(cells) / len(gt), 4),
             "N_samples": n_s, "Classes": "|".join(present_labels), "n_classes": int(present.sum()),
             "Coverage": m["coverage"], "Accuracy": m["acc"], "MacroF1": m["macro_f1"],
             "Kappa": m["kappa"],
             "N_called_by_all": int(called_all.sum()),
             "Classes_selective": "|".join(sel_labels),
             "Accuracy_selective": ms["acc"], "MacroF1_selective": ms["macro_f1"]}
        for c, v in zip(vocab, m["f1"]):
            r[f"F1_{c}"] = v if c in present_labels else np.nan
        rows.append(r)

    # ── bootstrap (cell-level and sample-level), paired across tools ──
    rng = np.random.default_rng(rng_seed)
    N = len(cells)
    boot_cell = {t: np.empty(B) for t in tools}
    boot_acc = {t: np.empty(B) for t in tools}
    for b in range(B):
        idx = rng.integers(0, N, N)
        ytb = yt[idx]
        pres_b = present & (np.bincount(ytb, minlength=K) > 0)
        for t in tools:
            mb = metrics_from_cm(confmat(ytb, enc[t][idx], K), pres_b)
            boot_cell[t][b] = mb["macro_f1"]
            boot_acc[t][b] = mb["acc"]
    boot_smp = None
    if n_s >= 5:
        psc = {t: per_sample_cms(yt, enc[t], s_codes, n_s, K) for t in tools}
        boot_smp = {t: np.empty(B) for t in tools}
        for b in range(B):
            sidx = rng.integers(0, n_s, n_s)
            for t in tools:
                cm = psc[t][sidx].sum(axis=0)
                pres_b = present & (cm.sum(axis=1) > 0)
                boot_smp[t][b] = metrics_from_cm(cm, pres_b)["macro_f1"]

    # ── permutation null: shuffle truth within the evaluated cells ──
    perm_rows = []
    rngp = np.random.default_rng(rng_seed + 1)
    perms = [rngp.permutation(N) for _ in range(B)]
    for r in rows:
        t = r["Tool"]
        null = np.array([metrics_from_cm(confmat(yt[p], enc[t], K), present)["macro_f1"] for p in perms])
        r["Null_MacroF1_mean"] = null.mean()
        r["Null_MacroF1_p99"] = np.quantile(null, 0.99)
        r["Perm_p"] = (1 + (null >= r["MacroF1"]).sum()) / (B + 1)
        r["MacroF1_CI_cell_lo"], r["MacroF1_CI_cell_hi"] = np.quantile(boot_cell[t], [0.025, 0.975])
        r["Acc_CI_cell_lo"], r["Acc_CI_cell_hi"] = np.quantile(boot_acc[t], [0.025, 0.975])
        if boot_smp is not None:
            r["MacroF1_CI_sample_lo"], r["MacroF1_CI_sample_hi"] = np.quantile(boot_smp[t], [0.025, 0.975])
        else:
            r["MacroF1_CI_sample_lo"] = r["MacroF1_CI_sample_hi"] = np.nan

    pair_rows = []
    for a, b_ in itertools.combinations(tools, 2):
        obs = rows[tools.index(a)]["MacroF1"] - rows[tools.index(b_)]["MacroF1"]
        dc = boot_cell[a] - boot_cell[b_]
        pr = {"Dataset": ds, "GT": gt_type, "Vocab": vocab_name, "ToolSet": toolset_name,
              "Tool_A": a, "Tool_B": b_, "Diff_MacroF1": obs,
              "CI_cell_lo": np.quantile(dc, 0.025), "CI_cell_hi": np.quantile(dc, 0.975)}
        if boot_smp is not None:
            ds_ = boot_smp[a] - boot_smp[b_]
            pr["CI_sample_lo"], pr["CI_sample_hi"] = np.quantile(ds_, [0.025, 0.975])
        else:
            pr["CI_sample_lo"] = pr["CI_sample_hi"] = np.nan
        pair_rows.append(pr)
    return rows, pair_rows


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--mapping", choices=["canonical", "original"], default="canonical")
    ap.add_argument("--predictions", choices=["rev2", "original"], default="rev2",
                    help="rev2: SignacX, CellAssign, scGPT from the revision-2 re-runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mapping == "original" and args.predictions == "rev2":
        ap.error("--mapping original only applies to the original predictions")
    if args.out is None:
        base = os.path.join(os.path.dirname(HERE), "expected_results_rev2")
        args.out = (os.path.join(base, "original_mapping") if args.mapping == "original"
                    else os.path.join(base, "original_predictions") if args.predictions == "original"
                    else base)
    os.makedirs(args.out, exist_ok=True)
    print(f"mapping = {args.mapping}; predictions = {args.predictions}; output = {args.out}")
    map_rows, map_seen = [], set()

    evals = [("GSE125449", "published", "broad6"), ("GSE149614", "published", "broad6"),
             ("GSE156625", "published", "broad6"), ("GSE125449", "published", "broad7")]
    evals += [(ds, "enrichR", "broad6") for ds in COHORTS]

    all_rows, all_pairs, audit, cellsets = [], [], [], []
    for k, (ds, gtt, voc) in enumerate(evals):
        print(f"\n=== {ds} | {gtt} | {voc}", flush=True)
        gt, samples = load_gt_and_samples(ds, gtt, voc, args.mapping)
        if args.mapping == "canonical":
            raw_lab, not_run = load_raw_tools(ds, voc, args.predictions)
            raw = {}
            for t, s in raw_lab.items():
                raw[t] = map_raw(t, s, voc)
                if (ds, voc, t) not in map_seen:
                    map_seen.add((ds, voc, t))
                    vc = pd.DataFrame({"raw": s.values, "mapped": raw[t].values}).value_counts()
                    for (rl, ml), n in vc.items():
                        map_rows.append({"Dataset": ds, "Vocab": voc, "Tool": t, "Raw_label": rl,
                                         "Mapped": ml, "Rule": cb.map_label(rl)[1] if t != "scGPT" else "broad",
                                         "N_cells": int(n)})
        else:
            raw, not_run = load_tools(ds, voc)
        vocab = BROAD6 if voc == "broad6" else BROAD7
        preds = {}
        for t in TOOLS:
            if t not in raw:
                continue
            p = raw[t].reindex(gt.index)
            n_missing = int(p.isna().sum())
            p = p.fillna("NotRun").astype(str)
            if t in not_run:
                p[p.index.isin(not_run[t])] = "NotRun"
            oov = p[~p.isin(vocab + ["Unknown", "NotRun"])]
            audit.append({"Dataset": ds, "GT": gtt, "Vocab": voc, "Tool": t,
                          "N_GT_labelled": len(gt), "N_NotRun": int((p == "NotRun").sum()),
                          "N_NotRun_missing_rows": n_missing,
                          "N_Unknown": int((p == "Unknown").sum()),
                          "N_out_of_vocab_as_Unknown": len(oov),
                          "out_of_vocab_labels": "|".join(f"{a}:{b}" for a, b in oov.value_counts().head(8).items())})
            p[~p.isin(vocab + ["NotRun"])] = "Unknown"
            if (p != "NotRun").sum() == 0:
                print(f"  {t}: no output on this cohort; excluded")
                continue
            preds[t] = p
        ran_frac = {t: (p != "NotRun").mean() for t, p in preds.items()}
        orig = [t for t in preds if t in ORIGINAL7]
        toolsets = {"all7": orig}                      # the seven tools of the original submission
        if len(preds) > len(orig):
            toolsets["all"] = list(preds)              # every tool with output, incl. revision-2 additions
        core = [t for t in orig if ran_frac[t] >= 0.99]
        if set(core) != set(orig):
            toolsets["core"] = core
        if gtt == "enrichR":
            toolsets["noSignacX"] = [t for t in orig if t != "SignacX"]
            toolsets["noncircular"] = [t for t in preds if t not in CIRCULAR_ENRICHR]
        for name, ts in toolsets.items():
            rows, pairs = evaluate(ds, gtt, voc, name, gt, samples, {t: preds[t] for t in ts},
                                   args.B, SEED + k)
            all_rows += rows
            all_pairs += pairs
            r0 = rows[0]
            cellsets.append({"Dataset": ds, "GT": gtt, "Vocab": voc, "ToolSet": name,
                             "Tools": "|".join(ts), "N_GT_labelled": r0["N_GT_labelled"],
                             "N_common": r0["N_common"], "Frac_common": r0["Frac_common"],
                             "N_called_by_all": r0["N_called_by_all"], "N_samples": r0["N_samples"],
                             "Classes": r0["Classes"]})
            for r in rows:
                print(f"  [{name}] {r['Tool']:10s} N={r['N_common']:6d} cov={r['Coverage']:.3f} "
                      f"acc={r['Accuracy']:.3f} mF1={r['MacroF1']:.3f} "
                      f"(sel {r['MacroF1_selective']:.3f})", flush=True)

    scores = pd.DataFrame(all_rows)
    scores.to_csv(f"{args.out}/rev2_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(all_pairs).to_csv(f"{args.out}/rev2_pairwise_bootstrap.csv", index=False, float_format="%.4f")
    pd.DataFrame(audit).to_csv(f"{args.out}/rev2_label_audit.csv", index=False)
    pd.DataFrame(cellsets).to_csv(f"{args.out}/rev2_cellsets.csv", index=False)
    if map_rows:
        pd.DataFrame(map_rows).to_csv(f"{args.out}/rev2_label_mapping.csv", index=False)

    # ── Friedman / Nemenyi across enrichR cohorts ──
    import scikit_posthocs as sp
    e = scores[scores.GT == "enrichR"]
    fr_rows = []

    def friedman(name, ds_list, tool_list, table, toolset):
        # Each cohort contributes the scores computed on that tool set's own common cells.
        sub = table[table.Dataset.isin(ds_list) & table.Tool.isin(tool_list) & (table.ToolSet == toolset)]
        mat = sub.pivot(index="Dataset", columns="Tool", values="MacroF1").loc[ds_list, tool_list]
        if mat.isna().any().any():
            print(f"  {name}: missing cells, skipped"); return
        stat, p = friedmanchisquare(*[mat[c].values for c in mat.columns])
        ranks = mat.rank(axis=1, ascending=False).mean()
        fr_rows.append({"Analysis": name, "n_datasets": len(ds_list), "k_tools": len(tool_list),
                        "chi2": stat, "p": p,
                        "mean_rank": "|".join(f"{t}:{ranks[t]:.2f}" for t in ranks.sort_values().index),
                        "mean_MacroF1": "|".join(f"{t}:{mat[t].mean():.3f}" for t in mat.mean().sort_values(ascending=False).index)})
        nem = sp.posthoc_nemenyi_friedman(mat.values)
        nem.index = nem.columns = tool_list
        nem.to_csv(f"{args.out}/rev2_nemenyi_{name}.csv", float_format="%.4f")
        mat.to_csv(f"{args.out}/rev2_friedman_input_{name}.csv", float_format="%.4f")

    six_no_signacx = [t for t in ORIGINAL7 if t != "SignacX"]
    friedman("A_6cohorts_6tools_noSignacX", COHORTS, six_no_signacx, e, "noSignacX")
    friedman("B_5cohorts_7tools", [c for c in COHORTS if c != "GSE223204"], ORIGINAL7, e, "all7")
    nc = [t for t in TOOLS if t not in CIRCULAR_ENRICHR
          and set(COHORTS) <= set(e[(e.Tool == t) & (e.ToolSet == "noncircular")].Dataset)]
    friedman("C_6cohorts_noncircular", COHORTS, nc, e, "noncircular")
    # Primary analysis of revision 2: investigator-curated (published) labels, broad-6.
    pub = scores[(scores.GT == "published") & (scores.Vocab == "broad6")]
    pub_ds = [d for d in ["GSE125449", "GSE149614", "GSE156625"] if d in set(pub.Dataset)]
    friedman("D_published_3cohorts_7tools", pub_ds, ORIGINAL7, pub, "all7")
    if (pub.ToolSet == "all").any():
        allt = [t for t in TOOLS if set(pub_ds) <= set(pub[(pub.Tool == t) & (pub.ToolSet == "all")].Dataset)]
        friedman(f"E_published_3cohorts_{len(allt)}tools", pub_ds, allt, pub, "all")
    pd.DataFrame(fr_rows).to_csv(f"{args.out}/rev2_friedman.csv", index=False, float_format="%.4g")
    print("\n", pd.DataFrame(fr_rows).to_string())

    # ── side-by-side with the originally reported numbers ──
    orig = pd.read_csv(os.path.join(os.path.dirname(HERE), "expected_results", "benchmark_summary.csv"))
    orig["Tool"] = orig["Tool"].replace({"signacX": "SignacX"})
    orig = orig.rename(columns={"GT_type": "GT", "Macro_F1": "MacroF1_original",
                                "Accuracy": "Accuracy_original", "N_evaluated": "N_evaluated_original",
                                "Unknown_rate": "Unknown_rate_original"})
    new = scores[scores.ToolSet == "all7"].copy()
    new = new[~((new.Dataset == "GSE125449") & (new.GT == "published") & (new.Vocab == "broad6"))]
    cmp_ = new.merge(orig[["Tool", "Dataset", "GT", "N_evaluated_original", "Unknown_rate_original",
                           "Accuracy_original", "MacroF1_original"]],
                     on=["Tool", "Dataset", "GT"], how="left")
    cmp_["Delta_MacroF1"] = cmp_["MacroF1"] - cmp_["MacroF1_original"]
    cmp_[["Dataset", "GT", "Vocab", "Tool", "N_evaluated_original", "N_common", "Unknown_rate_original",
          "Coverage", "MacroF1_original", "MacroF1", "Delta_MacroF1", "MacroF1_selective",
          "Accuracy_original", "Accuracy"]].to_csv(f"{args.out}/rev2_vs_original.csv", index=False,
                                                   float_format="%.4f")
    print("\nwritten to", args.out)


if __name__ == "__main__":
    sys.exit(main())
