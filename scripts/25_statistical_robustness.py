"""
Statistical robustness analysis for the HCC scRNA-seq cell-type annotation benchmark.

Three outputs requested by reviewers of a benchmark paper:
  1. Cell-level bootstrap 95% CI for Macro F1, Accuracy, Cohen's Kappa
     (B = 1000, sampling cells with replacement within each evaluation).
  2. Per-dataset label-permutation null distribution for Macro F1
     (B = 1000, shuffling y_true within the dataset, recomputing Macro F1).
     Reported: mean null Macro F1, 95th and 99th percentile, and an empirical
     p-value for the observed Macro F1 against the null.
  3. Paired-intersection means across the datasets that are evaluated by
     ALL seven tools, plus Friedman + Nemenyi post-hoc tests on Macro F1
     across that paired set.

The script reuses the per-tool, per-dataset prediction loaders defined in
02_benchmark_evaluation.py and writes four CSVs into
release/expected_results/ that can be diff-checked against the manuscript.

Usage: python 25_statistical_robustness.py [--n-bootstrap 1000] [--n-permute 1000]

Random seed is fixed at 42.
"""

from __future__ import annotations
import argparse, os, sys, importlib.util, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.stats import friedmanchisquare

try:
    import scikit_posthocs as sp_post  # pip install scikit-posthocs
    HAS_POSTHOC = True
except ImportError:
    HAS_POSTHOC = False

warnings.filterwarnings("ignore")
SEED = 42

# Reuse the data loading machinery from 02_benchmark_evaluation.py without
# triggering its __main__ block.
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "eval_module", os.path.join(HERE, "02_benchmark_evaluation.py"))
EM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EM)

OUT_DIR = "/mnt/10t/holiday/hcc_paper/release/expected_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical primitives
# ─────────────────────────────────────────────────────────────────────────────

# ── Vectorised macro F1 / accuracy / kappa from confusion-matrix counts ──
# Encoding labels as integers once and computing the confusion matrix via
# bincount is ~10x faster than repeatedly calling sklearn.metrics.f1_score
# on string arrays inside a B=1000 loop.

def _encode_labels(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cls_to_int = {c: i for i, c in enumerate(classes)}
    yt = np.fromiter((cls_to_int[v] for v in y_true), dtype=np.int32, count=len(y_true))
    yp = np.fromiter((cls_to_int[v] for v in y_pred), dtype=np.int32, count=len(y_pred))
    return yt, yp, len(classes)


def _confmat(yt: np.ndarray, yp: np.ndarray, K: int) -> np.ndarray:
    """K x K confusion matrix (row=true, col=pred) via flat bincount."""
    flat = yt.astype(np.int64) * K + yp.astype(np.int64)
    out  = np.bincount(flat, minlength=K * K).reshape(K, K)
    return out


def _macro_f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1_per = np.where(denom > 0, 2 * tp / denom, 0.0)
    # Only average over classes that appear in y_true OR y_pred
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
    if not present.any():
        return 0.0
    return float(f1_per[present].mean())


def _accuracy_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.diag(cm).sum() / total) if total else 0.0


def _kappa_from_cm(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.diag(cm).sum() / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    return float((po - pe) / (1 - pe)) if (1 - pe) > 0 else 0.0


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    yt_i, yp_i, K = _encode_labels(y_true, y_pred)
    n = len(yt_i)
    f1s   = np.empty(B); accs  = np.empty(B); kappas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        cm  = _confmat(yt_i[idx], yp_i[idx], K)
        f1s[b]    = _macro_f1_from_cm(cm)
        accs[b]   = _accuracy_from_cm(cm)
        kappas[b] = _kappa_from_cm(cm)
    return {
        "macro_f1_mean":  float(f1s.mean()),
        "macro_f1_se":    float(f1s.std(ddof=1)),
        "macro_f1_2.5":   float(np.percentile(f1s,  2.5)),
        "macro_f1_97.5":  float(np.percentile(f1s, 97.5)),
        "accuracy_mean":  float(accs.mean()),
        "accuracy_2.5":   float(np.percentile(accs,  2.5)),
        "accuracy_97.5":  float(np.percentile(accs, 97.5)),
        "kappa_mean":     float(kappas.mean()),
        "kappa_2.5":      float(np.percentile(kappas,  2.5)),
        "kappa_97.5":     float(np.percentile(kappas, 97.5)),
    }


def permutation_null(y_true: np.ndarray, y_pred: np.ndarray, B: int, seed: int) -> dict:
    """Permute y_true within the dataset; y_pred fixed. Returns chance-level
    Macro F1 distribution + empirical one-sided p-value of observed."""
    rng = np.random.default_rng(seed)
    yt_i, yp_i, K = _encode_labels(y_true, y_pred)
    cm_obs   = _confmat(yt_i, yp_i, K)
    observed = _macro_f1_from_cm(cm_obs)
    null_vals = np.empty(B)
    yt_perm = yt_i.copy()
    for b in range(B):
        rng.shuffle(yt_perm)
        cm = _confmat(yt_perm, yp_i, K)
        null_vals[b] = _macro_f1_from_cm(cm)
    pval = (np.sum(null_vals >= observed) + 1) / (B + 1)
    return {
        "observed_macro_f1": float(observed),
        "null_mean":         float(null_vals.mean()),
        "null_sd":           float(null_vals.std(ddof=1)),
        "null_95p":          float(np.percentile(null_vals, 95)),
        "null_99p":          float(np.percentile(null_vals, 99)),
        "p_value":           float(pval),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build evaluation jobs (tool, dataset, gt_type) and align (y_true, y_pred)
# ─────────────────────────────────────────────────────────────────────────────

def _align(gt: pd.Series, pred: pd.Series):
    """Inner-join on barcode index; drop Unknown in either side; return
    aligned numpy arrays. Returns (yt, yp) or (None, None) if too few cells."""
    if pred is None or gt is None:
        return None, None
    df = gt.to_frame("gt").join(pred.rename("pred"), how="inner")
    df = df[(df["gt"] != "Unknown") & (df["pred"] != "Unknown")]
    if len(df) < 50:
        return None, None
    return df["gt"].astype(str).values, df["pred"].astype(str).values


def build_jobs():
    """Return list of dicts with {tool, dataset, gt_type, yt, yp}."""
    jobs = []

    # Published GT — GSE149614 (broad6)
    gt_df = pd.read_csv(f"{EM.GT_DIR}/GSE149614_ground_truth.tsv", sep="\t", index_col=0)
    gt149 = gt_df["celltype"]
    for tool, pred in {
        "CellTypist": EM.load_celltypist("GSE149614"),
        "CellAssign":  EM.load_cellassign("GSE149614"),
        "SignacX":     EM.load_signacx_new("GSE149614"),
        "CyteType":    EM.load_cytetype_new("GSE149614"),
        "scGPT":       EM.load_scgpt("GSE149614"),
        "SingleR":     EM.load_singler("GSE149614"),
        "ScType":      EM.load_sctype("GSE149614"),
    }.items():
        yt, yp = _align(gt149, pred)
        if yt is not None:
            jobs.append({"Tool": tool, "Dataset": "GSE149614",
                         "GT_type": "published", "yt": yt, "yp": yp})

    # Published GT — GSE125449 (broad7)
    gt_df2 = pd.read_csv(f"{EM.GT_DIR}/GSE125449_ground_truth.tsv", sep="\t", index_col=0)
    gt125 = gt_df2.loc[gt_df2["celltype"] != "unclassified", "celltype"]
    sx125_path = f"{EM.ANNOT_DIR}/GSE125449_signacX_pred.csv"
    sx_pred = None
    if os.path.exists(sx125_path):
        df_sx = pd.read_csv(sx125_path, index_col=0)
        col = "pred_broad" if "pred_broad" in df_sx.columns else "pred_celltype"
        sx_pred = EM.harmonise(df_sx[col].fillna("Unknown").astype(str),
                               EM.SIGNACX_TO_BROAD7)
    for tool, pred in {
        "CellTypist": EM.load_celltypist("GSE125449"),
        "CellAssign":  EM.load_cellassign("GSE125449"),
        "SignacX":     sx_pred,
        "CyteType":    EM.load_cytetype_new("GSE125449", broad7=True),
        "scGPT":       EM.load_scgpt("GSE125449", broad7=True),
        "SingleR":     EM.load_singler("GSE125449", broad7=True),
        "ScType":      EM.load_sctype("GSE125449", broad7=True),
    }.items():
        yt, yp = _align(gt125, pred)
        if yt is not None:
            jobs.append({"Tool": tool, "Dataset": "GSE125449",
                         "GT_type": "published", "yt": yt, "yp": yp})

    # enrichR GT — GSE149614, GSE125449 (broad6)
    for ds in ["GSE149614", "GSE125449"]:
        gt = EM.load_enrichr_gt(ds)
        if ds == "GSE125449":
            sx_path = f"{EM.ANNOT_DIR}/GSE125449_signacX_pred.csv"
            sx_pred6 = None
            if os.path.exists(sx_path):
                df_sx = pd.read_csv(sx_path, index_col=0)
                col = "pred_broad" if "pred_broad" in df_sx.columns else "pred_celltype"
                sx_pred6 = EM.harmonise(df_sx[col].fillna("Unknown").astype(str),
                                        EM.SIGNACX_TO_BROAD6)
            tools = {
                "CellTypist": EM.load_celltypist_broad6(ds),
                "CellAssign":  EM.load_cellassign_broad6(ds),
                "SignacX":     sx_pred6,
                "CyteType":    EM.load_cytetype_new(ds),
                "scGPT":       EM.load_scgpt(ds),
                "SingleR":     EM.load_singler(ds),
                "ScType":      EM.load_sctype(ds),
            }
        else:
            tools = {
                "CellTypist": EM.load_celltypist(ds),
                "CellAssign":  EM.load_cellassign(ds),
                "SignacX":     EM.load_signacx_new(ds),
                "CyteType":    EM.load_cytetype_new(ds),
                "scGPT":       EM.load_scgpt(ds),
                "SingleR":     EM.load_singler(ds),
                "ScType":      EM.load_sctype(ds),
            }
        for tool, pred in tools.items():
            yt, yp = _align(gt, pred)
            if yt is not None:
                jobs.append({"Tool": tool, "Dataset": ds, "GT_type": "enrichR",
                             "yt": yt, "yp": yp})

    # enrichR GT — GSE156625 + 3 originals
    # CellAssign GSE156625 added in revision (script 27, drug_sc env Python 3.11
    # + scvi-tools 1.4.2). Previously excluded due to Python 3.13 incompatibility.
    gt156 = EM.load_enrichr_gt("GSE156625")
    for tool, pred in {
        "CellTypist": EM.load_celltypist("GSE156625"),
        "SignacX":     EM.load_signacx_new("GSE156625"),
        "CyteType":    EM.load_cytetype_new("GSE156625"),
        "CellAssign":  EM.load_cellassign("GSE156625"),
        "scGPT":       EM.load_scgpt("GSE156625"),
        "SingleR":     EM.load_singler("GSE156625"),
        "ScType":      EM.load_sctype("GSE156625"),
    }.items():
        yt, yp = _align(gt156, pred)
        if yt is not None:
            jobs.append({"Tool": tool, "Dataset": "GSE156625",
                         "GT_type": "enrichR", "yt": yt, "yp": yp})

    for ds in ["GSE162616", "GSE202642", "GSE223204"]:
        gt = EM.load_enrichr_gt(ds)
        for tool, pred in {
            "CellTypist": EM.load_celltypist(ds),
            "SignacX":     EM.load_signacx_ori(ds),
            "CyteType":    EM.load_cytetype_ori(ds),
            "CellAssign":  EM.load_cellassign_ori(ds),
            "scGPT":       EM.load_scgpt(ds),
            "SingleR":     EM.load_singler(ds),
            "ScType":      EM.load_sctype(ds),
        }.items():
            yt, yp = _align(gt, pred)
            if yt is not None:
                jobs.append({"Tool": tool, "Dataset": ds, "GT_type": "enrichR",
                             "yt": yt, "yp": yp})

    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--n-permute",   type=int, default=1000)
    args = ap.parse_args()

    print(f"[seed={SEED}] B_boot={args.n_bootstrap}, B_perm={args.n_permute}")
    print("\n>> Building evaluation jobs from prediction tables ...")
    jobs = build_jobs()
    print(f"   {len(jobs)} (tool × dataset × gt_type) combinations available\n")

    # ── B1 + B2 ─────────────────────────────────────────────────────────────
    boot_rows, perm_rows = [], []
    for i, j in enumerate(jobs, 1):
        tag = f"{j['Tool']:11s} | {j['Dataset']} | {j['GT_type']}"
        print(f"[{i:3d}/{len(jobs)}] {tag}  (n={len(j['yt'])})")
        sub_seed = SEED + i

        # Bootstrap
        bci = bootstrap_ci(j["yt"], j["yp"], args.n_bootstrap, sub_seed)
        bci.update({"Tool": j["Tool"], "Dataset": j["Dataset"],
                    "GT_type": j["GT_type"], "N_evaluated": len(j["yt"])})
        boot_rows.append(bci)

        # Permutation null
        pn = permutation_null(j["yt"], j["yp"], args.n_permute, sub_seed + 100000)
        pn.update({"Tool": j["Tool"], "Dataset": j["Dataset"],
                   "GT_type": j["GT_type"], "N_evaluated": len(j["yt"])})
        perm_rows.append(pn)

    boot_df = pd.DataFrame(boot_rows)
    perm_df = pd.DataFrame(perm_rows)

    boot_path = f"{OUT_DIR}/bootstrap_ci.csv"
    perm_path = f"{OUT_DIR}/permutation_null.csv"
    boot_df.to_csv(boot_path, index=False)
    perm_df.to_csv(perm_path, index=False)
    print(f"\n  → {boot_path}")
    print(f"  → {perm_path}")

    # ── B3a: paired-intersection means (Macro F1) ───────────────────────────
    # Restrict to enrichR GT (this is where the "front cluster" claim lives)
    e = boot_df[boot_df["GT_type"] == "enrichR"]
    pivot = e.pivot_table(index="Tool", columns="Dataset",
                          values="macro_f1_mean", aggfunc="mean")
    # Datasets evaluated by ALL tools
    all_tools = pivot.dropna(axis=1, how="any")
    print(f"\n  Datasets common to all tools: {list(all_tools.columns)}")
    paired = all_tools.mean(axis=1).round(4).to_frame("paired_mean_macro_f1")
    paired["n_datasets_used"] = all_tools.shape[1]
    paired["datasets"] = ";".join(all_tools.columns.tolist())
    paired_path = f"{OUT_DIR}/paired_intersection_means.csv"
    paired.reset_index().to_csv(paired_path, index=False)
    print(f"  → {paired_path}")
    print(paired)

    # ── B3b: Friedman + Nemenyi across paired datasets ──────────────────────
    fri_rows = []
    if all_tools.shape[1] >= 3:
        # Friedman expects (n_subjects, n_treatments) but with subjects = datasets and
        # treatments = tools. friedmanchisquare takes one array per treatment.
        groups = [all_tools.loc[t].values for t in all_tools.index]
        stat, pval = friedmanchisquare(*groups)
        fri_rows.append({"test": "Friedman", "statistic": stat,
                         "p_value": pval, "n_datasets": all_tools.shape[1],
                         "n_tools": all_tools.shape[0]})
        print(f"\n  Friedman chi^2 = {stat:.4f}, p = {pval:.4g}")

        if HAS_POSTHOC:
            data_for_nem = all_tools.T   # rows = datasets, cols = tools
            nem = sp_post.posthoc_nemenyi_friedman(data_for_nem.values)
            nem.index   = all_tools.index
            nem.columns = all_tools.index
            nem_path = f"{OUT_DIR}/friedman_nemenyi.csv"
            nem.round(4).to_csv(nem_path)
            print(f"  → Nemenyi p-values → {nem_path}")
            print(nem.round(3))
        else:
            print("  (scikit-posthocs not installed; Nemenyi skipped)")
    else:
        print("  Too few paired datasets for Friedman; skipped")

    pd.DataFrame(fri_rows).to_csv(f"{OUT_DIR}/friedman_test.csv", index=False)
    print(f"\nAll done. seed={SEED}.")


if __name__ == "__main__":
    main()
