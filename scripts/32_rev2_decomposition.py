"""
32_rev2_decomposition.py — how much each revision-2 change moves every score.

Columns per (cohort, GT, vocabulary, tool), tool set all7:
  MacroF1_reported   expected_results/benchmark_summary.csv (manuscript)
  MacroF1_rules      31 --mapping original --predictions original
                     (fixed class set, common cells, Unknown = error)
  MacroF1_mapping    31 --mapping canonical --predictions original
  MacroF1_reruns     31 --mapping canonical --predictions rev2
                     (SignacX, CellAssign, scGPT re-run; other tools unchanged)
Delta_* columns give each step's change. Writes expected_results_rev2/rev2_decomposition.csv.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
E = f"{R}/expected_results_rev2"
keys = ["Dataset", "GT", "Vocab", "Tool"]

rep = pd.read_csv(f"{R}/expected_results/benchmark_summary.csv")
rep["Tool"] = rep["Tool"].replace({"signacX": "SignacX"})
rep = rep.rename(columns={"GT_type": "GT", "Macro_F1": "MacroF1_reported",
                          "Unknown_rate": "UnknownRate_reported", "N_evaluated": "N_reported"})
rep["Vocab"] = ["broad7" if (d == "GSE125449" and g == "published") else "broad6"
                for d, g in zip(rep.Dataset, rep.GT)]


def step(path, tag, extra=()):
    s = pd.read_csv(path)
    s = s[s.ToolSet == "all7"]
    cols = keys + ["N_common", "Coverage", "MacroF1"] + list(extra)
    return s[cols].rename(columns={"N_common": f"N_common_{tag}", "Coverage": f"Coverage_{tag}",
                                   "MacroF1": f"MacroF1_{tag}"})


m = rep[keys + ["N_reported", "UnknownRate_reported", "MacroF1_reported"]]
m = m.merge(step(f"{E}/original_mapping/rev2_scores.csv", "rules"), on=keys, how="outer")
m = m.merge(step(f"{E}/original_predictions/rev2_scores.csv", "mapping"), on=keys, how="outer")
m = m.merge(step(f"{E}/rev2_scores.csv", "reruns",
                 ["MacroF1_selective", "MacroF1_CI_cell_lo", "MacroF1_CI_cell_hi",
                  "MacroF1_CI_sample_lo", "MacroF1_CI_sample_hi"]), on=keys, how="outer")
m["Delta_rules"] = m["MacroF1_rules"] - m["MacroF1_reported"]
m["Delta_mapping"] = m["MacroF1_mapping"] - m["MacroF1_rules"]
m["Delta_reruns"] = m["MacroF1_reruns"] - m["MacroF1_mapping"]
m["Delta_total"] = m["MacroF1_reruns"] - m["MacroF1_reported"]
order = keys + ["N_reported", "N_common_rules", "N_common_mapping", "N_common_reruns",
                "UnknownRate_reported", "Coverage_rules", "Coverage_mapping", "Coverage_reruns",
                "MacroF1_reported", "MacroF1_rules", "MacroF1_mapping", "MacroF1_reruns",
                "Delta_rules", "Delta_mapping", "Delta_reruns", "Delta_total", "MacroF1_selective",
                "MacroF1_CI_cell_lo", "MacroF1_CI_cell_hi", "MacroF1_CI_sample_lo", "MacroF1_CI_sample_hi"]
m = m[order].sort_values(["GT", "Dataset", "Vocab", "Tool"], ascending=[False, True, True, True])
m.to_csv(f"{E}/rev2_decomposition.csv", index=False, float_format="%.4f")
print(m[keys + ["MacroF1_reported", "MacroF1_rules", "MacroF1_mapping", "MacroF1_reruns",
                "Coverage_reruns"]].round(3).to_string(index=False))
