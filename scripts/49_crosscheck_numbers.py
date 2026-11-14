"""49_crosscheck_numbers.py — list every number in the revised texts that does not occur in the results files.

Collects all numeric values in expected_results_rev2/*.csv, rev2_manuscript_numbers.json and the
supplementary inputs, in the roundings used in the text (3/2/1 decimals, percentages, integers), and
reports each number token of the documents that matches none of them, with context.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

E = "/mnt/10t/holiday/hcc_paper/release/expected_results_rev2"
R = "/mnt/10t/holiday/hcc_paper/rev2"
EXTRA = ["/mnt/10t/assi_result/HCC/benchmark_datasets/annotation_results_rev2/scgpt_reference_exclusion.csv"]

vals = set()


def add(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return
    if not np.isfinite(v):
        return
    for x in (v, abs(v)):
        vals.add(f"{x:.3f}"); vals.add(f"{x:.2f}"); vals.add(f"{x:.1f}"); vals.add(f"{x:.0f}")
        vals.add(f"{x*100:.1f}"); vals.add(f"{x*100:.2f}"); vals.add(f"{x*100:.0f}")
        vals.add(f"{round(x, 3)+0:.3f}")


for f in glob.glob(f"{E}/*.csv") + glob.glob(f"{E}/cnv/*.csv") + EXTRA:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    for c in df.columns:
        col = pd.to_numeric(df[c], errors="coerce")
        for v in col.dropna().unique():
            add(v)
        if df[c].dtype == object:   # numbers embedded in strings, e.g. "ScType:0.965"
            for sv in df[c].dropna().astype(str).unique()[:5000]:
                for t in re.findall(r"\d+\.\d+|\d+", sv):
                    add(t)
js = json.load(open(f"{E}/rev2_manuscript_numbers.json"))


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    else:
        add(o)
        if isinstance(o, str):
            for t in re.findall(r"\d+\.\d+|\d+", o):
                add(t)


walk(js)
md = open(f"{E}/rev2_manuscript_numbers.md").read()
for t in re.findall(r"\d[\d,]*\.?\d*", md):
    add(t.replace(",", ""))

# numbers that are design constants, versions, dates, settings — not results
ALLOW = set("""0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
41 42 43 44 45 46 47 48 49 50 54 55 66 75 100 120 200 500 512 1000 1200 2000 3000 5000 10000 60697 2019 2020 2021 2022
2023 2024 2025 2026 0.05 0.3 0.7 0.55 1.5 0.01 0.2 0.8 0.03 1.000""".split())


def tokens(text):
    text = re.sub(r"\\url\{[^}]*\}|\\href\{[^}]*\}\{[^}]*\}|\\cite\{[^}]*\}|\\label\{[^}]*\}|\\ref\{[^}]*\}", " ", text)
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}|\\input\{[^}]*\}", " ", text)
    text = re.sub(r"v\d+(\.\d+)+|\d+\.\d+\.\d+|20\d\d-\d\d(-\d\d)?|GSE\d+|GSM\d+|10\.\d{4,}/\S+|NSTC [\d-]+E-[\d-]+MY3", " ", text)
    text = text.replace("{,}", "")
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)", text):
        yield m.group(1), text[max(0, m.start() - 60):m.end() + 30].replace("\n", " ")


def body(tex):
    a = tex.find("\\begin{document}")
    b = tex.find("\\begin{thebibliography}")
    return tex[a:b if b > 0 else len(tex)]


bad = 0
for doc in sys.argv[1:]:
    tex = open(f"{R}/{doc}").read()
    if doc.startswith("manuscript"):
        tex = body(tex)
        tex = tex[tex.find("\\begin{abstract}"):]       # skip title block / affiliations
    else:
        tex = tex[tex.find("\\begin{document}"):]
    miss = []
    for tok, ctx in tokens(tex):
        t = tok.lstrip("0") if False else tok
        if t in ALLOW or t in vals:
            continue
        miss.append((tok, ctx))
    print(f"=== {doc}: {len(miss)} numbers not found in result files")
    for tok, ctx in miss:
        print(f"  {tok:>10}  …{ctx}…")
    bad += len(miss)
