"""
One fine-to-broad label mapping for every tool and for the enrichR ground truth
(revision 2). Replaces the four separate mappings used before
(02_benchmark_evaluation.py: SIGNACX_TO_BROAD6, CYTETYPE_TO_BROAD6,
ORI_TO_BROAD6, cytetype_fuzzy_broad; _broad_label.BROAD_MAP), which disagreed
on several boundaries (e.g. vascular smooth muscle -> Endothelial in one,
Fibroblast in another) and contained substring bugs ("mast cell" contains
"t cell" and was mapped to T/NK; "Plasmacytoid dendritic cells" -> B).

Broad-6 classes (lineage level):
  T/NK         T, NK, NKT, MAIT, gamma-delta, ILC
  B            B cells, plasma cells, plasmablasts
  Myeloid      monocytes, macrophages/Kupffer/TAM, DC (cDC/pDC), neutrophils,
               mast cells, basophils, eosinophils
  Endothelial  all endothelial cells incl. LSEC, lymphatic, tumour EC
  Fibroblast   mesenchymal and mural cells: fibroblasts/CAF, stellate cells,
               pericytes, vascular smooth muscle
  Hepatocyte   hepatic parenchymal/epithelial lineage: normal and malignant
               hepatocytes, HCC cells, hepatoblast/fetal-like, HPC-like,
               cholangiocytes and other epithelial labels

Rules, applied in order:
  1. Exact matches for closed vocabularies (SignacX levels, broad names).
  2. Quality-control flags (doublet, artifact, low quality, ambient RNA,
     mixed population, dying) -> Unknown: the tool declined to assign a type.
  3. Non-liver / non-resolvable lineages (erythroid, platelets, stem and
     haematopoietic progenitors, neural, bone, cartilage, skin, germ cells)
     -> Unknown.
  4. Descriptors of contamination are removed ("Hepatocyte-contaminated T
     cell" is a T cell; "Kupffer cell with hepatocyte contamination" is a
     Kupffer cell).
  5. Lineage keywords (word-boundary regular expressions) are matched on the
     main label outside parentheses; if several classes match, the class
     whose keyword ends last wins (English head noun). Only if the main label
     matches nothing are parenthetical qualifiers used.
  6. Labels that name no single broad-6 class (SignacX "NonImmune",
     "Lymphocytes", "Unclassified"; generic "lymphocyte") -> Unknown.

Broad-7 (GSE125449 published labels): the broad-6 class is renamed
(T cell, B cell, TAM, TEC, CAF) and the Hepatocyte class is split into
HPC-like (main label names a progenitor / cholangiocyte / biliary / ductal /
oval / hepatoblast cell and does not say malignant/HCC/carcinoma) and
Malignant cell (everything else).
"""
from __future__ import annotations

import re

BROAD6 = ["T/NK", "Myeloid", "Hepatocyte", "B", "Endothelial", "Fibroblast"]
BROAD7 = ["T cell", "TAM", "B cell", "Malignant cell", "HPC-like", "TEC", "CAF"]
B6_TO_B7 = {"T/NK": "T cell", "Myeloid": "TAM", "B": "B cell", "Endothelial": "TEC",
            "Fibroblast": "CAF", "Hepatocyte": "Malignant cell", "Unknown": "Unknown"}

_EXACT = {
    # broad names and the broad-7 names
    "t/nk": "T/NK", "myeloid": "Myeloid", "hepatocyte": "Hepatocyte", "b": "B",
    "endothelial": "Endothelial", "fibroblast": "Fibroblast",
    "t cell": "T/NK", "tam": "Myeloid", "b cell": "B", "malignant cell": "Hepatocyte",
    "hpc-like": "Hepatocyte", "tec": "Endothelial", "caf": "Fibroblast",
    # SignacX (L1/L2 and cell-type level)
    "tnk": "T/NK", "nk": "T/NK", "mph": "Myeloid", "plasma.cells": "B",
    "b.memory": "B", "b.naive": "B", "macrophages": "Myeloid",
    "mon.classical": "Myeloid", "mon.nonclassical": "Myeloid", "dc": "Myeloid",
    "pdc": "Myeloid", "neutrophils": "Myeloid", "mast": "Myeloid",
    "epithelial": "Hepatocyte", "fibroblasts": "Fibroblast",
    "nonimmune": "Unknown", "lymphocytes": "Unknown", "unclassified": "Unknown",
    # abstention labels of scmap / scPred ("unassigned") and SingleCellNet ("rand")
    "unassigned": "Unknown", "rand": "Unknown",
    # missing values
    "unknown": "Unknown", "na": "Unknown", "nan": "Unknown", "none": "Unknown", "": "Unknown",
}

_QC = re.compile(r"doublet|multiplet|artifact|artefact|low[- ]?quality|ambient|"
                 r"mixed population|\bdying\b|debris")
_EXCLUDE = re.compile(r"erythro|red blood|platelet|megakaryocyte|\bstem cells?\b|"
                      r"\bips cells?\b|\bhsc\b|\bgmp\b|\bcmp\b|\bmep\b|^bm\b|bm & prog|"
                      r"neuro|astrocyte|chondro|osteo|keratinocyte|gametocyte|germ cell")
_CONTAM = [re.compile(r"\b[\w+/]+-contaminated\b"),
           re.compile(r"\bwith [\w\s+/-]*contamination\b")]

_LINEAGE = [
    ("T/NK", re.compile(r"\b(t|nkt)[ -]?(cells?|lymphocytes?)\b|\bt-like\b|\bcd[48]\b|\btregs?\b|"
                        r"\bregulatory t\b|\bmait\b|\bnkt?\b|\bnatural killer\b|\bgamma[- ]?delta\b|"
                        r"γδ|\bilcs?\b|\binnate lymphoid\b|\btemra\b|\btfh\b|\bthymocytes?\b|^t\.")),
    ("B", re.compile(r"\bb[ -]?cells?\b|\bplasma[ -]?cells?\b|\bplasmablasts?\b|\bb lymphocytes?\b|"
                     r"\bantibody-secreting\b|\bmemory b\b|\bnaive b\b|\bpre-b\b|\bpro-b\b")),
    ("Myeloid", re.compile(r"macrophage|kupffer|\bmono\b|monocyte|dendritic|\b(c|p|mreg)?dcs?[12]?s?\b|"
                           r"neutrophil|\bmast\b|basophil|eosinophil|\bmyeloid\b|myelocyte|\btams?\b|"
                           r"microglia")),
    ("Endothelial", re.compile(r"endotheli|\blsecs?\b|sinusoidal|\bt?ecs?\b|vessel")),
    ("Fibroblast", re.compile(r"fibroblast|stellate|pericyte|smooth muscle|\bv?smcs?\b|\bmural\b|"
                              r"\bcafs?\b|mesenchymal|\bmscs?\b|perivascular")),
    ("Hepatocyte", re.compile(r"hepatocyte|hepatocellular|\bhcc\b|hepatoblast|liver bud hepatic|"
                              r"cholangio|biliary|ductal|epitheli|\bhpc\b|liver progenitor|"
                              r"hepatic progenitor|oval cell|malignant|carcinoma|\btumou?r cells?\b")),
]
_HPC = re.compile(r"progenitor|cholangio|biliary|ductal|\bhpc\b|oval cell|hepatoblast")
_MALIGNANT = re.compile(r"malignant|\bhcc\b|carcinoma|hepatocellular|\btumou?r cells?\b")


def _norm(label) -> str:
    s = "" if label is None else str(label)
    s = s.split("::")[-1]                      # ScType "Liver::Hepatocytes"
    return s.strip()


def _match(text: str):
    best, best_end = None, -1
    for cls, rx in _LINEAGE:
        for m in rx.finditer(text):
            if m.end() > best_end:
                best, best_end = cls, m.end()
    return best


def map_label(label) -> tuple[str, str]:
    """Return (broad6 class, rule) for one raw label."""
    raw = _norm(label)
    lo = raw.lower()
    if lo in _EXACT:
        return _EXACT[lo], "exact"
    if lo.startswith("t.") and lo.count(".") >= 1:      # SignacX T.CD8.em etc.
        return "T/NK", "signacx-T"
    if _QC.search(lo):
        return "Unknown", "qc-flag"
    t = lo.replace("_", " ")
    if _EXCLUDE.search(t):
        return "Unknown", "excluded-lineage"
    for rx in _CONTAM:
        t = rx.sub(" ", t)
    head = re.sub(r"\([^)]*\)", " ", t)
    cls = _match(head)
    if cls:
        return cls, "head"
    paren = " ".join(re.findall(r"\(([^)]*)\)", t))
    cls = _match(paren)
    if cls:
        return cls, "parenthetical"
    return "Unknown", "no-match"


def broad6(label) -> str:
    return map_label(label)[0]


def broad7(label) -> str:
    b6, _ = map_label(label)
    if b6 != "Hepatocyte":
        return B6_TO_B7[b6]
    lo = _norm(label).lower().replace("_", " ")
    if lo in ("hpc-like",):
        return "HPC-like"
    head = re.sub(r"\([^)]*\)", " ", lo)
    if _HPC.search(head) and not _MALIGNANT.search(head):
        return "HPC-like"
    return "Malignant cell"


def gt_broad6(enrichr_term, reject_keywords) -> str:
    """enrichR ground truth: keep the published rejection list, then map."""
    lo = str(enrichr_term).lower()
    if any(k in lo for k in reject_keywords):
        return "Unknown"
    return broad6(enrichr_term)
