"""
Single canonical implementation of the enrichR-derived ground-truth
fine-to-broad keyword mapping. Shared by:
  - 10_build_enrichr_groundtruth.py
  - 16_prepare_gse156625_external.py

Previously these two scripts maintained independent copies of REJECT_KEYWORDS
and BROAD_MAP which drifted (notably: cholangiocyte/biliary/hpc were mapped
to different broad classes between the two scripts, producing inconsistent
ground truth between GSE156625 and the other five cohorts). This module
removes the drift surface; any future changes happen once here.

The mapper produces fine-grained broad labels (Cholangiocyte, HPC-like,
Erythroid in addition to the broad-six). Downstream evaluation
(02_benchmark_evaluation.py) keeps only the broad-six (T/NK, Myeloid,
Hepatocyte, B, Endothelial, Fibroblast) and the broad-seven HCC-specific
set (T cell, CAF, Malignant cell, TAM, B cell, TEC, HPC-like); anything
else is coerced to Unknown at evaluation time. This separation is
deliberate: the snapshot files retain the finer labels so a future
broader-vocabulary evaluation can re-use the same snapshots without
re-querying enrichR.
"""

from __future__ import annotations

# Curated list of cell-type terms that are tissue-irrelevant or
# developmentally non-applicable in the HCC TME and are therefore
# excluded before broad mapping.
REJECT_KEYWORDS = [
    "goblet", "intestinal crypt", "paneth", "microglia", "microglial",
    "mesangial", "medullary", "leydig", "spermatogon", "trophoblast",
    "villous", "pancreatic acinar", "pancreatic ductal", "pancreatic duct",
    "osteoblast", "osteoclast", "chondrocyte", "adipocyte", "keratinocyte",
    "melanocyte", "secretory cell lung", "transit amplifying",
    "cardiac", "cardiomyocyte", "podocyte", "proximal tubule", "distal tubule",
    "collecting duct", "loop of henle", "bowman", "glomerular",
    "enterocyte", "colonocyte", "crypt", "epithelial cell lung",
    "epithelial cell ovary", "fetal germ", "germinal center",
    "multilymphoid progenitor", "megakaryocyte progenitor",
    "stromal cell bone", "schwann", "oligodendrocyte", "astrocyte",
    "dopaminergic", "purkinje", "granule cell", "cortical neuron",
    "leydig", "sertoli", "granulocyte progenitor", "erythroid progenitor",
    "common myeloid progenitor", "common lymphoid progenitor",
]

# Fine-grained → broad mapping. Cholangiocyte / Hepatoblast / HPC-like /
# Erythroid are kept as their own labels (not collapsed into Hepatocyte
# or B) so that the snapshot files can be re-used under a future broader
# vocabulary; downstream broad-6 evaluation coerces these to Unknown.
BROAD_MAP = {
    "t cell": "T/NK", "cd4": "T/NK", "cd8": "T/NK", "treg": "T/NK",
    "regulatory t": "T/NK", "mait": "T/NK", "nkt": "T/NK",
    "exhausted": "T/NK", "cytotoxic t": "T/NK", "gamma delta": "T/NK",
    "gdelta": "T/NK", "intraepithelial lymphocyte": "T/NK",
    "mucosal": "T/NK", "natural killer": "T/NK", "nk cell": "T/NK",
    "nk-cell": "T/NK", "innate lymphoid": "T/NK", "ilc": "T/NK",

    "macrophage": "Myeloid", "kupffer": "Myeloid",
    "tumor-associated macrophage": "Myeloid", "monocyte": "Myeloid",
    "dendritic cell": "Myeloid", "dendritic": "Myeloid",
    "plasmacytoid dendritic": "Myeloid", "conventional dc": "Myeloid",
    " dc": "Myeloid", "neutrophil": "Myeloid", "mast cell": "Myeloid",
    "basophil": "Myeloid", "eosinophil": "Myeloid",

    "b cell": "B", "b-cell": "B", "plasma cell": "B", "plasmablast": "B",
    "memory b": "B", "naive b": "B", "germinal centre b": "B",

    "hepatocyte": "Hepatocyte", "liver bud hepatic": "Hepatocyte",
    # "hepatic progenitor" maps to HPC-like, not Hepatocyte. This is
    # consistent with Cell Ontology (CL hepatic stem cell CL_0019029 is
    # distinct from mature hepatocyte CL_0000182) and with the module's
    # design intent (HPC-like preserved as its own label; demoted to
    # Unknown only at the broad-six evaluation stage).
    "hepatic progenitor": "HPC-like",

    "endothelial": "Endothelial", "vascular smooth muscle": "Endothelial",
    "lymphatic": "Endothelial",

    "fibroblast": "Fibroblast", "stellate cell": "Fibroblast",
    "myofibroblast": "Fibroblast",
    "cancer-associated fibroblast": "Fibroblast", "caf": "Fibroblast",
    "pericyte": "Fibroblast", "smooth muscle cell": "Fibroblast",

    # Distinct lineages kept as their own labels; demoted to Unknown by
    # downstream broad-6 evaluation but preserved in snapshot files for
    # broader-vocabulary re-evaluation.
    "cholangiocyte": "Cholangiocyte", "biliary": "Cholangiocyte",
    "hpc": "HPC-like",
    "erythroid": "Erythroid", "erythrocyte": "Erythroid",
    "red blood cell": "Erythroid",
}


def broad_label(cell_type_str: str) -> str:
    """Map a fine-grained enrichR cell-type term to its broad class label.
    Returns "Unknown" if rejected or no match.
    """
    ct_lower = str(cell_type_str).lower()
    for rej in REJECT_KEYWORDS:
        if rej in ct_lower:
            return "Unknown"
    for key, broad in BROAD_MAP.items():
        if key.lower() in ct_lower:
            return broad
    return "Unknown"
