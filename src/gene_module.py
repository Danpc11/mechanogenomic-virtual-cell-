"""
================================================================================
 gene_module.py  —  Mechanosensitive gene layer of the virtual cell
================================================================================

Maps the cell's mechanical state (nuclear drive sigma, YAP N/C) to per-gene
activation scores, using an explicit RESPONSE-SHAPE model for each gene:

    linear        : score ∝ drive                (graded, no threshold)
    weak_power    : score ∝ drive^p, p<1          (saturating, early responders)
    sigmoid       : score = Hill(drive; K, n)     (threshold / switch-like)

This is the mechanogenomic output layer and the HYPOTHESIS GENERATOR: genes
with a switch-like (sigmoid) response above the patient's stiffness, and that
are pharmacologically actionable, are flagged as candidate intervention points.

IMPORTANT (honest scope): the response-shape assignment below is the model's
PREDICTION for each gene, to be tested against RNA-seq/qPCR — not a fit to the
data. Treating it as a prediction (assign shape from the mechanotransduction
role, THEN validate) is what gives it causal, not merely descriptive, force.

Author: Daniel Pérez-Calixto (INMEGEN / UNAM)
================================================================================
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Gene catalog: mechanosensitive genes with predicted response shape + role
# ---------------------------------------------------------------------------
#   shape params:
#     linear     : uses drive / drive_ref
#     weak_power : exponent p (<1)
#     sigmoid    : K (half-max drive), n (Hill coefficient)
#   actionable : True if a drug can target it (hypothesis generator)
# ---------------------------------------------------------------------------
class Gene:
    def __init__(self, symbol, shape, role, actionable=False,
                 p=0.5, K=30.0, n=3.0, drive_ref=55.0, note=""):
        self.symbol = symbol
        self.shape = shape           # "linear" | "weak_power" | "sigmoid"
        self.role = role
        self.actionable = actionable
        self.p = p
        self.K = K
        self.n = n
        self.drive_ref = drive_ref
        self.note = note

    def score(self, drive):
        if self.shape == "linear":
            return float(np.clip(drive / self.drive_ref, 0, 1.5))
        if self.shape == "weak_power":
            return float(np.clip((drive / self.drive_ref) ** self.p, 0, 1.5))
        if self.shape == "sigmoid":
            return float(drive ** self.n / (self.K ** self.n + drive ** self.n))
        raise ValueError(self.shape)


# Predicted response shapes from mechanotransduction role:
#   core YAP/TEAD targets & matrix genes -> switch-like (sigmoid, threshold)
#   early cytoskeletal responders        -> weak power (saturating)
#   graded housekeeping-adjacent         -> linear
GENES = {
    # --- YAP/TAZ-TEAD core targets: threshold (sigmoid) ---
    "CTGF":   Gene("CTGF (CCN2)", "sigmoid", "YAP/TEAD target", actionable=True,
                   K=32, n=4, note="canonical YAP output; verteporfin-sensitive"),
    "CYR61":  Gene("CYR61 (CCN1)", "sigmoid", "YAP/TEAD target", actionable=True,
                   K=32, n=4),
    "ANKRD1": Gene("ANKRD1", "sigmoid", "YAP/TEAD target", actionable=False,
                   K=34, n=5),
    # --- fibrogenic / matrix: threshold ---
    "ACTA2":  Gene("ACTA2 (α-SMA)", "sigmoid", "myofibroblast/contractile",
                   actionable=True, K=30, n=3, note="ROCK/myosin-linked"),
    "COL1A1": Gene("COL1A1", "sigmoid", "collagen I", actionable=True,
                   K=33, n=3, note="LOX-crosslinked matrix"),
    "LOX":    Gene("LOX", "sigmoid", "matrix crosslinker", actionable=True,
                   K=31, n=3, note="feeds back on stiffness; PXS-5505"),
    # --- cytoskeletal early responders: weak power (saturating) ---
    "TAGLN":  Gene("TAGLN (SM22)", "weak_power", "cytoskeletal", p=0.5),
    "TPM1":   Gene("TPM1", "weak_power", "cytoskeletal", p=0.45),
    "FN1":    Gene("FN1 (fibronectin)", "weak_power", "matrix", p=0.6,
                   actionable=True),
    # --- nuclear envelope: linear graded ---
    "LMNA":   Gene("LMNA (lamin A/C)", "linear", "nuclear envelope",
                   drive_ref=60, note="validated vs inferred lamin"),
    "LMNB1":  Gene("LMNB1", "linear", "nuclear envelope", drive_ref=70),
    # --- hepatocyte identity (falls with drive): inverse linear ---
    "HNF4A":  Gene("HNF4A", "linear", "hepatocyte identity (inverse)",
                   drive_ref=55, note="declines as drive rises (dedifferentiation)"),
}
# HNF4A declines: handled via inverse in score_genes.
_INVERSE = {"HNF4A"}


# ---------------------------------------------------------------------------
# Tissue-specific gene panels
# ---------------------------------------------------------------------------
# The CORE genes above (YAP/TEAD targets, matrix, cytoskeleton, envelope) are
# broadly mechanosensitive and apply to every phenotype. On top of them, each
# cell type has its OWN identity markers and lineage-specific mechano-responsive
# genes. Identity markers typically FALL as stiffness rises (dedifferentiation);
# lineage effectors typically RISE. These are the model's per-phenotype
# predictions, to be validated against that cell type's RNA-seq.
#
#   direction: "up" (rises with drive) or "down" (identity loss, inverse)
# ---------------------------------------------------------------------------
CORE_GENES = ["CTGF", "CYR61", "ANKRD1", "ACTA2", "COL1A1", "LOX",
              "TAGLN", "TPM1", "FN1", "LMNA", "LMNB1"]

PHENOTYPE_GENES = {
    "hepatocyte": {
        "HNF4A":  Gene("HNF4A", "linear", "hepatocyte identity (inverse)",
                       drive_ref=55, note="master hepatic TF; lost on stiff"),
        "ALB":    Gene("ALB (albumin)", "linear", "hepatocyte function (inverse)",
                       drive_ref=55, note="secretory function marker"),
        "CYP3A4": Gene("CYP3A4", "linear", "hepatic metabolism (inverse)",
                       drive_ref=58, note="drug-metabolizing identity"),
        "MKI67":  Gene("MKI67 (Ki-67)", "sigmoid", "proliferation", K=33, n=3,
                       note="proliferative re-entry with stiffness"),
    },
    "A549": {  # lung adenocarcinoma epithelial
        "SFTPC": Gene("SFTPC (surfactant C)", "linear", "AT2 identity (inverse)",
                      drive_ref=55, note="alveolar epithelial identity"),
        "NKX2-1": Gene("NKX2-1 (TTF-1)", "linear", "lung lineage TF (inverse)",
                       drive_ref=58),
        "VIM":   Gene("VIM (vimentin)", "sigmoid", "EMT/mesenchymal", K=30, n=3,
                      actionable=True, note="EMT program with stiffening"),
        "SNAI1": Gene("SNAI1 (Snail)", "sigmoid", "EMT master TF", K=32, n=4,
                      actionable=True),
    },
    "NHLF": {  # normal lung fibroblast
        "COL3A1": Gene("COL3A1", "sigmoid", "fibrillar collagen", K=31, n=3,
                       actionable=True, note="fibrotic ECM output"),
        "FAP":    Gene("FAP", "sigmoid", "activated fibroblast", K=33, n=4,
                       actionable=True, note="myofibroblast activation marker"),
        "POSTN":  Gene("POSTN (periostin)", "sigmoid", "matricellular", K=32, n=3,
                       actionable=True, note="IPF/fibrosis marker"),
        "PDGFRA": Gene("PDGFRA", "weak_power", "fibroblast identity", p=0.5),
    },
    "AT2_lung": {  # alveolar type-2 epithelial
        "SFTPC": Gene("SFTPC (surfactant C)", "linear", "AT2 identity (inverse)",
                      drive_ref=55, note="canonical AT2 marker; lost in fibrosis"),
        "SFTPB": Gene("SFTPB (surfactant B)", "linear", "AT2 function (inverse)",
                      drive_ref=56),
        "AGER":  Gene("AGER (AT1 marker)", "sigmoid", "AT2->AT1 transition",
                      K=31, n=3, note="aberrant differentiation on stiff"),
        "KRT8":  Gene("KRT8", "sigmoid", "transitional/aberrant epithelium",
                      K=32, n=3, note="KRT8+ transitional state in IPF"),
    },
    "MCF10A": {  # normal mammary epithelial
        "CDH1":  Gene("CDH1 (E-cadherin)", "linear", "epithelial identity (inverse)",
                      drive_ref=55, note="epithelial junction; lost in EMT"),
        "KRT18": Gene("KRT18", "linear", "luminal epithelial (inverse)",
                      drive_ref=57),
        "VIM":   Gene("VIM (vimentin)", "sigmoid", "EMT/mesenchymal", K=30, n=3,
                      actionable=True, note="EMT with matrix stiffening"),
        "SNAI2": Gene("SNAI2 (Slug)", "sigmoid", "EMT TF", K=32, n=4,
                      actionable=True),
    },
    "MDA": {  # MDA-MB-231 invasive breast cancer
        "VIM":   Gene("VIM (vimentin)", "weak_power", "mesenchymal (already high)",
                      p=0.4, note="constitutively mesenchymal"),
        "MMP9":  Gene("MMP9", "sigmoid", "invasion/ECM degradation", K=30, n=3,
                      actionable=True, note="stiffness-driven invasiveness"),
        "MMP2":  Gene("MMP2", "sigmoid", "invasion", K=31, n=3, actionable=True),
        "ZEB1":  Gene("ZEB1", "sigmoid", "EMT/stemness TF", K=33, n=4,
                      actionable=True, note="metastatic program"),
    },
    "fibroblast": {  # generic fibroblast
        "ACTA2":  Gene("ACTA2 (α-SMA)", "sigmoid", "myofibroblast", K=30, n=3,
                       actionable=True),
        "COL3A1": Gene("COL3A1", "sigmoid", "fibrillar collagen", K=31, n=3,
                       actionable=True),
        "FAP":    Gene("FAP", "sigmoid", "activated fibroblast", K=33, n=4,
                       actionable=True),
        "S100A4": Gene("S100A4 (FSP1)", "weak_power", "fibroblast activation",
                       p=0.5),
    },
}
# genes in the per-phenotype panels that represent identity LOST with stiffness
_PHENO_INVERSE = {"HNF4A", "ALB", "CYP3A4", "SFTPC", "SFTPB", "NKX2-1",
                  "CDH1", "KRT18", "PDGFRA"}


def genes_for(phenotype="hepatocyte"):
    """Return the effective gene set for a phenotype: shared CORE mechanosensitive
    genes + that cell type's identity/lineage markers."""
    genes = {k: GENES[k] for k in CORE_GENES}
    genes.update(PHENOTYPE_GENES.get(phenotype, {}))
    return genes


def _inverse_set(phenotype="hepatocyte"):
    inv = set(_INVERSE)
    for k in PHENOTYPE_GENES.get(phenotype, {}):
        if k in _PHENO_INVERSE:
            inv.add(k)
    return inv


def score_genes(nuclear_drive, yap_nc=None, phenotype="hepatocyte"):
    """Return {gene_symbol: activation_score} at a given drive, using the gene
    panel for `phenotype` (shared core genes + that cell type's markers)."""
    genes = genes_for(phenotype)
    inverse = _inverse_set(phenotype)
    out = {}
    for key, g in genes.items():
        s = g.score(nuclear_drive)
        if key in inverse:
            s = float(np.clip(1.0 - s, 0, 1.5))
        out[g.symbol] = round(s, 3)
    return out


def response_shape_table(phenotype="hepatocyte"):
    """The model's PREDICTED response-shape class per gene for a phenotype
    (for pre-registration before looking at that cell type's RNA-seq)."""
    genes = genes_for(phenotype)
    rows = []
    for key, g in genes.items():
        rows.append(dict(gene=g.symbol, shape=g.shape, role=g.role,
                         actionable=g.actionable))
    return rows


def actionable_hypotheses(nuclear_drive, threshold=0.5, phenotype="hepatocyte"):
    """HYPOTHESIS GENERATOR: actionable genes whose predicted activation exceeds
    `threshold` at the given drive, for this phenotype's panel."""
    genes = genes_for(phenotype)
    inverse = _inverse_set(phenotype)
    scored = score_genes(nuclear_drive, phenotype=phenotype)
    hits = []
    for key, g in genes.items():
        if not g.actionable or key in inverse:
            continue
        s = scored[g.symbol]
        if s >= threshold:
            hits.append(dict(gene=g.symbol, score=s, shape=g.shape,
                             role=g.role, note=g.note))
    return sorted(hits, key=lambda r: r["score"], reverse=True)


def qpcr_panel():
    """Suggested qPCR validation panel: one gene per response-shape class plus
    the inverse identity marker, at the most informative stiffness/time."""
    return {
        "sigmoid (threshold)": ["CTGF (CCN2)", "ACTA2 (α-SMA)", "COL1A1"],
        "weak_power (saturating)": ["TAGLN (SM22)", "FN1 (fibronectin)"],
        "linear (graded)": ["LMNA (lamin A/C)"],
        "inverse (identity loss)": ["HNF4A"],
        "recommended_conditions": "1 & 23 kPa at 120 h (steady state) + 36 h "
                                  "(transient), matching the imaging conditions",
    }


def _demo():
    import mvirtual_cell as mvc
    from mvirtual_cell import PHENOTYPES
    print("=" * 68)
    print("  GENE MODULE  —  mechanosensitive gene layer")
    print("=" * 68)
    hep = PHENOTYPES["hepatocyte"]
    print("\n  Predicted response shapes (pre-registered, then validated):")
    for r in response_shape_table():
        a = "  [actionable]" if r["actionable"] else ""
        print(f"    {r['gene']:>18}  {r['shape']:>11}  {r['role']}{a}")

    print("\n  Gene activation across fibrosis stages:")
    stages = list(mvc.FIBROSIS_STIFFNESS.items())
    genes_show = ["CTGF (CCN2)", "ACTA2 (α-SMA)", "TAGLN (SM22)",
                  "LMNA (lamin A/C)", "HNF4A"]
    hdr = "    " + f"{'gene':>18}" + "".join(f"{s:>6}" for s, _ in stages)
    print(hdr)
    for gsym in genes_show:
        row = f"    {gsym:>18}"
        for stg, E in stages:
            sig = mvc.nuclear_stress(E, hep, reps=4)
            row += f"{score_genes(sig)[gsym]:>6.2f}"
        print(row)

    print("\n  Actionable hypotheses at F4 (26 kPa):")
    sig = mvc.nuclear_stress(26.0, hep, reps=4)
    for h in actionable_hypotheses(sig):
        print(f"    {h['gene']:>18}  score={h['score']:.2f}  ({h['role']})")
    print("=" * 68)


if __name__ == "__main__":
    _demo()
