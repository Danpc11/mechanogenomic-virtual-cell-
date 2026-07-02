"""
================================================================================
 pharmacology.py  —  Clinical connection, drug perturbation, and toxicity
================================================================================

Extends the mechanogenomic virtual cell toward application. THREE things,
each grounded in the physics and each with explicit limits:

  1. CLINICAL CONNECTION
     Liver elastography (FibroScan) measures patient liver stiffness in kPa —
     the model's input variable. A patient's stiffness places them on the
     mechanogenomic trajectory (map_patient).

  2. DRUG PERTURBATION (in silico screening)
     Mechanotransduction-targeting drugs map to specific physical parameters
     of the model (contractility, clutches, tissue stiffness, YAP output).
     apply_drug() predicts the effect on nuclear stress, YAP, and the
     mechanosensitive gene signature.

  3. TOXICITY as MECHANICAL-STATE-DEPENDENT FUNCTIONAL DECLINE
     Hepatocyte function (albumin, urea, CYP450, HNF4a) falls on stiff
     substrates (well documented). hepatocyte_function() predicts a functional
     index from the mechanical state.

--------------------------------------------------------------------------------
IMPORTANT — HONEST LIMITS (read before using):
  * This is a HYPOTHESIS-GENERATING tool, not a validated pharmacology or
    toxicity predictor. It does NOT model drug metabolism, off-target effects,
    pharmacokinetics, or chemical/idiosyncratic hepatotoxicity (DILI).
  * The "toxicity" axis is mechanical-microenvironment-driven functional
    decline only — one axis among many.
  * Clinical history demands caution: mechanistically sound antifibrotics
    (e.g. simtuzumab/anti-LOXL2) have failed in trials. Model predictions
    prioritize and explain; they do not replace experiments or trials.
--------------------------------------------------------------------------------

Author: Daniel Pérez-Calixto (INMEGEN / UNAM)
Dependencies: numpy; uses mvirtual_cell (and optionally fast_model for speed).
================================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import replace
import mvirtual_cell as mvc
from mvirtual_cell import PHENOTYPES, FIBROSIS_STIFFNESS


# ============================================================================
# 1.  DRUG LIBRARY  —  mechanism mapped to physical parameters
# ============================================================================
#   Each drug is a perturbation of the model's physical parameters. The mapping
#   reflects the drug's established mechanism of action (literature-based).
#   Parameters that can be perturbed:
#     nm    : myosin motors / contractility     (ROCK, myosin inhibitors)
#     nc    : substrate clutches / adhesions    (FAK, integrin inhibitors)
#     E_mod : multiplies TISSUE stiffness E      (LOX/LOXL2 crosslink inhibitors)
#     laminAC: nuclear lamin A/C level
#     yap_output: multiplies YAP transcriptional OUTPUT downstream of nuclear
#                 localization (YAP/TEAD inhibitors act here, NOT on stress)
# ---------------------------------------------------------------------------
class Drug:
    def __init__(self, name, target, organ="multi", nm_mod=1.0, nc_mod=1.0,
                 E_mod=1.0, laminAC_mod=1.0, yap_output_mod=1.0,
                 fibroblast_mod=1.0, axis="mechanical", status="research", note=""):
        self.name = name
        self.target = target
        self.organ = organ                 # lung / liver / kidney / heart / multi
        self.nm_mod = nm_mod               # contractility multiplier
        self.nc_mod = nc_mod               # clutch/adhesion multiplier
        self.E_mod = E_mod                 # tissue stiffness multiplier (microenv.)
        self.laminAC_mod = laminAC_mod
        self.yap_output_mod = yap_output_mod   # downstream YAP output multiplier
        self.fibroblast_mod = fibroblast_mod   # fibroblast activation / ECM output
        self.axis = axis                   # "mechanical" or "growth-factor/signaling"
        self.status = status               # approved / trial / research / failed
        self.note = note


# The model's physics is the MECHANOTRANSDUCTION machinery (motor-clutch, YAP,
# lamin). Approved antifibrotics mostly act on an UPSTREAM/PARALLEL axis:
# growth-factor signaling (PDGFR/FGFR/VEGFR, TGF-β, PDE4B) that drives fibroblast
# activation and matrix deposition -> tissue stiffening (E) over time. We map
# those to a reduction in tissue stiffness E (their net mechanical consequence)
# and flag their axis honestly, since the model does NOT resolve their direct
# signaling targets. Mechanotransduction-targeting drugs map to motor/clutch/YAP.
DRUGS = {
    # ---- APPROVED (growth-factor / signaling axis; act via reduced stiffening) ----
    "nintedanib": Drug("Nintedanib", "PDGFR/FGFR/VEGFR tyrosine-kinase inhibitor",
        organ="lung", E_mod=0.75, fibroblast_mod=0.6,
        axis="growth-factor/signaling", status="approved (IPF, PF-ILD, SSc-ILD)",
        note="Blocks fibroblast growth-factor signaling. Net effect: less stiffening. "
             "Model captures the mechanical consequence, not the kinase targets."),
    "pirfenidone": Drug("Pirfenidone", "TGF-β / TNF-α / IL-6 modulator",
        organ="lung", E_mod=0.8, fibroblast_mod=0.65,
        axis="growth-factor/signaling", status="approved (IPF)",
        note="Reduces TGF-β-driven collagen. Acts upstream of mechanics."),
    "nerandomilast": Drug("Nerandomilast", "PDE4B inhibitor",
        organ="lung", E_mod=0.8, fibroblast_mod=0.7,
        axis="growth-factor/signaling", status="approved 2025 (IPF, PPF)",
        note="First new IPF mechanism in >10 yr; raises cAMP, immunomodulatory."),
    # ---- MECHANOTRANSDUCTION axis (what the model resolves directly) ----
    "fasudil": Drug("Fasudil", "ROCK inhibitor", organ="multi", nm_mod=0.55,
        axis="mechanical", status="approved (Japan, other indications)",
        note="Lowers actomyosin contractility. Antifibrotic in lung/liver/kidney models."),
    "Y-27632": Drug("Y-27632", "ROCK inhibitor", organ="multi", nm_mod=0.5,
        axis="mechanical", status="research",
        note="Research ROCK inhibitor; reduces contractility."),
    "blebbistatin": Drug("Blebbistatin", "Myosin-II inhibitor", organ="multi",
        nm_mod=0.35, axis="mechanical", status="research",
        note="Directly inhibits non-muscle myosin II (contractility)."),
    "defactinib": Drug("Defactinib", "FAK inhibitor", organ="multi", nc_mod=0.55,
        axis="mechanical", status="trial (fibrosis/oncology)",
        note="Focal adhesion kinase inhibitor; reduces adhesion clutches."),
    "cilengitide": Drug("Cilengitide", "Integrin (αvβ3/αvβ5) inhibitor",
        organ="multi", nc_mod=0.5, axis="mechanical", status="research",
        note="Reduces integrin-based clutches."),
    "verteporfin": Drug("Verteporfin", "YAP/TEAD inhibitor", organ="multi",
        yap_output_mod=0.3, axis="mechanical", status="research (repurposing)",
        note="Blocks YAP transcriptional OUTPUT downstream; nuclear stress unchanged."),
    "simvastatin": Drug("Simvastatin", "Statin (YAP cytoplasmic retention)",
        organ="multi", yap_output_mod=0.6, nm_mod=0.85, axis="mechanical",
        status="approved (other indication); antifibrotic repurposing",
        note="Promotes YAP inactivation. Repurposing candidate."),
    # ---- MATRIX-crosslinking (stiffness axis directly) ----
    "PXS-5505": Drug("PXS-5505", "Pan-LOX inhibitor", organ="multi", E_mod=0.65,
        axis="mechanical", status="trial (myelofibrosis, liver)",
        note="Reduces collagen crosslinking -> lowers tissue stiffness directly."),
    "simtuzumab": Drug("Simtuzumab", "anti-LOXL2 antibody", organ="liver/lung",
        E_mod=0.9, axis="mechanical", status="FAILED (no stage improvement)",
        note="Mechanistically sound but failed in trials — the humility benchmark."),
    # ---- KIDNEY / HEART candidates (signaling axis) ----
    "finerenone": Drug("Finerenone", "Non-steroidal MR antagonist",
        organ="kidney/heart", E_mod=0.82, fibroblast_mod=0.7,
        axis="growth-factor/signaling", status="approved (CKD in T2D)",
        note="Reduces cardiorenal fibrosis/inflammation. Upstream of mechanics."),
    # ---- METABOLIC axis (upstream of everything; model only sees net stiffness) ----
    "resmetirom": Drug("Resmetirom (Rezdiffra)", "THR-β agonist (liver-directed)",
        organ="liver", E_mod=0.82, fibroblast_mod=0.75,
        axis="metabolic", status="approved 2024 (first for MASH F2-F3 fibrosis)",
        note="Reduces hepatic fat/lipotoxicity -> less inflammation -> less fibrosis. "
             "Acts on METABOLISM, far upstream of mechanics. Model sees only the "
             "net stiffness consequence — cannot predict its metabolic benefit."),
    "lanifibranor": Drug("Lanifibranor", "Pan-PPAR agonist",
        organ="liver", E_mod=0.85, fibroblast_mod=0.75,
        axis="metabolic", status="trial (Phase 3, MASH)",
        note="Metabolic + HSC-deactivating. Upstream of mechanics."),
    "semaglutide": Drug("Semaglutide", "GLP-1 receptor agonist",
        organ="liver", E_mod=0.88, fibroblast_mod=0.8,
        axis="metabolic", status="trial (Phase 3, MASH)",
        note="Metabolic (weight loss / insulin). Indirect antifibrotic."),
    "control": Drug("Vehicle control", "none", organ="-",
        axis="-", status="-", note="No perturbation (baseline)."),
}


def apply_drug(drug, ph=None, E=None, reps=6, use_fast=True):
    """Predict a drug's effect on the mechanotransduction chain.

    Returns a dict comparing baseline vs treated: nuclear stress, YAP N/C
    (including downstream output modulation), and nuclear area. `E` is the
    tissue stiffness (kPa); if None, uses the F4 cirrhotic value.
    """
    if isinstance(drug, str):
        drug = DRUGS[drug]
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    if E is None:
        E = FIBROSIS_STIFFNESS["F4"]

    # baseline
    sig0 = _stress(E, ph, reps, use_fast)
    yap0 = mvc.yap_nc_ratio(E, ph, reps=reps)
    area0 = mvc.nuclear_area_ss(E, ph, reps=reps)

    # treated: apply perturbations
    ph_t = replace(ph, nm=max(int(round(ph.nm * drug.nm_mod)), 1),
                   nc=max(int(round(ph.nc * drug.nc_mod)), 5),
                   laminAC=ph.laminAC * drug.laminAC_mod)
    E_t = E * drug.E_mod
    sig1 = _stress(E_t, ph_t, reps, use_fast)
    # YAP: nuclear localization from treated chain, then downstream output block
    yap1_loc = mvc.yap_nc_ratio(E_t, ph_t, reps=reps)
    yap1 = 1.0 + (yap1_loc - 1.0) * drug.yap_output_mod   # output modulation
    area1 = mvc.nuclear_area_ss(E_t, ph_t, reps=reps)

    return dict(
        drug=drug.name, target=drug.target, E_baseline=E, E_treated=E_t,
        nuclear_stress=(sig0, sig1), yap_nc=(yap0, yap1),
        nuclear_area=(area0, area1),
        yap_reduction_pct=100 * (yap0 - yap1) / max(yap0 - 1, 1e-6),
        note=drug.note)


def _stress(E, ph, reps, use_fast):
    if use_fast:
        try:
            import fast_model as fm
            # use fast form only when phenotype params are unchanged from a
            # calibrated key; otherwise fall back to the stochastic motor
            key = _phenotype_key(ph)
            if key is not None:
                return float(fm.nuclear_stress_fast(E, key))
        except Exception:
            pass
    return mvc.nuclear_stress(E, ph, reps=reps)


def _phenotype_key(ph):
    """Return the SATURATING_PARAMS key if ph matches a calibrated phenotype
    with unperturbed motor params, else None (so we fall back to simulation)."""
    import fast_model as fm
    base = PHENOTYPES.get("hepatocyte")
    if (ph.nm == base.nm and ph.nc == base.nc and abs(ph.alpha - base.alpha) < 1e-9
            and abs(ph.kc - base.kc) < 1e-9):
        return "hepatocyte"
    return None


# ============================================================================
# 2.  TOXICITY  =  MECHANICAL-STATE-DEPENDENT HEPATOCYTE FUNCTION
# ============================================================================
#   Hepatocyte function (albumin, urea, CYP450, HNF4a) declines on stiff
#   substrates (Natarajan 2015; Desai 2016; You 2019; Deegan 2021). We model a
#   functional index that is HIGH on soft substrate and falls as stiffness /
#   nuclear stress rises. This is a proxy for functional toxicity of the
#   mechanical microenvironment — NOT chemical/DILI toxicity.
# ---------------------------------------------------------------------------
def hepatocyte_function(E, ph=None, E_half=12.0, hill=2.0, reps=6, use_fast=True):
    """Hepatocyte functional index in [0, 1] as a function of tissue stiffness.

    1.0 = fully differentiated/functional (soft, healthy ~2-4 kPa);
    ->0 = dedifferentiated/dysfunctional (stiff, fibrotic/cirrhotic).
    Falls with stiffness with half-max at E_half (~12 kPa, ~F3), consistent
    with reported loss of albumin/urea/CYP450/HNF4a on stiff matrices."""
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    # function declines with stiffness (Hill), modulated by nuclear stress
    f_stiff = 1.0 / (1.0 + (E / E_half) ** hill)
    return float(np.clip(f_stiff, 0.0, 1.0))


def function_markers(E, ph=None):
    """Illustrative relative levels of hepatocyte function markers vs stiffness
    (albumin, urea, CYP3A4, HNF4a), all scaled to the functional index.
    Directional prediction (all fall with stiffness), to be checked by assay."""
    f = hepatocyte_function(E, ph)
    return dict(functional_index=f,
                albumin=f, urea=0.3 + 0.7 * f, CYP3A4=f ** 1.2, HNF4A=f ** 0.8)


def toxicity_flag(drug, ph=None, E=None, reps=6):
    """Does a drug move hepatocytes toward the functional or dysfunctional
    regime? Positive delta = protective (restores function); negative = harmful
    (in the mechanical sense only)."""
    if isinstance(drug, str):
        drug = DRUGS[drug]
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    if E is None:
        E = FIBROSIS_STIFFNESS["F4"]
    f_base = hepatocyte_function(E, ph)
    f_treated = hepatocyte_function(E * drug.E_mod, ph)
    delta = f_treated - f_base
    verdict = ("protective (restores function)" if delta > 0.02
               else "harmful (reduces function)" if delta < -0.02
               else "neutral (mechanical function unchanged)")
    return dict(drug=drug.name, function_baseline=f_base,
                function_treated=f_treated, delta=delta, verdict=verdict,
                caveat="Mechanical-function axis only; not chemical/DILI toxicity.")


# ============================================================================
# 3.  CLINICAL CONNECTION  —  elastography -> trajectory position
# ============================================================================
def map_patient(liver_stiffness_kPa, ph=None, reps=6):
    """Map a patient's measured liver stiffness (FibroScan/elastography, kPa)
    onto the mechanogenomic trajectory: fibrosis stage, predicted YAP activity,
    predicted hepatocyte function, and mechanosensitive drive.

    Stage cutoffs are approximate literature values (VCTE, MASLD/NAFLD);
    they vary by etiology and device and are indicative, not diagnostic."""
    E = float(liver_stiffness_kPa)
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    # approximate VCTE stage cutoffs (kPa)
    if E < 6.0:       stage = "F0 (no/minimal fibrosis)"
    elif E < 8.0:     stage = "F1 (mild)"
    elif E < 10.0:    stage = "F2 (significant)"
    elif E < 14.0:    stage = "F3 (advanced)"
    else:             stage = "F4 (cirrhosis)"
    return dict(
        liver_stiffness_kPa=E, fibrosis_stage=stage,
        yap_activity=mvc.yap_nc_ratio(E, ph, reps=reps),
        hepatocyte_function=hepatocyte_function(E, ph),
        nuclear_stress=mvc.nuclear_stress(E, ph, reps=reps),
        note="Indicative mapping; cutoffs vary by etiology/device, not diagnostic.")


# ============================================================================
# 4.  IN SILICO DRUG SCREEN
# ============================================================================
def screen_drugs(ph=None, E=None, drugs=None, reps=6):
    """Rank drugs by predicted benefit at a given tissue stiffness: how much
    they reduce the mechanosensitive/YAP drive while restoring (or not harming)
    hepatocyte function. Returns a sorted list (best first)."""
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    if E is None:
        E = FIBROSIS_STIFFNESS["F4"]
    if drugs is None:
        drugs = [k for k in DRUGS if k != "control"]

    rows = []
    for key in drugs:
        eff = apply_drug(key, ph=ph, E=E, reps=reps)
        tox = toxicity_flag(key, ph=ph, E=E, reps=reps)
        yap0, yap1 = eff["yap_nc"]
        yap_drop = (yap0 - yap1) / max(yap0 - 1, 1e-6)     # fraction of drive removed
        # composite score: remove profibrotic YAP drive + restore function
        score = 0.6 * yap_drop + 0.4 * max(tox["delta"], 0)
        rows.append(dict(drug=DRUGS[key].name, target=DRUGS[key].target,
                         organ=DRUGS[key].organ, axis=DRUGS[key].axis,
                         status=DRUGS[key].status,
                         yap_drive_removed=yap_drop, function_delta=tox["delta"],
                         score=score, note=DRUGS[key].note))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


# ============================================================================
# 5.  DEMO
# ============================================================================
def _demo():
    print("=" * 74)
    print("  PHARMACOLOGY / CLINICAL / TOXICITY  —  hypothesis-generating layer")
    print("=" * 74)
    print("  NOTE: in silico hypotheses only; not a validated drug/tox predictor.")

    ph = PHENOTYPES["hepatocyte"]

    # --- 1. clinical mapping ---
    print("\n[1] Clinical: patient liver stiffness -> trajectory")
    print(f"    {'kPa':>6}{'stage':>26}{'YAP':>7}{'function':>10}")
    for E in [4, 7, 9.5, 13, 26]:
        m = map_patient(E, ph, reps=4)
        print(f"    {E:>6.1f}{m['fibrosis_stage']:>26}"
              f"{m['yap_activity']:>7.2f}{m['hepatocyte_function']:>10.2f}")

    # --- 2. toxicity / function vs stiffness ---
    print("\n[2] Hepatocyte function markers vs stiffness (all fall on stiff):")
    print(f"    {'kPa':>6}{'f_index':>9}{'albumin':>9}{'CYP3A4':>8}{'HNF4A':>8}")
    for E in [2, 7, 13, 26, 45]:
        mk = function_markers(E, ph)
        print(f"    {E:>6.1f}{mk['functional_index']:>9.2f}{mk['albumin']:>9.2f}"
              f"{mk['CYP3A4']:>8.2f}{mk['HNF4A']:>8.2f}")

    # --- 3. single drug effect ---
    print("\n[3] Drug effect at F4 (26 kPa) — fasudil (ROCK inhibitor, mechanical axis):")
    eff = apply_drug("fasudil", ph=ph, E=26.0, reps=4)
    print(f"    nuclear stress: {eff['nuclear_stress'][0]:.1f} -> {eff['nuclear_stress'][1]:.1f}")
    print(f"    YAP N/C:        {eff['yap_nc'][0]:.2f} -> {eff['yap_nc'][1]:.2f}")

    # --- 4. in silico screen across organs ---
    print("\n[4] In silico screen at 26 kPa — current antifibrotics, ranked:")
    print(f"    {'drug':>20}  {'organ':>12}  {'axis':>12}  {'YAP↓':>5}  {'score':>5}")
    for r in screen_drugs(ph=ph, E=26.0, reps=4):
        print(f"    {r['drug']:>20}  {r['organ']:>12}  {r['axis']:>12}"
              f"  {r['yap_drive_removed']*100:>4.0f}%  {r['score']:>5.2f}")

    print("\n  THREE AXES OF ACTION (the model resolves only the first directly):")
    print("  • mechanical    — motor-clutch, YAP, lamin, matrix crosslinking.")
    print("                    Model predicts these directly (high confidence).")
    print("  • growth-factor — nintedanib, pirfenidone, nerandomilast (PDGFR/")
    print("                    FGFR/TGF-β/PDE4B). Model sees net stiffness only.")
    print("  • metabolic     — resmetirom, lanifibranor, semaglutide (lipid/")
    print("                    metabolism). FURTHEST upstream. Resmetirom is the")
    print("                    first approved MASH-fibrosis drug, yet the physical")
    print("                    model cannot predict its metabolic benefit — only")
    print("                    its downstream mechanical consequence. Honest limit.")

    print("\n" + "=" * 74)
    print("  Reminder: mechanistically sound antifibrotics have failed in trials")
    print("  (e.g. simtuzumab). Use these as prioritized hypotheses, not answers.")
    print("=" * 74)


if __name__ == "__main__":
    _demo()
