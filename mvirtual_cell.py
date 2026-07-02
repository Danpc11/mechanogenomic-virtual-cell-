"""
================================================================================
 virtual_cell.py  —  Mechanogenomic virtual-cell model (calibrated)
================================================================================

A minimal, first-principles physical model of nuclear mechanotransduction:

    substrate stiffness E  ->  traction T  ->  nuclear stress sigma
                                            ->  { nuclear area, YAP N/C, lamin A/C }

The model couples a stochastic motor-clutch engine (Chan & Odde 2008;
Bangasser & Odde 2013) to a lamin-A/C-gated nucleus, with a two-population
structure (basal binucleate + mechanosensitive) and a contact-inhibition
switch (integrin -> cadherin). Parameters are CALIBRATED against projected
nuclear area of primary rat hepatocytes on hydrogels (0.5-23 kPa, 2-36 h,
~40,000 nuclei) and cross-validated against qPCR.

Author: Daniel Pérez-Calixto (INMEGEN / UNAM)

Dependencies: numpy, scipy. Optional: numba (≈50-100x speed-up).
================================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace, field

# ------------------------------------------------------------------ numba (optional)
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:                                  # pure-python fallback
    HAS_NUMBA = False
    def njit(f=None, **k):
        if f is None:
            return lambda g: g
        return f


# ============================================================================
# 1.  MOTOR-CLUTCH ENGINE  (cell-substrate interface)
# ============================================================================
@njit
def _mc_kernel(kappa_sub, nm, Fm, vu, nc, kon, koff0, Fb, kc,
               T, dt, burn, seed):
    """
    Stochastic motor-clutch simulation. Returns steady-state mean substrate
    traction <F_sub> (pN), averaged over the post-burn-in window.

    Physics per step:
      F_sub = sum_bound kc*x_i
      v     = vu * (1 - F_sub/F_stall)_+                 (load-dependent flow)
      dx_i  = v*dt * kappa/(kappa+kc)                    (compliance partition)
      koff  = koff0 * exp(kc|x_i|/Fb)                    (slip bond)
      kon                                                (binding, x=0)
    """
    np.random.seed(seed)
    Fstall = nm * Fm
    bound = np.zeros(nc, dtype=np.bool_)
    xc = np.zeros(nc)
    nsteps = int(T / dt)
    start = int(burn * nsteps)
    acc = 0.0
    cnt = 0
    for s in range(nsteps):
        nb = 0
        Fsum = 0.0
        for i in range(nc):
            if bound[i]:
                nb += 1
                Fsum += kc * xc[i]
        F_sub = Fsum if nb > 0 else 0.0
        v = vu * (1.0 - F_sub / Fstall)
        v = v if v > 0.0 else 0.0
        if nb > 0:
            load = v * dt * kappa_sub / (kappa_sub + kc)
            for i in range(nc):
                if bound[i]:
                    xc[i] += load
        for i in range(nc):
            if bound[i]:
                Fc = kc * abs(xc[i])
                koff = koff0 * np.exp(Fc / Fb)
                if np.random.random() < koff * dt:
                    bound[i] = False
                    xc[i] = 0.0
            else:
                if np.random.random() < kon * dt:
                    bound[i] = True
                    xc[i] = 0.0
        if s >= start:
            acc += F_sub
            cnt += 1
    return acc / cnt if cnt > 0 else 0.0


# ============================================================================
# 2.  PHENOTYPE  =  physical parameter vector
# ============================================================================
@dataclass
class Phenotype:
    """A cell phenotype is a point in the model's physical parameter space."""
    name: str
    # --- motor-clutch ---
    nm: int    = 50      # myosin motors (contractility)
    Fm: float  = 2.0     # force per motor (pN)
    vu: float  = 110.0   # unloaded actin retrograde velocity (nm/s)
    nc: int    = 100     # substrate clutches (focal adhesions)
    kon: float = 0.5     # clutch binding rate (1/s)
    koff0: float = 0.1   # base clutch unbinding rate (1/s)
    Fb: float  = 2.0     # slip-bond rupture force (pN)
    kc: float  = 1.0     # clutch stiffness (pN/nm)
    # --- nuclear ---
    laminAC: float = 1.0 # relative lamin A/C level (nuclear stiffness); INFERRED from area
    alpha: float   = 0.15# E -> kappa coupling  (kappa = alpha * E)
    # --- geometry (nuclear area, um^2) ---
    A_min: float = 40.0  # basal (unstressed) projected nuclear area
    A_max: float = 120.0 # maximally spread area
    s0: float    = 15.0  # half-saturation stress scale (x laminAC)
    # --- dynamics ---
    tau: float   = 35.0  # nuclear flattening time constant (h)  [calibrated]


# ---------------------------------------------------------------------------
# 2.1  CALIBRATED PHENOTYPE LIBRARY
#      Hepatocyte params are calibrated against the hydrogel data (see §7).
#      Other lines use literature-anchored starting points; laminAC is the
#      key discriminating parameter (inferred from area, validated by LMNA qPCR).
# ---------------------------------------------------------------------------
PHENOTYPES = {
    # --- primary calibration target ---
    "hepatocyte": Phenotype("Primary hepatocyte (rat)",
        nm=45, nc=90, kc=1.1, alpha=0.13, laminAC=1.10,
        A_min=38.0, A_max=90.0, s0=15.0, tau=35.0),
    # --- lung ---
    "A549": Phenotype("A549 (lung adenocarcinoma)",
        nm=48, nc=95, kc=0.9, alpha=0.16, laminAC=0.80),
    "NHLF": Phenotype("NHLF (normal lung fibroblast)",
        nm=60, nc=120, kc=1.0, alpha=0.15, laminAC=1.00),
    "AT2_lung": Phenotype("AT2 alveolar epithelial",
        nm=40, nc=90, kc=1.2, alpha=0.12, laminAC=1.30),
    # --- breast ---
    "MCF10A": Phenotype("MCF10A (normal breast)",
        nm=52, nc=105, kc=1.0, alpha=0.15, laminAC=1.20),
    "MDA": Phenotype("MDA-MB-231 (invasive breast)",
        nm=45, nc=80, kc=0.7, alpha=0.18, laminAC=0.50),
    # --- generic ---
    "fibroblast": Phenotype("Generic fibroblast",
        nm=60, nc=120, kc=1.0, alpha=0.15, laminAC=1.00),
}


# ============================================================================
# 3.  MECHANOTRANSDUCTION CHAIN   E -> sigma -> {area, YAP}
# ============================================================================
def traction(E, ph: Phenotype, reps=6, T=10.0, dt=2e-4, burn=0.4):
    """Steady-state mean substrate traction T(E) (pN) via motor-clutch."""
    kappa = ph.alpha * E
    vals = [_mc_kernel(kappa, ph.nm, ph.Fm, ph.vu, ph.nc, ph.kon, ph.koff0,
                       ph.Fb, ph.kc, T, dt, burn, s) for s in range(reps)]
    return float(np.mean(vals))


def nuclear_stress(E, ph: Phenotype, reps=6):
    """Nuclear stress sigma(E) = T(E) * kappa/(kappa+kc)."""
    kappa = ph.alpha * E
    T = traction(E, ph, reps=reps)
    return T * kappa / (kappa + ph.kc)


def nuclear_area_ss(E, ph: Phenotype, reps=6):
    """Steady-state projected nuclear area (um^2). s_half = s0 * laminAC:
    a stiffer lamina resists flattening -> saturates at higher stress."""
    sig = nuclear_stress(E, ph, reps=reps)
    s_half = ph.s0 * ph.laminAC
    return ph.A_min + (ph.A_max - ph.A_min) * sig / (sig + s_half)


def yap_nc_ratio(E, ph: Phenotype,
                 s_thresh=8.0, s_width=2.5, s_scale=12.0, NC_max=5.0, reps=6):
    """YAP nucleocytoplasmic ratio. Lamin A/C gates the unwrinkling threshold
    (sigma*/lamin) and amplifies sustained tension. Resting ~1, up to ~NC_max."""
    sig = nuclear_stress(E, ph, reps=reps)
    unwrinkled = 1.0 / (1.0 + np.exp(-(sig - s_thresh / ph.laminAC) / s_width))
    drive = sig / (sig + s_scale)
    return 1.0 + (NC_max - 1.0) * unwrinkled * ph.laminAC * drive


def lamin_expected(E, ph: Phenotype, exponent=0.3, E_ref=0.5):
    """Expected lamin A/C level vs stiffness (Swift-Discher power-law scaling).
    Prediction to compare against LMNA qPCR."""
    return (E / E_ref) ** exponent


# ============================================================================
# 4.  TEMPORAL DYNAMICS  +  CONTACT INHIBITION  (integrin -> cadherin)
# ============================================================================
def confluence(t, t_c=18.0):
    """Confluence fraction c(t) = 1 - exp(-t/t_c)  (t in hours)."""
    return 1.0 - np.exp(-t / t_c)


def nc_effective(ph: Phenotype, t, beta=0.5, t_c=18.0):
    """Effective substrate clutches decrease as cell-cell contacts engage:
    nc_eff = nc0 * (1 - beta * confluence(t)).  (integrin -> cadherin switch)"""
    return max(int(round(ph.nc * (1.0 - beta * confluence(t, t_c)))), 5)


def nuclear_area_time(E, t, ph: Phenotype, A0=None, reps=6, contact_inhibition=False,
                      beta=0.5, t_c=18.0):
    """Nuclear area at time t (h): first-order relaxation toward the
    stiffness-set steady state with time constant tau. Optionally include
    contact inhibition (reduces effective clutches as confluence rises)."""
    if A0 is None:
        A0 = ph.A_min + 15.0
    ph_t = ph
    if contact_inhibition:
        ph_t = replace(ph, nc=nc_effective(ph, t, beta, t_c))
    A_ss = nuclear_area_ss(E, ph_t, reps=reps)
    return A_ss + (A0 - A_ss) * np.exp(-t / ph.tau)


# ============================================================================
# 5.  TWO-POPULATION MODEL  (basal binucleate + mechanosensitive)
# ============================================================================
#   Calibrated from hydrogel data (GMM deconvolution, BIC-selected):
#     basal population:          mean 37.9 um^2, CV 6%, CONSTANT (binucleate)
#     mechanosensitive pop:      mean ~68.8 um^2, grows with stiffness & time
# ---------------------------------------------------------------------------
BASAL_POP = dict(mean=37.9, sd=2.3, cv=0.06)         # binucleate, mechanically inert

def population_mixture(E, t, ph: Phenotype, phi=None, reps=6,
                       contact_inhibition=True):
    """Return (mu_basal, mu_mecano, phi) for the two-population area distribution.
    phi = basal fraction (rises with confluence if not given)."""
    mu_basal = BASAL_POP["mean"]
    mu_mecano = nuclear_area_time(E, t, ph, reps=reps,
                                  contact_inhibition=contact_inhibition)
    if phi is None:
        phi = 0.45 + 0.30 * confluence(t)             # basal fraction rises with time
    return mu_basal, mu_mecano, phi


# ============================================================================
# 6.  FIBROSIS  ->  STIFFNESS  ->  MECHANOTRANSDUCTION PREDICTION
# ============================================================================
#   METAVIR stage -> tissue stiffness (median SWE/TE elastography values, kPa)
# ---------------------------------------------------------------------------
FIBROSIS_STIFFNESS = {"F0": 2.5, "F1": 7.0, "F2": 9.5, "F3": 13.0, "F4": 26.0}
#   F0 healthy liver spans ~1-4 kPa; 2.5 kPa (midpoint) used as the single
#   representative value for the simulation. F4 cirrhosis: ~26 kPa (up to 48-69).

def fibrosis_prediction(ph: Phenotype = None, reps=8):
    """Predict mechanotransduction output across fibrosis stages F0->F4.
    Returns a dict of arrays keyed by stage list."""
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    stages = list(FIBROSIS_STIFFNESS.keys())
    E = np.array([FIBROSIS_STIFFNESS[s] for s in stages])
    sigma = np.array([nuclear_stress(e, ph, reps=reps) for e in E])
    yap = np.array([yap_nc_ratio(e, ph, reps=reps) for e in E])
    lamin = np.array([lamin_expected(e, ph) for e in E])
    return dict(stages=stages, E_kPa=E, sigma=sigma, yap=yap, lamin=lamin)


# ============================================================================
# 7.  CALIBRATION SUMMARY  (values fitted to the hydrogel data)
# ============================================================================
CALIBRATION = {
    "phenotype": "primary_rat_hepatocyte",
    "data": "~40,000 nuclei, DAPI Z-projection, 0.5/1/5/23 kPa x 2/12/24/36 h",
    "motor_clutch": dict(nm=45, Fm=2.0, vu=110.0, nc=90,
                         kon=0.5, koff0=0.1, Fb=2.0, kc=1.1, alpha=0.13),
    "two_populations": dict(
        basal_mean_um2=37.9, basal_cv=0.06,          # constant (binucleate)
        mechano_mean_um2=68.8,                        # grows with stiffness & time
        mechano_time_corr=0.53,                       # r vs time (p=0.04)
        basal_time_corr=-0.19,                        # r vs time (p=0.48, n.s.)
        one_pop_rejected_by_BIC="16/16 conditions"),
    "time_constant_tau_h": 35.3,                      # +/- 2.6 h
    "mechanism": "temporal dynamics (R2=0.56) > variable coupling (R2=0.33)",
    "lamin_validation": "inferred lamin vs LMNA qPCR: r=0.84",
    "stiffness_effect_pooled_eta2": 0.022,            # masked by averaging both pops
    "fibrosis_mapping_kPa": FIBROSIS_STIFFNESS,
    "fibrosis_rnaseq": "31/31 mechanosensitive genes up with stage; 17/31 convex",
}


# ============================================================================
# 8.  DEMO  /  SELF-TEST
# ============================================================================
def _demo():
    print("=" * 72)
    print("  MECHANOGENOMIC VIRTUAL CELL — calibrated model")
    print(f"  numba acceleration: {'ON' if HAS_NUMBA else 'OFF (pure python)'}")
    print("=" * 72)

    hep = PHENOTYPES["hepatocyte"]

    # --- 1. Nuclear stress & area vs stiffness (hepatocyte) ---
    print("\n[1] Hepatocyte: stiffness -> nuclear stress -> area")
    print(f"    {'E(kPa)':>7}{'sigma':>9}{'area(um2)':>11}{'YAP N/C':>9}")
    for E in [0.5, 1, 5, 23]:
        sig = nuclear_stress(E, hep, reps=6)
        area = nuclear_area_ss(E, hep, reps=6)
        yap = yap_nc_ratio(E, hep, reps=6)
        print(f"    {E:>7.1f}{sig:>9.1f}{area:>11.1f}{yap:>9.2f}")

    # --- 2. Two-population structure over time (23 kPa) ---
    print("\n[2] Two populations at 23 kPa (basal constant, mechano grows):")
    print(f"    {'t(h)':>6}{'mu_basal':>10}{'mu_mecano':>11}{'phi_basal':>11}")
    for t in [2, 12, 24, 36]:
        mb, mm, phi = population_mixture(23.0, t, hep, reps=4)
        print(f"    {t:>6d}{mb:>10.1f}{mm:>11.1f}{phi:>11.2f}")

    # --- 3. Phenotype discrimination (YAP N/C, soft vs stiff) ---
    print("\n[3] Phenotype discrimination (YAP N/C at 0.5 vs 20 kPa):")
    for key in ["MDA", "hepatocyte", "NHLF", "AT2_lung"]:
        ph = PHENOTYPES[key]
        lo = yap_nc_ratio(0.5, ph, reps=4)
        hi = yap_nc_ratio(20.0, ph, reps=4)
        print(f"    {ph.name:32s} {lo:.2f} -> {hi:.2f}  (lamin={ph.laminAC})")

    # --- 4. Fibrosis prediction ---
    print("\n[4] Fibrosis F0->F4 (stiffness -> mechanotransduction):")
    pred = fibrosis_prediction(hep, reps=6)
    print(f"    {'stage':>6}{'E(kPa)':>8}{'sigma':>8}{'YAP':>7}{'lamin':>8}")
    for i, s in enumerate(pred["stages"]):
        print(f"    {s:>6}{pred['E_kPa'][i]:>8.1f}{pred['sigma'][i]:>8.1f}"
              f"{pred['yap'][i]:>7.2f}{pred['lamin'][i]:>8.2f}")

    print("\n" + "=" * 72)
    print("  Calibration summary:")
    for k, v in CALIBRATION.items():
        print(f"    {k}: {v}")
    print("=" * 72)


if __name__ == "__main__":
    _demo()
