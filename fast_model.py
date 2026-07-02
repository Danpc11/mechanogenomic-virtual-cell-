"""
================================================================================
 fast_model.py  —  Fast analytic surrogate of the motor-clutch stress map
================================================================================

Symbolic regression (symbolic.py) showed that the stochastic motor-clutch
engine's stiffness->stress map is well described by a saturating (Michaelis-
Menten-like) form:

        sigma(E) = Vmax * E / (K + E)

This module fits that form to the real simulator once per phenotype, then uses
the closed-form expression for all downstream predictions. The result is a
~10^5x speed-up (microseconds vs seconds) with R^2 ~ 0.97-0.997, which makes
parameter sweeps, sensitivity analysis, and inference essentially free.

The stochastic engine in mvirtual_cell.py remains the ground truth; this is a
fast, interpretable approximation for when speed matters. Use `calibrate()` to
(re)fit Vmax and K for any phenotype.

Author: Daniel Pérez-Calixto (INMEGEN / UNAM)
================================================================================
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
import mvirtual_cell as mvc
from mvirtual_cell import PHENOTYPES


# ============================================================================
# 1.  PRE-CALIBRATED SATURATING PARAMETERS  (fit to the stochastic motor)
# ============================================================================
#   sigma(E) = Vmax * E / (K + E).  Fit R^2 in parentheses.
# ---------------------------------------------------------------------------
SATURATING_PARAMS = {
    "hepatocyte": dict(Vmax=65.0, K=5.01),   # R2 = 0.982
    "MDA":        dict(Vmax=61.1, K=2.12),   # R2 = 0.975
    "AT2_lung":   dict(Vmax=72.0, K=8.50),   # R2 = 0.997
    "NHLF":       dict(Vmax=87.8, K=3.96),   # R2 = 0.982
}


# ============================================================================
# 2.  FAST ANALYTIC MAPS
# ============================================================================
def nuclear_stress_fast(E, phenotype_key="hepatocyte", Vmax=None, K=None):
    """Closed-form nuclear stress sigma(E) = Vmax*E/(K+E). Instant, deterministic.

    Provide either a phenotype key (uses pre-calibrated Vmax, K) or explicit
    Vmax and K (e.g. from calibrate())."""
    if Vmax is None or K is None:
        p = SATURATING_PARAMS.get(phenotype_key)
        if p is None:
            raise KeyError(f"No saturating params for '{phenotype_key}'; "
                           f"call calibrate() first.")
        Vmax, K = p["Vmax"], p["K"]
    E = np.asarray(E, dtype=float)
    return Vmax * E / (K + E)


def nuclear_area_fast(E, ph, phenotype_key="hepatocyte", Vmax=None, K=None):
    """Steady-state nuclear area using the fast analytic stress.
    Uses the phenotype's A_min, A_max, s0, laminAC for the area map."""
    sig = nuclear_stress_fast(E, phenotype_key, Vmax, K)
    s_half = ph.s0 * ph.laminAC
    return ph.A_min + (ph.A_max - ph.A_min) * sig / (sig + s_half)


def yap_nc_fast(E, ph, phenotype_key="hepatocyte", Vmax=None, K=None,
                s_thresh=8.0, s_width=2.5, s_scale=12.0, NC_max=5.0):
    """YAP N/C using the fast analytic stress (mirrors mvc.yap_nc_ratio)."""
    sig = nuclear_stress_fast(E, phenotype_key, Vmax, K)
    unwrinkled = 1.0 / (1.0 + np.exp(-(sig - s_thresh / ph.laminAC) / s_width))
    drive = sig / (sig + s_scale)
    return 1.0 + (NC_max - 1.0) * unwrinkled * ph.laminAC * drive


# ============================================================================
# 3.  CALIBRATION  (fit Vmax, K to the stochastic motor for any phenotype)
# ============================================================================
def calibrate(ph, n_points=25, reps=10, E_range=(0.3, 40.0), seed=0):
    """Fit the saturating form to the stochastic motor for a phenotype.
    Returns dict(Vmax, K, R2)."""
    E = np.logspace(np.log10(E_range[0]), np.log10(E_range[1]), n_points)
    sig = np.array([mvc.nuclear_stress(e, ph, reps=reps) for e in E])
    popt, _ = curve_fit(lambda x, Vm, Kk: Vm * x / (Kk + x), E, sig,
                        p0=[sig.max(), np.median(E)], maxfev=10000)
    pred = popt[0] * E / (popt[1] + E)
    R2 = 1.0 - np.sum((sig - pred) ** 2) / np.sum((sig - sig.mean()) ** 2)
    return dict(Vmax=float(popt[0]), K=float(popt[1]), R2=float(R2))


def validate_against_motor(ph, phenotype_key="hepatocyte",
                           Es=(0.5, 1, 5, 23), reps=8):
    """Compare the fast analytic stress to the stochastic motor at test points."""
    out = []
    for E in Es:
        stoch = mvc.nuclear_stress(E, ph, reps=reps)
        fast = float(nuclear_stress_fast(E, phenotype_key))
        out.append(dict(E=E, stochastic=stoch, fast=fast,
                        rel_err=abs(fast - stoch) / max(stoch, 1e-9)))
    return out


# ============================================================================
# 4.  DEMO
# ============================================================================
def _demo():
    print("=" * 72)
    print("  FAST ANALYTIC MODEL  —  sigma(E) = Vmax*E/(K+E)")
    print("=" * 72)

    ph = PHENOTYPES["hepatocyte"]

    print("\n  Pre-calibrated saturating parameters:")
    for k, p in SATURATING_PARAMS.items():
        print(f"    {k:>12}: Vmax={p['Vmax']:.1f}, K={p['K']:.2f}")

    print("\n  Fast vs stochastic nuclear stress (hepatocyte):")
    print(f"    {'E(kPa)':>7}{'stochastic':>12}{'fast':>8}{'rel.err':>9}")
    for r in validate_against_motor(ph):
        print(f"    {r['E']:7.1f}{r['stochastic']:12.1f}{r['fast']:8.1f}"
              f"{r['rel_err']*100:8.1f}%")

    import time
    t0 = time.time()
    for _ in range(1000):
        nuclear_stress_fast(np.array([0.5, 1, 5, 23]))
    dt = time.time() - t0
    print(f"\n  1000 evaluations (4 stiffnesses each) in {dt*1000:.1f} ms")
    print(f"  -> ~{dt/1000*1e6:.1f} µs per call (vs ~1 s for the stochastic motor)")

    print("\n  Recalibrating for a phenotype from scratch:")
    fit = calibrate(PHENOTYPES["MCF10A"], reps=8)
    print(f"    MCF10A: Vmax={fit['Vmax']:.1f}, K={fit['K']:.2f}, R2={fit['R2']:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    _demo()
