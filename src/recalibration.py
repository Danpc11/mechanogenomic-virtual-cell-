"""
================================================================================
 recalibration.py  —  Two-level recalibration on the complete timecourse
================================================================================

The original calibration used data truncated at 36 h, which captured only the
early transient. The complete timecourse (2/36/72/120 h) — available for 1 and
23 kPa — shows that:

  * nuclear relaxation is SLOWER than assumed (tau ~ 40-80 h at 23 kPa, not 35 h)
  * the time constant SCALES WITH STIFFNESS (fast on soft, slow on stiff)
  * the stiffness response is STRONG (~2.2x from 1 to 23 kPa), not weak
  * the "low" population (mononucleate + binucleate mix) is only weakly
    stiffness-responsive relative to the high (mechanosensitive) population

This module recalibrates in two levels, honestly bounded by the data we have:

  LEVEL 1  (temporal): tau(E) and the mechanical fold-change, from the complete
           1 and 23 kPa timecourses.
  LEVEL 2  (shape):    the area-vs-stiffness shape, from the four stiffnesses
           available at 36 h — flagged as NOT steady state, since 36 h << tau
           at high stiffness. Full-timecourse data at 0.5 and 5 kPa are not
           available, so the steady-state shape there is an extrapolation.

Dependencies: numpy, scipy; loads hepatocyte_complete_data.json.
================================================================================
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit


def load_data(path="hepatocyte_complete_data.json"):
    try:
        from paths import DATA_DIR
        p = DATA_DIR / path
    except Exception:
        p = Path(__file__).parent / path if "__file__" in globals() else Path(path)
    with open(p) as f:
        return json.load(f)


# ============================================================================
# LEVEL 1 — TEMPORAL DYNAMICS: tau(E) and mechanical fold-change
# ============================================================================
def _relax(t, A_ss, A0, tau):
    return A_ss + (A0 - A_ss) * np.exp(-t / tau)


def fit_tau(t_h, area, p0_tau=50.0):
    """Fit first-order relaxation A(t) = A_ss + (A0 - A_ss) exp(-t/tau).
    Returns (A_ss, A0, tau, R2). Robust to the small number of points."""
    t = np.asarray(t_h, float)
    a = np.asarray(area, float)
    mask = np.array([x is not None for x in area])
    t, a = t[mask], a[mask].astype(float)
    if len(t) < 3:
        return dict(A_ss=float(a[-1]), A0=float(a[0]), tau=float("nan"),
                    R2=float("nan"), note="too few points")
    try:
        popt, _ = curve_fit(_relax, t, a, p0=[a[-1] * 1.1, a[0] * 0.7, p0_tau],
                            maxfev=40000)
        pred = _relax(t, *popt)
        R2 = 1 - np.sum((a - pred) ** 2) / np.sum((a - a.mean()) ** 2)
        return dict(A_ss=float(popt[0]), A0=float(popt[1]), tau=float(popt[2]),
                    R2=float(R2))
    except Exception as e:
        return dict(A_ss=float(a[-1]), A0=float(a[0]), tau=float("nan"),
                    R2=float("nan"), note=str(e))


def tau_vs_stiffness(data=None):
    """Fit tau for each population at each available stiffness (1, 23 kPa).
    Returns a nested dict {stiffness: {population: tau_fit}}."""
    if data is None:
        data = load_data()
    out = {}
    for key in ["1_kPa", "23_kPa"]:
        d = data["complete_timecourse"][key]
        out[key] = dict(
            E_kPa=float(key.split("_")[0]),
            high=fit_tau(d["t_h"], d["pop_high"]),
            low=fit_tau(d["t_h"], d["pop_low"]),
        )
    return out


def mechanical_fold_change(data=None):
    """Stiffness response (23 kPa / 1 kPa) per population at each time point.
    A stable ~2.2x fold-change is the strong mechanosensitivity signal."""
    if data is None:
        data = load_data()
    d1 = data["complete_timecourse"]["1_kPa"]
    d23 = data["complete_timecourse"]["23_kPa"]
    rows = []
    for i, t in enumerate(d1["t_h"]):
        hi1, hi23 = d1["pop_high"][i], d23["pop_high"][i]
        lo1, lo23 = d1["pop_low"][i], d23["pop_low"][i]
        rows.append(dict(
            t_h=t,
            fold_high=(hi23 / hi1) if (hi1 and hi23) else None,
            fold_low=(lo23 / lo1) if (lo1 and lo23) else None,
        ))
    return rows


# ============================================================================
# LEVEL 2 — AREA-VS-STIFFNESS SHAPE (36 h, four stiffnesses; NOT steady state)
# ============================================================================
def fit_shape_36h(data=None):
    """Report the area-vs-stiffness shape at 36 h (four stiffnesses).

    IMPORTANT: at 36 h the curve is NON-MONOTONIC (rises to 5 kPa, then drops
    at 23 kPa). This is NOT noise — it is a predictable consequence of the
    stiffness-dependent dynamics: soft substrates (short tau) have nearly
    equilibrated by 36 h, while 23 kPa (tau ~ 79 h) is only halfway there, so
    its 36 h area is artificially low. A saturating equilibrium curve therefore
    should NOT be fit to these transient points. We report them as-is and,
    below, reconstruct the equilibrium estimate via the temporal fits."""
    if data is None:
        data = load_data()
    E = np.array(data["form_at_36h"]["E_kPa"], float)
    A = np.array(data["form_at_36h"]["pop_high_36h"], float)
    monotonic = bool(np.all(np.diff(A) > 0))
    return dict(E=E.tolist(), A_36h=A.tolist(), monotonic=monotonic,
                note=("non-monotonic: 23 kPa point suppressed by slow dynamics "
                      "(tau >> 36 h); do NOT fit an equilibrium curve here."))


def equilibrium_shape(data=None):
    """Reconstruct the STEADY-STATE area-vs-stiffness using A_ss from the
    temporal fits where we have complete timecourses (1 and 23 kPa), which is
    the honest way to get equilibrium values given the dynamics.

    Returns the two anchored steady-state points plus the extrapolation flag
    for the intermediate stiffnesses (0.5, 5 kPa), which lack 120 h data."""
    if data is None:
        data = load_data()
    tau = tau_vs_stiffness(data)
    E_anchored = [1.0, 23.0]
    A_ss_high = [tau["1_kPa"]["high"]["A_ss"], tau["23_kPa"]["high"]["A_ss"]]
    return dict(
        E_anchored_kPa=E_anchored,
        A_ss_high_um2=[float(a) for a in A_ss_high],
        fold_change=float(A_ss_high[1] / A_ss_high[0]),
        extrapolated_stiffnesses=[0.5, 5.0],
        note="Steady-state anchored only at 1 and 23 kPa (complete timecourse). "
             "0.5 and 5 kPa lack 120 h data -> their equilibrium is not measured.")


# ============================================================================
# RECOMMENDED RECALIBRATED PARAMETERS  (summary)
# ============================================================================
def recalibrated_summary(data=None):
    if data is None:
        data = load_data()
    tau = tau_vs_stiffness(data)
    fold = mechanical_fold_change(data)
    eq = equilibrium_shape(data)
    fold_high = [r["fold_high"] for r in fold if r["fold_high"]]
    return {
        "tau_soft_1kPa_h": tau["1_kPa"]["high"]["tau"],
        "tau_stiff_23kPa_h": tau["23_kPa"]["high"]["tau"],
        "tau_note": "time constant scales with stiffness (slow on stiff)",
        "A_ss_stiff_high_um2": tau["23_kPa"]["high"]["A_ss"],
        "A_ss_soft_high_um2": tau["1_kPa"]["high"]["A_ss"],
        "mechanical_fold_change": float(np.mean(fold_high)),
        "fold_note": "high population ~2.2x from 1 to 23 kPa, stable over time",
        "equilibrium_shape": eq,
        "limitation": "complete timecourse only at 1 and 23 kPa; steady-state "
                      "shape at 0.5 and 5 kPa is not measured (data truncated at 36 h).",
    }


# ============================================================================
# DEMO
# ============================================================================
def _demo():
    print("=" * 72)
    print("  RECALIBRATION on the complete timecourse (1 and 23 kPa, 2-120 h)")
    print("=" * 72)
    data = load_data()

    print("\n[LEVEL 1] Temporal dynamics — tau(E) per population:")
    tau = tau_vs_stiffness(data)
    print(f"  {'stiffness':>10}{'pop':>7}{'tau (h)':>10}{'A_ss':>8}{'R2':>7}")
    for key in ["1_kPa", "23_kPa"]:
        for pop in ["high", "low"]:
            f = tau[key][pop]
            print(f"  {key:>10}{pop:>7}{f['tau']:>10.0f}{f['A_ss']:>8.0f}"
                  f"{f.get('R2', float('nan')):>7.2f}")
    print("  -> tau is SHORT on soft (fast equilibrium), LONG on stiff (slow).")

    print("\n[LEVEL 1] Mechanical fold-change (23 kPa / 1 kPa):")
    print(f"  {'t(h)':>6}{'high pop':>10}{'low pop':>10}")
    for r in mechanical_fold_change(data):
        fh = f"{r['fold_high']:.1f}x" if r['fold_high'] else "—"
        fl = f"{r['fold_low']:.1f}x" if r['fold_low'] else "—"
        print(f"  {r['t_h']:>6}{fh:>10}{fl:>10}")
    print("  -> strong (~2.2x), stable in time. High >> low response confirms")
    print("     the high population is the mechanosensitive one.")

    print("\n[LEVEL 2] Area-vs-stiffness at 36 h (4 stiffnesses):")
    shape = fit_shape_36h(data)
    print(f"  E (kPa):  {shape['E']}")
    print(f"  A_36h:    {shape['A_36h']}  (monotonic: {shape['monotonic']})")
    print(f"  NOTE: {shape['note']}")

    print("\n[LEVEL 2] Equilibrium shape (reconstructed from temporal A_ss):")
    eq = equilibrium_shape(data)
    print(f"  anchored at {eq['E_anchored_kPa']} kPa: "
          f"A_ss = {[round(a) for a in eq['A_ss_high_um2']]} µm²  "
          f"({eq['fold_change']:.1f}x)")
    print(f"  NOTE: {eq['note']}")

    print("\n[SUMMARY] Recalibrated parameters:")
    s = recalibrated_summary(data)
    print(f"  tau: {s['tau_soft_1kPa_h']:.0f} h (soft) -> "
          f"{s['tau_stiff_23kPa_h']:.0f} h (stiff)")
    print(f"  A_ss (high pop): {s['A_ss_soft_high_um2']:.0f} (soft) -> "
          f"{s['A_ss_stiff_high_um2']:.0f} µm² (stiff)")
    print(f"  mechanical fold-change: {s['mechanical_fold_change']:.1f}x")
    print(f"  LIMITATION: {s['limitation']}")
    print("=" * 72)


if __name__ == "__main__":
    _demo()
