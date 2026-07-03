"""
================================================================================
 symbolic.py  —  Symbolic regression on the virtual-cell model
================================================================================

Two of the model's maps are phenomenological rather than derived:

  (a) stiffness -> nuclear stress:   sigma(E)   [emergent from the stochastic
      motor-clutch engine; we currently have no closed form]
  (b) stiffness -> mechanotransduction output that drives gene expression.

Symbolic regression searches the space of mathematical expressions for a
compact closed-form formula that reproduces the simulator's behavior. Unlike a
neural network, the output is an interpretable equation a physicist can inspect,
sanity-check against limits, and put in a paper.

This is "AI discovering the functional form", not a black box: we run the
expensive stochastic simulator to generate (E -> sigma) data, then let genetic
programming find sigma ≈ f(E) in elementary functions.

Engine: gplearn (pure-python genetic programming; no Julia/torch needed).
Author: Daniel Pérez-Calixto (INMEGEN / UNAM)
================================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import replace
import mvirtual_cell as mvc
from mvirtual_cell import PHENOTYPES

try:
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except ImportError:
    HAS_GPLEARN = False


# ============================================================================
# 1.  GENERATE TRAINING DATA FROM THE SIMULATOR
# ============================================================================
def generate_stress_data(ph=None, E_range=(0.3, 40.0), n_points=40, reps=10,
                         log_spaced=True, seed=0):
    """Run the motor-clutch engine to produce (E, sigma) training pairs.
    Averaged over replicates for stable targets."""
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    if log_spaced:
        E = np.logspace(np.log10(E_range[0]), np.log10(E_range[1]), n_points)
    else:
        E = np.linspace(E_range[0], E_range[1], n_points)
    sigma = np.array([mvc.nuclear_stress(e, ph, reps=reps) for e in E])
    return E, sigma


# ============================================================================
# 2.  SYMBOLIC REGRESSION
# ============================================================================
def fit_symbolic(X, y, generations=20, population=2000, seed=0, verbose=True):
    """Run genetic-programming symbolic regression y ≈ f(X).

    X : (n, d) feature array (e.g. E, or [E, nc, ...])
    y : (n,) target (e.g. sigma)
    Returns the fitted SymbolicRegressor and the best program as a string."""
    if not HAS_GPLEARN:
        raise ImportError("gplearn is required (pip install gplearn)")
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    est = SymbolicRegressor(
        population_size=population,
        generations=generations,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "inv"),
        metric="rmse",
        parsimony_coefficient=0.05,     # strong complexity penalty -> compact formulas
        const_range=(-5.0, 5.0),
        random_state=seed,
        verbose=1 if verbose else 0,
        n_jobs=1,
    )
    est.fit(X, y)
    return est, str(est._program)


def evaluate_formula(est, X, y):
    """R^2 of a fitted symbolic model on (X, y)."""
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    pred = est.predict(X)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ============================================================================
# 3.  COMPARE AGAINST CANDIDATE PHYSICAL FORMS
# ============================================================================
def compare_physical_forms(E, sigma):
    """Fit a few interpretable candidate forms and report R^2, so the symbolic
    result can be judged against physically motivated baselines:
        - linear:      sigma = a*E + b
        - power law:   sigma = a*E^b
        - saturating:  sigma = Vmax*E/(K+E)   (Michaelis-Menten-like)
        - log:         sigma = a*log(E) + b
    """
    from scipy.optimize import curve_fit
    E = np.asarray(E, float); sigma = np.asarray(sigma, float)

    def r2(pred):
        return 1 - np.sum((sigma - pred) ** 2) / np.sum((sigma - sigma.mean()) ** 2)

    out = {}
    # linear
    a, b = np.polyfit(E, sigma, 1)
    out["linear (aE+b)"] = r2(a * E + b)
    # power law
    try:
        p, _ = curve_fit(lambda x, a, b: a * np.power(x, b), E, sigma,
                         p0=[1, 0.5], maxfev=10000)
        out["power (a·E^b)"] = (r2(p[0] * E ** p[1]), f"a={p[0]:.2f}, b={p[1]:.2f}")
    except Exception:
        out["power (a·E^b)"] = (float("nan"), "fit failed")
    # saturating
    try:
        p, _ = curve_fit(lambda x, Vm, K: Vm * x / (K + x), E, sigma,
                         p0=[sigma.max(), np.median(E)], maxfev=10000)
        out["saturating (Vmax·E/(K+E))"] = (r2(p[0] * E / (p[1] + E)),
                                            f"Vmax={p[0]:.1f}, K={p[1]:.2f}")
    except Exception:
        out["saturating (Vmax·E/(K+E))"] = (float("nan"), "fit failed")
    # log
    a, b = np.polyfit(np.log(E), sigma, 1)
    out["log (a·lnE+b)"] = r2(a * np.log(E) + b)
    return out


# ============================================================================
# 4.  DEMO
# ============================================================================
def _demo():
    print("=" * 72)
    print("  SYMBOLIC REGRESSION on the motor-clutch stress map sigma(E)")
    print("=" * 72)

    print("\n  Generating (E, sigma) data from the simulator...")
    E, sigma = generate_stress_data(n_points=30, reps=8)
    print(f"    {len(E)} points, E in [{E.min():.1f}, {E.max():.1f}] kPa, "
          f"sigma in [{sigma.min():.1f}, {sigma.max():.1f}]")

    # --- candidate physical forms first (interpretable baselines) ---
    print("\n  Candidate physical forms (R^2):")
    forms = compare_physical_forms(E, sigma)
    for name, val in forms.items():
        if isinstance(val, tuple):
            print(f"    {name:28s} R2={val[0]:.3f}   {val[1]}")
        else:
            print(f"    {name:28s} R2={val:.3f}")

    # --- symbolic regression search ---
    if not HAS_GPLEARN:
        print("\n  gplearn not installed; skipping symbolic search.")
        print("  (pip install gplearn)")
        print("=" * 72)
        return

    print("\n  Running symbolic regression (genetic programming)...")
    # normalize E to help the search; keep it simple with single feature
    est, program = fit_symbolic(E.reshape(-1, 1), sigma,
                                generations=15, population=1500, verbose=False)
    r2 = evaluate_formula(est, E.reshape(-1, 1), sigma)
    print(f"\n  Discovered formula (X0 = E):")
    print(f"    sigma ≈ {program}")
    print(f"    R^2 = {r2:.3f}")

    print("\n  Interpretation:")
    print("    Compare the discovered form and its R^2 against the physical")
    print("    baselines above. A saturating form typically wins, consistent")
    print("    with clutch-limited force transmission at high stiffness.")
    print("=" * 72)


if __name__ == "__main__":
    _demo()
