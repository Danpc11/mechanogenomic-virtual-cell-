"""
================================================================================
 inference.py  —  Simulation-based inference for the virtual-cell model
================================================================================

The motor-clutch engine is a stochastic simulator with an intractable
likelihood, so point-fitting (curve_fit) gives no uncertainty and cannot tell
which parameters are actually identified by the data. This module infers the
POSTERIOR distribution over physical parameters using likelihood-free /
simulation-based inference.

Two engines are provided, both built directly on mvirtual_cell (no retraining,
no black box — the physical simulator IS the model):

  * ABC-SMC   : sequential Monte Carlo Approximate Bayesian Computation.
                Dependencies: numpy, scipy only (lightweight, always available).
  * (optional) neural posterior via `sbi` if installed — see `try_sbi_npe`.

The posterior tells you, e.g., "nc = 90 ± 15, and kon is NOT identified by area
data alone" — exactly the identifiability statement a reviewer wants, and the
quantitative version of the lamin-identifiability claim in the manuscript.
================================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import replace
import mvirtual_cell as mvc
from mvirtual_cell import PHENOTYPES


# ============================================================================
# 0.  FAST EMULATOR OF THE MOTOR  (makes SBI tractable)
# ============================================================================
#   The motor-clutch engine costs ~1 s per evaluation, so ABC (thousands of
#   calls) would take hours. We build a cheap surrogate by simulating nuclear
#   stress on a grid of (E, nc, nm, kc, alpha, laminAC) once, then interpolate.
#   ABC then runs on the emulator (microseconds), not the raw simulator.
#   This is the standard SBI pattern: amortize the expensive simulator up front.
# ---------------------------------------------------------------------------
class MotorEmulator:
    """Grid-based emulator of nuclear_area_ss(E; phenotype params).

    Trains once on the real simulator over a coarse grid, caches it, and
    provides instant predictions by multilinear interpolation. Accuracy is
    traded for a ~10^4x speed-up, which is what makes posterior inference
    feasible without GPUs."""

    def __init__(self, base=None, reps=6):
        self.base = base or PHENOTYPES["hepatocyte"]
        self.reps = reps
        self.trained = False

    def train(self, Es, nc_grid=None, lamin_grid=None, verbose=True):
        """Build the emulator over (E, nc, laminAC) — the parameters that most
        shape the area curve. Other params stay at base values.

        Note: nuclear_area_ss = A_min + (A_max-A_min)*sigma/(sigma+s0*lamin),
        and sigma depends on the motor only through (E, nc, ...). We tabulate
        sigma(E, nc) once, then apply the (cheap, analytic) lamin/area map at
        query time — so laminAC needs no extra simulation."""
        from mvirtual_cell import nuclear_stress
        self.Es = np.asarray(Es, dtype=float)
        self.nc_grid = np.asarray(nc_grid if nc_grid is not None
                                  else np.linspace(40, 160, 9), dtype=float)
        # tabulate sigma(E, nc) on the grid (the only expensive part)
        self.sigma_tab = np.empty((len(self.Es), len(self.nc_grid)))
        for j, nc in enumerate(self.nc_grid):
            ph = replace(self.base, nc=int(round(nc)))
            for i, E in enumerate(self.Es):
                self.sigma_tab[i, j] = nuclear_stress(E, ph, reps=self.reps)
        self.trained = True
        if verbose:
            print(f"  emulator trained on {len(self.Es)}x{len(self.nc_grid)} grid "
                  f"({self.sigma_tab.size} simulations)")
        return self

    def sigma(self, E, nc):
        """Interpolate nuclear stress at (E, nc)."""
        i = np.interp(E, self.Es, np.arange(len(self.Es)))
        j = np.interp(nc, self.nc_grid, np.arange(len(self.nc_grid)))
        i0, j0 = int(np.floor(i)), int(np.floor(j))
        i1 = min(i0 + 1, len(self.Es) - 1)
        j1 = min(j0 + 1, len(self.nc_grid) - 1)
        di, dj = i - i0, j - j0
        s = (self.sigma_tab[i0, j0] * (1 - di) * (1 - dj)
             + self.sigma_tab[i1, j0] * di * (1 - dj)
             + self.sigma_tab[i0, j1] * (1 - di) * dj
             + self.sigma_tab[i1, j1] * di * dj)
        return s

    def area(self, E, nc, laminAC, A_min=None, A_max=None, s0=None):
        """Predict steady-state nuclear area (analytic lamin/area map on top of
        the interpolated stress) — instant, no simulation."""
        A_min = self.base.A_min if A_min is None else A_min
        A_max = self.base.A_max if A_max is None else A_max
        s0 = self.base.s0 if s0 is None else s0
        sig = self.sigma(E, nc)
        return A_min + (A_max - A_min) * sig / (sig + s0 * laminAC)

    def area_curve(self, nc, laminAC):
        return np.array([self.area(E, nc, laminAC) for E in self.Es])


# ============================================================================
# 1.  PARAMETER SPACE & PRIORS
# ============================================================================
#   Each parameter: (name, low, high) uniform prior.
#   NOTE: the fast ABC-SMC path (abc_smc + MotorEmulator) currently supports
#   ONLY priors over {nc, laminAC}. The wider set below (nm, kc, alpha) is the
#   target space for the optional neural path (try_sbi_npe) or a future full
#   emulator grid; abc_smc() will raise NotImplementedError for those.
# ---------------------------------------------------------------------------
ABC_SUPPORTED = {"nc", "laminAC"}          # what abc_smc() can infer today
DEFAULT_PRIORS = {
    "nc":      (40.0, 160.0),   # substrate clutches      [abc_smc supported]
    "nm":      (25.0, 90.0),    # myosin motors           [sbi/neural only]
    "kc":      (0.5, 2.0),      # clutch stiffness (pN/nm)[sbi/neural only]
    "alpha":   (0.05, 0.25),    # E -> kappa coupling     [sbi/neural only]
    "laminAC": (0.3, 2.5),      # relative lamin A/C      [abc_smc supported]
}


def sample_prior(priors, n, rng):
    """Draw n samples from independent uniform priors. Returns (n, d) array."""
    names = list(priors)
    lows = np.array([priors[k][0] for k in names])
    highs = np.array([priors[k][1] for k in names])
    return names, rng.uniform(lows, highs, size=(n, len(names)))


def _make_phenotype(base, names, theta):
    """Build a Phenotype from a base + a parameter vector (ints where needed)."""
    kw = {}
    for k, v in zip(names, theta):
        kw[k] = int(round(v)) if k in ("nc", "nm") else float(v)
    return replace(base, **kw)


# ============================================================================
# 2.  SUMMARY STATISTICS & DISTANCE
# ============================================================================
def area_summary(ph, Es, reps=4):
    """Summary statistic vector: steady-state nuclear area at each stiffness.
    (This is the observable the lab measures.)"""
    return np.array([mvc.nuclear_area_ss(E, ph, reps=reps) for E in Es])


def simulate_summary(base, names, theta, Es, reps=4):
    ph = _make_phenotype(base, names, theta)
    return area_summary(ph, Es, reps=reps)


def distance(s_sim, s_obs):
    """Euclidean distance between summary vectors (normalized by observed scale)."""
    s_obs = np.asarray(s_obs, dtype=float)
    scale = np.maximum(np.abs(s_obs), 1.0)
    return float(np.sqrt(np.mean(((s_sim - s_obs) / scale) ** 2)))


# ============================================================================
# 3.  ABC-SMC  (sequential Monte Carlo ABC)
# ============================================================================
def abc_smc(observed_summary, Es, emulator=None, priors=None, base=None,
            n_particles=400, n_rounds=5, alpha=0.5, seed=0, verbose=True):
    """Infer the posterior over (nc, laminAC) by ABC-SMC, run on a fast
    emulator of the motor (see MotorEmulator). Because the emulator is
    instantaneous, thousands of ABC evaluations take seconds.

    Returns dict with the posterior sample, per-parameter mean/std/CI, and the
    identifiability score (1 - posterior_std/prior_std: ~0 unidentified,
    ~1 well constrained)."""
    if priors is None:
        priors = {"nc": (40.0, 160.0), "laminAC": (0.3, 2.5)}
    if set(priors) != {"nc", "laminAC"}:
        raise NotImplementedError(
            "The current ABC emulator supports only priors over 'nc' and "
            "'laminAC' (it tabulates sigma(E, nc) and applies laminAC "
            "analytically). For nm/kc/alpha, use try_sbi_npe() or extend "
            "MotorEmulator to a full grid.")
    if base is None:
        base = PHENOTYPES["hepatocyte"]
    if emulator is None:
        emulator = MotorEmulator(base=base).train(Es, verbose=verbose)

    rng = np.random.default_rng(seed)
    names = list(priors)          # must be ["nc", "laminAC"] for the emulator
    d = len(names)
    lows = np.array([priors[k][0] for k in names])
    highs = np.array([priors[k][1] for k in names])
    obs = np.asarray(observed_summary, dtype=float)
    scale = np.maximum(np.abs(obs), 1.0)

    def sim_dist(theta):
        nc, lam = theta[names.index("nc")], theta[names.index("laminAC")]
        s = emulator.area_curve(nc, lam)
        return float(np.sqrt(np.mean(((s - obs) / scale) ** 2)))

    # round 0: prior sampling
    theta = rng.uniform(lows, highs, size=(n_particles * 8, d))
    dists = np.array([sim_dist(t) for t in theta])
    keep = np.argsort(dists)[:n_particles]
    particles = theta[keep]
    weights = np.ones(n_particles) / n_particles
    tol = np.quantile(dists[keep], alpha)
    if verbose:
        print(f"  round 0: tol={tol:.4f}  best d={dists[keep][0]:.4f}")

    for r in range(1, n_rounds):
        cov = np.cov(particles.T) + 1e-6 * np.eye(d)
        L = np.linalg.cholesky(cov)
        new, new_d, acc, tries = np.empty_like(particles), np.empty(n_particles), 0, 0
        while acc < n_particles and tries < n_particles * 500:
            cand = particles[rng.choice(n_particles, p=weights)] + L @ rng.standard_normal(d)
            tries += 1
            if np.any(cand < lows) or np.any(cand > highs):
                continue
            dc = sim_dist(cand)
            if dc <= tol:
                new[acc], new_d[acc] = cand, dc
                acc += 1
        if acc > 0:
            particles, dd = new[:acc], new_d[:acc]
            tol = np.quantile(dd, alpha)
            weights = np.ones(acc) / acc
        if verbose:
            print(f"  round {r}: accepted={acc}  tol={tol:.4f}  best d={dd.min():.4f}")

    prior_std = (highs - lows) / np.sqrt(12)
    post_mean, post_std = particles.mean(0), particles.std(0)
    ident = 1.0 - post_std / prior_std
    summary = {k: dict(mean=float(post_mean[i]), std=float(post_std[i]),
                       ci95=(float(np.percentile(particles[:, i], 2.5)),
                             float(np.percentile(particles[:, i], 97.5))),
                       identifiability=float(ident[i]))
               for i, k in enumerate(names)}
    return dict(names=names, posterior=particles, summary=summary, tol=tol,
                emulator=emulator)


# ============================================================================
# 4.  OPTIONAL: neural posterior estimation via `sbi` (if installed)
# ============================================================================
def try_sbi_npe(observed_summary, Es, priors=None, base=None,
                n_simulations=2000, reps=4, seed=0):
    """If the `sbi` package is available, run neural posterior estimation.
    Falls back with a clear message if sbi/torch are not installed."""
    try:
        import torch
        from sbi.inference import SNPE
        from sbi.utils import BoxUniform
    except ImportError:
        return {"available": False,
                "message": "sbi/torch not installed; use abc_smc() instead."}
    if priors is None:
        priors = DEFAULT_PRIORS
    if base is None:
        base = PHENOTYPES["hepatocyte"]
    names = list(priors)
    low = torch.tensor([priors[k][0] for k in names])
    high = torch.tensor([priors[k][1] for k in names])
    prior = BoxUniform(low=low, high=high)

    def simulator(theta):
        t = theta.numpy()
        return torch.tensor(simulate_summary(base, names, t, Es, reps),
                            dtype=torch.float32)

    theta = prior.sample((n_simulations,))
    x = torch.stack([simulator(t) for t in theta])
    inference = SNPE(prior=prior)
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()
    samples = posterior.sample((2000,), x=torch.tensor(observed_summary,
                                                       dtype=torch.float32))
    s = samples.numpy()
    return {"available": True, "names": names,
            "posterior_mean": s.mean(0).tolist(),
            "posterior_std": s.std(0).tolist()}


# ============================================================================
# 5.  DEMO  (recover known parameters from synthetic data, with uncertainty)
# ============================================================================
def _demo():
    print("=" * 72)
    print("  SIMULATION-BASED INFERENCE (ABC-SMC) on the motor-clutch model")
    print("=" * 72)

    base = PHENOTYPES["hepatocyte"]
    Es = [0.5, 1, 2, 5, 10, 23]

    # ground-truth phenotype -> synthetic "observed" areas
    truth = replace(base, nc=110, laminAC=1.3)
    rng = np.random.default_rng(1)
    obs = area_summary(truth, Es, reps=8) + rng.normal(0, 1.0, len(Es))
    print(f"\n  Ground truth: nc={truth.nc}, laminAC={truth.laminAC}")
    print(f"  Observed areas: {np.round(obs,1)}")

    # train the fast emulator once, then run ABC on it
    print("\n  Training emulator (one-time cost)...")
    emu = MotorEmulator(base=base, reps=6).train(Es)

    print("  Running ABC-SMC on the emulator...")
    res = abc_smc(obs, Es, emulator=emu, base=base,
                  n_particles=400, n_rounds=5, seed=0)

    print("\n  Posterior (mean ± std, 95% CI, identifiability):")
    for k, s in res["summary"].items():
        print(f"    {k:>9}: {s['mean']:6.2f} ± {s['std']:5.2f}  "
              f"CI[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]  "
              f"ident={s['identifiability']:.2f}")

    print("\n  Interpretation:")
    print("    identifiability ~1 -> parameter pinned down by the data")
    print("    identifiability ~0 -> data do not constrain this parameter")
    print("=" * 72)


# ============================================================================
# 6.  TIMECOURSE INFERENCE  (uses the complete 2-120 h dynamics)
# ============================================================================
#   The recalibration data are TRAJECTORIES A(E, t), not static points. This
#   infers the posterior over (laminAC, A_max) from the full mechanosensitive
#   dynamics at 1 and 23 kPa, using the model's own nuclear_area_time with the
#   stiffness-dependent tau(E). More information than the steady-state-only fit,
#   and consistent with the recalibrated dynamics.
# ---------------------------------------------------------------------------
def abc_timecourse(observed, base=None, priors=None, n_particles=300,
                   n_rounds=5, reps=4, alpha=0.5, seed=0, verbose=True):
    """Infer (laminAC, A_max) from the complete mechanosensitive timecourse.

    observed : dict {(E_kPa, t_h): area_um2}  (mechanosensitive population)
    Returns the posterior sample, per-parameter mean/std/CI/identifiability,
    and the model fit at the posterior mean.
    """
    if base is None:
        base = PHENOTYPES["hepatocyte"]
    if priors is None:
        priors = {"laminAC": (0.5, 2.5), "A_max": (150.0, 350.0),
                  "A0": (40.0, 90.0)}
    rng = np.random.default_rng(seed)
    names = list(priors)
    lows = np.array([priors[k][0] for k in names])
    highs = np.array([priors[k][1] for k in names])
    keys = sorted(observed)
    y = np.array([observed[k] for k in keys])

    def sim(theta):
        lam = float(theta[names.index("laminAC")])
        A_max = float(theta[names.index("A_max")])
        A0 = float(theta[names.index("A0")]) if "A0" in names else base.A_min + 15.0
        ph = replace(base, laminAC=lam, A_max=A_max)
        out = []
        for E, t in keys:
            try:
                import fast_model as fm
                sig = float(fm.nuclear_stress_fast(E, "hepatocyte"))
            except Exception:
                sig = mvc.nuclear_stress(E, ph, reps=reps)
            A_ss = ph.A_min + (ph.A_max - ph.A_min) * sig / (sig + ph.s0 * lam)
            tau = mvc.tau_of_E(E, ph)
            out.append(A_ss + (A0 - A_ss) * np.exp(-t / tau))
        return np.array(out)

    def dist(theta):
        return float(np.sqrt(np.mean((sim(theta) - y) ** 2)))

    # round 0
    theta = rng.uniform(lows, highs, size=(n_particles * 6, len(names)))
    d = np.array([dist(t) for t in theta])
    keep = np.argsort(d)[:n_particles]
    particles = theta[keep]
    tol = np.quantile(d[keep], alpha)
    if verbose:
        print(f"  round 0: tol={tol:.2f}  best d={d[keep][0]:.2f}")
    for r in range(1, n_rounds):
        cov = np.cov(particles.T) + 1e-6 * np.eye(len(names))
        L = np.linalg.cholesky(cov)
        new, nd, acc, tries = [], [], 0, 0
        while acc < n_particles and tries < n_particles * 300:
            cand = particles[rng.integers(n_particles)] + L @ rng.standard_normal(len(names))
            tries += 1
            if np.any(cand < lows) or np.any(cand > highs):
                continue
            dc = dist(cand)
            if dc <= tol:
                new.append(cand); nd.append(dc); acc += 1
        if acc > 0:
            particles = np.array(new)
            tol = np.quantile(nd, alpha)
        if verbose:
            print(f"  round {r}: accepted={acc}  tol={tol:.2f}")

    prior_std = (highs - lows) / np.sqrt(12)
    mean, std = particles.mean(0), particles.std(0)
    ident = 1.0 - std / prior_std
    summary = {k: dict(mean=float(mean[i]), std=float(std[i]),
                       ci95=(float(np.percentile(particles[:, i], 2.5)),
                             float(np.percentile(particles[:, i], 97.5))),
                       identifiability=float(ident[i]))
               for i, k in enumerate(names)}
    fit = {f"{E}kPa_{t}h": float(v) for (E, t), v in zip(keys, sim(mean))}
    return dict(names=names, posterior=particles, summary=summary,
                observed=observed, fit=fit)


if __name__ == "__main__":
    _demo()
