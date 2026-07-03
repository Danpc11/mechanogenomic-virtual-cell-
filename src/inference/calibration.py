"""
================================================================================
 calibration.py  —  Fitting layer for the mechanogenomic virtual cell
================================================================================

Functions to CALIBRATE the model from experimental data, rather than relying on
the hard-coded constants in mvirtual_cell.py. This makes the calibration
reproducible: given nuclear-area measurements (and, optionally, qPCR), these
functions recover the two-population structure, the lamin A/C level, the
temporal time constant, and a full phenotype parameter vector.

Pipeline
--------
    raw nuclear areas (per E, t)
        -> deconvolve_two_populations()      # GMM + BIC: basal vs mechanosensitive
        -> fit_lamin_from_area()             # lamin A/C from area-vs-stiffness shape
        -> fit_temporal()                    # nuclear flattening time constant tau
        -> fit_phenotype()                   # assemble a calibrated Phenotype
        -> correlate_with_expression()       # validate vs RNA-seq (fibrosis)

Author: Daniel Pérez-Calixto (INMEGEN / UNAM)

Dependencies: numpy, scipy, scikit-learn, pandas (pandas only for CSV I/O).
================================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import replace
from scipy.optimize import curve_fit
from scipy import stats

import mvirtual_cell as mvc
from mvirtual_cell import Phenotype, nuclear_stress, PHENOTYPES

try:
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# 0.  DATA I/O
# ============================================================================
def load_hydrogel_csv(path, E_col="E_kPa", t_col="t_h", area_col="nuclear_area"):
    """Load a long-format CSV of single-cell areas into a dict keyed by (E, t).

    Expected columns: stiffness (kPa), time (h), nuclear area (um^2), one row
    per cell. Returns {(E, t): np.ndarray of areas}.
    """
    import pandas as pd
    df = pd.read_csv(path)
    out = {}
    for (E, t), g in df.groupby([E_col, t_col]):
        out[(float(E), float(t))] = g[area_col].to_numpy(dtype=float)
    return out


def clean_areas(areas, iqr_k=1.5, floor=20.0):
    """Remove outliers by a per-condition IQR rule with a detection floor.
    Returns the filtered array (doublets/debris removed)."""
    a = np.asarray(areas, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    hi = q3 + iqr_k * (q3 - q1)
    return a[(a >= floor) & (a <= hi)]


# ============================================================================
# 1.  TWO-POPULATION DECONVOLUTION  (basal binucleate + mechanosensitive)
# ============================================================================
def deconvolve_two_populations(areas, max_components=3, random_state=0):
    """Fit Gaussian-mixture models with 1..max_components and select by BIC.

    Returns a dict with:
        n_best        : BIC-selected number of components
        one_rejected  : True if the 1-component model is NOT selected
        mu_basal, sd_basal, w_basal      : low-area (basal) component
        mu_mecano, sd_mecano, w_mecano   : high-area (mechanosensitive) component
    The 2-component parameters are always returned (ordered low->high) so the
    basal/mechanosensitive means can be tracked across conditions even when
    BIC marginally prefers 3 components (extra components capture skew).
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for deconvolution "
                          "(pip install scikit-learn)")
    a = np.asarray(areas, dtype=float).reshape(-1, 1)
    bics = []
    models = {}
    for k in range(1, max_components + 1):
        gm = GaussianMixture(n_components=k, n_init=4,
                             random_state=random_state).fit(a)
        bics.append(gm.bic(a))
        models[k] = gm
    n_best = int(np.argmin(bics)) + 1

    gm2 = models[2]
    order = np.argsort(gm2.means_.flatten())
    mu = gm2.means_.flatten()[order]
    sd = np.sqrt(gm2.covariances_.flatten())[order]
    w = gm2.weights_[order]
    return dict(
        n_best=n_best,
        one_rejected=(n_best != 1),
        bic=dict(zip(range(1, max_components + 1), bics)),
        mu_basal=float(mu[0]), sd_basal=float(sd[0]), w_basal=float(w[0]),
        mu_mecano=float(mu[1]), sd_mecano=float(sd[1]), w_mecano=float(w[1]),
    )


def two_population_table(data, clean=True):
    """Run the deconvolution across all (E, t) conditions.

    `data` is {(E, t): areas}. Returns a list of per-condition dicts with
    E, t and the two-population parameters (basal + mechanosensitive)."""
    rows = []
    for (E, t) in sorted(data):
        a = clean_areas(data[(E, t)]) if clean else np.asarray(data[(E, t)])
        d = deconvolve_two_populations(a)
        d["E_kPa"] = E
        d["t_h"] = t
        d["n_cells"] = len(a)
        rows.append(d)
    return rows


def population_stats(rows):
    """Summarize a two_population_table: are the two populations behaving as
    the model predicts (basal constant, mechanosensitive growing with time)?"""
    t = np.array([r["t_h"] for r in rows])
    mu_b = np.array([r["mu_basal"] for r in rows])
    mu_m = np.array([r["mu_mecano"] for r in rows])
    r_b = stats.pearsonr(t, mu_b)
    r_m = stats.pearsonr(t, mu_m)
    n_rejected = sum(r["one_rejected"] for r in rows)
    return dict(
        basal_mean=float(mu_b.mean()), basal_cv=float(mu_b.std() / mu_b.mean()),
        basal_time_r=float(r_b[0]), basal_time_p=float(r_b[1]),
        mecano_mean=float(mu_m.mean()), mecano_cv=float(mu_m.std() / mu_m.mean()),
        mecano_time_r=float(r_m[0]), mecano_time_p=float(r_m[1]),
        one_pop_rejected=f"{n_rejected}/{len(rows)}",
    )


# ============================================================================
# 2.  LAMIN A/C  FROM  AREA-VS-STIFFNESS  SHAPE
# ============================================================================
#   A_ss(E) = A_min + (A_max - A_min) * sigma(E) / (sigma(E) + s0*lamin)
#   The half-saturation stress s_half = s0*lamin controls the SHAPE, so lamin
#   is identifiable from the area-vs-stiffness curve (validated by LMNA qPCR).
# ---------------------------------------------------------------------------
def fit_lamin_from_area(E_array, area_array, ph: Phenotype = None,
                        s0=15.0, reps=6, fit_bounds=None):
    """Fit (A_min, A_max, lamin) to a steady-state area-vs-stiffness curve.

    Returns dict with fitted A_min, A_max, laminAC, s_half, R2, and the
    predicted curve. `ph` supplies the motor parameters used to compute
    nuclear stress (defaults to the hepatocyte phenotype).

    Note on identifiability: laminAC enters only through s_half = s0*lamin, so
    it is recovered up to the absolute stress scale of the motor. What is
    robustly identified is the *shape* of the area-vs-stiffness curve and the
    lamin value *relative* to other cell lines fit with the same motor — which
    is exactly what is needed to correlate inferred lamin with LMNA qPCR."""
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    E_array = np.asarray(E_array, dtype=float)
    area_array = np.asarray(area_array, dtype=float)

    # Pre-compute nuclear stress at each stiffness (motor is stochastic)
    sigma = np.array([nuclear_stress(E, ph, reps=reps) for E in E_array])

    def model(sig, A_min, A_max, lamin):
        return A_min + (A_max - A_min) * sig / (sig + s0 * lamin)

    if fit_bounds is None:
        fit_bounds = ([20.0, 40.0, 0.2], [60.0, 300.0, 3.0])
    p0 = [area_array.min(), area_array.max() * 1.5, 1.0]
    popt, pcov = curve_fit(model, sigma, area_array, p0=p0,
                           bounds=fit_bounds, maxfev=20000)
    pred = model(sigma, *popt)
    ss_res = np.sum((area_array - pred) ** 2)
    ss_tot = np.sum((area_array - area_array.mean()) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    perr = np.sqrt(np.diag(pcov))
    return dict(A_min=float(popt[0]), A_max=float(popt[1]), laminAC=float(popt[2]),
                s_half=float(s0 * popt[2]), s0=s0,
                A_min_err=float(perr[0]), A_max_err=float(perr[1]),
                laminAC_err=float(perr[2]), R2=float(R2),
                sigma=sigma, pred=pred)


def validate_lamin_vs_qpcr(lamin_inferred, lmna_qpcr):
    """Correlate model-inferred lamin (per cell line / condition) with LMNA
    qPCR. Returns Pearson r and p. This is the independent molecular check."""
    r, p = stats.pearsonr(np.asarray(lamin_inferred), np.asarray(lmna_qpcr))
    return dict(pearson_r=float(r), p_value=float(p))


# ============================================================================
# 3.  TEMPORAL DYNAMICS  (nuclear flattening time constant tau)
# ============================================================================
def fit_temporal(t_array, area_array, A_ss=None):
    """Fit the first-order relaxation  A(t) = A_ss + (A0 - A_ss) exp(-t/tau).

    If A_ss is None it is fit jointly with A0 and tau. Returns tau (h),
    A0, A_ss and R2."""
    t_array = np.asarray(t_array, dtype=float)
    area_array = np.asarray(area_array, dtype=float)

    if A_ss is None:
        def model(t, A0, A_ss, tau):
            return A_ss + (A0 - A_ss) * np.exp(-t / tau)
        p0 = [area_array[0], area_array[-1], 20.0]
        bounds = ([10, 10, 1.0], [300, 300, 300])
        popt, _ = curve_fit(model, t_array, area_array, p0=p0,
                            bounds=bounds, maxfev=20000)
        A0, A_ss_fit, tau = popt
        pred = model(t_array, *popt)
    else:
        def model(t, A0, tau):
            return A_ss + (A0 - A_ss) * np.exp(-t / tau)
        p0 = [area_array[0], 20.0]
        popt, _ = curve_fit(model, t_array, area_array, p0=p0,
                            bounds=([10, 1.0], [300, 300]), maxfev=20000)
        A0, tau = popt
        A_ss_fit = A_ss
        pred = model(t_array, *popt)

    ss_res = np.sum((area_array - pred) ** 2)
    ss_tot = np.sum((area_array - area_array.mean()) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(tau_h=float(tau), A0=float(A0), A_ss=float(A_ss_fit), R2=float(R2))


# ============================================================================
# 4.  FULL PHENOTYPE FIT
# ============================================================================
def fit_phenotype(data, name="fitted", base=None, reps=6):
    """Assemble a calibrated Phenotype from a full {(E, t): areas} dataset.

    Steps: (1) deconvolve two populations per condition; (2) fit lamin, A_min,
    A_max from the mechanosensitive steady-state area vs stiffness (using the
    latest time point per stiffness as the steady-state estimate);
    (3) fit tau from the mechanosensitive temporal trajectory. Returns
    (Phenotype, report dict)."""
    if base is None:
        base = PHENOTYPES["hepatocyte"]
    rows = two_population_table(data)

    # steady-state area per stiffness = mechanosensitive mean at the latest time
    Es = sorted(set(r["E_kPa"] for r in rows))
    ts = sorted(set(r["t_h"] for r in rows))
    t_last = max(ts)
    E_ss, A_ss_obs = [], []
    for E in Es:
        cand = [r for r in rows if r["E_kPa"] == E and r["t_h"] == t_last]
        if cand:
            E_ss.append(E)
            A_ss_obs.append(cand[0]["mu_mecano"])
    lam = fit_lamin_from_area(E_ss, A_ss_obs, ph=base, reps=reps)

    # temporal fit at the stiffest condition (strongest dynamics)
    E_stiff = max(Es)
    tt = [r["t_h"] for r in rows if r["E_kPa"] == E_stiff]
    aa = [r["mu_mecano"] for r in rows if r["E_kPa"] == E_stiff]
    order = np.argsort(tt)
    temporal = fit_temporal(np.array(tt)[order], np.array(aa)[order])

    ph = replace(base, name=name,
                 laminAC=lam["laminAC"], A_min=lam["A_min"],
                 A_max=lam["A_max"], s0=lam["s0"], tau=temporal["tau_h"])
    report = dict(lamin_fit=lam, temporal_fit=temporal,
                  population_stats=population_stats(rows), n_conditions=len(rows))
    return ph, report


# ============================================================================
# 5.  RNA-SEQ VALIDATION  (fibrosis gene trajectories vs model prediction)
# ============================================================================
def correlate_with_expression(stage_labels, gene_expression, ph: Phenotype = None,
                              predictor="sigma", reps=8):
    """Correlate observed gene expression across fibrosis stages with the
    model's predicted mechanotransduction output.

    Parameters
    ----------
    stage_labels : list of stage names, e.g. ["F0","F1","F2","F3","F4"]
    gene_expression : dict {gene_name: array of expression per stage}
                      (same order and length as stage_labels)
    predictor : which model output to correlate against
                ("sigma", "yap", or "lamin")

    Returns a dict {gene: {pearson_r, spearman_r, p_value}} plus the model
    prediction vector used. Genes whose trajectory matches the predicted
    (typically convex) stiffness response score highest."""
    if ph is None:
        ph = PHENOTYPES["hepatocyte"]
    pred = mvc.fibrosis_prediction(ph, reps=reps)
    # align to requested stages
    idx = [pred["stages"].index(s) for s in stage_labels]
    yhat = np.asarray(pred[predictor])[idx]

    out = {}
    for gene, expr in gene_expression.items():
        expr = np.asarray(expr, dtype=float)
        pr = stats.pearsonr(yhat, expr)
        sr = stats.spearmanr(yhat, expr)
        out[gene] = dict(pearson_r=float(pr[0]), p_value=float(pr[1]),
                         spearman_r=float(sr[0]))
    return dict(predictor=predictor, model_prediction=yhat.tolist(),
                stages=stage_labels, genes=out)


def rank_genes_by_fit(correlation_result, by="pearson_r"):
    """Rank genes by how well their trajectory matches the model prediction."""
    genes = correlation_result["genes"]
    return sorted(genes.items(), key=lambda kv: kv[1][by], reverse=True)


# ============================================================================
# 6.  DEMO  (synthetic data round-trip: fit recovers known parameters)
# ============================================================================
def _demo():
    print("=" * 72)
    print("  CALIBRATION LAYER — self-test (synthetic round-trip)")
    print("=" * 72)

    rng = np.random.default_rng(0)
    truth = replace(PHENOTYPES["hepatocyte"], laminAC=1.3, A_min=40.0,
                    A_max=95.0, tau=30.0)

    # --- generate synthetic two-population data from the known phenotype ---
    Es = [0.5, 1, 2, 5, 10, 23]
    ts = [2, 12, 24, 36]
    data = {}
    for E in Es:
        for t in ts:
            mu_m = mvc.nuclear_area_time(E, t, truth, reps=4)
            mecano = rng.normal(mu_m, 8.0, 220)
            basal = rng.normal(mvc.BASAL_POP["mean"], 6.0, 180)   # constant
            data[(E, t)] = np.concatenate([mecano, basal])

    # --- 1. deconvolution ---
    print("\n[1] Two-population deconvolution (should reject 1-pop):")
    rows = two_population_table(data)
    ps = population_stats(rows)
    print(f"    one-population rejected: {ps['one_pop_rejected']}")
    print(f"    basal:  mean={ps['basal_mean']:.1f}  CV={ps['basal_cv']*100:.0f}%"
          f"  (time r={ps['basal_time_r']:+.2f})")
    print(f"    mecano: mean={ps['mecano_mean']:.1f}  CV={ps['mecano_cv']*100:.0f}%"
          f"  (time r={ps['mecano_time_r']:+.2f})")

    # --- 2. fit phenotype (recover truth) ---
    print("\n[2] Phenotype fit (recover known parameters):")
    ph_fit, report = fit_phenotype(data, name="recovered", reps=5)
    print(f"    {'param':>10}{'true':>9}{'fitted':>9}")
    for p in ["laminAC", "A_min", "A_max", "tau"]:
        print(f"    {p:>10}{getattr(truth, p):>9.1f}{getattr(ph_fit, p):>9.1f}")
    print(f"    lamin fit R2 = {report['lamin_fit']['R2']:.3f}")

    # --- 3. lamin vs qPCR (synthetic) ---
    print("\n[3] Lamin-vs-qPCR validation (synthetic):")
    lam_inf = [0.5, 0.8, 1.0, 1.2, 1.3]
    lmna = [0.6, 0.75, 1.05, 1.15, 1.35]
    v = validate_lamin_vs_qpcr(lam_inf, lmna)
    print(f"    Pearson r = {v['pearson_r']:.2f}  (p={v['p_value']:.3f})")

    # --- 4. RNA-seq correlation (synthetic convex genes) ---
    print("\n[4] RNA-seq correlation (synthetic convex genes):")
    stages = ["F0", "F1", "F2", "F3", "F4"]
    pred = mvc.fibrosis_prediction(PHENOTYPES["hepatocyte"], reps=5)
    sig = np.array(pred["sigma"])
    genes = {
        "CCN2": sig / sig[0] + rng.normal(0, 0.03, 5),
        "LOX":  (sig / sig[0]) ** 1.3 + rng.normal(0, 0.03, 5),
        "flat_control": np.ones(5) + rng.normal(0, 0.03, 5),
    }
    corr = correlate_with_expression(stages, genes, predictor="sigma", reps=5)
    for gene, res in rank_genes_by_fit(corr):
        print(f"    {gene:>14}: r={res['pearson_r']:+.2f}")

    print("\n" + "=" * 72)
    print("  Calibration layer OK.")
    print("=" * 72)


if __name__ == "__main__":
    _demo()
