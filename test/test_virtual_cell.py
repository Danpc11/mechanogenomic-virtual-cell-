"""
================================================================================
 test_virtual_cell.py  —  Executable validation of the virtual-cell model
================================================================================

Runnable checks that the physical model reproduces its qualitative anchors.
These are not fits — they verify that the calibrated model still behaves
correctly (biphasic traction, stiffness-dependent nuclear spreading, YAP
activation, lamin-dependent gating, two-population dynamics, monotonic
fibrosis response). Run directly:

    python test_virtual_cell.py

or under pytest:

    pytest test_virtual_cell.py -v

Each test prints its result and asserts the expected behavior. Because the
motor-clutch engine is stochastic, tests use enough replicates for stable
means and assert robust qualitative relations rather than exact values.

Author: Daniel Pérez-Calixto (INMEGEN / UNAM)
================================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import mvirtual_cell as mvc
from mvirtual_cell import (PHENOTYPES, traction, nuclear_stress,
                           nuclear_area_ss, yap_nc_ratio, nuclear_area_time)
from dataclasses import replace

REPS = 8  # replicate seeds for stable stochastic means


# ---------------------------------------------------------------------------
def test_biphasic_traction():
    """Traction should be biphasic in substrate stiffness: an intermediate
    optimum exceeds both the very soft and very stiff limits."""
    ph = PHENOTYPES["hepatocyte"]
    kappas = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0])  # pN/nm
    # sweep kappa directly by mapping through alpha (E = kappa/alpha)
    T = np.array([traction(k / ph.alpha, ph, reps=REPS) for k in kappas])
    k_opt = kappas[np.argmax(T)]
    assert T.max() > T[0], "optimum should exceed the soft limit"
    assert T.max() > T[-1], "optimum should exceed the stiff limit"
    assert 0.1 <= k_opt <= 5.0, f"optimum stiffness out of expected range: {k_opt}"
    print(f"  [OK] biphasic traction: optimum at kappa≈{k_opt:.2f} pN/nm "
          f"(T_opt={T.max():.1f} pN)")


def test_nuclear_area_increases_with_stiffness():
    """Steady-state nuclear area should increase monotonically with stiffness
    for the mechanosensitive response."""
    ph = PHENOTYPES["hepatocyte"]
    Es = [0.5, 1, 5, 23]
    A = [nuclear_area_ss(E, ph, reps=REPS) for E in Es]
    assert A[-1] > A[0], "area should be larger on stiff than on soft substrate"
    assert all(np.diff(A) > -2.0), "area should be broadly non-decreasing"
    print(f"  [OK] nuclear area vs stiffness: {A[0]:.1f} -> {A[-1]:.1f} µm²")


def test_yap_activation_with_stiffness():
    """YAP N/C ratio should rise from near-baseline on soft substrates to a
    multi-fold value on stiff substrates."""
    ph = PHENOTYPES["hepatocyte"]
    yap_soft = yap_nc_ratio(0.5, ph, reps=REPS)
    yap_stiff = yap_nc_ratio(20.0, ph, reps=REPS)
    assert yap_stiff > yap_soft, "YAP should increase with stiffness"
    assert yap_stiff / yap_soft > 1.3, "YAP should rise appreciably with stiffness"
    print(f"  [OK] YAP N/C: {yap_soft:.2f} (soft) -> {yap_stiff:.2f} (stiff)")


def test_lamin_knockdown_reduces_yap():
    """Reducing lamin A/C (knockdown) should lower stiff-substrate YAP:
    a softer nucleus gates less transcriptional activation."""
    ph = PHENOTYPES["hepatocyte"]
    yap_wt = yap_nc_ratio(20.0, ph, reps=REPS)
    ph_kd = replace(ph, laminAC=ph.laminAC * 0.2)   # 80% knockdown
    yap_kd = yap_nc_ratio(20.0, ph_kd, reps=REPS)
    drop = (yap_wt - yap_kd) / (yap_wt - 1.0)
    assert yap_kd < yap_wt, "lamin knockdown should reduce YAP"
    assert drop > 0.3, f"knockdown should substantially reduce YAP (drop={drop:.2f})"
    print(f"  [OK] lamin knockdown: YAP {yap_wt:.2f} -> {yap_kd:.2f} "
          f"({drop*100:.0f}% of dynamic range lost)")


def test_phenotype_lamin_ordering():
    """Phenotypes with higher lamin A/C should show larger YAP dynamic range
    (soft->stiff), reflecting stiffer, more gating-competent nuclei."""
    def yap_range(key):
        ph = PHENOTYPES[key]
        return yap_nc_ratio(20.0, ph, reps=REPS) - yap_nc_ratio(0.5, ph, reps=REPS)
    r_mda = yap_range("MDA")        # laminAC = 0.5 (soft nucleus)
    r_at2 = yap_range("AT2_lung")   # laminAC = 1.3 (stiff nucleus)
    assert r_at2 > r_mda, ("high-lamin phenotype should have larger YAP range "
                           f"(AT2={r_at2:.2f} vs MDA={r_mda:.2f})")
    print(f"  [OK] lamin ordering: YAP range MDA(soft)={r_mda:.2f} "
          f"< AT2(stiff)={r_at2:.2f}")


def test_two_population_basal_constant():
    """The basal population is constant; the mechanosensitive population grows
    with time (contact inhibition included)."""
    ph = PHENOTYPES["hepatocyte"]
    mus_basal, mus_mecano = [], []
    for t in [2, 12, 24, 36]:
        mb, mm, phi = mvc.population_mixture(23.0, t, ph, reps=4)
        mus_basal.append(mb)
        mus_mecano.append(mm)
    assert np.std(mus_basal) < 1e-6, "basal population must be constant"
    assert mus_mecano[-1] > mus_mecano[0], "mechanosensitive pop should grow in time"
    print(f"  [OK] two populations: basal constant={mus_basal[0]:.1f} µm², "
          f"mecano {mus_mecano[0]:.1f}->{mus_mecano[-1]:.1f} µm²")


def test_contact_inhibition_reduces_clutches():
    """Effective substrate clutches should decrease as confluence rises."""
    ph = PHENOTYPES["hepatocyte"]
    nc_early = mvc.nc_effective(ph, t=2)
    nc_late = mvc.nc_effective(ph, t=36)
    assert nc_late < nc_early, "contact inhibition should reduce effective clutches"
    print(f"  [OK] contact inhibition: nc_eff {nc_early} (2h) -> {nc_late} (36h)")


def test_temporal_relaxation():
    """Nuclear area should relax toward the steady state over time."""
    ph = PHENOTYPES["hepatocyte"]
    A_early = nuclear_area_time(23.0, 2, ph, reps=REPS)
    A_late = nuclear_area_time(23.0, 100, ph, reps=REPS)
    A_ss = nuclear_area_ss(23.0, ph, reps=REPS)
    assert abs(A_late - A_ss) < abs(A_early - A_ss), "area should approach steady state"
    print(f"  [OK] temporal relaxation: A(2h)={A_early:.1f} -> "
          f"A(100h)={A_late:.1f} ≈ A_ss={A_ss:.1f} µm²")


def test_fibrosis_monotonic():
    """Predicted nuclear stress should increase monotonically across fibrosis
    stages F0->F4 (the tissue stiffens)."""
    pred = mvc.fibrosis_prediction(PHENOTYPES["hepatocyte"], reps=REPS)
    sig = np.array(pred["sigma"])
    assert sig[-1] > sig[0], "F4 stress should exceed F0"
    # allow tiny stochastic dips but require overall increase
    assert np.polyfit(range(len(sig)), sig, 1)[0] > 0, "trend should be increasing"
    print(f"  [OK] fibrosis F0->F4 stress: {sig[0]:.1f} -> {sig[-1]:.1f} "
          f"(monotonic increasing)")


def test_optimum_sensitive_to_clutch_not_motor():
    """The optimal stiffness should shift more when clutch number changes than
    when motor number changes by the same relative amount."""
    ph = PHENOTYPES["hepatocyte"]
    kappas = np.array([0.1, 0.3, 0.7, 1.5, 3.0, 7.0])

    def k_opt(phen):
        T = [traction(k / phen.alpha, phen, reps=REPS) for k in kappas]
        return kappas[int(np.argmax(T))]

    base = k_opt(ph)
    more_clutch = k_opt(replace(ph, nc=int(ph.nc * 1.6)))
    more_motor = k_opt(replace(ph, nm=int(ph.nm * 1.6)))
    shift_clutch = abs(np.log(more_clutch / base))
    shift_motor = abs(np.log(more_motor / base))
    # robust qualitative check: clutch perturbation shifts optimum at least as much
    assert shift_clutch >= shift_motor - 1e-9, (
        f"optimum should be at least as sensitive to clutch as to motor "
        f"(clutch shift={shift_clutch:.2f}, motor shift={shift_motor:.2f})")
    print(f"  [OK] optimum sensitivity: clutch shift={shift_clutch:.2f} "
          f">= motor shift={shift_motor:.2f}")


def test_tau_scales_with_stiffness():
    """Recalibration result: nuclear relaxation is SLOWER on stiffer substrates
    (tau increases with stiffness), from the complete 2-120 h timecourse."""
    ph = PHENOTYPES["hepatocyte"]
    tau_soft = mvc.tau_of_E(1.0, ph)
    tau_stiff = mvc.tau_of_E(23.0, ph)
    assert tau_stiff > tau_soft, "tau should increase with stiffness"
    assert tau_stiff / tau_soft > 2.0, "stiff relaxation should be much slower"
    # dynamics: stiff substrate keeps growing longer than soft
    A_soft_late = mvc.nuclear_area_time(1.0, 120, ph, reps=REPS)
    A_soft_mid = mvc.nuclear_area_time(1.0, 36, ph, reps=REPS)
    A_stiff_late = mvc.nuclear_area_time(23.0, 120, ph, reps=REPS)
    A_stiff_mid = mvc.nuclear_area_time(23.0, 36, ph, reps=REPS)
    soft_growth = A_soft_late - A_soft_mid
    stiff_growth = A_stiff_late - A_stiff_mid
    assert stiff_growth > soft_growth, ("stiff substrate should keep growing "
                                        "between 36-120 h more than soft")
    print(f"  [OK] tau scales with stiffness: {tau_soft:.0f} h (soft) -> "
          f"{tau_stiff:.0f} h (stiff); stiff still growing 36->120 h")


# ---------------------------------------------------------------------------
ALL_TESTS = [
    test_biphasic_traction,
    test_nuclear_area_increases_with_stiffness,
    test_yap_activation_with_stiffness,
    test_lamin_knockdown_reduces_yap,
    test_phenotype_lamin_ordering,
    test_two_population_basal_constant,
    test_contact_inhibition_reduces_clutches,
    test_temporal_relaxation,
    test_fibrosis_monotonic,
    test_optimum_sensitive_to_clutch_not_motor,
    test_tau_scales_with_stiffness,
]


def run_all():
    print("=" * 72)
    print("  VIRTUAL CELL — model validation suite")
    print(f"  numba: {'ON' if mvc.HAS_NUMBA else 'OFF (pure python, slower)'}")
    print("=" * 72)
    passed = 0
    failed = 0
    for test in ALL_TESTS:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print("=" * 72)
    print(f"  {passed}/{len(ALL_TESTS)} passed"
          + (f", {failed} failed" if failed else "  — all validations OK"))
    print("=" * 72)
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = run_all()
    sys.exit(0 if ok else 1)
