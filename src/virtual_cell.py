"""
================================================================================
 virtual_cell.py  —  The mechanogenomic virtual cell (central interface)
================================================================================

Wraps the physical model into a single reusable object with internal state, so
the framework behaves like a *virtual cell* rather than a set of loose
functions. This is the reusable-platform layer:

    from virtual_cell import VirtualCell
    cell = VirtualCell("hepatocyte")          # instantiate an avatar
    state = cell.simulate(E=23.0, t=72.0)     # advance to a mechanical context
    state.yap_nc, state.nuclear_area          # read observables
    cell.state_vector()                       # numeric state vector
    cell.gene_scores()                        # mechanogenomic output

The single physical input is the tissue/substrate stiffness E (kPa) — the same
quantity measured clinically by elastography — plus optional time t (h). The
cell exposes a full observable state and a numeric state vector for downstream
analysis (sensitivity, benchmarking, visualization).
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np

import mvirtual_cell as mvc
from mvirtual_cell import PHENOTYPES


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------
@dataclass
class CellState:
    """A snapshot of the virtual cell's mechanical + mechanogenomic state.

    All fields are model OUTPUTS given the physical input (E, t). Units noted.
    """
    phenotype: str
    E_kPa: float                    # input: substrate/tissue stiffness (kPa)
    t_h: Optional[float]            # input: time in culture (h); None = steady state
    traction: float                 # cytoskeletal traction force (pN)
    nuclear_drive: float            # transmitted nuclear load / drive (a.u., force)
    nuclear_area: float             # projected nuclear area (um^2)
    yap_nc: float                   # YAP nucleocytoplasmic ratio
    laminAC: float                  # relative lamin A/C (nuclear stiffness)
    nc_eff: float                   # effective engaged clutches / adhesions
    tau_h: float                    # relaxation time constant at this stiffness (h)
    function_index: float           # hepatocyte functional index [0,1] (mech. axis)
    fibrosis_stage: str             # nearest fibrosis stage label
    gene_scores: dict = field(default_factory=dict)   # per-gene activation [0,1]

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# The virtual cell
# ---------------------------------------------------------------------------
class VirtualCell:
    """A reusable, calibratable mechanogenomic virtual cell.

    Parameters
    ----------
    phenotype : str | Phenotype
        Key in PHENOTYPES (e.g. "hepatocyte") or a Phenotype instance. The same
        physical core re-parameterized per phenotype = a distinct virtual cell.
    reps : int
        Stochastic replicate seeds for the motor (higher = smoother means).
    use_fast : bool
        Use the analytic surrogate for the stress map when available (fast).
    """

    def __init__(self, phenotype="hepatocyte", reps=6, use_fast=True):
        if isinstance(phenotype, str):
            if phenotype not in PHENOTYPES:
                raise KeyError(f"Unknown phenotype '{phenotype}'. "
                               f"Available: {list(PHENOTYPES)}")
            self.phenotype_key = phenotype
            self.phenotype = PHENOTYPES[phenotype]
        else:
            self.phenotype_key = getattr(phenotype, "name", "custom")
            self.phenotype = phenotype
        self.reps = reps
        self.use_fast = use_fast
        self._state: Optional[CellState] = None
        self._gene_module = None      # lazy import to avoid circular dependency

    # -- core: advance the cell to a mechanical context ---------------------
    def simulate(self, E, t=None, contact_inhibition=False) -> CellState:
        """Compute the full state at stiffness E (kPa) and optional time t (h).

        If t is None, returns the steady-state nuclear area; otherwise the
        transient area with the stiffness-dependent relaxation tau(E)."""
        ph = self.phenotype
        sig = self._drive(E)
        yap = mvc.yap_nc_ratio(E, ph, reps=self.reps)
        lam = mvc.lamin_expected(E, ph)
        nc_eff = (mvc.nc_effective(ph, t) if (contact_inhibition and t is not None)
                  else float(ph.nc))
        tau = mvc.tau_of_E(E, ph)
        if t is None:
            area = mvc.nuclear_area_ss(E, ph, reps=self.reps)
        else:
            area = mvc.nuclear_area_time(E, t, ph, reps=self.reps,
                                         contact_inhibition=contact_inhibition)
        func = self._function_index(sig)
        state = CellState(
            phenotype=self.phenotype_key, E_kPa=float(E),
            t_h=(None if t is None else float(t)),
            traction=float(mvc.traction(E, ph, reps=self.reps)),
            nuclear_drive=float(sig), nuclear_area=float(area),
            yap_nc=float(yap), laminAC=float(ph.laminAC),
            nc_eff=float(nc_eff), tau_h=float(tau),
            function_index=float(func), fibrosis_stage=mvc.stage_of_stiffness(E),
            gene_scores={})
        # attach mechanogenomic output
        state.gene_scores = self._gene_scores_from_state(state)
        self._state = state
        return state

    # -- observable state vector (numeric) ---------------------------------
    STATE_FIELDS = ("E_kPa", "traction", "nuclear_drive", "nuclear_area",
                    "yap_nc", "laminAC", "nc_eff", "tau_h", "function_index")

    def state_vector(self, state: Optional[CellState] = None) -> np.ndarray:
        """Numeric state vector (for sensitivity / benchmarking / embedding)."""
        s = state or self._state
        if s is None:
            raise RuntimeError("Call simulate() before state_vector().")
        return np.array([getattr(s, f) for f in self.STATE_FIELDS], float)

    # -- mechanogenomic output ---------------------------------------------
    def gene_scores(self, state: Optional[CellState] = None) -> dict:
        s = state or self._state
        if s is None:
            raise RuntimeError("Call simulate() before gene_scores().")
        return s.gene_scores

    def _gene_scores_from_state(self, state: CellState) -> dict:
        """Delegate to the gene module (lazy import)."""
        if self._gene_module is None:
            try:
                import gene_module
                self._gene_module = gene_module
            except Exception:
                return {}
        key = getattr(self, "phenotype_key", "hepatocyte")
        try:
            return self._gene_module.score_genes(
                state.nuclear_drive, state.yap_nc, phenotype=key)
        except TypeError:
            # older signature without phenotype
            return self._gene_module.score_genes(state.nuclear_drive, state.yap_nc)

    # -- trajectory across a stiffness or time sweep -----------------------
    def trajectory(self, Es=None, t=None):
        """Sweep stiffness (or use the fibrosis stages) and return a list of
        states — the mechanogenomic trajectory."""
        if Es is None:
            Es = list(mvc.FIBROSIS_STIFFNESS.values())
        return [self.simulate(E, t=t) for E in Es]

    # -- helpers -----------------------------------------------------------
    def _drive(self, E):
        if self.use_fast:
            try:
                import fast_model as fm
                if (self.phenotype_key == "hepatocyte"):
                    return float(fm.nuclear_stress_fast(E, "hepatocyte"))
            except Exception:
                pass
        return mvc.nuclear_stress(E, self.phenotype, reps=self.reps)

    def _function_index(self, sig, sigma_half=25.0, hill=2.0):
        """Hepatocyte functional index from the nuclear drive (mechanical axis).
        1 = differentiated/functional (low drive); ->0 = dysfunctional (high)."""
        return float(np.clip(1.0 / (1.0 + (sig / sigma_half) ** hill), 0.0, 1.0))

    def __repr__(self):
        return f"VirtualCell(phenotype='{self.phenotype_key}')"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo():
    print("=" * 70)
    print("  MECHANOGENOMIC VIRTUAL CELL  —  central interface")
    print("=" * 70)
    cell = VirtualCell("hepatocyte")
    print(f"\n  {cell!r}\n")
    print(f"  {'E(kPa)':>7}{'t(h)':>6}{'drive':>8}{'area':>8}{'YAP':>7}"
          f"{'tau':>7}{'func':>7}{'stage':>7}")
    for E, t in [(1, 120), (5, 120), (13, 120), (23, 120), (23, 36)]:
        s = cell.simulate(E, t=t)
        print(f"  {E:>7.1f}{t:>6.0f}{s.nuclear_drive:>8.1f}{s.nuclear_area:>8.0f}"
              f"{s.yap_nc:>7.2f}{s.tau_h:>7.0f}{s.function_index:>7.2f}"
              f"{s.fibrosis_stage:>7}")

    print("\n  State vector at E=23 kPa, t=120 h:")
    s = cell.simulate(23.0, t=120.0)
    print("   ", dict(zip(cell.STATE_FIELDS,
                          np.round(cell.state_vector(), 2))))
    ng = len(s.gene_scores)
    print(f"\n  Mechanogenomic output: {ng} genes scored"
          + (f" (top: {max(s.gene_scores, key=s.gene_scores.get)})" if ng else ""))
    print("=" * 70)


if __name__ == "__main__":
    _demo()
