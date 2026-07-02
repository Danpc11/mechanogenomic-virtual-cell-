# Mechanogenomic virtual-cell model

[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-3776ab.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.24-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.10-8caae6.svg)](https://scipy.org/)

<p align="left">
  <img src="assets/mvirtual_cell_logo.png" alt="mVirtual Cell logo" width="160"/>
</p>

A minimal, first-principles physical model of nuclear mechanotransduction that
links substrate stiffness to nuclear deformation, YAP/TAZ activity, and
fibrosis-associated transcriptional trajectories. Calibrated against nuclear
area of primary hepatocytes on hydrogels and validated against human liver
RNA-seq cohorts across fibrosis stages.

📖 **Full documentation:** see the [project Wiki](../../wiki).

## Quick start

```bash
pip install -r requirements.txt
python mvirtual_cell.py          # runs the model self-test / demo
python test_virtual_cell.py      # runs the validation suite (10 checks)
```

```python
from mvirtual_cell import (PHENOTYPES, nuclear_stress, nuclear_area_ss,
                          yap_nc_ratio, nuclear_area_time, population_mixture,
                          fibrosis_prediction, CALIBRATION)

hep = PHENOTYPES["hepatocyte"]          # calibrated phenotype

nuclear_stress(23.0, hep)               # nuclear stress at 23 kPa
nuclear_area_ss(23.0, hep)              # steady-state nuclear area
yap_nc_ratio(23.0, hep)                 # YAP nucleocytoplasmic ratio
nuclear_area_time(23.0, t=36, ph=hep, contact_inhibition=True)  # area at 36 h
population_mixture(23.0, t=36, ph=hep)  # (mu_basal, mu_mecano, phi)
fibrosis_prediction(hep)                # F0->F4 prediction
```

## Repository contents

| File | Purpose |
|---|---|
| `mvirtual_cell.py` | The physical model and its calibrated parameters (predict). |
| `calibration.py` | Fitting layer: recover parameters from data (calibrate). |
| `test_virtual_cell.py` | Executable validation suite (10 qualitative checks). |
| `Theory_draft.md` | Manuscript draft (model, calibration, RNA-seq validation). |
| `Datasets.md` | Description of the RNA-seq cohorts used for validation. |
| `assets/` | Diagram and logo. |

## Model structure (`mvirtual_cell.py`)

1. **Motor-clutch engine** (`_mc_kernel`) — stochastic kernel, numba-accelerated.
2. **Phenotype** (dataclass) + **PHENOTYPES** — calibrated library (hepatocyte, A549, NHLF, MCF10A, MDA, AT2, fibroblast).
3. **Mechanotransduction chain** — `traction` → `nuclear_stress` → `nuclear_area_ss`, `yap_nc_ratio`, `lamin_expected`.
4. **Temporal dynamics + contact inhibition** — `nuclear_area_time`, `nc_effective`, `confluence`.
5. **Two-population model** — `population_mixture`, `BASAL_POP` (binucleate cells).
6. **Fibrosis → stiffness → prediction** — `FIBROSIS_STIFFNESS`, `fibrosis_prediction`.
7. **CALIBRATION** — summary of all values fitted to the data.

## Calibrating from your own data (`calibration.py`)

The fitting layer recovers the model parameters from experimental data, so the
calibration is reproducible rather than hard-coded.

```python
import calibration as cal

data = cal.load_hydrogel_csv("paper_data_two_pop.csv")   # {(E, t): areas}

# two-population deconvolution (GMM + BIC): basal vs mechanosensitive
rows = cal.two_population_table(data)
print(cal.population_stats(rows))

# fit a full phenotype (lamin A/C, A_min, A_max, tau) from the data
phenotype, report = cal.fit_phenotype(data, name="my_hepatocyte")

# validate model prediction against RNA-seq (fibrosis)
corr = cal.correlate_with_expression(["F0","F1","F2","F3","F4"],
                                      gene_expression, predictor="sigma")
ranked = cal.rank_genes_by_fit(corr)
```

Key functions: `deconvolve_two_populations`, `fit_lamin_from_area`,
`fit_temporal`, `fit_phenotype`, `correlate_with_expression`.

## Validation (`test_virtual_cell.py`)

Runnable checks that the calibrated model reproduces its qualitative anchors
(not fits — behavioral verification):

```bash
python test_virtual_cell.py      # or: pytest test_virtual_cell.py -v
```

Covers: biphasic traction, stiffness-dependent nuclear spreading, YAP
activation, lamin-knockdown collapse of YAP, phenotype lamin ordering,
two-population dynamics, contact inhibition, temporal relaxation, monotonic
fibrosis response, and clutch-vs-motor sensitivity of the optimum.

## Calibrated parameters (primary hepatocyte)

- **Motor-clutch:** nm=45, Fm=2.0, vu=110, nc=90, kon=0.5, koff0=0.1, Fb=2.0, kc=1.1, α=0.13.
- **Two populations:** basal 37.9 µm² (CV 6%, constant) + mechanosensitive ~68.8 µm² (grows).
- **Dynamics:** τ = 35.3 ± 2.6 h · inferred lamin vs LMNA qPCR: r=0.84.
- **Fibrosis stages:** F0 ~1–4, F1 ~7, F2 ~9.5, F3 ~13, F4 ~26 kPa.

## Notes

- Runs without numba, but ~50-100x slower; install it for parameter sweeps.
- Simulations are stochastic: use a larger `reps` (6-8) for stable means.
- Hepatocyte parameters are calibrated against real data; the other cell lines
  use literature-anchored starting points (laminAC is inferred from area and
  validated against qPCR).

## Diagram

![Study diagram](assets/Diagram_mvirtual_cell.png)

## Citation

If you use this model or repository, please cite it via the
[`CITATION.cff`](CITATION.cff) file (GitHub's "Cite this repository" button).
