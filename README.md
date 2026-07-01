# Mechanogenomic virtual-cell model

Self-contained module with the complete physical model and its calibrated parameters.

## Quick start

```bash
pip install -r requirements.txt
python mvirtual_cell.py          # runs the self-test / demo
```

```python
from virtual_cell import (PHENOTYPES, nuclear_stress, nuclear_area_ss,
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

## Module structure

1. **Motor-clutch engine** (`_mc_kernel`) — stochastic kernel, numba-accelerated.
2. **Phenotype** (dataclass) + **PHENOTYPES** — calibrated library (hepatocyte, A549, NHLF, MCF10A, MDA, AT2, fibroblast).
3. **Mechanotransduction chain** — `traction` → `nuclear_stress` → `nuclear_area_ss`, `yap_nc_ratio`, `lamin_expected`.
4. **Temporal dynamics + contact inhibition** — `nuclear_area_time`, `nc_effective`, `confluence`.
5. **Two-population model** — `population_mixture`, `BASAL_POP` (binucleate cells).
6. **Fibrosis → stiffness → prediction** — `FIBROSIS_STIFFNESS`, `fibrosis_prediction`.
7. **CALIBRATION** — summary of all values fitted to the data.

## Calibrated parameters (primary hepatocyte)

- **Motor-clutch:** nm=45, Fm=2.0, vu=110, nc=90, kon=0.5, koff0=0.1, Fb=2.0, kc=1.1, α=0.13.

- **Two populations:** basal 37.9 µm² (CV 6%, constant) + mechanosensitive ~68.8 µm² (grows).

- **Viscoelasticity** τ = 35.3 ± 2.6 h · inferred lamin vs LMNA qPCR: r=0.84.

- **Fibrosis stages:** F0~4, F1~7, F2~9.5, F3~13, F4~26 kPa.

## Notes

- Runs without numba, but ~50-100x slower; install it for parameter sweeps.
- Simulations are stochastic: use a larger `reps` (6-8) for stable means.
- Hepatocyte parameters are calibrated against real data; the other cell lines
  use literature-anchored starting points (laminAC is inferred from area and
  validated against qPCR)

## Diagram

![Study diagram](Diagram/Diagram_mvirtual_cell.png)
