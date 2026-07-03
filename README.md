<h1>
  <img src="assets/mvirtual_cell_logo.png" alt="mVirtual Cell logo" width="40">
  Mechanogenomic virtual-cell model
</h1>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-supported-blue)
![SciPy](https://img.shields.io/badge/SciPy-supported-blue)

**Mechanogenomic Virtual Cell** is a phenotype-aware physical-computational framework for modeling how extracellular stiffness is converted into cellular traction, nuclear mechanotransduction, YAP/TAZ activity and mechanosensitive transcriptional trajectories.

The current reference case study focuses on **primary hepatocytes and hepatic fibrosis**, where tissue stiffening is used as a mechanical axis to predict nuclear remodeling and fibrosis-associated transcriptional trajectories. The framework is designed to support additional mechanically distinct cell phenotypes.

Full documentation: see the [project Wiki](https://github.com/Danpc11/mechanogenomic-virtual-cell/wiki).

---

## Conceptual overview

The model represents the cell as a physically constrained mechanotransduction system:

```text
extracellular stiffness
        ↓
motor–clutch traction
        ↓
nuclear mechanical drive
        ↓
lamin A/C-gated nuclear deformation
        ↓
YAP/TAZ nuclear activity
        ↓
mechanosensitive gene trajectories
```

The central hypothesis is that extracellular and tissue stiffness can be treated as a physical input that drives nuclear remodeling and mechanogenomic activation.

In the current reference implementation, hepatic fibrosis is used as the first disease case study. Fibrosis progression is represented as a tissue-stiffness axis, and the hepatocyte virtual cell is calibrated using nuclear-area dynamics from primary hepatocytes cultured on hydrogels.

---

## Phenotype-aware modeling

The framework is not restricted to a single cell type. Each phenotype is represented by a parameter set controlling mechanical sensing, nuclear mechanics and mechanotranscriptional response.

Phenotype-specific parameters may include:

- actomyosin contractility;
- effective clutch number;
- clutch stiffness;
- nuclear-area range;
- lamin A/C level;
- YAP/TAZ mechanosensitivity;
- contact inhibition;
- time-dependent nuclear relaxation.

Current and planned phenotypes include:

| Phenotype | Status | Intended use |
|---|---|---|
| Hepatocyte | Reference case study | Hepatic fibrosis and stiffness-driven mechanogenomics |
| Fibroblast | Exploratory | Matrix remodeling and fibrotic activation |
| A549 | Exploratory | Lung epithelial mechanotransduction |
| AT2 | Exploratory | Alveolar epithelial mechanics |
| MCF10A | Exploratory | Mammary epithelial mechanobiology |
| MDA-MB-231 | Exploratory | Cancer-cell mechanosensing |
| NHLF | Exploratory | Lung fibroblast stiffness response |

The hepatocyte model is currently the most developed phenotype because it is supported by hydrogel nuclear-area dynamics and fibrosis-stage transcriptomic validation.

---

## Quick start

Clone the repository and install dependencies:

```bash
git clone https://github.com/Danpc11/mechanogenomic-virtual-cell.git
cd mechanogenomic-virtual-cell

pip install -r requirements.txt
```

Run the core model demo:

```bash
python src/mvirtual_cell.py
```

Run the validation suite:

```bash
python test/test_virtual_cell.py
```

Regenerate figures and outputs:

```bash
python results/make_figures.py
```

Use the model from Python:

```python
import sys
sys.path.insert(0, "src")

from mvirtual_cell import (
    PHENOTYPES,
    nuclear_stress,
    nuclear_area_ss,
    yap_nc_ratio,
    nuclear_area_time,
    population_mixture,
    fibrosis_prediction,
    CALIBRATION,
)

hep = PHENOTYPES["hepatocyte"]

# Nuclear mechanical drive at fibrosis-like stiffness
drive = nuclear_stress(23.0, hep)

# Steady-state nuclear area
area_ss = nuclear_area_ss(23.0, hep)

# YAP nuclear/cytoplasmic ratio
yap = yap_nc_ratio(23.0, hep)

# Time-dependent nuclear area at 36 h
area_36h = nuclear_area_time(
    23.0,
    t=36,
    ph=hep,
    contact_inhibition=True,
)

# Two-population nuclear-area model
mu_low, mu_mecano, phi = population_mixture(23.0, t=36, ph=hep)

# Fibrosis-stage prediction
fibrosis = fibrosis_prediction(hep)
```

---

## Repository structure

```text
mechanogenomic-virtual-cell/
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── Theory_draft.md
│
├── assets/
│   ├── Diagram_mvirtual_cell.png
│   ├── mvc_logo.png
│   ├── mvc_logo_3.png
│   └── mvirtual_cell_logo.png
│
├── src/
│   ├── paths.py
│   ├── mvirtual_cell.py
│   ├── fast_model.py
│   ├── calibration.py
│   ├── recalibration.py
│   ├── inference.py
│   ├── symbolic.py
│   └── pharmacology.py
│
├── data/
│   ├── RANseq_datasets_info.md
│   ├── genes_nucleo_all.tsv
│   ├── hepatocyte_complete_data.json
│   ├── hepatocyte_two_populations.csv
│   └── saturating_params.json
│
├── results/
│   ├── hepatocyte_posterior.json
│   └── make_figures.py
│
└── test/
    └── test_virtual_cell.py
```

Scripts resolve file paths through `src/paths.py`, so data and results are located relative to the repository root rather than the current working directory.

---

## Main files

### `src/mvirtual_cell.py`

Core physical model.

It contains:

1. **Motor–clutch engine**  
   A stochastic actomyosin–integrin clutch model that converts substrate stiffness into traction.

2. **Phenotype library**  
   A `Phenotype` dataclass and `PHENOTYPES` dictionary containing calibrated or literature-anchored cell phenotypes.

3. **Mechanotransduction chain**  
   Functions linking traction to nuclear mechanical drive, nuclear area, YAP/TAZ activity and expected lamin A/C behavior.

4. **Temporal dynamics**  
   Time-dependent nuclear area relaxation and contact-inhibition effects.

5. **Two-population nuclear-area model**  
   A basal population plus a mechanosensitive population.

6. **Fibrosis-stage prediction**  
   A stiffness mapping from fibrosis stages F0–F4 to predicted mechanotransduction outputs.

7. **Calibration summary**  
   The `CALIBRATION` object records fitted and inferred values.

> Note on terminology: `nuclear_stress` returns a **nuclear mechanical drive**, not a physical stress in Pa. The function name is kept for continuity.

---

### `src/calibration.py`

Fitting and calibration layer.

Use this file to recover model parameters from experimental nuclear-area data.

Key functions include:

- `load_hydrogel_csv`
- `deconvolve_two_populations`
- `two_population_table`
- `population_stats`
- `fit_lamin_from_area`
- `fit_temporal`
- `fit_phenotype`
- `correlate_with_expression`

Example:

```python
import calibration as cal

data = cal.load_hydrogel_csv("areas.csv")

rows = cal.two_population_table(data)
print(cal.population_stats(rows))

phenotype, report = cal.fit_phenotype(
    data,
    name="my_hepatocyte",
)
```

---

### `src/recalibration.py`

Recalibration using the complete 2–120 h hepatocyte timecourse.

This module performs a two-level fit:

1. time-dependent fitting from the complete 1 kPa and 23 kPa curves;
2. stiffness-shape fitting from additional hydrogel points.

Example:

```python
import recalibration as rc

rc.tau_vs_stiffness()
rc.mechanical_fold_change()
rc.recalibrated_summary()
```

Current interpretation:

- the complete timecourse supports a strong mechanical response from 1 to 23 kPa;
- the mechanosensitive nuclear-area population increases approximately 2.2-fold between soft and stiff substrates;
- intermediate stiffness conditions are currently less constrained because complete timecourses are available only for 1 and 23 kPa.

---

### `src/inference.py`

Simulation-based inference for uncertainty and identifiability.

This module estimates posterior distributions over physical parameters rather than relying only on point estimates.

It includes:

- ABC-SMC inference for static area-vs-stiffness data;
- timecourse inference for dynamic nuclear-area data;
- posterior summaries for lamin A/C and other effective physical parameters.

Example:

```python
import inference as inf

res = inf.abc_smc(observed_area, Es)
res_t = inf.abc_timecourse(observed_dynamics)
```

The posterior output is stored in:

```text
results/hepatocyte_posterior.json
```

---

### `src/symbolic.py`

Symbolic regression module.

This module searches for compact analytic expressions that approximate the stiffness-to-nuclear-drive relationship generated by the stochastic motor–clutch model.

The current discovered form is a saturating response:

```text
sigma(E) = Vmax * E / (K + E)
```

This form is stored in:

```text
data/saturating_params.json
```

---

### `src/fast_model.py`

Fast analytic surrogate for parameter sweeps and inference.

The stochastic motor–clutch model is the mechanistic reference model.  
The fast model uses the saturating expression discovered by symbolic regression to provide an instant approximation.

Use this for:

- parameter sweeps;
- sensitivity analysis;
- inference;
- figure generation;
- repeated simulations.

Example:

```python
import fast_model as fm

fm.nuclear_stress_fast(23.0, "hepatocyte")
fm.calibrate(PHENOTYPES["MCF10A"])
```

---

### `src/pharmacology.py`

Hypothesis-generating clinical and pharmacological extension.

This is **not** a validated pharmacology, PK, DILI or toxicity model.

It maps mechanical disease state to exploratory predictions involving:

- elastography stiffness;
- mechanotransduction-targeting interventions;
- mechanical-function axes such as albumin, urea, CYP450 and HNF4A behavior.

Example:

```python
import pharmacology as ph

ph.map_patient(13.0)
ph.screen_drugs(E=26.0)
ph.toxicity_flag("fasudil", E=26.0)
```

Use this module only for exploratory hypothesis generation.

---

### `src/paths.py`

Centralized path resolver.

This keeps scripts portable by resolving paths to:

- repository root;
- `data/`;
- `results/`;
- `assets/`.

---

## Data files

### `data/hepatocyte_complete_data.json`

Complete primary-hepatocyte nuclear-area timecourse.

Current complete timecourse conditions:

- 1 kPa;
- 23 kPa;
- 2 h, 36 h, 72 h and 120 h.

This file anchors the recalibrated time-dependent model.

---

### `data/hepatocyte_two_populations.csv`

Two-population deconvolution of nuclear-area distributions.

The model separates:

1. a low-area basal population;
2. a mechanosensitive population whose nuclear area increases with stiffness and time.

---

### `data/saturating_params.json`

Parameters for the fast saturating stiffness-to-drive surrogate:

```text
sigma(E) = Vmax * E / (K + E)
```

These parameters are used by `src/fast_model.py`.

---

### `data/genes_nucleo_all.tsv`

Mechanosensitive and nuclear-associated gene list used for mechanogenomic analysis.

Representative modules include:

- YAP/TAZ–TEAD signaling;
- nuclear envelope and lamina;
- adhesion and cytoskeleton;
- extracellular matrix and fibrosis.

---

### `data/RANseq_datasets_info.md`

RNA-seq cohort notes for fibrosis-stage validation.

This file documents the human liver RNA-seq datasets used to compare model-predicted mechanotransduction outputs with fibrosis-associated transcriptional trajectories.

---

## Results files

### `results/hepatocyte_posterior.json`

Posterior parameter estimates from simulation-based inference.

This file summarizes uncertainty in the recalibrated hepatocyte model.

---

### `results/make_figures.py`

Figure-generation script.

Use it to regenerate project figures and visual outputs from the model and data files:

```bash
python results/make_figures.py
```

---

## Assets

The `assets/` folder contains visual material for documentation and presentation:

```text
assets/Diagram_mvirtual_cell.png
assets/mvc_logo.png
assets/mvc_logo_3.png
assets/mvirtual_cell_logo.png
```

These files are used for the README, Wiki and conceptual model diagrams.

---

## Model outputs

The model predicts the following physical and biological quantities:

| Output | Meaning |
|---|---|
| `traction(E)` | Cell-generated traction from the motor–clutch system |
| `nuclear_stress(E)` | Nuclear mechanical drive transmitted from the substrate |
| `nuclear_area_ss(E)` | Steady-state projected nuclear area |
| `nuclear_area_time(E, t)` | Time-dependent nuclear area |
| `yap_nc_ratio(E)` | YAP nuclear-to-cytoplasmic ratio |
| `lamin_expected(E)` | Expected lamin A/C-linked nuclear response |
| `population_mixture(E, t)` | Basal and mechanosensitive nuclear-area populations |
| `fibrosis_prediction()` | Predicted F0–F4 mechanotransduction trajectory |

---

## Case study: hepatic fibrosis

The first biological application of the mechanogenomic virtual-cell framework is hepatic fibrosis.

In this case study, fibrosis progression is treated as a tissue-stiffness axis. The hepatocyte virtual cell is calibrated using nuclear-area dynamics from primary hepatocytes cultured on soft and stiff hydrogels, and the resulting model outputs are compared with fibrosis-associated RNA-seq trajectories from human liver cohorts.

Current model interpretation:

- the mechanosensitive population shows a strong stiffness-dependent nuclear-area increase;
- the 1→23 kPa response is approximately 2.2-fold for the mechanosensitive population;
- nuclear adaptation is time-dependent and stiffness-dependent;
- high-stiffness substrates relax more slowly than soft substrates;
- the stiffness-to-drive relation is saturating rather than purely linear.

Approximate fibrosis stiffness mapping used by the model:

| Fibrosis stage | Approximate stiffness |
|---|---|
| F0 | ~1–4 kPa |
| F1 | ~7 kPa |
| F2 | ~9.5 kPa |
| F3 | ~13 kPa |
| F4 | ~23–26 kPa |

---

## Validation

Run:

```bash
python test/test_virtual_cell.py
```

The validation suite checks qualitative anchors of the model, including:

- stiffness-dependent traction;
- stiffness-dependent nuclear spreading;
- YAP/TAZ activation;
- lamin A/C perturbation behavior;
- phenotype-level lamin ordering;
- two-population nuclear dynamics;
- contact inhibition;
- temporal relaxation;
- monotonic fibrosis-stage response;
- sensitivity of the traction optimum to clutch and motor parameters.

---

## Current limitations

This repository is an active research model. Important limitations:

1. `nuclear_stress` is a nuclear mechanical-drive scalar, not a stress in Pa.
2. Complete long-time-course nuclear-area data are currently available only for 1 and 23 kPa.
3. Intermediate stiffnesses require additional long-time measurements to fully constrain the steady-state stiffness curve.
4. The RNA-seq validation uses tissue-level fibrosis datasets and may include cell-composition effects.
5. The pharmacology module is hypothesis-generating and should not be interpreted as a validated drug-response or toxicity predictor.
6. qPCR validation in hepatocytes on fibrosis-like hydrogels is the next experimental step for closing the mechanogenomic loop.
7. Additional phenotypes are exploratory until phenotype-specific calibration data are incorporated.

---

## Suggested paper storyline

Working title:

```text
A mechanogenomic virtual-cell model predicts stiffness-driven transcriptional trajectories
```

Core claim:

```text
A phenotype-aware physical virtual-cell state predicts stiffness-driven mechanogenomic trajectories better than stiffness alone.
```

Current reference case study:

```text
Primary hepatocytes and hepatic fibrosis.
```

Planned extension:

```text
Additional phenotypes will be incorporated through phenotype-specific mechanical, nuclear and transcriptional parameter sets.
```

Planned figure logic:

1. **Framework** — define the phenotype-aware virtual-cell state.
2. **Physical model** — motor–clutch and nuclear mechanics.
3. **Hydrogel calibration** — nuclear-area dynamics and two-population fitting.
4. **Fibrosis prediction** — F0–F4 stiffness mapping and transcriptomic trajectories.
5. **Benchmark and ablation** — compare against linear, power-law and Hill/sigmoid baselines.
6. **qPCR validation** — test predicted genes in hepatocytes on fibrosis-like hydrogels.

---

## Documentation

Additional documentation is available in the [project Wiki](https://github.com/Danpc11/mechanogenomic-virtual-cell/wiki), including:

- Motor–Clutch Model;
- Nuclear Mechanics Model;
- model architecture;
- fibrosis stiffness mapping;
- gene trajectory interpretation.

---

## Citation

If you use this model or repository, please cite it using the repository citation file:

```text
CITATION.cff
```

GitHub can automatically generate a citation from the **Cite this repository** button.

---

## License

This project is released under the MIT License.

See:

```text
LICENSE
```
