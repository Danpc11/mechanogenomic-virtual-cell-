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

| Phenotype | Key | Status | Fast surrogate | Intended use |
|---|---|---|---|---|
| Hepatocyte | `hepatocyte` | **Reference case study** | calibrated (R²≈0.98) | Hepatic fibrosis and stiffness-driven mechanogenomics |
| A549 | `A549` | Exploratory | — | Lung epithelial mechanotransduction |
| NHLF | `NHLF` | Exploratory | calibrated (R²≈0.98) | Lung fibroblast stiffness response |
| AT2 | `AT2_lung` | Exploratory | calibrated (R²≈1.00) | Alveolar epithelial mechanics |
| MCF10A | `MCF10A` | Exploratory | — | Mammary epithelial mechanobiology |
| MDA-MB-231 | `MDA` | Exploratory | calibrated (R²≈0.98) | Cancer-cell mechanosensing |
| Fibroblast | `fibroblast` | Exploratory | — | Matrix remodeling and fibrotic activation |

All phenotypes are instantiable today via `VirtualCell(<key>)` or
`PHENOTYPES[<key>]`; the hepatocyte is the fully calibrated **reference case
study**, and the others are literature-anchored starting points (with a fitted
fast surrogate where noted). New phenotypes are added by supplying a parameter
set — the physical core is shared, so the framework is genuinely phenotype-aware
rather than hepatocyte-specific.

The hepatocyte model is currently the most developed phenotype because it is supported by hydrogel nuclear-area dynamics and fibrosis-stage transcriptomic validation.

---

## Quick start

Clone the repository and install dependencies:

```bash
git clone https://github.com/Danpc11/mechanogenomic-virtual-cell.git
cd mechanogenomic-virtual-cell

pip install -r requirements.txt
```

Or install it as a package (recommended — makes it importable anywhere and adds
console commands):

```bash
pip install -e .              # core
pip install -e ".[all]"       # + numba, gplearn, figures, and 3D visualization
```

Installed, the model is importable under the `mvcell` namespace and exposes
console commands:

```bash
mvcell-demo          # VirtualCell demo
mvcell-benchmark     # full model vs simple baselines
mvcell-sensitivity   # local + global sensitivity
```

```python
from mvcell import VirtualCell
cell = VirtualCell("hepatocyte")
state = cell.simulate(E=23.0, t=72.0)
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
├── pyproject.toml                 # installable package (pip install -e .)
│
├── src/                           # importable as the `mvcell` package
│   ├── __init__.py
│   ├── paths.py
│   ├── mvirtual_cell.py           # core physical model + phenotype library
│   ├── virtual_cell.py            # VirtualCell class (stateful interface)
│   ├── gene_module.py             # mechanosensitive gene layer / hypotheses
│   ├── fast_model.py
│   ├── calibration.py
│   ├── recalibration.py
│   ├── inference.py
│   ├── symbolic.py
│   ├── benchmark.py               # full model vs simple baselines
│   ├── sensitivity.py             # local (OAT) + global (Sobol)
│   ├── stats_ci.py                # bootstrap CIs + resampling tests
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
├── visualization/                 # model-generated visualization
│   ├── make_state_grid.py
│   └── render_pyvista_virtual_cell.py   # 3D (PyVista + Trame)
│
├── docs/                          # web demo (GitHub Pages)
│   ├── virtual_cell_demo.html
│   └── states.json
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

### `src/virtual_cell.py`

The central `VirtualCell` interface. Wraps the physical model into a stateful,
reusable object — the same physical core, re-parameterized per phenotype, is a
distinct in-silico avatar. Its single physical input is tissue stiffness (the
quantity elastography measures clinically).

```python
from virtual_cell import VirtualCell

cell = VirtualCell("hepatocyte")
state = cell.simulate(E=23.0, t=72.0)   # advance to a mechanical context
state.yap_nc, state.nuclear_area        # observables
cell.state_vector()                     # numeric state vector (for analysis)
cell.gene_scores()                      # mechanogenomic output
cell.trajectory()                       # F0->F4 trajectory
```

`CellState` carries the full observable state (traction, nuclear drive, area,
YAP, lamin, effective clutches, tau(E), function index, fibrosis stage, gene
scores).

---

### `src/gene_module.py`

The mechanosensitive gene layer and hypothesis generator. Maps the mechanical
state to per-gene activation using an explicit response-shape model — **sigmoid
(threshold), weak-power (saturating), or linear (graded)** — assigned from each
gene\'s mechanotransduction role *before* looking at RNA-seq, so it is a
falsifiable prediction rather than a post-hoc fit.

```python
import gene_module as gm

gm.response_shape_table()        # predicted shape per gene (pre-registered)
gm.score_genes(nuclear_drive)    # activation scores
gm.actionable_hypotheses(drive)  # candidate intervention points
gm.qpcr_panel()                  # suggested validation panel (one per shape)
```

Actionable genes whose predicted activation crosses threshold are flagged as
candidate intervention points — the hypothesis-generation output.

---

### `src/benchmark.py`

Benchmarks the full mechanistic model against simple baselines on the same
data: (a) linear stiffness→area, (b) motor-clutch stress without the
nuclear/temporal layer, (c) the full model. Uses leave-one-condition-out
cross-validation and AIC/BIC (complexity-penalized).

```python
import benchmark as bm
bm.run_benchmark()
```

The full model generalizes better (CV-R² ≈ 0.85 vs 0.78) and, crucially, is the
only one that captures the temporal rise on stiff substrate — the tau(E)
dynamical law the simple models structurally cannot represent. This is the
evidence that the physics buys predictive structure, not just fit flexibility.

---

### `src/sensitivity.py`

Local one-at-a-time elasticities and global variance-based Sobol indices (a
lightweight Saltelli/Jansen estimator, no external SALib dependency).

```python
import sensitivity as sa
sa.run_sensitivity()             # local (OAT) + global (Sobol)
```

Identifies the nuclear gate (`laminAC`) and the adhesion/clutch group (`nc`,
`kc`) as the parameters the data must constrain — matching what inference
identifies — while the `alpha` coupling is low-impact and safe to fix. Provides
the robustness evidence.

---

### `src/stats_ci.py`

Statistical rigor: nonparametric bootstrap confidence intervals and
resampling-based tests.

```python
import stats_ci as st

st.bootstrap_ci(sample)                       # BCa/percentile CI for any statistic
st.fold_change_ci(soft_vals, stiff_vals)      # CI on the stiffness fold-change
st.bootstrap_parameter(fit_fn, data_rows)     # CI on a fitted parameter
st.permutation_test(a, b)                     # difference between conditions
```

The headline mechanosensitivity result comes with uncertainty: the 1→23 kPa
nuclear-area fold-change is **≈2.1× (95% CI ≈ [1.8, 2.5])**, a real mechanical
response (the CI excludes 1).

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

## Visualization

Visualizations are **generated from the model\'s own state outputs** (not
hand-drawn cartoons). Every visual feature maps to a model variable:

| Model output | Visual encoding |
|---|---|
| `nuclear_area` A(t) | nucleus size |
| `nuclear_drive` sigma | nucleus flattening + color warmth |
| `yap_nc` | dots / spheres inside the nucleus |
| `laminAC` | nuclear-envelope thickness |
| `nc_eff` | number of adhesions |
| `traction` T(E) | traction cones at adhesions |
| gene scores | activation bars |

Two levels:

**Web demo (`docs/`)** — a self-contained `virtual_cell_demo.html` with sliders
for stiffness and time, publishable via GitHub Pages. Regenerate its states
with `python visualization/make_state_grid.py`.

**3D scientific rendering (`visualization/render_pyvista_virtual_cell.py`)** —
PyVista + Trame. Static export works headless; the interactive Trame app serves
stiffness/time sliders in the browser.

```bash
# static image (headless)
python visualization/render_pyvista_virtual_cell.py --E 23 --t 120 --save cell.png

# interactive 3D app (run locally, needs a browser)
python visualization/render_pyvista_virtual_cell.py --serve
```

Requires the visualization extra: `pip install -e ".[viz]"`.

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

Hepatic fibrosis is the **first, fully-calibrated case study** of the general mechanogenomic virtual-cell framework — chosen because tissue stiffening provides a clean physical axis and because complete nuclear-area timecourses and fibrosis RNA-seq cohorts are available. The framework itself is phenotype-agnostic (see the phenotype table); the hepatocyte is where it is validated in depth.

In this case study, fibrosis progression is treated as a tissue-stiffness axis. The hepatocyte virtual cell is calibrated using nuclear-area dynamics from primary hepatocytes cultured on soft and stiff hydrogels, and the resulting model outputs are compared with fibrosis-associated RNA-seq trajectories from human liver cohorts.

Current model interpretation:

- the mechanosensitive population shows a strong stiffness-dependent nuclear-area increase;
- the 1→23 kPa response is approximately 2.1–2.2-fold (bootstrap 95% CI ≈ [1.8, 2.5]) for the mechanosensitive population;
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
- sensitivity of the traction optimum to clutch and motor parameters;
- stiffness-dependent relaxation time tau(E);
- the VirtualCell interface and state vector;
- gene response-shape predictions (sigmoid / weak-power / linear);
- benchmark: full model generalizes and captures the temporal law;
- sensitivity: nc and laminAC dominate (matching inference);
- bootstrap confidence interval on the stiffness fold-change.

The suite currently contains **16 validations** and runs in CI.

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

