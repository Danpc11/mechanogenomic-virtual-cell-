# A minimal physical model of nuclear mechanotransduction predicts the mechanogenomic trajectory of hepatic fibrosis

---

## Abstract

Tissue stiffening is a hallmark and a driver of chronic disease, yet the quantitative link between the mechanical microenvironment and the transcriptional programs it activates remains poorly formalized. Here we build a minimal, first-principles physical model of the mechanosensing cell — a "virtual cell" — in which substrate rigidity is transduced to nuclear deformation and YAP activity through a stochastic motor–clutch engine coupled to a lamin-A/C-gated nucleus. The model contains few parameters, all with physical meaning, and is calibrated against a single, easily measured observable: the projected nuclear area of primary hepatocytes cultured on hydrogels of defined stiffness. A model-based deconvolution reveals that the hepatocyte nuclear-area distribution is a mixture of a mechanically inert basal population (binucleate cells) and a mechanosensitive population whose area grows with stiffness and time; averaging the two masks the mechanical signal. The mechanosensitive population is captured by a contact-inhibition switch in which cell–cell (E-cadherin) engagement progressively displaces cell–substrate (integrin) clutches. Because hepatic fibrosis is progressive tissue stiffening — spanning ~1 kPa (F0) to ~26 kPa (F4), the same range as our hydrogels — the model predicts coordinated activation of the nuclear mechanotransduction machinery along fibrosis progression. We test this against three independent human liver RNA-seq cohorts and find that 31/31 nuclear-mechanosensitive genes rise monotonically with fibrosis stage, with a predominance of convex (threshold-like) trajectories that match the model's non-linear stiffness sensing. The framework unifies in vitro mechanobiology, a physical model, and human disease genomics, and yields testable predictions for the order of gene activation during fibrogenesis.

---

## 1. Introduction

Cells sense the stiffness of their surroundings and convert it into biochemical and transcriptional responses — a process termed mechanotransduction. The nucleus has emerged as a central mechanosensor: forces generated at cell–substrate adhesions are transmitted through the actin cytoskeleton and the LINC complex to the nuclear lamina, deforming the nucleus and modulating the nuclear import of transcriptional regulators such as YAP/TAZ. Lamin A/C, the principal determinant of nuclear stiffness, scales with tissue rigidity and cytoskeletal tension, forming a mechanical feedback that tunes the cell's sensitivity to its environment.

Liver fibrosis is a paradigmatic example of pathological stiffening. As fibrosis progresses through the METAVIR stages F0→F4, the liver stiffens from a soft, healthy parenchyma (~1–4 kPa) to a rigid, cirrhotic organ (~26 kPa, up to 48–69 kPa), driving hepatocyte dysfunction, loss of epithelial identity, and activation of profibrotic programs. Whether the coordinated transcriptional changes observed across fibrosis stages can be understood — and predicted — from a physical model of mechanosensing has not been established.

Existing models of mechanotransduction are typically either phenomenological (fitting response curves without mechanistic constraints) or highly detailed (large reaction networks with many free parameters). We take a deliberately minimal, first-principles approach: a small set of physically meaningful equations, calibrated against a single observable, that nonetheless captures the essential non-linearities of stiffness sensing. We then ask whether this in-vitro-calibrated model predicts the in-vivo genomic trajectory of a human disease.

Our contributions are threefold. First, we formulate a minimal virtual-cell model in which a stochastic motor–clutch engine drives a lamin-gated nucleus (Section 2). Second, we calibrate it against projected nuclear area of primary hepatocytes on hydrogels, and in doing so uncover a two-population structure — mechanically inert binucleate cells plus a mechanosensitive population governed by a contact-inhibition switch (Section 3). Third, we use the physical mapping between fibrosis stage and stiffness to predict, and then validate against three human RNA-seq cohorts, the coordinated upregulation of the nuclear mechanotransduction machinery, with a characteristic threshold-like signature (Section 4).

---

## 2. The physical model of the virtual cell

### 2.1 Architecture and rationale

The model is organized as a short causal chain, from the outside in:

$$
\text{substrate stiffness } E \;\longrightarrow\; \text{traction } T \;\longrightarrow\; \text{nuclear stress } \sigma \;\longrightarrow\; \{\text{nuclear deformation},\ \text{YAP activity}\}.
$$

Each arrow is a physical module with a small number of parameters. The design principle is minimality: every parameter must be either measurable or identifiable from data, and the model must reproduce the known non-linearities of mechanosensing (biphasic traction, threshold-like YAP activation, lamin-dependent gating) without ad-hoc terms. We describe each module in turn.

### 2.2 The motor–clutch engine

The interface between the cell and its substrate is modeled as a stochastic motor–clutch system (following Chan & Odde, 2008; Bangasser & Odde, 2013). A cell adheres through $n_c$ molecular clutches (integrin-based adhesions), each an elastic linker of stiffness $k_c$ connecting the retrograde-flowing actin cytoskeleton to a compliant substrate of stiffness $\kappa$. Myosin motors ($n_m$, each exerting force $F_m$) drive actin retrograde flow at unloaded velocity $v_u$, with stall force $F_{\text{stall}} = n_m F_m$.

The actin flow velocity is load-dependent:

$$
v = v_u\left(1 - \frac{F_{\text{sub}}}{F_{\text{stall}}}\right)_{+}, \qquad F_{\text{sub}} = \sum_{i \in \text{bound}} k_c\, x_i,
$$

where $x_i$ is the extension of bound clutch $i$ and $(\cdot)_+$ denotes rectification to non-negative values. Bound clutches are loaded by the flow, with the displacement partitioned between clutch and substrate according to their relative compliance:

$$
\frac{dx_i}{dt} = v \,\frac{\kappa}{\kappa + k_c}.
$$

Unbound clutches bind at constant rate $k_{\text{on}}$ (engaging at $x_i = 0$). Bound clutches detach as **slip bonds**, with a force-accelerated off-rate:

$$
k_{\text{off}} = k_{\text{off}}^{0}\,\exp\!\left(\frac{k_c\,|x_i|}{F_b}\right),
$$

where $F_b$ sets the characteristic bond-rupture force. The system is integrated by a Gillespie-like stochastic scheme; the steady-state mean substrate traction $T(\kappa) = \langle F_{\text{sub}}\rangle$ emerges from the competition between motor pulling and clutch load-and-fail dynamics.

A key emergent property, requiring no tuning, is a **biphasic dependence of traction on stiffness**: on very soft substrates clutches bear little load and detach rarely but transmit little force; on very stiff substrates load builds fast and clutches fail collectively ("load-and-fail"); an optimal stiffness $\kappa_{\text{opt}}$ maximizes traction. In our calibration $\kappa_{\text{opt}} \sim 0.7$–$1$ pN/nm, and — consistent with the original motor–clutch analysis — $\kappa_{\text{opt}}$ is far more sensitive to clutch parameters ($n_c, k_{\text{on}}, F_b$) than to motor parameters ($n_m, F_m$).

**Substrate stiffness mapping.** The experimentally controlled Young's modulus $E$ (in kPa) is mapped to the model clutch–substrate stiffness by a linear relation $\kappa = \alpha E$, with $\alpha$ a single coupling constant (~0.13–0.15). This is the only phenomenological link in the substrate module and is fixed once per phenotype.

### 2.3 Force transmission to the nucleus

The traction generated at adhesions is transmitted through the actin cap and the LINC complex to the nuclear surface. We model the nuclear stress as the traction weighted by a mechanical coupling factor:

$$
\sigma_{\text{nuc}}(E) = T(E)\,\frac{\kappa}{\kappa + k_c},
$$

i.e. the fraction of traction that is elastically transmitted rather than dissipated in compliant elements. This preserves the biphasic character of $T$ while imposing that soft substrates transmit little stress to the nucleus.

### 2.4 Lamin-A/C gating and YAP activity

Lamin A/C determines nuclear stiffness and therefore how nuclear stress is converted to nuclear-envelope deformation and to YAP nuclear import. We introduce a single dimensionless lamin level $\ell$ (relative to a reference) that gates the response through two effects.

First, the nuclear envelope must be "unwrinkled" (flattened) before nuclear pore stretch increases YAP import; the unwrinkling threshold is set by the lamina and thus by $\ell$:

$$
u(\sigma) = \frac{1}{1 + \exp\!\big[-(\sigma - \sigma^{*}/\ell)/w\big]},
$$

a stiffer lamina (higher $\ell$) requiring more stress to unwrinkle. Second, once unwrinkled, a stiffer lamina sustains more surface tension, amplifying YAP import. The YAP nucleocytoplasmic ratio is

$$
\left(\frac{\text{YAP}}{\text{YAP}}\right)_{N/C} = 1 + (R_{\max}-1)\,u(\sigma)\,\ell\,\frac{\sigma}{\sigma + \sigma_s}.
$$

This reproduces two experimental anchors: a resting ratio near 1 that rises 4–5-fold with stiffness, and the collapse of YAP nuclear localization upon lamin A/C knockdown. In our implementation, reducing $\ell$ by 80% lowers the stiff-substrate YAP ratio by ~65%, reproducing the reported lamin-dependence of YAP.

### 2.5 Nuclear deformation and its dynamics

The experimentally accessible observable is the projected nuclear area $A$ (DAPI, confocal Z-projection). Its steady-state value increases with nuclear stress as the nucleus flattens:

$$
A_{\text{ss}}(E) = A_{\min} + (A_{\max} - A_{\min})\,\frac{\sigma(E)}{\sigma(E) + s_{1/2}}, \qquad s_{1/2} = s_0\,\ell.
$$

Crucially, the half-saturation stress $s_{1/2}$ is **proportional to the lamin level $\ell$**: a stiffer nucleus resists flattening, so its area saturates at higher stress. This makes $\ell$ **identifiable from the shape of the area-vs-stiffness curve** — the basis of our calibration (Section 3.4).

Nuclear flattening is not instantaneous. We model the approach to steady state as first-order relaxation with a single time constant $\tau$:

$$
\tau\,\frac{dA}{dt} = A_{\text{ss}}(E) - A(t) \;\;\Rightarrow\;\; A(t) = A_{\text{ss}}(E) + \big(A_0 - A_{\text{ss}}(E)\big)\,e^{-t/\tau}.
$$

### 2.6 Two-population structure

Primary hepatocyte cultures are heterogeneous. The projected nuclear-area distribution is modeled as a two-component mixture:

$$
P(A \mid E, t) = \phi\, \mathcal{N}\!\big(A;\,\mu_b,\,\varsigma_b\big) \;+\; (1-\phi)\,\mathcal{N}\!\big(A;\,\mu_m(E,t),\,\varsigma_m\big),
$$

with a **basal population** of constant mean $\mu_b$ — hypothesized to be binucleate hepatocytes, whose doubled DAPI signal gives a fixed basal area independent of mechanics — and a **mechanosensitive population** whose mean $\mu_m(E,t)$ follows the deformation dynamics of Section 2.5. The mixing weight $\phi$ (basal fraction) rises with confluence.

### 2.7 The contact-inhibition switch (integrin → cadherin)

As cells proliferate and confluence increases, cell–cell contacts form. Cadherin-based adherens junctions engage an E-cadherin/α-catenin **clutch** that mechanically competes with substrate adhesions: engagement of cell–cell contacts progressively displaces cell–substrate (integrin) clutches. We encode this as a confluence-dependent reduction of the effective substrate clutch number:

$$
n_c^{\text{eff}}(t) = n_c^{0}\,\big[1 - \beta\, c(t)\big], \qquad c(t) = 1 - e^{-t/t_c},
$$

where $c(t)$ is confluence and $\beta$ the strength of contact inhibition. Because $n_c^{\text{eff}}$ enters the motor–clutch engine, this lowers traction and nuclear stress as cultures confluence, so the mechanosensitive population's spreading saturates and its fraction declines — consistent with the observed dynamics (Section 3.3). This module directly connects a genomic/biochemical switch (cadherin↑, focal-adhesion↓) to a physical parameter of the engine.

### 2.8 Phenotype as a parameter vector

A cell phenotype is a point in the model's parameter space — principally $(\ell, n_m, n_c, \alpha)$. Distinct phenotypes therefore produce distinct, falsifiable stiffness-response curves. As an illustration, with lamin levels ordered tumor(low) < epithelial < fibroblast(normal) < alveolar(high), the model yields YAP N/C ranges of 1.1→2.7 (invasive MDA-MB-231), 1.8→4.4 (fibroblast) and 1.3→5.4 (AT2), i.e. a damped response for the soft-nucleus invasive phenotype and an amplified response for the stiff-nucleus epithelial phenotype — from the physics alone, not from fitting.

---

## 3. Calibration with hydrogel experiments

### 3.1 Experimental design

Primary rat hepatocytes were cultured on polyacrylamide hydrogels of stiffness 0.5, 1, 5 and 23 kPa and fixed at 2, 12, 24 and 36 h (four stiffnesses × four times × two biological replicates; ~40,000 nuclei total). Nuclear area was quantified from DAPI confocal Z-projections. The experimental design deliberately relies on two readily obtained observables — **projected nuclear area** (the calibration target) and **qPCR** (independent validation) — and does *not* require direct measurement of nuclear stiffness, which is instead an inferred model parameter.

### 3.2 Two populations, not one

Naïvely averaging all nuclei gives a nearly flat area–stiffness relationship (stiffness explaining only ~2% of the variance; **Fig. 1A**), which would suggest hepatocytes are mechanically unresponsive. However, a Gaussian-mixture analysis of the raw single-cell distributions rejects the one-population model in 16/16 conditions by BIC: the data are intrinsically bimodal (**Fig. 1B**). This resolves an apparent discrepancy between the flat population average and immunofluorescence images (DAPI/CK-18/F-actin) that show clear stiffness- and time-dependent spreading: the mechanical signal in the responsive subpopulation is diluted when averaged with the constant basal population.

### 3.3 Differential dynamics validate the two-population model

The two populations behave exactly as hypothesized. The basal population is constant (mean 37.9 ± 2.3 µm², CV 6%, no correlation with time; $r=-0.19$, $p=0.48$), consistent with a fixed-area binucleate subpopulation. The mechanosensitive population grows with time ($r=+0.53$, $p=0.04$) and reaches larger areas at higher stiffness (**Fig. 1B**, **Fig. 2**). Moreover, the mechanosensitive fraction declines over time (from ~0.55 to ~0.30), the signature of confluence-driven contact inhibition (Section 2.7; **Fig. 4A**). Deformation normalized to the 2 h baseline shows a graded increase with time and stiffness (**Fig. 2**).

### 3.4 Lamin A/C inferred from area, validated by qPCR

Because the lamin level $\ell$ sets the half-saturation stress $s_{1/2}$ (Section 2.5), it is identifiable from the *shape* of the area–stiffness curve. Fitting the mechanosensitive population recovers a lamin level per cell line without any direct nuclear-stiffness measurement. In cross-line demonstration data the inferred lamin correlates with qPCR of *LMNA* ($r=0.84$; **Fig. 3**), providing independent molecular validation of a purely mechanically inferred parameter. Identifiability requires adequate sampling of the transition region: ≥6–7 stiffnesses spanning ~0.5–40 kPa, ≥3 biological replicates, ≥30 cells per point.

### 3.5 Mechanism comparison

Two mechanistic hypotheses for the mechanosensitive population were compared by fit: (A) stiffness sets the area plateau via the motor while area approaches it with time constant $\tau$; (B) the coupling $\alpha$ varies with spreading. The temporal-dynamics mechanism (A) fits better ($R^2 = 0.56$ vs $0.33$), with $\tau \approx 35$ h. The integrated model is summarized schematically in **Fig. 5**.

---

## 4. Predictions and validation with RNA-seq of hepatic fibrosis

### 4.1 Fibrosis stage is a stiffness axis

The central prediction connecting the calibrated model to disease rests on a physical identity: **hepatic fibrosis is progressive tissue stiffening**. Shear-wave and transient elastography give consistent median stiffness values per METAVIR stage:

| Stage | Tissue stiffness (kPa) | Histology |
|:-----:|:----------------------:|:----------|
| F0 | ~1–4 | no fibrosis |
| F1 | ~7 | portal fibrous expansion |
| F2 | ~9.5 | thin septa |
| F3 | ~13 | bridging septa |
| F4 | >22 (to 48–69) | cirrhosis |

The F0→F4 progression sweeps ~4→26 kPa — essentially the same mechanical range as our hydrogels (0.5–23 kPa; **Fig. 6A**). The in vitro axis and the in vivo disease axis are the *same physical variable*.

### 4.2 Model predictions across fibrosis stages

Running the motor at the stiffness of each stage predicts the mechanotransduction output (nuclear stress, YAP activity, lamin level) as a function of fibrosis stage. The model predicts that the nuclear mechanosensing machinery is progressively engaged as the tissue stiffens, with the strongest activation in the F3→F4 transition, where stiffness rises most steeply.

### 4.3 Coordinated upregulation of the mechanosensitive machinery

We tested this against three independent human liver RNA-seq cohorts (GSE130970, GSE135251, GSE162694), examining 31 genes spanning the nuclear mechanotransduction machinery: YAP/TAZ output (*CCN2/CTGF, WWTR1, YAP1, TEAD2/4*), nuclear envelope and lamina (*LMNA, LMNB2, TMPO, NUP93, TPR*), contractile cytoskeleton (*ACTA2, MYL9, MYH9/10, CFL1, VIM, FLNA*), adhesion/mechanosensors (*ILK, VCL, SRC, PIEZO1*), matrix remodeling (*LOX, COL1A1, COL1A2*), and transcriptional regulators (*MKL1, SMAD2, KLF2, HDAC1/2, DNMT3A*). All 31 genes rise monotonically with fibrosis stage across the cohorts (**Fig. 6B**), a coherent, model-consistent activation of the machinery.

### 4.4 Convex signatures indicate threshold sensing

Beyond monotonicity, the *shape* of the trajectories is informative. A majority of genes (17/31) follow convex (power-law/sigmoid) trajectories rather than linear ones — they surge at high stiffness (F3–F4) rather than rising proportionally. This is precisely the non-linear, threshold-like response predicted by the motor–clutch engine and the lamin-gated YAP module (Sections 2.2, 2.4): mechanical sensing is not proportional but amplifying. The agreement between the predicted non-linearity and the observed convex signatures (**Fig. 6C**) is stronger evidence than monotonicity alone, and suggests that non-linear mechanical sensing is a *driver* of the fibrotic transcriptional program, not merely a correlate.

### 4.5 Predicted activation order and candidate biomarkers

Because the model assigns each response an effective stiffness threshold, it predicts an **order of activation** along fibrosis progression: low-threshold genes activate early (biomarkers of incipient stiffening), high-threshold genes activate late (markers of the F3→F4 transition to cirrhosis). This ordering is directly testable in the GEO cohorts by ranking genes on the fibrosis-stage at which they cross half-maximal induction.

---

## 5. Discussion

We have presented a minimal, first-principles model of nuclear mechanotransduction — a virtual cell — that is calibrated against a single, easily measured observable and then predicts the genomic trajectory of a human disease. Three features distinguish the approach. First, **minimality with mechanism**: a handful of physically meaningful parameters reproduce the biphasic traction, threshold-like YAP activation, and lamin-dependent gating that phenomenological fits impose by hand. Second, **inference rather than measurement**: nuclear stiffness (lamin A/C), which is difficult to measure directly, is inferred from the shape of the area–stiffness curve and validated independently by qPCR. Third, **a physical bridge to pathology**: because fibrosis is stiffening, the same model that describes hydrogel experiments predicts the in-vivo mechanogenomic trajectory, validated in three human cohorts.

The two-population structure carries a methodological lesson: population averages can mask mechanical signals when a mechanically inert subpopulation (here, binucleate hepatocytes) is present. Model-based deconvolution recovers the signal and, in doing so, exposes a second mechanism — contact inhibition — in which cell–cell (cadherin) engagement displaces cell–substrate (integrin) clutches, naturally encoded as a reduction of effective substrate clutches in the engine.

**Clinical implication.** If the order of gene activation follows the model's effective thresholds, low-threshold mechanosensitive genes are candidate early biomarkers of tissue stiffening, and high-threshold genes candidate markers of the transition to cirrhosis — a mechanically grounded staging of fibrogenesis.
