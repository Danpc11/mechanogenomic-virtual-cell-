"""
================================================================================
 make_nature_figures.py  —  Nature-style figures for the mechanogenomic model
================================================================================

Regenerates the paper figures in a Nature-journal visual style, adapted from a
reference publication script (ticks-in, minor-tick suppression, bold mathtext,
lowercase panel letters, curated palette, insets, twin axes). Uses the real
hepatocyte data and the model modules.

Style choices (matching the reference):
  * sans-serif body font, bold axis labels/titles
  * ticks pointing IN on both axes, minor ticks length 0
  * lowercase bold panel letters placed at (x=-0.06, y=0.99)
  * curated indigo/cyan/amber/red/green/violet palette
  * thin spines (linewidth 1), no top/right clutter where appropriate
  * insets via inset_axes; twin y-axes for dual quantities

Note: matplotlib mathtext is used for symbols (LaTeX-free) so it runs anywhere.
Set USE_TEX=True if a full LaTeX install with type1cm/amsmath is available.

Outputs: figuras_nature/Fig1..Fig4 (pdf + png, 300 dpi).
Author: Daniel Pérez-Calixto (INMEGEN / UNAM)
================================================================================
"""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import mvirtual_cell as mvc
from mvirtual_cell import PHENOTYPES
import fast_model as fm
from paths import DATA_DIR, RESULTS_DIR, FIGURES_DIR

# ----------------------------------------------------------------------------
# GLOBAL STYLE  (adapted from the reference Nature-style script)
# ----------------------------------------------------------------------------
USE_TEX = False   # set True only with a complete LaTeX (type1cm, amsmath, bm)

BWITH = 1.0                                   # spine linewidth
PALETTE = ['#6366F1', '#06B6D4', '#F59E0B',   # indigo, cyan, amber
           '#EF4444', '#10B981', '#8B5CF6']   # red, green, violet
MARKERS = ['o', 'P', 'H', 'D', 'v', 's']

plt.rcParams.update({
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'Nimbus Sans', 'DejaVu Sans'],
    'savefig.dpi': 300,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'DejaVu Sans',
    'mathtext.it': 'DejaVu Sans:italic',
    'mathtext.bf': 'DejaVu Sans:bold',
})
if USE_TEX:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"


def style_axis(ax):
    """Apply the reference tick/spine style to an axis."""
    for s in ax.spines.values():
        s.set_linewidth(BWITH)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='y', which='minor', length=0)
    ax.tick_params(axis='x', which='both', bottom=True, top=False,
                   labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', which='both', left=True, right=False,
                   labelleft=True, labelright=False)
    ax.tick_params(labelsize=12)


def panel(ax, letter, x=-0.06, y=0.99, fontsize=15):
    """Lowercase bold panel label, reference placement."""
    ax.set_title(letter, x=x, y=y, fontsize=fontsize)


OUT = str(FIGURES_DIR)
os.makedirs(OUT, exist_ok=True)


# ============================================================================
# FIGURE 1 — Calibration: two populations + area vs stiffness + fibrosis
# ============================================================================
def figure1():
    import json as _json
    cdata = _json.load(open(DATA_DIR / 'hepatocyte_complete_data.json'))
    tp = pd.read_csv(DATA_DIR / 'hepatocyte_two_populations.csv')
    Es = sorted(tp.E_kPa.unique())
    ts = sorted(tp.t_h.unique())

    fig = plt.figure(figsize=(13.5, 3.4))
    gs = plt.GridSpec(1, 4, wspace=0.55)

    # --- a: complete timecourse (mechanosensitive pop, 1 vs 23 kPa) ---
    ax = fig.add_subplot(gs[0, 0]); style_axis(ax)
    for key, col, lab in [("1_kPa", PALETTE[4], "1 kPa"),
                          ("23_kPa", PALETTE[3], "23 kPa")]:
        d = cdata["complete_timecourse"][key]
        tt = d["t_h"]
        hi = [x if x else np.nan for x in d["pop_high"]]
        ax.plot(tt, hi, marker=MARKERS[0], color=col, lw=2.5, ms=7, label=lab)
    ax.set_xlabel(r'time (h)', fontsize=13.5)
    ax.set_ylabel(r'nuclear area ($\mu$m$^2$)', fontsize=13.5)
    ax.legend(frameon=False, fontsize=10, loc='upper left',
              title='mechanosens.', title_fontsize=9)
    panel(ax, 'a')

    # --- b: tau SCALES WITH STIFFNESS (key recalibration result) ---
    ax = fig.add_subplot(gs[0, 1]); style_axis(ax)
    hep = PHENOTYPES["hepatocyte"]
    E_anchor = [1.0, 23.0]
    tau_anchor = [mvc.tau_of_E(1.0, hep), mvc.tau_of_E(23.0, hep)]
    E_line = np.logspace(np.log10(0.5), np.log10(23), 40)
    tau_line = [mvc.tau_of_E(e, hep) for e in E_line]
    ax.plot(E_line, tau_line, '-', color=PALETTE[0], lw=2, alpha=0.5, zorder=1)
    ax.plot(E_anchor, tau_anchor, 'o', color=PALETTE[0], ms=13, zorder=3)
    for e, t in zip(E_anchor, tau_anchor):
        ax.annotate(f'{t:.0f} h', (e, t), textcoords="offset points",
                    xytext=(9, -3), fontsize=11, fontweight='bold')
    ax.axhline(35, ls='--', color='#94A3B8', lw=1)
    ax.text(1.4, 37.5, 'old fixed (35 h)', fontsize=8.5, color='#64748B')
    ax.set_xscale('log'); ax.set_ylim(0, 92)
    ax.set_xlabel(r'stiffness $E$ (kPa)', fontsize=13.5)
    ax.set_ylabel(r'relaxation $\tau$ (h)', fontsize=13.5)
    panel(ax, 'b')

    # --- c: nuclear drive sigma(E) — motor vs saturating form (twin: YAP) ---
    ax = fig.add_subplot(gs[0, 2]); style_axis(ax)
    E_fine = np.logspace(np.log10(0.4), np.log10(30), 60)
    sig_c = fm.nuclear_stress_fast(E_fine, "hepatocyte")
    sig_stoch = np.array([mvc.nuclear_stress(e, hep, reps=6) for e in E_fine])
    ax.scatter(E_fine, sig_stoch, s=14, color='#94A3B8', alpha=0.7,
               label='motor', zorder=2)
    ax.plot(E_fine, sig_c, '-', color=PALETTE[2], lw=2.5, label='saturating',
            zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel(r'stiffness $E$ (kPa)', fontsize=13.5)
    ax.set_ylabel(r'nuclear drive $\sigma$', fontsize=13.5, color=PALETTE[2])
    ax.tick_params(axis='y', labelcolor=PALETTE[2])
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax2 = ax.twinx()
    ax2.tick_params(axis='x', which='minor', length=0)
    ax2.tick_params(axis='y', which='minor', length=0)
    yap = np.array([mvc.yap_nc_ratio(e, hep, reps=6) for e in E_fine])
    ax2.plot(E_fine, yap, '--', color=PALETTE[4], lw=2.5)
    ax2.set_ylabel(r'YAP N/C', fontsize=13.5, color=PALETTE[4])
    ax2.tick_params(axis='y', labelcolor=PALETTE[4], labelsize=12)
    panel(ax, 'c')

    # --- d: fibrosis trajectory F0->F4 (drive) with inset (YAP) ---
    ax = fig.add_subplot(gs[0, 3]); style_axis(ax)
    pred = mvc.fibrosis_prediction(hep, reps=6)
    stages = pred["stages"]
    xpos = np.arange(len(stages))
    ax.plot(xpos, pred["sigma"], marker=MARKERS[3], color=PALETTE[0], lw=2.5,
            ms=7)
    ax.set_xticks(xpos); ax.set_xticklabels(stages)
    ax.set_xlabel(r'fibrosis stage', fontsize=13.5)
    ax.set_ylabel(r'nuclear drive $\sigma$', fontsize=13.5)
    panel(ax, 'd')
    axins = inset_axes(ax, width="42%", height="42%", loc="upper left",
                       bbox_to_anchor=(0.13, 0.02, 1.0, 1.0),
                       bbox_transform=ax.transAxes, borderpad=0.9)
    axins.plot(xpos, pred["yap"], marker='o', ms=3, lw=1.8, color=PALETTE[4])
    axins.set_xticks(xpos); axins.set_xticklabels(stages, fontsize=7)
    axins.set_ylabel('YAP', labelpad=1, fontsize=9)
    axins.tick_params(labelsize=7)
    for s in axins.spines.values():
        s.set_linewidth(BWITH * 0.8)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.18, wspace=0.55)
    fig.savefig(f"{OUT}/Fig1_calibration.pdf", bbox_inches='tight')
    fig.savefig(f"{OUT}/Fig1_calibration.png", bbox_inches='tight')
    plt.close(fig)
    print("  Fig1 (calibration) done")


# ============================================================================
# FIGURE 2 — AI: posterior + identifiability + symbolic form
# ============================================================================
def figure2():
    post = json.load(open(RESULTS_DIR / 'hepatocyte_posterior.json'))["parameters"]
    sat = json.load(open(DATA_DIR / 'saturating_params.json'))

    fig = plt.figure(figsize=(13.5, 3.4))
    gs = plt.GridSpec(1, 3, wspace=0.48)

    # --- a: posterior of lamin A/C (timecourse; supersedes 1.66) ---
    ax = fig.add_subplot(gs[0, 0]); style_axis(ax)
    lam = post["laminAC"]
    rng = np.random.default_rng(0)
    samp = rng.normal(lam["mean"], lam["std"], 4000)
    ax.hist(samp, bins=32, color=PALETTE[5], alpha=0.75, density=True)
    ax.axvline(lam["mean"], color='#2d004d', lw=2)
    ax.axvspan(lam["ci95"][0], lam["ci95"][1], alpha=0.15, color=PALETTE[5])
    ax.axvline(1.66, ls='--', color='#94A3B8', lw=1.5)
    ax.text(1.66, ax.get_ylim()[1]*0.5, ' old (1.66,\n truncated)', fontsize=8,
            color='#64748B', ha='left')
    ax.set_xlabel(r'lamin A/C (posterior)', fontsize=13.5)
    ax.set_ylabel(r'density', fontsize=13.5)
    ax.text(0.97, 0.95, f"mean {lam['mean']:.2f}\n(complete\ntimecourse)",
            transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
            bbox=dict(boxstyle='round', fc='white', ec='#ccc', lw=0.8))
    panel(ax, 'a')

    # --- b: identifiability (lollipop) of timecourse params ---
    ax = fig.add_subplot(gs[0, 1]); style_axis(ax)
    keys = [k for k in ["laminAC", "A_max", "A0"] if k in post]
    labels = {'laminAC': 'lamin A/C', 'A_max': r'$A_{max}$', 'A0': r'$A_0$'}
    names = [labels[k] for k in keys]
    idents = [post[k]["identifiability"] for k in keys]
    ypos = np.arange(len(names))
    cols_b = [PALETTE[5], PALETTE[1], PALETTE[2]]
    for y, v, c in zip(ypos, idents, cols_b):
        ax.plot([0, v], [y, y], '-', color=c, lw=2.5, zorder=2)
        ax.plot(v, y, 'o', color=c, ms=13, zorder=3)
        ax.text(v + 0.03, y, f'{v:.2f}', va='center', fontsize=11, fontweight='bold')
    ax.axvline(0.8, ls='--', color='#94A3B8', lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.4); ax.set_xlim(0, 1.15)
    ax.set_xlabel(r'identifiability', fontsize=13.5)
    panel(ax, 'b')

    # --- c: functional form comparison (lollipop, saturating highlighted) ---
    ax = fig.add_subplot(gs[0, 2]); style_axis(ax)
    forms = ['linear', 'power', 'log', 'saturating']
    r2 = [0.70, 0.90, 0.96, 0.982]
    ypos = np.arange(len(forms))
    for y, (fm_name, v) in enumerate(zip(forms, r2)):
        hi = (fm_name == 'saturating')
        c = PALETTE[2] if hi else '#94A3B8'
        ax.plot([0.6, v], [y, y], '-', color=c, lw=2.5 if hi else 1.8, zorder=2)
        ax.plot(v, y, 'o', color=c, ms=14 if hi else 10, zorder=3)
        ax.text(v + 0.006, y, f'{v:.2f}', va='center', fontsize=10,
                fontweight='bold' if hi else 'normal')
    ax.set_yticks(ypos); ax.set_yticklabels(forms)
    ax.set_xlim(0.6, 1.03); ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel(r'$R^2$ to motor', fontsize=13.5)
    hp = sat['hepatocyte']
    ax.text(0.62, 3.2, r'$\sigma = V_{max}\,E/(K+E)$'
            + f"\nVmax={hp['Vmax']:.0f}, K={hp['K']:.1f}",
            ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='#FEF3C7', ec=PALETTE[2], lw=1))
    panel(ax, 'c')

    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.20, wspace=0.48)
    fig.savefig(f"{OUT}/Fig2_inference.pdf", bbox_inches='tight')
    fig.savefig(f"{OUT}/Fig2_inference.png", bbox_inches='tight')
    plt.close(fig)
    print("  Fig2 (AI inference) done")


# ============================================================================
# FIGURE 3 — Application: clinical mapping + function + drug screen
# ============================================================================
def figure3():
    import pharmacology as ph
    hep = PHENOTYPES["hepatocyte"]

    fig = plt.figure(figsize=(13.5, 3.4))
    gs = plt.GridSpec(1, 3, wspace=0.5)

    # --- a: hepatocyte function vs stiffness (markers) ---
    ax = fig.add_subplot(gs[0, 0]); style_axis(ax)
    E = np.linspace(1, 40, 40)
    f = [ph.hepatocyte_function(e, hep, use_fast=False) for e in E]
    ax.plot(E, f, '-', color=PALETTE[0], lw=3, label='functional index')
    for stg, ekpa, mk, col in [('F0', 2.5, 'o', PALETTE[4]),
                                ('F2', 9.5, 'D', PALETTE[2]),
                                ('F4', 26, 'v', PALETTE[3])]:
        fi = ph.hepatocyte_function(ekpa, hep, use_fast=False)
        ax.plot(ekpa, fi, mk, color=col, ms=9, zorder=4)
        ax.annotate(stg, (ekpa, fi), textcoords="offset points",
                    xytext=(6, 8), fontsize=10, fontweight='bold', color=col)
    ax.set_xlabel(r'stiffness $E$ (kPa)', fontsize=13.5)
    ax.set_ylabel(r'hepatocyte function', fontsize=13.5)
    panel(ax, 'a')

    # --- b: clinical trajectory — YAP & function vs stiffness (twin) ---
    ax = fig.add_subplot(gs[0, 1]); style_axis(ax)
    yap = [mvc.yap_nc_ratio(e, hep, reps=5) for e in E]
    ax.plot(E, yap, '-', color=PALETTE[0], lw=3)
    ax.set_xlabel(r'liver stiffness (kPa)', fontsize=13.5)
    ax.set_ylabel(r'YAP N/C', fontsize=13.5, color=PALETTE[0])
    ax.tick_params(axis='y', labelcolor=PALETTE[0])
    for cut in [6, 8, 10, 14]:
        ax.axvline(cut, ls=':', color='#CBD5E1', lw=1.2)
    ax2 = ax.twinx()
    ax2.tick_params(axis='x', which='minor', length=0)
    ax2.tick_params(axis='y', which='minor', length=0)
    ax2.plot(E, f, '--', color=PALETTE[3], lw=3)
    ax2.set_ylabel(r'function', fontsize=13.5, color=PALETTE[3])
    ax2.tick_params(axis='y', labelcolor=PALETTE[3], labelsize=12)
    panel(ax, 'b')

    # --- c: drug screen at F4 (lollipop, colored by axis; all 3 axes shown) ---
    ax = fig.add_subplot(gs[0, 2]); style_axis(ax)
    all_rows = ph.screen_drugs(ph=hep, E=26.0, reps=4)
    # ensure at least one metabolic + one signaling drug are shown alongside
    # the top mechanical ones, so all three axes are visible
    mech = [r for r in all_rows if r['axis'] == 'mechanical'][:5]
    signaling = [r for r in all_rows if r['axis'] == 'growth-factor/signaling'][:2]
    metabolic = [r for r in all_rows if r['axis'] == 'metabolic'][:2]
    rows = mech + signaling + metabolic
    rows = sorted(rows, key=lambda r: r['yap_drive_removed'])   # ascending for barh
    names = [r['drug'].split(' (')[0] for r in rows]
    scores = [r['yap_drive_removed'] * 100 for r in rows]
    axis_col = {'mechanical': PALETTE[3], 'growth-factor/signaling': PALETTE[1],
                'metabolic': PALETTE[2]}
    cols = [axis_col.get(r['axis'], '#CBD5E1') for r in rows]
    ypos = np.arange(len(names))
    for y, v, c in zip(ypos, scores, cols):
        ax.plot([0, v], [y, y], '-', color=c, lw=2, zorder=2, alpha=0.7)
        ax.plot(v, y, 'o', color=c, ms=10, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(-2, max(scores) + 8)
    ax.set_xlabel('YAP drive removed (%)', fontsize=13.5)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[3],
                  label='mechanical', ms=9),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[1],
                  label='signaling', ms=9),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[2],
                  label='metabolic', ms=9)]
    ax.legend(handles=leg, frameon=False, fontsize=8.5, loc='lower right')
    panel(ax, 'c')

    fig.subplots_adjust(left=0.07, right=0.95, top=0.88, bottom=0.18, wspace=0.5)
    fig.savefig(f"{OUT}/Fig3_application.pdf", bbox_inches='tight')
    fig.savefig(f"{OUT}/Fig3_application.png", bbox_inches='tight')
    plt.close(fig)
    print("  Fig3 (application) done")


if __name__ == "__main__":
    print("Generating Nature-style figures...")
    figure1()
    figure2()
    figure3()
    print(f"Done. Figures in {OUT}/ (pdf + png, 300 dpi).")
