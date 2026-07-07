"""
================================================================================
 2_validate_predictions.py — model predictions vs. observed transcriptomics
================================================================================

Closes the transcriptomic validation loop. The R pipeline (scripts 0–1_3)
produces `conserved_genes.tsv`: for each panel gene, the empirically-observed
fibrosis-stage response (direction up/down and best-fit shape linear/power/
sigmoid) that is concordant across the three GEO cohorts.

This script cross-checks that OBSERVED table against the model's PREDICTIONS,
read live from `gene_module.py` — so the prediction and its test can never drift
apart. The response shape and direction were assigned from each gene's
mechanotransduction role *before* looking at RNA-seq, so this is a falsifiable
test, not a fit.

Reports, per gene and in aggregate:
  * DIRECTION agreement  (predicted up/down  vs observed dominant_dir)
  * SHAPE agreement       (predicted linear/power/sigmoid vs observed best-fit)
  * per-gene concordance flags
  * aggregate concordance rates + a binomial test against chance for direction,
    and a shape-agreement rate with its own binomial test (chance = 1/3).

Usage:
    python transcriptomics/2_validate_predictions.py \
        --conserved transcriptomics/results/model_fits/conserved_genes.tsv \
        --phenotype hepatocyte \
        --out transcriptomics/results/validation_results

Outputs {out}.tsv (per-gene) and {out}.json (summary + statistics).
================================================================================
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import stats

import gene_module as gm


# shape-name harmonization: model (gene_module) <-> R pipeline (1_3_Fit_filter.R)
_SHAPE_MODEL_TO_OBS = {
    "linear": "linear",
    "weak_power": "power_law",
    "sigmoid": "sigmoid",
}

# synonym groups: symbols that refer to the same gene are collapsed to one key,
# so the model panel and the observed table match regardless of naming vintage.
_SYNONYMS = {
    "CCN2": "CTGF", "CTGF": "CTGF",
    "CCN1": "CYR61", "CYR61": "CYR61",
    "MRTFA": "MKL1", "MKL1": "MKL1", "MRTF-A": "MKL1",
    "TAZ": "WWTR1", "WWTR1": "WWTR1",
    "SM22": "TAGLN", "TAGLN": "TAGLN",
}


def _norm(symbol):
    """Normalize a gene symbol for matching: drop any parenthetical annotation
    (e.g. 'ACTA2 (α-SMA)' -> 'ACTA2'), uppercase, then collapse known synonyms
    to a canonical key so the model panel and the observed table align."""
    import re
    base = re.sub(r"\s*\(.*?\)\s*", "", str(symbol)).strip().upper()
    return _SYNONYMS.get(base, base)


# ---------------------------------------------------------------------------
# Model predictions (read live from gene_module)
# ---------------------------------------------------------------------------
def model_predictions(phenotype="hepatocyte", extended=False):
    """Return {NORM_KEY: {"symbol","pred_dir","pred_shape","role"}} from the model.

    If extended=True and phenotype is hepatocyte, use the rule-based extended
    hepatic panel (hepatic_panel.py): shape from functional category, direction
    from role — both set from prior biology, independent of the observed data.
    """
    if extended and phenotype == "hepatocyte":
        try:
            import hepatic_panel as hp
            panel = hp.hepatic_panel()
            dirs = hp.hepatic_directions()
            cats = hp.hepatic_categories()
            preds = {}
            for sym, g in panel.items():
                preds[_norm(sym)] = dict(
                    symbol=sym, pred_dir=dirs[sym],
                    pred_shape=_SHAPE_MODEL_TO_OBS.get(g.shape, g.shape),
                    role=cats[sym].replace("_", " "))
            return preds
        except Exception as exc:
            print(f"[warn] extended panel unavailable ({exc}); using core panel")

    table = gm.response_shape_table(phenotype)
    inverse = gm._inverse_set(phenotype)
    genes = gm.genes_for(phenotype)
    sym_to_key = {g.symbol: k for k, g in genes.items()}
    preds = {}
    for row in table:
        sym = row["gene"]
        key = sym_to_key.get(sym, sym)
        pred_dir = "down" if key in inverse else "up"
        preds[_norm(sym)] = dict(
            symbol=sym,
            pred_dir=pred_dir,
            pred_shape=_SHAPE_MODEL_TO_OBS.get(row["shape"], row["shape"]),
            role=row["role"],
        )
    return preds


# ---------------------------------------------------------------------------
# Observed table (from the R pipeline)
# ---------------------------------------------------------------------------
def load_observed(path):
    """Load conserved_genes.tsv and reduce to one observed direction + shape per
    gene. Direction = dominant_dir. Observed shape = majority best-fit model
    across the per-dataset `{dataset}_model` columns (ties -> highest mean R²)."""
    df = pd.read_csv(path, sep="\t")
    sym_col = "Gene_symbol" if "Gene_symbol" in df.columns else df.columns[0]
    model_cols = [c for c in df.columns if c.endswith("_model")]
    r2_cols = [c for c in df.columns if c.endswith("_r2")]

    obs = {}
    for _, r in df.iterrows():
        sym = r[sym_col]
        # observed direction
        obs_dir = r["dominant_dir"] if "dominant_dir" in df.columns else None
        # observed shape: majority vote across datasets, tie-broken by mean R²
        shapes = [str(r[c]) for c in model_cols if pd.notna(r[c])]
        if shapes:
            vals, counts = np.unique(shapes, return_counts=True)
            top = vals[counts == counts.max()]
            if len(top) == 1:
                obs_shape = top[0]
            else:  # tie -> pick the shape with the highest mean R²
                best, best_r2 = top[0], -1
                for s in top:
                    r2s = [r[c] for c, m in zip(r2_cols, model_cols)
                           if str(r[m]) == s and pd.notna(r[c])]
                    m = np.mean(r2s) if r2s else 0
                    if m > best_r2:
                        best, best_r2 = s, m
                obs_shape = best
        else:
            obs_shape = None
        mean_r2 = np.nanmean([r[c] for c in r2_cols]) if r2_cols else np.nan
        obs[_norm(sym)] = dict(symbol=sym, obs_dir=obs_dir,
                               obs_shape=obs_shape, mean_r2=mean_r2)
    return obs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(conserved_path, phenotype="hepatocyte", extended=False):
    preds = model_predictions(phenotype, extended=extended)
    obs = load_observed(conserved_path)

    rows = []
    for nkey, p in preds.items():
        o = obs.get(nkey)
        if o is None or o["obs_dir"] is None:
            rows.append(dict(gene=p["symbol"], tested=False,
                             pred_dir=p["pred_dir"], pred_shape=p["pred_shape"],
                             role=p["role"]))
            continue
        dir_ok = (p["pred_dir"] == o["obs_dir"])
        shape_ok = (p["pred_shape"] == o["obs_shape"])
        rows.append(dict(
            gene=p["symbol"], obs_gene=o["symbol"], tested=True, role=p["role"],
            pred_dir=p["pred_dir"], obs_dir=o["obs_dir"], dir_match=dir_ok,
            pred_shape=p["pred_shape"], obs_shape=o["obs_shape"],
            shape_match=shape_ok, mean_r2=round(float(o["mean_r2"]), 3)
            if not np.isnan(o["mean_r2"]) else None,
        ))
    df = pd.DataFrame(rows)

    tested = df[df["tested"]]
    n = len(tested)
    n_dir = int(tested["dir_match"].sum()) if n else 0
    n_shape = int(tested["shape_match"].sum()) if n else 0

    # direction: binomial test vs chance = 0.5
    dir_p = stats.binomtest(n_dir, n, 0.5, alternative="greater").pvalue if n else None
    # shape: binomial test vs chance = 1/3 (three shape classes)
    shape_p = stats.binomtest(n_shape, n, 1/3, alternative="greater").pvalue if n else None

    summary = dict(
        phenotype=phenotype,
        n_panel_genes=len(preds),
        n_tested=n,
        n_untested=len(preds) - n,
        direction=dict(
            n_correct=n_dir, rate=round(n_dir / n, 3) if n else None,
            chance=0.5, p_value=dir_p),
        shape=dict(
            n_correct=n_shape, rate=round(n_shape / n, 3) if n else None,
            chance=round(1/3, 3), p_value=shape_p),
        both_correct=int((tested["dir_match"] & tested["shape_match"]).sum())
        if n else 0,
    )
    return df, summary


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Validate model predictions vs "
                                             "observed transcriptomics")
    ap.add_argument("--conserved", required=True,
                    help="conserved_genes.tsv from 1_3_Fit_filter.R")
    ap.add_argument("--phenotype", default="hepatocyte")
    ap.add_argument("--extended", action="store_true", help="use the extended hepatic panel")
    ap.add_argument("--out", default="transcriptomics/results/validation_results")
    args = ap.parse_args()

    df, summary = validate(args.conserved, args.phenotype, extended=args.extended)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}.tsv", sep="\t", index=False)
    Path(f"{out}.json").write_text(json.dumps(summary, indent=2))

    # console report
    print("=" * 68)
    print(f"  TRANSCRIPTOMIC VALIDATION — {summary['phenotype']}")
    print("=" * 68)
    print(f"  panel genes: {summary['n_panel_genes']}  |  tested "
          f"(concordant across cohorts): {summary['n_tested']}")
    d, s = summary["direction"], summary["shape"]
    print(f"\n  DIRECTION (up/down):  {d['n_correct']}/{summary['n_tested']} "
          f"correct = {d['rate']}   (chance {d['chance']}, "
          f"p = {d['p_value']:.2e})" if summary["n_tested"] else "  no tested genes")
    print(f"  SHAPE (lin/pow/sig):  {s['n_correct']}/{summary['n_tested']} "
          f"correct = {s['rate']}   (chance {s['chance']}, "
          f"p = {s['p_value']:.2e})" if summary["n_tested"] else "")
    print(f"  both correct:         {summary['both_correct']}/{summary['n_tested']}")
    print(f"\n  wrote {out}.tsv and {out}.json")
    print("=" * 68)


if __name__ == "__main__":
    main()
