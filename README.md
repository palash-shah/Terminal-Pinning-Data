RUN INSTRUCTIONS
============================================


STEP 1: Collect the panel
=========================

  python3 collect_polymarket_panel.py --target 50

Optional smoke test first:
  python3 collect_polymarket_panel.py --probe

You should see output like:
    fidelity=1 min: status=200, n=0
    fidelity=10 min: status=200, n=0
    fidelity=60 min: status=200, n=120
  ...confirming the cascade is doing its job.

Tunables (defaults shown):
  --target 30           target panel size
  --scan 8000           how deep to page through closed markets
  --min-volume 500000   $500K floor on cumulative volume
  --min-ticks 50        minimum ticks at finest available fidelity
  --min-interior 0.20   minimum fraction of ticks in [0.05, 0.95]
  --max-per-topic 6     dedup cap per topic
  --max-per-week 10     dedup cap per resolution week

Expected runtime: 5-15 minutes.
Output: polymarket_panel/ folder with manifest.csv and ~30-50 csv files.

The manifest now also records fidelity_used_min for each market, so the
analysis can adapt.


STEP 2: Run the analysis
========================

  pip install --upgrade numpy scipy matplotlib pandas
  python3 analyze_panel.py --panel-dir polymarket_panel --out out

The analysis automatically adapts the (t, t+Delta) pair tolerance to the
local tick spacing, so 12-hour-fidelity markets still produce pairs.

Expected runtime: 1-3 minutes.
Output: out/ folder with:
  summary.json           per-market stats (beta_0, sigma_J, lambda, etc.)
  panel_table.tex        LaTeX rows for Table 1
  sigma_pooled.tex       LaTeX rows for Table 2 (Sigma* convergence)
  sigma_pooled.json      raw cell ratios with se's and n's
  smile_table.tex        LaTeX rows for Table 3 (smile comparison)
  smile_rows.json        raw smile data
  paths.png              all market price paths
  logit_validation.png   bridge model overlay
  smiles.png             smile comparison
  pinning_rate.png       pooled log-log P(1-P) scatter
